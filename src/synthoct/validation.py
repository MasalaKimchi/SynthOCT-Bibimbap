from __future__ import annotations

import csv
import tempfile
import time
from pathlib import Path

import numpy as np
from skimage import io

from .generators import (
    HYPOTHESIS_CONFIGS,
    HYPOTHESIS_WAVES,
    PROMISING_PIPELINE_CONFIGS,
    PROMISING_PIPELINE_WAVES,
    VISUAL_PIPELINE_CONFIGS,
    heuristic_layer_phantom,
    hypothesis_phantom,
    official_baseline_phantom,
    pipeline_phantom,
    physics_guided_phantom,
)
from .dataset import ScanRecord, iter_records, load_scan_from_zip, make_grouped_folds
from .evaluation import (
    calculate_metrics,
    competition_proxy_score,
    composite_score,
    evaluate_feature_map_metrics,
    profile_scores,
    summarize_challenge_metrics,
    summarize_rows,
    summarize_sample_wins,
    write_rows,
)
from .scanners import render_with_api, write_api_config
from .submission import to_gray_png

HYPOTHESIS_METHODS = ("H0_official", *HYPOTHESIS_CONFIGS.keys())
PROMISING_PIPELINE_METHODS = (*PROMISING_PIPELINE_CONFIGS.keys(), *VISUAL_PIPELINE_CONFIGS.keys())
METHODS = (
    "official",
    "heuristic",
    "physics-guided",
    *HYPOTHESIS_METHODS,
    *PROMISING_PIPELINE_METHODS,
)
METHOD_WAVES = tuple({**HYPOTHESIS_WAVES, **PROMISING_PIPELINE_WAVES}.keys())


def resolve_method_wave(wave: str | None, methods: list[str] | None = None) -> list[str] | None:
    if wave is None:
        return methods
    if methods is not None:
        raise ValueError("Pass either explicit methods or a method wave, not both.")
    waves = {**HYPOTHESIS_WAVES, **PROMISING_PIPELINE_WAVES}
    return list(waves[wave])


def make_folds(records: list[ScanRecord], folds: int = 3) -> dict[int, list[ScanRecord]]:
    return make_grouped_folds(records, folds=folds)


def write_reference_scan(zip_path: str | Path, record: ScanRecord, out_path: Path) -> Path:
    arr = load_scan_from_zip(zip_path, record.archive_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if arr.max(initial=0) <= 1.0:
        arr = arr * 255
    io.imsave(out_path, np.clip(arr, 0, 255).astype(np.uint8))
    return out_path


def run_method(method: str, ref_path: Path, phantom_path: Path, scatterers_count: int, seed: int) -> Path:
    if method in HYPOTHESIS_METHODS:
        return hypothesis_phantom(ref_path, phantom_path, method, seed=seed, scatterers_count=scatterers_count)
    if method in PROMISING_PIPELINE_METHODS:
        return pipeline_phantom(ref_path, phantom_path, method, seed=seed, scatterers_count=scatterers_count)
    if method == "official":
        return official_baseline_phantom(phantom_path, seed=seed, scatterers_count=scatterers_count)
    if method == "heuristic":
        return heuristic_layer_phantom(ref_path, phantom_path, seed=seed, scatterers_count=scatterers_count)
    if method == "physics-guided":
        return physics_guided_phantom(ref_path, phantom_path, seed=seed, scatterers_count=scatterers_count)
    raise ValueError(f"Unknown method: {method}")


def run_internal_validation(
    zip_path: str | Path,
    out_dir: str | Path,
    methods: list[str] | None = None,
    folds: int = 3,
    max_per_fold: int = 4,
    scatterers_count: int = 300_000,
    include_maps: bool = True,
    include_lpips: bool = False,
    seed: int = 7,
    api_key_file: str | Path | None = None,
    poll_interval_seconds: float = 10.0,
    max_polls: int = 60,
    skip_existing: bool = True,
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = methods or list(METHODS)
    api_config_path = write_api_config(out_dir / "Configuration_api.ini", scatterers_count=scatterers_count)

    records = [r for r in iter_records(zip_path) if r.modality == "png"]
    folded = make_folds(records, folds=folds)
    rows: list[dict[str, float | str]] = []
    work_root = out_dir / "samples"
    with tempfile.TemporaryDirectory(prefix="synthoct-val-") as tmp:
        tmp = Path(tmp)
        for fold_idx in range(folds):
            selected = sorted(folded[fold_idx], key=lambda r: r.archive_path)[:max_per_fold]
            for sample_idx, record in enumerate(selected):
                ref_path = write_reference_scan(zip_path, record, tmp / f"fold{fold_idx}_sample{sample_idx}_ref.png")
                for method in methods:
                    sample_dir = work_root / f"fold_{fold_idx}" / f"sample_{sample_idx}" / method
                    phantom_path = sample_dir / "phantom.txt"
                    api_scan_path = sample_dir / "api_scan.png"
                    pred_path = sample_dir / "api_scan_gray.png"
                    start = time.perf_counter()
                    run_method(method, ref_path, phantom_path, scatterers_count=scatterers_count, seed=seed + fold_idx * 100 + sample_idx)
                    generation_seconds = time.perf_counter() - start
                    if skip_existing and api_scan_path.exists() and pred_path.exists():
                        request_id = "existing"
                        render_seconds = 0.0
                        poll_count = 0
                    else:
                        render_start = time.perf_counter()
                        request_id, rendered_path, render_seconds, poll_count = render_with_api(
                            phantom_path,
                            api_config_path,
                            api_scan_path,
                            api_key_file=api_key_file,
                            poll_interval_seconds=poll_interval_seconds,
                            max_polls=max_polls,
                        )
                        render_seconds = time.perf_counter() - render_start if render_seconds == 0.0 else render_seconds
                        to_gray_png(rendered_path, pred_path)

                    row: dict[str, float | str] = {
                        "fold": fold_idx,
                        "sample": sample_idx,
                        "method": method,
                        "archive_path": record.archive_path,
                        "generation_seconds": generation_seconds,
                        "render_seconds": render_seconds,
                        "poll_count": poll_count,
                        "request_id": request_id,
                    }
                    for key, value in calculate_metrics(ref_path, pred_path, include_lpips=include_lpips).items():
                        row[f"Struct_{key}"] = value
                    row.update(profile_scores(ref_path, pred_path))
                    if include_maps:
                        row.update(evaluate_feature_map_metrics(ref_path, pred_path, sample_dir / "ref_maps", sample_dir / "pred_maps"))
                    row["Composite"] = composite_score(row)
                    row["CompetitionProxy"] = competition_proxy_score(row)
                    rows.append(row)

    detail_path = out_dir / "internal_validation_detail.csv"
    summary_path = out_dir / "internal_validation_summary.csv"
    challenge_summary_path = out_dir / "challenge_metrics_summary.csv"
    wins_path = out_dir / "hypothesis_wins.csv"
    write_rows(detail_path, rows)
    summary_rows = summarize_rows(rows)
    write_rows(summary_path, summary_rows)
    write_rows(challenge_summary_path, summarize_challenge_metrics(rows))
    write_rows(wins_path, summarize_sample_wins(rows, score_key="Struct_MS-SSIM"))
    return detail_path, summary_path


def plot_hypothesis_progress(summary_path: str | Path, output_path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(Path(summary_path).open()))
    h_rows = [r for r in rows if str(r["method"]).startswith("H")]
    h_rows.sort(key=lambda r: int(str(r["method"]).split("_", 1)[0][1:]))
    labels = [r["method"].split("_", 1)[0] for r in h_rows]
    scores = [float(r.get("Struct_MS-SSIM_mean") or r.get("Struct_SSIM_mean")) for r in h_rows]
    names = [r["method"].split("_", 1)[1].replace("_", " ") for r in h_rows]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5.8))
    colors = ["#64748b"] + ["#2563eb" if i == int(np.argmax(scores)) else "#0f766e" for i in range(1, len(scores))]
    plt.plot(labels, scores, color="#111827", linewidth=1.5, alpha=0.55)
    plt.scatter(labels, scores, s=88, c=colors, zorder=3)
    for label, score, name in zip(labels, scores, names):
        plt.text(label, score + 0.003, name[:18], ha="center", va="bottom", fontsize=8, rotation=20)
    plt.ylabel("MS-SSIM")
    plt.xlabel("Hypothesis iteration")
    plt.title("SynthOCT Internal Validation Progression")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path
