from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage import io

from .dataset import iter_records, load_scan_from_zip, make_grouped_folds
from .evaluation import calculate_metrics, competition_proxy_score, evaluate_feature_map_metrics, profile_scores, summarize_rows, write_rows
from .scanners import render_with_api, write_api_config
from .submission import to_gray_png
from .generators import HYPOTHESIS_CONFIGS, physics_guided_phantom


@dataclass(frozen=True)
class Candidate:
    name: str
    config: dict[str, float | bool]


def generate_candidate_configs(seed: int = 0, random_count: int = 48) -> list[Candidate]:
    rng = np.random.default_rng(seed)
    candidates: list[Candidate] = []
    anchors = {
        "H61": HYPOTHESIS_CONFIGS["H61_api_low_depth_prelim"],
        "H67": HYPOTHESIS_CONFIGS["H67_coarse_to_fine_crisp"],
        "H68": HYPOTHESIS_CONFIGS["H68_layer_map_prior"],
        "H11": HYPOTHESIS_CONFIGS["H11_low_depth_comp"],
    }
    for anchor_name, cfg in anchors.items():
        tuned = dict(cfg)
        tuned.setdefault("oac_percentile", 70.0)
        tuned.setdefault("base_energy_mix", 0.55)
        tuned.setdefault("texture_sigma_scale", 1.0)
        tuned.setdefault("lateral_smooth", 0.8)
        candidates.append(Candidate(f"O_anchor_{anchor_name}", tuned))

    base = dict(HYPOTHESIS_CONFIGS["H61_api_low_depth_prelim"])
    grid = [
        ("depth", "depth_compensation", [-1.40, -1.10, -0.80, -0.50, 0.00, 0.60]),
        ("oac", "oac_weight", [0.80, 1.00, 1.20, 1.60, 2.00, 2.40]),
        ("density", "density_power", [0.75, 0.90, 1.05, 1.20, 1.35]),
        ("energy", "energy_oac_scale", [6.0, 10.0, 16.0, 24.0, 34.0]),
        ("percentile", "oac_percentile", [58.0, 65.0, 70.0, 78.0, 84.0]),
        ("mix", "base_energy_mix", [0.30, 0.40, 0.55, 0.70]),
        ("texture", "texture_sigma_scale", [0.55, 0.70, 0.90, 1.15]),
        ("smooth", "lateral_smooth", [0.28, 0.55, 0.90, 1.20]),
        ("band", "band_boost", [0.0, 0.12, 0.30, 0.50]),
    ]
    for label, key, values in grid:
        for value in values:
            cfg = dict(base)
            cfg[key] = value
            candidates.append(Candidate(f"O_{label}_{value}", cfg))

    for idx in range(random_count):
        cfg = {
            "density_power": float(rng.uniform(0.72, 1.38)),
            "depth_compensation": float(rng.uniform(-1.50, 1.40)),
            "oac_weight": float(rng.uniform(0.70, 2.60)),
            "texture_weight": float(rng.uniform(0.12, 0.42)),
            "energy_oac_scale": float(rng.uniform(5.0, 38.0)),
            "band_boost": float(rng.choice([0.0, rng.uniform(0.04, 0.55)])),
            "lateral_smooth": float(rng.uniform(0.25, 1.35)),
            "oac_percentile": float(rng.uniform(55.0, 86.0)),
            "base_energy_mix": float(rng.uniform(0.25, 0.80)),
            "texture_sigma_scale": float(rng.uniform(0.50, 1.25)),
        }
        candidates.append(Candidate(f"O_random_{idx:02d}", cfg))
    return candidates


def run_candidate_search(
    zip_path: str | Path,
    out_dir: str | Path,
    folds: int = 3,
    max_per_fold: int = 1,
    scatterers_count: int = 300_000,
    random_count: int = 48,
    seed: int = 23,
    include_maps: bool = True,
    api_key_file: str | Path | None = None,
    poll_interval_seconds: float = 10.0,
    max_polls: int = 60,
    skip_existing: bool = True,
) -> tuple[Path, Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = generate_candidate_configs(seed=seed, random_count=random_count)
    candidate_map = {candidate.name: candidate.config for candidate in candidates}
    (out_dir / "candidate_configs.json").write_text(json.dumps(candidate_map, indent=2, sort_keys=True))
    api_config_path = write_api_config(out_dir / "Configuration_api.ini", scatterers_count=scatterers_count)

    records = [r for r in iter_records(zip_path) if r.modality == "png"]
    folded = make_grouped_folds(records, folds=folds)
    rows: list[dict[str, float | str]] = []

    with tempfile.TemporaryDirectory(prefix="synthoct-opt-") as tmp_name:
        tmp = Path(tmp_name)
        for fold_idx in range(folds):
            selected = sorted(folded[fold_idx], key=lambda r: r.archive_path)[:max_per_fold]
            for sample_idx, record in enumerate(selected):
                ref_path = tmp / f"fold{fold_idx}_sample{sample_idx}_ref.png"
                arr = load_scan_from_zip(zip_path, record.archive_path)
                io.imsave(ref_path, np.clip(arr, 0, 255).astype(np.uint8))
                for candidate in candidates:
                    sample_dir = out_dir / "samples" / f"fold_{fold_idx}" / f"sample_{sample_idx}" / candidate.name
                    phantom_path = sample_dir / "phantom.txt"
                    api_scan_path = sample_dir / "api_scan.png"
                    pred_path = sample_dir / "api_scan_gray.png"
                    start = time.perf_counter()
                    physics_guided_phantom(
                        ref_path,
                        phantom_path,
                        seed=seed + fold_idx * 100 + sample_idx,
                        scatterers_count=scatterers_count,
                        **candidate.config,
                    )
                    generation_seconds = time.perf_counter() - start
                    if skip_existing and api_scan_path.exists() and pred_path.exists():
                        request_id = "existing"
                        render_seconds = 0.0
                        poll_count = 0
                    else:
                        request_id, rendered_path, render_seconds, poll_count = render_with_api(
                            phantom_path,
                            api_config_path,
                            api_scan_path,
                            api_key_file=api_key_file,
                            poll_interval_seconds=poll_interval_seconds,
                            max_polls=max_polls,
                        )
                        to_gray_png(rendered_path, pred_path)

                    row: dict[str, float | str] = {
                        "fold": fold_idx,
                        "sample": sample_idx,
                        "method": candidate.name,
                        "archive_path": record.archive_path,
                        "generation_seconds": generation_seconds,
                        "render_seconds": render_seconds,
                        "poll_count": poll_count,
                        "request_id": request_id,
                    }
                    for key, value in calculate_metrics(ref_path, pred_path, include_lpips=False).items():
                        row[f"Struct_{key}"] = value
                    row.update(profile_scores(ref_path, pred_path))
                    if include_maps:
                        row.update(evaluate_feature_map_metrics(ref_path, pred_path, sample_dir / "ref_maps", sample_dir / "pred_maps"))
                    row["CompetitionProxy"] = competition_proxy_score(row)
                    rows.append(row)

    detail_path = out_dir / "optimizer_detail.csv"
    summary_path = out_dir / "optimizer_summary.csv"
    write_rows(detail_path, rows)
    summary = summarize_rows(rows)
    write_rows(summary_path, summary)
    best = summary[0]
    best_config_path = out_dir / "best_config.json"
    best_config_path.write_text(
        json.dumps(
            {"method": best["method"], "config": candidate_map[str(best["method"])]},
            indent=2,
            sort_keys=True,
        )
    )
    return detail_path, summary_path, best_config_path


def plot_optimizer_top(summary_path: str | Path, output_path: str | Path, top_n: int = 20) -> Path:
    import csv
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(Path(summary_path).open()))[:top_n]
    labels = [row["method"].replace("O_", "") for row in rows]
    scores = [float(row["CompetitionProxy_mean"]) for row in rows]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(rows)), scores, color="#0f766e")
    plt.xticks(range(len(rows)), labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Competition proxy score")
    plt.title("Top Physics-Optimizer Candidates")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path
