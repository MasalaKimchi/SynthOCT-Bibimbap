from __future__ import annotations

import csv
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from skimage import io

from .generators import (
    HYPOTHESIS_CONFIGS,
    HYPOTHESIS_WAVES,
    heuristic_layer_phantom,
    hypothesis_phantom,
    official_baseline_phantom,
    physics_guided_phantom,
)
from .dataset import ScanRecord, iter_records, load_scan_from_zip
from .scanners import render_with_api, write_api_config
from .submission.preliminary import _to_gray_png
from .evaluation import calculate_metrics
from .features import calculate_oac, calculate_speckle_contrast_map, generate_maps, load_and_linearize_image, load_scan

HYPOTHESIS_METHODS = ("H0_official", *HYPOTHESIS_CONFIGS.keys())
METHODS = (
    "official",
    "heuristic",
    "physics-guided",
    *HYPOTHESIS_METHODS,
)
METHOD_WAVES = tuple(HYPOTHESIS_WAVES.keys())


def resolve_method_wave(wave: str | None, methods: list[str] | None = None) -> list[str] | None:
    if wave is None:
        return methods
    if methods is not None:
        raise ValueError("Pass either explicit methods or a method wave, not both.")
    return list(HYPOTHESIS_WAVES[wave])


def make_folds(records: list[ScanRecord], folds: int = 3) -> dict[int, list[ScanRecord]]:
    groups: dict[str, list[ScanRecord]] = defaultdict(list)
    for record in records:
        groups[f"{record.sex}/{record.age_band}/{record.body_site}/{record.subject_key}"].append(record)
    folded: dict[int, list[ScanRecord]] = {i: [] for i in range(folds)}
    for idx, key in enumerate(sorted(groups)):
        folded[idx % folds].extend(groups[key])
    return folded


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
    if method == "official":
        return official_baseline_phantom(phantom_path, seed=seed, scatterers_count=scatterers_count)
    if method == "heuristic":
        return heuristic_layer_phantom(ref_path, phantom_path, seed=seed, scatterers_count=scatterers_count)
    if method == "physics-guided":
        return physics_guided_phantom(ref_path, phantom_path, seed=seed, scatterers_count=scatterers_count)
    raise ValueError(f"Unknown method: {method}")


def profile_scores(ref_path: Path, pred_path: Path) -> dict[str, float]:
    ref = load_scan(ref_path)
    pred = load_scan(pred_path)
    if ref.shape != pred.shape:
        pred = resize_like(pred, ref.shape)
    ref_depth = ref.mean(axis=1)
    pred_depth = pred.mean(axis=1)
    ref_lat = ref.mean(axis=0)
    pred_lat = pred.mean(axis=0)
    depth_corr = safe_corr(ref_depth, pred_depth)
    lateral_corr = safe_corr(ref_lat, pred_lat)

    ref_oac = calculate_oac(load_and_linearize_image(ref_path))
    pred_oac = calculate_oac(load_and_linearize_image(pred_path))
    ref_sc = calculate_speckle_contrast_map(load_and_linearize_image(ref_path))
    pred_sc = calculate_speckle_contrast_map(load_and_linearize_image(pred_path))
    if ref_oac.shape != pred_oac.shape:
        pred_oac = resize_like(pred_oac, ref_oac.shape)
    if ref_sc.shape != pred_sc.shape:
        pred_sc = resize_like(pred_sc, ref_sc.shape)
    return {
        "DepthCorr": depth_corr,
        "LateralCorr": lateral_corr,
        "OACProfileCorr": safe_corr(ref_oac.mean(axis=1), pred_oac.mean(axis=1)),
        "SCMeanAbsErr": float(abs(np.mean(ref_sc) - np.mean(pred_sc))),
    }


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def resize_like(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    from skimage.transform import resize

    return resize(arr, shape, anti_aliasing=True, preserve_range=True)


def composite_score(row: dict[str, float | str]) -> float:
    ms = float(row.get("Struct_MS-SSIM", np.nan))
    lpips = float(row.get("Struct_LPIPS", np.nan))
    lpips_proxy = float(row.get("Struct_LPIPS_PROXY", np.nan))
    ssim = float(row.get("Struct_SSIM", np.nan))
    depth = float(row.get("DepthCorr", 0.0))
    oac = float(row.get("OACProfileCorr", 0.0))
    sc_err = float(row.get("SCMeanAbsErr", 1.0))
    oac_ssim = float(row.get("OAC_SSIM", np.nan))
    sc_ssim = float(row.get("SC_SSIM", np.nan))
    rsc_ssim = float(row.get("RSC_SSIM", np.nan))
    if np.isnan(ms):
        ms = ssim
    if np.isnan(lpips):
        lpips = lpips_proxy if not np.isnan(lpips_proxy) else 0.5
    map_terms = [v for v in (oac_ssim, sc_ssim, rsc_ssim) if not np.isnan(v)]
    map_score = float(np.mean(map_terms)) if map_terms else oac
    return float(
        0.25 * ms
        + 0.15 * (1.0 - lpips)
        + 0.10 * ssim
        + 0.15 * depth
        + 0.10 * oac
        + 0.20 * map_score
        + 0.05 * (1.0 / (1.0 + sc_err))
    )


def competition_proxy_score(row: dict[str, float | str]) -> float:
    """Leaderboard-oriented score: MS-SSIM up, LPIPS down, physics maps as proxy guardrails."""
    struct_ms = float(row.get("Struct_MS-SSIM", np.nan))
    struct_ssim = float(row.get("Struct_SSIM", np.nan))
    lpips = float(row.get("Struct_LPIPS", np.nan))
    lpips_proxy = float(row.get("Struct_LPIPS_PROXY", np.nan))
    if np.isnan(struct_ms):
        struct_ms = struct_ssim
    if np.isnan(lpips):
        lpips = lpips_proxy if not np.isnan(lpips_proxy) else 0.5
    map_ms_values = [
        float(row.get("OAC_MS-SSIM", np.nan)),
        float(row.get("SC_MS-SSIM", np.nan)),
        float(row.get("RSC_MS-SSIM", np.nan)),
    ]
    map_ssim_values = [
        float(row.get("OAC_SSIM", np.nan)),
        float(row.get("SC_SSIM", np.nan)),
        float(row.get("RSC_SSIM", np.nan)),
    ]
    map_terms = [v for v in map_ms_values if not np.isnan(v)] or [v for v in map_ssim_values if not np.isnan(v)]
    map_score = float(np.mean(map_terms)) if map_terms else float(row.get("OACProfileCorr", 0.0))
    return float(0.45 * struct_ms + 0.35 * (1.0 - lpips) + 0.20 * map_score)


def _challenge_lpips_key(row: dict[str, float | str], suffix: str = "") -> str:
    lpips_key = f"Struct_LPIPS{suffix}"
    proxy_key = f"Struct_LPIPS_PROXY{suffix}"
    lpips_value = _finite_float(row.get(lpips_key), np.nan)
    return lpips_key if np.isfinite(lpips_value) else proxy_key


def _finite_float(value: float | str | None, default: float = np.nan) -> float:
    try:
        out = float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


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
                        _to_gray_png(rendered_path, pred_path)

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
                        ref_maps = generate_maps(ref_path, output_dir=sample_dir / "ref_maps")
                        pred_maps = generate_maps(pred_path, output_dir=sample_dir / "pred_maps")
                        for map_name in ("OAC", "SC", "RSC"):
                            map_metrics = calculate_metrics(ref_maps[map_name], pred_maps[map_name], include_lpips=False)
                            row[f"{map_name}_SSIM"] = map_metrics["SSIM"]
                            row[f"{map_name}_MS-SSIM"] = map_metrics["MS-SSIM"]
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


def summarize_rows(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    numeric_keys = sorted(k for row in rows for k, v in row.items() if k not in {"method", "archive_path"} and isinstance(v, (int, float)))
    by_method: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)
    summary = []
    for method, method_rows in sorted(by_method.items()):
        out: dict[str, float | str] = {"method": method, "n": len(method_rows)}
        for key in numeric_keys:
            values = np.array([float(r[key]) for r in method_rows if key in r and np.isfinite(float(r[key]))], dtype=float)
            if values.size:
                out[f"{key}_mean"] = float(values.mean())
                out[f"{key}_std"] = float(values.std(ddof=0))
        summary.append(out)
    summary.sort(
        key=lambda r: (
            _finite_float(r.get("Struct_MS-SSIM_mean"), -1.0),
            -_finite_float(r.get(_challenge_lpips_key(r, "_mean")), np.inf),
        ),
        reverse=True,
    )
    return summary


def summarize_challenge_metrics(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    by_method: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    by_sample: dict[tuple[str, str, str], list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)
        by_sample[(str(row["fold"]), str(row["sample"]), str(row["archive_path"]))].append(row)

    ms_wins: dict[str, int] = defaultdict(int)
    lpips_wins: dict[str, int] = defaultdict(int)
    for sample_rows in by_sample.values():
        ms_sorted = sorted(sample_rows, key=lambda row: _finite_float(row.get("Struct_MS-SSIM"), -1.0), reverse=True)
        if ms_sorted:
            ms_wins[str(ms_sorted[0]["method"])] += 1
        lpips_sorted = sorted(sample_rows, key=lambda row: _finite_float(row.get(_challenge_lpips_key(row)), np.inf))
        if lpips_sorted:
            lpips_wins[str(lpips_sorted[0]["method"])] += 1

    out: list[dict[str, float | str]] = []
    for method, method_rows in by_method.items():
        lpips_key = _challenge_lpips_key(method_rows[0])
        ms_values = np.array([_finite_float(row.get("Struct_MS-SSIM")) for row in method_rows], dtype=float)
        lpips_values = np.array([_finite_float(row.get(lpips_key)) for row in method_rows], dtype=float)
        ms_values = ms_values[np.isfinite(ms_values)]
        lpips_values = lpips_values[np.isfinite(lpips_values)]
        row: dict[str, float | str] = {
            "method": method,
            "n": len(method_rows),
            "MS-SSIM_mean": float(ms_values.mean()) if ms_values.size else np.nan,
            "MS-SSIM_std": float(ms_values.std(ddof=0)) if ms_values.size else np.nan,
            "LPIPS_metric": lpips_key.replace("Struct_", ""),
            "LPIPS_or_proxy_mean": float(lpips_values.mean()) if lpips_values.size else np.nan,
            "LPIPS_or_proxy_std": float(lpips_values.std(ddof=0)) if lpips_values.size else np.nan,
            "MS-SSIM_wins": int(ms_wins[method]),
            "LPIPS_wins": int(lpips_wins[method]),
        }
        out.append(row)
    out.sort(key=lambda row: (_finite_float(row.get("MS-SSIM_mean"), -1.0), -_finite_float(row.get("LPIPS_or_proxy_mean"), np.inf)), reverse=True)
    return out


def summarize_sample_wins(rows: list[dict[str, float | str]], score_key: str = "Struct_MS-SSIM") -> list[dict[str, float | str]]:
    by_sample: dict[tuple[str, str, str], list[dict[str, float | str]]] = defaultdict(list)
    by_method: dict[str, list[float]] = defaultdict(list)
    margins: dict[str, list[float]] = defaultdict(list)
    wins: dict[str, int] = defaultdict(int)

    for row in rows:
        sample_key = (str(row["fold"]), str(row["sample"]), str(row["archive_path"]))
        by_sample[sample_key].append(row)
        by_method[str(row["method"])].append(float(row.get(score_key, row.get("Composite", 0.0))))

    for sample_rows in by_sample.values():
        scored = [(str(row["method"]), float(row.get(score_key, row.get("Composite", 0.0)))) for row in sample_rows]
        if not scored:
            continue
        scored.sort(key=lambda item: item[1], reverse=True)
        best_method, best_score = scored[0]
        wins[best_method] += 1
        for method, score in scored:
            margins[method].append(score - best_score)

    out: list[dict[str, float | str]] = []
    sample_count = max(1, len(by_sample))
    for method in sorted(by_method):
        scores = np.array(by_method[method], dtype=float)
        method_margins = np.array(margins[method], dtype=float)
        out.append(
            {
                "method": method,
                "n": int(scores.size),
                "wins": int(wins[method]),
                "win_rate": float(wins[method] / sample_count),
                f"{score_key}_mean": float(scores.mean()),
                "mean_margin_to_sample_best": float(method_margins.mean()) if method_margins.size else 0.0,
            }
        )
    out.sort(key=lambda row: (int(row["wins"]), float(row[f"{score_key}_mean"])), reverse=True)
    return out


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


def write_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
