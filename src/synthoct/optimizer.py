from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage import io

from .baselines import HYPOTHESIS_CONFIGS, physics_guided_baseline
from .dataset import iter_records, load_scan_from_zip
from .metrics import calculate_metrics
from .processor import generate_maps
from .surrogate import phantom_to_surrogate_scan
from .validation import competition_proxy_score, make_folds, profile_scores, summarize_rows, write_rows


@dataclass(frozen=True)
class Candidate:
    name: str
    config: dict[str, float | bool]


def generate_candidate_configs(seed: int = 0, random_count: int = 48) -> list[Candidate]:
    rng = np.random.default_rng(seed)
    candidates: list[Candidate] = []
    anchors = {
        "H11": HYPOTHESIS_CONFIGS["H11_low_depth_comp"],
        "H29": HYPOTHESIS_CONFIGS["H29_low_depth_high_oac"],
        "H34": HYPOTHESIS_CONFIGS["H34_epidermal_emphasis"],
        "H17": HYPOTHESIS_CONFIGS["H17_superlinear_density"],
        "H40": HYPOTHESIS_CONFIGS["H40_conservative_winner"],
    }
    for anchor_name, cfg in anchors.items():
        tuned = dict(cfg)
        tuned.setdefault("oac_percentile", 70.0)
        tuned.setdefault("base_energy_mix", 0.55)
        tuned.setdefault("texture_sigma_scale", 1.0)
        tuned.setdefault("lateral_smooth", 0.8)
        candidates.append(Candidate(f"O_anchor_{anchor_name}", tuned))

    base = dict(HYPOTHESIS_CONFIGS["H11_low_depth_comp"])
    base.update({"oac_percentile": 70.0, "base_energy_mix": 0.55, "texture_sigma_scale": 1.0, "lateral_smooth": 0.8})
    grid = [
        ("depth", "depth_compensation", [1.20, 1.30, 1.40, 1.50, 1.60]),
        ("oac", "oac_weight", [1.75, 1.95, 2.15, 2.35, 2.55]),
        ("density", "density_power", [0.90, 1.00, 1.08, 1.16, 1.24]),
        ("percentile", "oac_percentile", [60.0, 65.0, 70.0, 75.0, 80.0]),
        ("mix", "base_energy_mix", [0.45, 0.55, 0.65, 0.75]),
        ("texture", "texture_sigma_scale", [0.70, 0.90, 1.10, 1.30]),
        ("smooth", "lateral_smooth", [0.55, 0.80, 1.05, 1.30]),
        ("band", "band_boost", [0.0, 0.08, 0.18, 0.30]),
    ]
    for label, key, values in grid:
        for value in values:
            cfg = dict(base)
            cfg[key] = value
            candidates.append(Candidate(f"O_{label}_{value}", cfg))

    for idx in range(random_count):
        cfg = {
            "density_power": float(rng.uniform(0.88, 1.24)),
            "depth_compensation": float(rng.uniform(1.15, 1.75)),
            "oac_weight": float(rng.uniform(1.75, 2.65)),
            "texture_weight": float(rng.uniform(0.18, 0.50)),
            "energy_oac_scale": float(rng.uniform(28.0, 46.0)),
            "band_boost": float(rng.choice([0.0, rng.uniform(0.04, 0.35)])),
            "lateral_smooth": float(rng.uniform(0.45, 1.45)),
            "oac_percentile": float(rng.uniform(58.0, 82.0)),
            "base_energy_mix": float(rng.uniform(0.42, 0.78)),
            "texture_sigma_scale": float(rng.uniform(0.65, 1.35)),
        }
        candidates.append(Candidate(f"O_random_{idx:02d}", cfg))
    return candidates


def run_candidate_search(
    zip_path: str | Path,
    out_dir: str | Path,
    folds: int = 3,
    max_per_fold: int = 1,
    scatterers_count: int = 4_000,
    random_count: int = 48,
    seed: int = 23,
    include_maps: bool = True,
) -> tuple[Path, Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = generate_candidate_configs(seed=seed, random_count=random_count)
    candidate_map = {candidate.name: candidate.config for candidate in candidates}
    (out_dir / "candidate_configs.json").write_text(json.dumps(candidate_map, indent=2, sort_keys=True))

    records = [r for r in iter_records(zip_path) if r.modality == "png"]
    folded = make_folds(records, folds=folds)
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
                    pred_path = sample_dir / "surrogate.png"
                    start = time.perf_counter()
                    physics_guided_baseline(
                        ref_path,
                        phantom_path,
                        seed=seed + fold_idx * 100 + sample_idx,
                        scatterers_count=scatterers_count,
                        **candidate.config,
                    )
                    generation_seconds = time.perf_counter() - start
                    phantom_to_surrogate_scan(phantom_path, pred_path, seed=seed + sample_idx)

                    row: dict[str, float | str] = {
                        "fold": fold_idx,
                        "sample": sample_idx,
                        "method": candidate.name,
                        "archive_path": record.archive_path,
                        "generation_seconds": generation_seconds,
                    }
                    for key, value in calculate_metrics(ref_path, pred_path, include_lpips=False).items():
                        row[f"Struct_{key}"] = value
                    row.update(profile_scores(ref_path, pred_path))
                    if include_maps:
                        ref_maps = generate_maps(ref_path, output_dir=sample_dir / "ref_maps")
                        pred_maps = generate_maps(pred_path, output_dir=sample_dir / "pred_maps")
                        for map_name in ("OAC", "SC", "RSC"):
                            metrics = calculate_metrics(ref_maps[map_name], pred_maps[map_name], include_lpips=False)
                            row[f"{map_name}_SSIM"] = metrics["SSIM"]
                            row[f"{map_name}_MS-SSIM"] = metrics["MS-SSIM"]
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
