"""Reproduce the main-paper fixed-regularization 2 x 2 ablation.

The four benchmark runs and their per-scan outputs are written below
``outputs/experiments/main_ablation`` by default, which is ignored by Git.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


SCRIPT_DIR = Path(__file__).resolve().parent


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "synthoct"
        ).is_dir():
            return candidate
    raise RuntimeError(f"could not locate repository root above {start}")


REPO = _find_repo_root(SCRIPT_DIR)
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthoct.benchmark import run_local_benchmark  # noqa: E402
from synthoct.holographic_inverse import HolographicInverseConfig  # noqa: E402


CONFIGURATIONS = {
    "zero_single": HolographicInverseConfig(
        phase_iterations=0,
        phase_encoding="single",
    ),
    "momentum50_single": HolographicInverseConfig(
        phase_iterations=50,
        phase_encoding="single",
    ),
    "zero_pair": HolographicInverseConfig(
        phase_iterations=0,
        phase_encoding="dispersion-canceling-pair",
    ),
    "momentum50_pair": HolographicInverseConfig(
        phase_iterations=50,
        phase_encoding="dispersion-canceling-pair",
    ),
}

CONTRASTS = {
    "phase_at_single": ("momentum50_single", "zero_single"),
    "pair_at_zero": ("zero_pair", "zero_single"),
    "phase_at_pair": ("momentum50_pair", "zero_pair"),
    "pair_at_phase50": ("momentum50_pair", "momentum50_single"),
    "combined_vs_zero_single": ("momentum50_pair", "zero_single"),
}

BOOTSTRAP_SEED = 20260713


def _series_key(reference: str) -> str:
    stem = str(Path(reference).with_suffix(""))
    return re.sub(r"_frame(?:50|250|450)$", "", stem)


def _load_scores(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return {row["reference"]: float(row["Struct_MS-SSIM"]) for row in rows}


def _metric_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _cluster_bootstrap_ci(
    series_differences: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> list[float]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    rng = np.random.default_rng(seed)
    samples = []
    remaining = replicates
    while remaining:
        batch = min(10_000, remaining)
        indices = rng.integers(
            0,
            len(series_differences),
            size=(batch, len(series_differences)),
        )
        samples.append(series_differences[indices].mean(axis=1))
        remaining -= batch
    lower, upper = np.percentile(np.concatenate(samples), [2.5, 97.5])
    return [float(lower), float(upper)]


def _contrast_summary(
    positive: dict[str, float],
    baseline: dict[str, float],
    *,
    bootstrap_replicates: int,
) -> dict[str, object]:
    if positive.keys() != baseline.keys():
        raise ValueError("configuration reference sets do not match")
    references = sorted(positive)
    image_differences = np.asarray(
        [positive[key] - baseline[key] for key in references], dtype=float
    )
    by_series: dict[str, list[float]] = {}
    for reference, difference in zip(references, image_differences, strict=True):
        by_series.setdefault(_series_key(reference), []).append(float(difference))
    series_differences = np.asarray(
        [np.mean(by_series[key]) for key in sorted(by_series)], dtype=float
    )
    method = "exact" if np.all(series_differences != 0) else "auto"
    p_value = wilcoxon(
        series_differences,
        alternative="greater",
        method=method,
    ).pvalue
    return {
        "mean": float(np.mean(image_differences)),
        "min": float(np.min(image_differences)),
        "max": float(np.max(image_differences)),
        "image_wins": int(np.sum(image_differences > 0)),
        "series_wins": int(np.sum(series_differences > 0)),
        "series_cluster_bootstrap_ci95": _cluster_bootstrap_ci(
            series_differences,
            replicates=bootstrap_replicates,
            seed=BOOTSTRAP_SEED,
        ),
        "one_sided_exact_wilcoxon_p": float(p_value),
    }


def build_summary(
    output_dir: Path,
    *,
    bootstrap_replicates: int = 200_000,
) -> dict[str, object]:
    scores = {
        name: _load_scores(output_dir / name / "detail.csv")
        for name in CONFIGURATIONS
    }
    reference_sets = [set(values) for values in scores.values()]
    if any(current != reference_sets[0] for current in reference_sets[1:]):
        raise ValueError("configuration reference sets do not match")

    run_summaries = {
        name: json.loads((output_dir / name / "summary.json").read_text())
        for name in CONFIGURATIONS
    }
    manifest_hashes = {
        summary["dataset_manifest_sha256"] for summary in run_summaries.values()
    }
    if len(manifest_hashes) != 1:
        raise ValueError("configuration dataset manifests do not match")

    first_references = sorted(reference_sets[0])
    n_series = len({_series_key(reference) for reference in first_references})
    configurations = {}
    for name, inverse in CONFIGURATIONS.items():
        values = np.asarray([scores[name][key] for key in first_references])
        configurations[name] = {
            "phase_iterations": inverse.phase_iterations,
            "phase_encoding": inverse.phase_encoding,
            "struct_ms_ssim": _metric_summary(values),
        }

    return {
        "purpose": "full-public fixed-regularization 2x2 phase-selection by phase-encoding ablation",
        "evidence_source": "local_published_forward_model",
        "official_or_hidden_score": False,
        "reference_root": str(run_summaries["zero_single"]["input_root"]),
        "dataset_manifest_sha256": manifest_hashes.pop(),
        "n_images": len(first_references),
        "n_filename_defined_series": n_series,
        "parameters_fixed": {
            "axial_regularization": 0.02,
            "lateral_regularization": 0.05,
            "phase_momentum": 1.0,
            "dynamic_range_db": 51.0,
            "max_reflection_amplitude": 0.001,
        },
        "configurations": configurations,
        "paired_contrasts": {
            label: _contrast_summary(
                scores[positive],
                scores[baseline],
                bootstrap_replicates=bootstrap_replicates,
            )
            for label, (positive, baseline) in CONTRASTS.items()
        },
        "inference": {
            "unit": "mean of the three frames in each filename-defined acquisition series",
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "interval": "two-sided percentile 95% confidence interval",
            "test": "one-sided exact Wilcoxon signed-rank test on series means",
        },
        "environment": run_summaries["zero_single"]["environment"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO / "DATASET" / "DATASET_PNG",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "outputs" / "experiments" / "main_ablation",
    )
    parser.add_argument("--limit", type=int, help="Development subset; omit for all 120 scans.")
    parser.add_argument("--bootstrap-replicates", type=int, default=200_000)
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild the aggregate from existing detail.csv and summary.json files.",
    )
    args = parser.parse_args()

    if not args.summarize_only:
        for name, inverse in CONFIGURATIONS.items():
            print(f"\n=== {name} ===", flush=True)
            run_local_benchmark(
                args.dataset,
                args.output_dir / name,
                inverse=inverse,
                include_maps=False,
                include_lpips=False,
                limit=args.limit,
            )

    result = build_summary(
        args.output_dir,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "ablation_results.json"
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
