"""Reproduce the camera-ready main-paper local-model comparisons.

By default this runs the four-cell fixed-regularization ablation and the
earlier-method versus final-method comparison. Per-scan outputs are written
below ``outputs/experiments/main_ablation``, which is ignored by Git.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict
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


ABLATION_CONFIGURATIONS = {
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

FULL_COMPARISON_CONFIGURATIONS = {
    "earlier_zero_single": HolographicInverseConfig(
        axial_regularization=0.03,
        lateral_regularization=0.20,
        phase_iterations=0,
        phase_momentum=0.0,
        phase_encoding="single",
    ),
    "final200_pair": HolographicInverseConfig(
        axial_regularization=0.02,
        lateral_regularization=0.05,
        phase_iterations=200,
        phase_momentum=1.0,
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

ABLATION_BOOTSTRAP_SEED = 20260713
FULL_COMPARISON_BOOTSTRAP_SEED = 20260711
PUBLISHED_PUBLIC_MANIFEST_SHA256 = (
    "7a245393fcbce95601bfe20b575364f0a489364022ff7413b96d50552f8ee55b"
)


def _series_key(reference: str) -> str:
    stem = str(Path(reference).with_suffix(""))
    return re.sub(r"_frame(?:50|250|450)$", "", stem)


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def _metric_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _timing_summary(rows: list[dict[str, str]]) -> dict[str, float]:
    values = np.asarray([float(row["generation_seconds"]) for row in rows])
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _portable_reference_root(value: str) -> str:
    """Avoid embedding a contributor's absolute checkout path in summaries."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        return path.as_posix()
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return resolved.name


def _scope(n_images: int, manifest_hash: str, study: str) -> tuple[str, str, bool]:
    matches_published_set = (
        n_images == 120 and manifest_hash == PUBLISHED_PUBLIC_MANIFEST_SHA256
    )
    if matches_published_set:
        return "full_public_120", f"full-public {study}", True
    return "development_or_external_set", f"development/external-set {study}", False


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
    bootstrap_seed: int,
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
            seed=bootstrap_seed,
        ),
        "one_sided_exact_wilcoxon_p": float(p_value),
    }


def _study_inputs(
    output_dir: Path,
    configurations: dict[str, HolographicInverseConfig],
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, list[dict[str, str]]],
    dict[str, dict[str, object]],
    list[str],
    str,
]:
    rows = {
        name: _load_rows(output_dir / name / "detail.csv") for name in configurations
    }
    scores = {
        name: {row["reference"]: float(row["Struct_MS-SSIM"]) for row in current_rows}
        for name, current_rows in rows.items()
    }
    reference_sets = [set(values) for values in scores.values()]
    if any(current != reference_sets[0] for current in reference_sets[1:]):
        raise ValueError("configuration reference sets do not match")

    run_summaries = {
        name: json.loads((output_dir / name / "summary.json").read_text())
        for name in configurations
    }
    manifest_hashes = {
        summary["dataset_manifest_sha256"] for summary in run_summaries.values()
    }
    if len(manifest_hashes) != 1:
        raise ValueError("configuration dataset manifests do not match")
    references = sorted(reference_sets[0])
    return scores, rows, run_summaries, references, manifest_hashes.pop()


def build_ablation_summary(
    output_dir: Path,
    *,
    bootstrap_replicates: int = 200_000,
) -> dict[str, object]:
    scores, _, run_summaries, first_references, manifest_hash = _study_inputs(
        output_dir,
        ABLATION_CONFIGURATIONS,
    )
    n_series = len({_series_key(reference) for reference in first_references})
    result_scope, purpose, matches_published_set = _scope(
        len(first_references),
        manifest_hash,
        "fixed-regularization 2x2 phase-selection by phase-encoding ablation",
    )
    configurations = {}
    for name, inverse in ABLATION_CONFIGURATIONS.items():
        values = np.asarray([scores[name][key] for key in first_references])
        configurations[name] = {
            "phase_iterations": inverse.phase_iterations,
            "phase_encoding": inverse.phase_encoding,
            "struct_ms_ssim": _metric_summary(values),
        }

    return {
        "purpose": purpose,
        "result_scope": result_scope,
        "evidence_source": "local_published_forward_model",
        "official_or_hidden_score": False,
        "reference_root": _portable_reference_root(
            str(run_summaries["zero_single"]["input_root"])
        ),
        "dataset_manifest_sha256": manifest_hash,
        "dataset_matches_published_public_set": matches_published_set,
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
                bootstrap_seed=ABLATION_BOOTSTRAP_SEED,
            )
            for label, (positive, baseline) in CONTRASTS.items()
        },
        "inference": {
            "unit": "mean of the three frames in each filename-defined acquisition series",
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": ABLATION_BOOTSTRAP_SEED,
            "interval": "two-sided percentile 95% confidence interval",
            "test": "one-sided exact Wilcoxon signed-rank test on series means",
        },
        "environment": run_summaries["zero_single"]["environment"],
    }


def build_full_comparison_summary(
    output_dir: Path,
    *,
    bootstrap_replicates: int = 200_000,
) -> dict[str, object]:
    scores, rows, run_summaries, references, manifest_hash = _study_inputs(
        output_dir,
        FULL_COMPARISON_CONFIGURATIONS,
    )
    n_series = len({_series_key(reference) for reference in references})
    result_scope, purpose, matches_published_set = _scope(
        len(references),
        manifest_hash,
        "earlier-method versus final-method comparison",
    )
    configurations = {}
    for name, inverse in FULL_COMPARISON_CONFIGURATIONS.items():
        values = np.asarray([scores[name][key] for key in references])
        configurations[name] = {
            "parameters": asdict(inverse),
            "struct_ms_ssim": _metric_summary(values),
            "generation_seconds": _timing_summary(rows[name]),
        }

    return {
        "purpose": purpose,
        "result_scope": result_scope,
        "evidence_source": "local_published_forward_model",
        "official_or_hidden_score": False,
        "reference_root": _portable_reference_root(
            str(run_summaries["earlier_zero_single"]["input_root"])
        ),
        "dataset_manifest_sha256": manifest_hash,
        "dataset_matches_published_public_set": matches_published_set,
        "n_images": len(references),
        "n_filename_defined_series": n_series,
        "configurations": configurations,
        "paired_comparison": _contrast_summary(
            scores["final200_pair"],
            scores["earlier_zero_single"],
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=FULL_COMPARISON_BOOTSTRAP_SEED,
        ),
        "inference": {
            "unit": "mean of the three frames in each filename-defined acquisition series",
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": FULL_COMPARISON_BOOTSTRAP_SEED,
            "interval": "two-sided percentile 95% confidence interval",
            "test": "one-sided exact Wilcoxon signed-rank test on series means",
        },
        "environment": run_summaries["earlier_zero_single"]["environment"],
    }


def _selected_configurations(
    study: str,
) -> dict[str, HolographicInverseConfig]:
    if study == "ablation":
        return ABLATION_CONFIGURATIONS
    if study == "full-comparison":
        return FULL_COMPARISON_CONFIGURATIONS
    return {**ABLATION_CONFIGURATIONS, **FULL_COMPARISON_CONFIGURATIONS}


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
    parser.add_argument(
        "--limit", type=int, help="Development subset; omit for all 120 scans."
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=200_000)
    parser.add_argument(
        "--study",
        choices=("all", "ablation", "full-comparison"),
        default="all",
        help="Select a paper study; default reproduces both local studies.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild the aggregate from existing detail.csv and summary.json files.",
    )
    args = parser.parse_args()

    if not args.summarize_only:
        for name, inverse in _selected_configurations(args.study).items():
            print(f"\n=== {name} ===", flush=True)
            run_local_benchmark(
                args.dataset,
                args.output_dir / name,
                inverse=inverse,
                include_maps=False,
                include_lpips=False,
                limit=args.limit,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    builders = []
    if args.study in {"all", "ablation"}:
        builders.append(("ablation_results.json", build_ablation_summary))
    if args.study in {"all", "full-comparison"}:
        builders.append(("full_method_comparison.json", build_full_comparison_summary))
    for filename, builder in builders:
        result = builder(
            args.output_dir,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        summary_path = args.output_dir / filename
        summary_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
