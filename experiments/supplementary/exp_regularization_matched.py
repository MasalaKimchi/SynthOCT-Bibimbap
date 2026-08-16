"""Matched-regularization diagnostic for Reviewer 2, Comment 2.

This analysis holds axial regularization fixed and crosses phase
selection with scatterer encoding at every lateral-regularization value. Raw
per-scan output and a stable compact summary are written beneath the ignored
experiment-output directory by default.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from analysis_lib import (
    OUT,
    encode,
    one_frame_per_series,
    operators,
    render_and_score,
    series_key,
    target_magnitude,
)
from synthoct.holographic_inverse import (
    HolographicInverseConfig,
    _solve_complex_coefficients,
)
from synthoct.phantom import ExperimentConfig


ALPHA_A = 0.02
ALPHA_L = (0.02, 0.05, 0.10, 0.20, 0.40)
PHASE_ITERATIONS = (0, 200)
ENCODINGS = ("single", "pair")


def lateral_mode_fraction(
    coefficients: np.ndarray,
    right_singular_vectors: np.ndarray,
    weak_mask: np.ndarray,
) -> float:
    projected = coefficients @ right_singular_vectors.conj()
    energy = np.sum(np.abs(projected) ** 2, axis=0)
    return float(energy[weak_mask].sum() / energy.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT / "regularization_matched_factorial.json",
    )
    args = parser.parse_args()

    paths = one_frame_per_series()
    if args.limit is not None:
        paths = paths[: args.limit]

    scanner = ExperimentConfig(scatterers_count=300_000)
    _, lateral, _, _ = operators(scanner, HolographicInverseConfig())
    _, singular_values, vh = np.linalg.svd(lateral, full_matrices=False)
    right_singular_vectors = vh.conj().T
    weak_mask = singular_values < 0.05

    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for scan_number, path in enumerate(paths, 1):
        reference, magnitude = target_magnitude(path)
        scan_started = time.perf_counter()
        row: dict[str, object] = {
            "scan": path.name,
            "series": series_key(path),
        }
        for alpha_l in ALPHA_L:
            for iterations in PHASE_ITERATIONS:
                inverse = HolographicInverseConfig(
                    axial_regularization=ALPHA_A,
                    lateral_regularization=alpha_l,
                    phase_iterations=iterations,
                )
                coefficients, magnitude_mae = _solve_complex_coefficients(
                    magnitude, scanner, inverse
                )
                phase_key = f"phase{iterations}_alphaL_{alpha_l}"
                row[phase_key] = {
                    "projected_magnitude_mae": magnitude_mae,
                    "weak_mode_energy_fraction": lateral_mode_fraction(
                        coefficients, right_singular_vectors, weak_mask
                    ),
                }
                for encoding in ENCODINGS:
                    encoded = encode(coefficients, scanner, mode=encoding)
                    score, _ = render_and_score(encoded.rows, reference)
                    row[f"{phase_key}_{encoding}"] = {"msssim": score}
        rows.append(row)
        elapsed = time.perf_counter() - started
        print(
            f"[{scan_number}/{len(paths)}] {path.name}: "
            f"{time.perf_counter() - scan_started:.1f}s; "
            f"mean {elapsed / scan_number:.1f}s/scan",
            flush=True,
        )

    summary: dict[str, object] = {
        "design": {
            "n_scans": len(paths),
            "selection": "frame-250 scan from each filename-defined acquisition series",
            "axial_regularization_fixed": ALPHA_A,
            "lateral_regularization": ALPHA_L,
            "phase_iterations": PHASE_ITERATIONS,
            "encodings": ENCODINGS,
            "phase_momentum": 1.0,
            "dynamic_range_db": 51.0,
            "weak_mode_definition": "lateral singular value < 0.05",
            "weak_mode_count": int(weak_mask.sum()),
            "lateral_mode_count": int(len(singular_values)),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "msssim": {},
        "weak_mode_energy_fraction": {},
    }
    msssim_summary = summary["msssim"]
    mode_summary = summary["weak_mode_energy_fraction"]
    assert isinstance(msssim_summary, dict)
    assert isinstance(mode_summary, dict)
    for alpha_l in ALPHA_L:
        for iterations in PHASE_ITERATIONS:
            phase_key = f"phase{iterations}_alphaL_{alpha_l}"
            mode_values = [
                float(row[phase_key]["weak_mode_energy_fraction"])  # type: ignore[index]
                for row in rows
            ]
            mode_summary[phase_key] = {
                "mean": float(np.mean(mode_values)),
                "min": float(np.min(mode_values)),
                "max": float(np.max(mode_values)),
            }
            for encoding in ENCODINGS:
                key = f"{phase_key}_{encoding}"
                values = [float(row[key]["msssim"]) for row in rows]  # type: ignore[index]
                msssim_summary[key] = {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stable_summary = {key: value for key, value in summary.items() if key != "elapsed_seconds"}
    summary_path = args.output.with_name("regularization_matched_summary.json")
    summary_path.write_text(
        json.dumps(stable_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output} and {summary_path}", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
