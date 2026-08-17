"""Finite-resolution overlap and artifact-band study (Supplement S3).

The geometric count follows from the point-spread widths; the participation
ratio accounts for unequal modeled contributions. For the field
F = A C L^T the per-pixel contributions have magnitude
|A[z,z']| |C[z',x']| L[x,x'], so both the sum and the sum of squares are matrix
products and the participation ratio is exact, not sampled.

The study also reports artifact-band energy and its local-rendering ablation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from analysis_lib import (
    configure_paths,
    dump,
    encode,
    one_frame_per_series,
    operators,
    render_and_score,
    require_scans,
    scan_paths,
    series_key,
    target_magnitude,
)
from synthoct.holographic_inverse import (
    HolographicInverseConfig,
    _solve_complex_coefficients,
)
from synthoct.phantom import ExperimentConfig


def participation_ratio(coefficients: np.ndarray, axial_abs, lateral_abs):
    """Effective number of coefficient sites contributing to each output pixel."""
    magnitude = np.abs(coefficients)
    total = axial_abs @ magnitude @ lateral_abs.T
    squared = (axial_abs**2) @ (magnitude**2) @ (lateral_abs**2).T
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(squared > 0, total**2 / squared, np.nan)
    self_share = np.where(
        total > 0, magnitude * 1.0 / np.maximum(total, 1e-300), np.nan
    )
    return ratio, self_share


def artifact_band(scan: np.ndarray, coefficients: np.ndarray, half_width: int = 2):
    """Locate the bright horizontal line in the lower image and weigh it."""
    depth = scan.shape[0]
    lower = scan[int(0.6 * depth) :].mean(axis=1)
    peak = int(np.argmax(lower)) + int(0.6 * depth)
    lo, hi = max(0, peak - half_width), min(depth, peak + half_width + 1)
    energy = np.abs(coefficients) ** 2
    return {
        "band_row": peak,
        "band_rows": [lo, hi],
        "band_row_fraction_of_depth": peak / depth,
        "energy_fraction_in_band": float(energy[lo:hi].sum() / energy.sum()),
        "rows_fraction_of_image": (hi - lo) / depth,
        "band_mean_display_value": float(scan[lo:hi].mean()),
        "image_mean_display_value": float(scan.mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        help="PNG dataset root (default: repository DATASET/DATASET_PNG)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="generated-result directory (default: outputs/experiments/supplementary)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="use every frame instead of one frame per acquisition series",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="run only the first N scans after subset selection",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    configure_paths(dataset=args.dataset, output_dir=args.output_dir)
    paths = scan_paths() if args.full else one_frame_per_series()
    if args.limit is not None:
        paths = paths[: args.limit]
    paths = require_scans(paths)
    scanner = ExperimentConfig(scatterers_count=300_000)
    inverse = HolographicInverseConfig(phase_iterations=200)
    axial, lateral, _, _ = operators(scanner, inverse)
    axial_abs, lateral_abs = np.abs(axial), np.abs(lateral)

    records = []
    started = time.time()
    for n, path in enumerate(paths, 1):
        scan, magnitude = target_magnitude(path)
        coefficients, _ = _solve_complex_coefficients(magnitude, scanner, inverse)
        ratio, _ = participation_ratio(coefficients, axial_abs, lateral_abs)
        finite = ratio[np.isfinite(ratio)]

        # Weight by modeled output-field magnitude so weak-field pixels do not
        # dominate the per-scan average.
        weights = np.abs(axial @ coefficients @ lateral.T)
        weights = weights[np.isfinite(ratio)]
        weighted = float(np.sum(finite * weights) / np.sum(weights))

        band = artifact_band(scan, coefficients)

        # Does the render depend on the scatterers fitted onto the artifact line?
        rows = encode(coefficients, scanner, mode="pair").rows
        depth_lo = band["band_rows"][0] * scanner.pixel_size_z
        depth_hi = band["band_rows"][1] * scanner.pixel_size_z
        keep = (rows[:, 2] < depth_lo) | (rows[:, 2] >= depth_hi)
        full_score, _ = render_and_score(rows, scan)
        ablated_score, _ = render_and_score(rows[keep], scan)

        records.append(
            {
                "scan": path.name,
                "series": series_key(path),
                "voxels_per_pixel": {
                    "mean": float(finite.mean()),
                    "median": float(np.median(finite)),
                    "p05": float(np.percentile(finite, 5)),
                    "p95": float(np.percentile(finite, 95)),
                    "brightness_weighted_mean": weighted,
                },
                "artifact": {
                    **band,
                    "msssim_full": full_score,
                    "msssim_without_band": ablated_score,
                    "msssim_drop": full_score - ablated_score,
                },
            }
        )
        print(
            f"[{n}/{len(paths)}] {path.name} "
            f"PR={records[-1]['voxels_per_pixel']['brightness_weighted_mean']:.2f} "
            f"band={band['band_row']} drop={records[-1]['artifact']['msssim_drop']:.4f} "
            f"({(time.time() - started) / n:.1f} s/scan)",
            flush=True,
        )

    dump("density_study_full.json" if args.full else "density_study.json", records)

    def agg(path_fn):
        values = np.array([path_fn(r) for r in records])
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    summary = {
        "n_scans": len(records),
        "effective_coefficient_sites_per_pixel": agg(
            lambda r: r["voxels_per_pixel"]["brightness_weighted_mean"]
        ),
        "median_coefficient_sites_per_pixel": agg(
            lambda r: r["voxels_per_pixel"]["median"]
        ),
        "artifact_band_row": agg(lambda r: r["artifact"]["band_row"]),
        "artifact_energy_fraction": agg(
            lambda r: r["artifact"]["energy_fraction_in_band"]
        ),
        "artifact_msssim_drop": agg(lambda r: r["artifact"]["msssim_drop"]),
    }
    dump("density_summary_full.json" if args.full else "density_summary.json", summary)
    print("\n" + "\n".join(f"{k}: {v}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
