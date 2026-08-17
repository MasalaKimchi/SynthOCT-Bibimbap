"""Amplitude-constraint study (Supplement S4).

Tests the cost of stricter amplitude constraints: one common amplitude or a
small discrete set. Each variant reuses the same 200-iteration complex field,
so the comparison isolates the encoding.
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

VARIANTS = [
    ("pair", 0),
    ("pair_equal", 0),
    ("single", 0),
    ("pair_quantized", 2),
    ("pair_quantized", 4),
    ("pair_quantized", 16),
    ("pair_quantized", 64),
    ("pair_quantized", 256),
    ("equal_count", 0),
]


def equal_amplitude_condition() -> dict:
    """Exactly which phases admit an equal-amplitude pair.

    With d1 = d0 - sign(d0) * lambda/2, the zero-first-moment condition
    w0 d0 + w1 d1 = 0 at w0 = w1 = 1/2 requires d0 + d1 = 0, i.e.
    |d0| = lambda/4.  Since d0 = -phi lambda / (4 pi), that is |phi| = pi.
    """
    wavelength = 1.3
    phases = np.linspace(-np.pi, np.pi, 200_001)
    d0 = -phases * wavelength / (4.0 * np.pi)
    d1 = d0 - np.where(d0 >= 0, 1.0, -1.0) * wavelength / 2.0
    moment = 0.5 * d0 + 0.5 * d1
    tol = 1e-9
    roots = phases[np.abs(moment) < tol]
    return {
        "derivation": "w0=w1=1/2 gives d0+d1=0, hence |d0|=lambda/4 and |phi|=pi",
        "numerical_roots_rad": [float(r) for r in roots[:8]],
        "root_over_pi": [float(r / np.pi) for r in roots[:8]],
        "max_abs_moment_um": float(np.abs(moment).max()),
        "moment_at_phi_zero_um": float(moment[len(moment) // 2]),
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

    records = []
    started = time.time()
    for n, path in enumerate(paths, 1):
        scan, magnitude = target_magnitude(path)
        coefficients, _ = _solve_complex_coefficients(magnitude, scanner, inverse)
        row = {"scan": path.name, "series": series_key(path)}
        for mode, levels in VARIANTS:
            enc = encode(coefficients, scanner, mode=mode, levels=levels)
            score, _ = render_and_score(enc.rows, scan)
            row[enc.label] = {"msssim": score, "rows": enc.n_rows}
        records.append(row)
        print(
            f"[{n}/{len(paths)}] {path.name} "
            + " ".join(
                f"{k}={v['msssim']:.5f}" for k, v in row.items() if isinstance(v, dict)
            )
            + f"  ({(time.time() - started) / n:.1f} s/scan)",
            flush=True,
        )

    dump("encoding_study_full.json" if args.full else "encoding_study.json", records)

    labels = [k for k, v in records[0].items() if isinstance(v, dict)]
    summary = {"equal_amplitude_condition": equal_amplitude_condition(), "variants": {}}
    reference = np.array([r["pair"]["msssim"] for r in records])
    for label in labels:
        values = np.array([r[label]["msssim"] for r in records])
        summary["variants"][label] = {
            "n": len(values),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "delta_vs_pair_mean": float((values - reference).mean()),
            "rows": int(records[0][label]["rows"]),
        }
    dump(
        "encoding_summary_full.json" if args.full else "encoding_summary.json", summary
    )
    print("\n" + "\n".join(f"{k}: {v}" for k, v in summary["variants"].items()))
    print("\nequal-amplitude condition:", summary["equal_amplitude_condition"])


if __name__ == "__main__":
    main()
