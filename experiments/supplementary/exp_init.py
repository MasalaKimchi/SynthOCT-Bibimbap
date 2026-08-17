"""Phase initialisation study (Supplement S1).

Compares the standardized zero-phase start with axial analytic-signal and
minimum-phase (Kramers-Kronig) initializations for a magnitude-only input.
Reports both the alternating-projection residual trace and the end-to-end
MS-SSIM of the rendered phantom at fixed iteration checkpoints.
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
    scan_label,
    scan_paths,
    series_key,
    solve_with_trace,
    target_magnitude,
)
from synthoct.holographic_inverse import HolographicInverseConfig
from synthoct.phantom import ExperimentConfig

INITS = ("zero", "minphase", "hilbert")
CHECKPOINTS = (0, 10, 25, 50, 200)


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
        help="use every frame instead of the default sparse one-frame-per-series subset",
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
    paths = scan_paths() if args.full else one_frame_per_series()[::3]
    if args.limit is not None:
        paths = paths[: args.limit]
    paths = require_scans(paths)
    scanner = ExperimentConfig(scatterers_count=300_000)
    inverse = HolographicInverseConfig(phase_iterations=200)

    records = []
    started = time.time()
    for n, path in enumerate(paths, 1):
        scan, magnitude = target_magnitude(path)
        for init in INITS:
            t0 = time.time()
            _, residuals, saved = solve_with_trace(
                magnitude, scanner, inverse, init=init, checkpoints=CHECKPOINTS
            )
            scores = {}
            for step in CHECKPOINTS:
                rows = encode(saved[step], scanner, mode="pair").rows
                scores[step], _ = render_and_score(rows, scan)
            records.append(
                {
                    "scan": scan_label(path),
                    "series": series_key(path),
                    "init": init,
                    "residual_trace": residuals.tolist(),
                    "msssim_by_iteration": scores,
                    "seconds": time.time() - t0,
                }
            )
        elapsed = time.time() - started
        print(
            f"[{n}/{len(paths)}] {path.name} "
            f"zero@200={records[-3]['msssim_by_iteration'][200]:.5f} "
            f"minphase@200={records[-2]['msssim_by_iteration'][200]:.5f} "
            f"({elapsed / n:.1f} s/scan)",
            flush=True,
        )

    dump("init_study_full.json" if args.full else "init_study.json", records)

    # --- aggregate --------------------------------------------------------
    summary = {}
    for init in INITS:
        subset = [r for r in records if r["init"] == init]
        summary[init] = {
            "n": len(subset),
            "msssim_mean": {
                str(step): float(
                    np.mean([r["msssim_by_iteration"][step] for r in subset])
                )
                for step in CHECKPOINTS
            },
            "residual_mean": {
                str(step): float(np.mean([r["residual_trace"][step] for r in subset]))
                for step in CHECKPOINTS
            },
        }

    # Iterations each initialisation needs to reach the quality that the baseline
    # zero-phase start reaches at 200 iterations, per scan.
    zero_by_scan = {
        r["scan"]: r["msssim_by_iteration"][200] for r in records if r["init"] == "zero"
    }
    for init in INITS:
        needed = []
        for r in (x for x in records if x["init"] == init):
            goal = zero_by_scan[r["scan"]]
            hit = [s for s in CHECKPOINTS if r["msssim_by_iteration"][s] >= goal]
            needed.append(min(hit) if hit else None)
        reached = [x for x in needed if x is not None]
        summary[init]["iterations_to_match_zero_at_200"] = {
            "reached_fraction": len(reached) / len(needed),
            "median": float(np.median(reached)) if reached else None,
            "max": float(np.max(reached)) if reached else None,
        }

    records_by_init = {
        init: {r["scan"]: r for r in records if r["init"] == init} for init in INITS
    }
    zero_records = records_by_init["zero"]
    summary["study"] = {
        "selection": {
            "description": (
                "all PNG scans in lexicographic path order"
                if args.full
                else (
                    "prefer filename-labeled frame250 within each acquisition "
                    "series, sort series keys, then select every third series"
                )
            ),
            "full": args.full,
            "limit": args.limit,
            "n_scans": len(paths),
        },
        "checkpoints": list(CHECKPOINTS),
    }
    summary["pairwise_vs_zero"] = {}
    for init in INITS[1:]:
        paired = [
            (records_by_init[init][scan], zero) for scan, zero in zero_records.items()
        ]
        summary["pairwise_vs_zero"][init] = {
            "n_pairs": len(paired),
            "msssim_higher_count": {
                str(step): sum(
                    candidate["msssim_by_iteration"][step]
                    > zero["msssim_by_iteration"][step]
                    for candidate, zero in paired
                )
                for step in CHECKPOINTS
            },
            "residual_lower_count": {
                str(step): sum(
                    candidate["residual_trace"][step] < zero["residual_trace"][step]
                    for candidate, zero in paired
                )
                for step in CHECKPOINTS
            },
            "mean_delta_at_200": {
                "msssim": float(
                    np.mean(
                        [
                            candidate["msssim_by_iteration"][200]
                            - zero["msssim_by_iteration"][200]
                            for candidate, zero in paired
                        ]
                    )
                ),
                "residual": float(
                    np.mean(
                        [
                            candidate["residual_trace"][200]
                            - zero["residual_trace"][200]
                            for candidate, zero in paired
                        ]
                    )
                ),
            },
        }
    summary["mean_residual_199_to_200"] = {
        init: {
            "at_199": float(
                np.mean(
                    [r["residual_trace"][199] for r in records_by_init[init].values()]
                )
            ),
            "at_200": float(
                np.mean(
                    [r["residual_trace"][200] for r in records_by_init[init].values()]
                )
            ),
            "change_200_minus_199": float(
                np.mean(
                    [
                        r["residual_trace"][200] - r["residual_trace"][199]
                        for r in records_by_init[init].values()
                    ]
                )
            ),
        }
        for init in INITS
    }

    dump("init_summary_full.json" if args.full else "init_summary.json", summary)
    for init in INITS:
        block = summary[init]
        print(f"\n{init}: {block['msssim_mean']}")
        print(f"   match-zero@200: {block['iterations_to_match_zero_at_200']}")


if __name__ == "__main__":
    main()
