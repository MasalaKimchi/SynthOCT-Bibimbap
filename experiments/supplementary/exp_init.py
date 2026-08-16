"""Phase initialisation study (Reviewer 3, comment 1).

Compares the shipped zero-phase start against the reviewer's suggested axial
Hilbert transform and against the minimum-phase (Kramers-Kronig) construction,
which is the physically meaningful form of the same idea for a magnitude-only
input.  Reports both the alternating-projection residual trace and the end-to-end
MS-SSIM of the rendered phantom, so the question "would this reduce the number
of required iterations?" is answered with measurements.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from analysis_lib import (
    dump,
    encode,
    one_frame_per_series,
    render_and_score,
    scan_paths,
    series_key,
    solve_with_trace,
    target_magnitude,
)
from synthoct.holographic_inverse import HolographicInverseConfig
from synthoct.phantom import ExperimentConfig

INITS = ("zero", "minphase", "hilbert")
CHECKPOINTS = (0, 10, 25, 50, 200)


def main() -> None:
    full = "--full" in sys.argv
    paths = scan_paths() if full else one_frame_per_series()[::3]
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
                    "scan": str(path.relative_to(path.parents[4])),
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
            f"({elapsed/n:.1f} s/scan)",
            flush=True,
        )

    dump("init_study_full.json" if full else "init_study.json", records)

    # --- aggregate --------------------------------------------------------
    summary = {}
    for init in INITS:
        subset = [r for r in records if r["init"] == init]
        summary[init] = {
            "n": len(subset),
            "msssim_mean": {
                str(step): float(np.mean([r["msssim_by_iteration"][step] for r in subset]))
                for step in CHECKPOINTS
            },
            "residual_mean": {
                str(step): float(np.mean([r["residual_trace"][step] for r in subset]))
                for step in CHECKPOINTS
            },
        }

    # Iterations each initialisation needs to reach the quality that the shipped
    # zero-phase start reaches at 200 iterations, per scan.
    zero_by_series = {r["series"]: r["msssim_by_iteration"][200] for r in records if r["init"] == "zero"}
    for init in INITS:
        needed = []
        for r in (x for x in records if x["init"] == init):
            goal = zero_by_series[r["series"]]
            hit = [s for s in CHECKPOINTS if r["msssim_by_iteration"][s] >= goal]
            needed.append(min(hit) if hit else None)
        reached = [x for x in needed if x is not None]
        summary[init]["iterations_to_match_zero_at_200"] = {
            "reached_fraction": len(reached) / len(needed),
            "median": float(np.median(reached)) if reached else None,
            "max": float(np.max(reached)) if reached else None,
        }

    dump("init_summary_full.json" if full else "init_summary.json", summary)
    for init, block in summary.items():
        print(f"\n{init}: {block['msssim_mean']}")
        print(f"   match-zero@200: {block['iterations_to_match_zero_at_200']}")


if __name__ == "__main__":
    main()
