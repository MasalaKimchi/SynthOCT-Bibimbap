"""Why the earlier zero-phase pipeline needed strong lateral damping.

Reviewer 2, comment 2 asks for the physical or mathematical reason that the
zero-phase single-scatterer configuration used alpha_L = 0.20 while the
phase-pair method uses 0.05.  Two measurements answer it:

1.  where the inverse puts its energy in the lateral singular basis, for a real
    zero-phase target versus a phase-selected target; and
2.  a sweep of alpha_L for both complete configurations, so the optimum of each
    is measured rather than asserted.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from analysis_lib import (
    dump,
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

ALPHA_L = (0.02, 0.05, 0.10, 0.20, 0.40)


def lateral_mode_energy(coefficients: np.ndarray, v_lateral: np.ndarray) -> np.ndarray:
    """Energy of the coefficient grid per right-singular mode of L."""
    projected = coefficients @ v_lateral.conj()
    return np.sum(np.abs(projected) ** 2, axis=0)


def lateral_roughness(image: np.ndarray) -> float:
    """Mean absolute lateral first difference: a ringing proxy."""
    return float(np.mean(np.abs(np.diff(image, axis=1))))


def main() -> None:
    paths = one_frame_per_series()
    if "--quick" in sys.argv:
        paths = paths[:8]
    scanner = ExperimentConfig(scatterers_count=300_000)

    _, lateral, _, _ = operators(scanner, HolographicInverseConfig())
    _, sv_lateral, vh_lateral = np.linalg.svd(lateral, full_matrices=False)
    v_lateral = vh_lateral.conj().T

    spectra, sweep = [], []
    started = time.time()
    for n, path in enumerate(paths, 1):
        scan, magnitude = target_magnitude(path)

        # --- 1. energy per lateral singular mode --------------------------
        base = HolographicInverseConfig(phase_iterations=0)
        zero_coefficients, _ = _solve_complex_coefficients(magnitude, scanner, base)
        selected = HolographicInverseConfig(phase_iterations=200)
        selected_coefficients, _ = _solve_complex_coefficients(magnitude, scanner, selected)
        entry = {"scan": path.name, "series": series_key(path)}
        for name, coefficients in (
            ("zero_phase", zero_coefficients),
            ("selected_phase", selected_coefficients),
        ):
            energy = lateral_mode_energy(coefficients, v_lateral)
            total = float(energy.sum())
            entry[name] = {
                "fraction_below_sigma_0.05": float(energy[sv_lateral < 0.05].sum() / total),
                "fraction_below_sigma_0.20": float(energy[sv_lateral < 0.20].sum() / total),
                "fraction_in_top_64_modes": float(energy[:64].sum() / total),
                "mode_participation_ratio": float(total**2 / np.sum(energy**2)),
            }
        spectra.append(entry)

        # --- 2. alpha_L sweep for both complete configurations ------------
        row = {"scan": path.name, "series": series_key(path)}
        for alpha in ALPHA_L:
            earlier = HolographicInverseConfig(
                axial_regularization=0.03,
                lateral_regularization=alpha,
                phase_iterations=0,
                phase_encoding="single",
            )
            coefficients, _ = _solve_complex_coefficients(magnitude, scanner, earlier)
            rows = encode(coefficients, scanner, mode="single").rows
            score, image = render_and_score(rows, scan)
            row[f"earlier_alphaL_{alpha}"] = {
                "msssim": score,
                "lateral_roughness": lateral_roughness(image),
            }

            current = HolographicInverseConfig(
                axial_regularization=0.02,
                lateral_regularization=alpha,
                phase_iterations=200,
            )
            coefficients, _ = _solve_complex_coefficients(magnitude, scanner, current)
            rows = encode(coefficients, scanner, mode="pair").rows
            score, image = render_and_score(rows, scan)
            row[f"current_alphaL_{alpha}"] = {
                "msssim": score,
                "lateral_roughness": lateral_roughness(image),
            }
        sweep.append(row)
        print(f"[{n}/{len(paths)}] {path.name} ({(time.time()-started)/n:.1f} s/scan)", flush=True)

    dump("regularization_spectra.json", spectra)
    dump("regularization_sweep.json", sweep)

    reference_roughness = float(
        np.mean([lateral_roughness(target_magnitude(p)[0]) for p in paths])
    )
    summary = {
        "reference_lateral_roughness": reference_roughness,
        "lateral_singular_values": {
            "n_below_0.05": int(np.sum(sv_lateral < 0.05)),
            "n_below_0.20": int(np.sum(sv_lateral < 0.20)),
            "min": float(sv_lateral.min()),
            "max": float(sv_lateral.max()),
        },
        "mode_energy": {
            name: {
                key: float(np.mean([e[name][key] for e in spectra]))
                for key in spectra[0]["zero_phase"]
            }
            for name in ("zero_phase", "selected_phase")
        },
        "sweep": {
            key: {
                "msssim_mean": float(np.mean([r[key]["msssim"] for r in sweep])),
                "roughness_mean": float(np.mean([r[key]["lateral_roughness"] for r in sweep])),
            }
            for key in sweep[0]
            if key.startswith(("earlier_", "current_"))
        },
    }
    for prefix in ("earlier", "current"):
        keys = [k for k in summary["sweep"] if k.startswith(prefix)]
        best = max(keys, key=lambda k: summary["sweep"][k]["msssim_mean"])
        summary[f"best_alpha_L_{prefix}"] = float(best.rsplit("_", 1)[1])
    dump("regularization_summary.json", summary)
    print("\n" + "\n".join(f"{k}: {v}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
