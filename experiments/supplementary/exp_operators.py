"""Operator geometry, conditioning, and effective degrees of freedom.

Supports Supplement S3 by quantifying ill-posedness, resolution-element
occupancy, and what the regularized scanner constrains.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from analysis_lib import configure_paths, dump, operators
from synthoct.holographic_inverse import HolographicInverseConfig
from synthoct.phantom import ExperimentConfig
from synthoct.scanners.reference import scanner_wavenumbers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="generated-result directory (default: outputs/experiments/supplementary)",
    )
    return parser.parse_args()


def fwhm_samples(profile: np.ndarray, spacing: float) -> float:
    """FWHM of a unimodal profile, linearly interpolated, in physical units."""
    peak = float(profile.max())
    idx = int(profile.argmax())
    half = peak / 2.0

    def cross(direction: int) -> float:
        i = idx
        while 0 <= i + direction < len(profile) and profile[i + direction] > half:
            i += direction
        j = i + direction
        if not (0 <= j < len(profile)):
            return float(abs(i - idx))
        num = profile[i] - half
        den = profile[i] - profile[j]
        frac = num / den if den else 0.0
        return abs(i + direction * frac - idx)

    return (cross(-1) + cross(+1)) * spacing


def main() -> None:
    args = parse_args()
    configure_paths(output_dir=args.output_dir)
    scanner = ExperimentConfig(scatterers_count=300_000)
    inverse = HolographicInverseConfig()
    axial, lateral, axial_inv, lateral_inv = operators(scanner, inverse)

    sv_axial = np.linalg.svd(axial, compute_uv=False)
    sv_lateral = np.linalg.svd(lateral, compute_uv=False)
    axial_rank = int(np.linalg.matrix_rank(axial))

    def dof(sv: np.ndarray, alpha: float) -> float:
        return float(np.sum(sv**2 / (sv**2 + alpha**2)))

    # --- lateral geometry -------------------------------------------------
    mid = scanner.n_lateral // 2
    lateral_row = lateral[mid]
    nonzero = np.flatnonzero(lateral_row)
    lateral_amp_fwhm = fwhm_samples(lateral_row, scanner.pixel_size_x)
    lateral_int_fwhm = fwhm_samples(lateral_row**2, scanner.pixel_size_x)
    # Two-way (illumination x collection) sensitivity is the square of the
    # single-pass amplitude profile used by the renderer.
    lateral_cols_nonzero = int(nonzero.size)

    # --- axial geometry ---------------------------------------------------
    axial_col = np.abs(axial[:, scanner.n_depth // 2])
    axial_amp_fwhm = fwhm_samples(axial_col, scanner.pixel_size_z)
    axial_int_fwhm = fwhm_samples(axial_col**2, scanner.pixel_size_z)
    # energy-based support: smallest window holding 95% of the axial response
    order = np.argsort(-axial_col)
    cumulative = np.cumsum(axial_col[order] ** 2) / np.sum(axial_col**2)
    axial_support_95 = int(np.searchsorted(cumulative, 0.95) + 1)

    k = scanner_wavenumbers(scanner)
    k0 = 2.0 * np.pi / scanner.wavelength
    half_wave_phase = 2.0 * (scanner.wavelength / 2.0) * k0

    # --- what a resolution element contains -------------------------------
    lat_cells = lateral_amp_fwhm / scanner.pixel_size_x
    ax_cells = axial_amp_fwhm / scanner.pixel_size_z
    payload = {
        "operator_shapes": {"axial": list(axial.shape), "lateral": list(lateral.shape)},
        "singular_values": {
            "axial": {
                "max": float(sv_axial.max()),
                "min": float(sv_axial.min()),
                "median": float(np.median(sv_axial)),
                "numerical_rank": axial_rank,
                "nullity": int(axial.shape[1] - axial_rank),
                "finite_precision_max_to_min_ratio": float(
                    sv_axial.max() / max(sv_axial.min(), 1e-300)
                ),
                "n_below_alpha": int(np.sum(sv_axial < inverse.axial_regularization)),
                "alpha": inverse.axial_regularization,
            },
            "lateral": {
                "max": float(sv_lateral.max()),
                "min": float(sv_lateral.min()),
                "median": float(np.median(sv_lateral)),
                "condition_number": float(
                    sv_lateral.max() / max(sv_lateral.min(), 1e-300)
                ),
                "n_below_alpha_005": int(np.sum(sv_lateral < 0.05)),
                "n_below_alpha_020": int(np.sum(sv_lateral < 0.20)),
            },
        },
        "effective_dof": {
            "axial_alpha_0.02": dof(sv_axial, 0.02),
            "axial_alpha_0.03": dof(sv_axial, 0.03),
            "axial_of": scanner.n_depth,
            "lateral_alpha_0.05": dof(sv_lateral, 0.05),
            "lateral_alpha_0.20": dof(sv_lateral, 0.20),
            "lateral_of": scanner.n_lateral,
        },
        "lateral_geometry": {
            "beam_radius_um": scanner.beam_radius,
            "pixel_size_x_um": scanner.pixel_size_x,
            "amplitude_fwhm_um": lateral_amp_fwhm,
            "intensity_fwhm_um": lateral_int_fwhm,
            "nonzero_columns_per_ascan": lateral_cols_nonzero,
            "truncation_radius_um": float(np.sqrt(3.0) * scanner.beam_radius),
            "columns_within_amplitude_fwhm": lat_cells,
            "row_weights": {
                str(int(round((j - mid) * scanner.pixel_size_x))): float(lateral_row[j])
                for j in nonzero
            },
        },
        "axial_geometry": {
            "amplitude_fwhm_um": axial_amp_fwhm,
            "intensity_fwhm_um": axial_int_fwhm,
            "samples_within_amplitude_fwhm": ax_cells,
            "samples_holding_95pct_energy": axial_support_95,
            "pixel_size_z_um": scanner.pixel_size_z,
        },
        "resolution_element_occupancy": {
            "definition": (
                "voxels whose axial and lateral responses overlap within the "
                "amplitude FWHM of the scanner point-spread function, times two "
                "encoding rows per voxel"
            ),
            "voxels_per_resolution_element": lat_cells * ax_cells,
            "encoded_rows_per_resolution_element": 2.0 * lat_cells * ax_cells,
        },
        "carrier": {
            "k0_rad_per_um": k0,
            "k0_is_a_scanner_sample": bool(np.isclose(k[0], k0)),
            "k_min": float(k.min()),
            "k_max": float(k.max()),
            "fractional_bandwidth": float((k.max() - k.min()) / k0),
            "two_way_phase_over_half_wavelength_rad": half_wave_phase,
            "two_way_phase_over_half_wavelength_in_2pi": half_wave_phase / (2 * np.pi),
        },
    }
    path = dump("operators.json", payload)
    print(f"wrote {path}")
    for key in (
        "singular_values",
        "effective_dof",
        "lateral_geometry",
        "axial_geometry",
        "resolution_element_occupancy",
        "carrier",
    ):
        print(f"\n== {key} ==")
        print(
            "\n".join(
                f"  {k}: {v}"
                for k, v in payload[key].items()
                if not isinstance(v, dict)
            )
            or ""
        )
        for k, v in payload[key].items():
            if isinstance(v, dict):
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
