from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.fft import ifft, ifftshift
from skimage.transform import resize

from .features import load_scan
from .phantom import ExperimentConfig, save_phantom
from .scanners.reference import scanner_wavenumbers


@dataclass(frozen=True)
class HolographicInverseConfig:
    """Regularization and physical scaling for coherent inverse synthesis."""

    axial_regularization: float = 0.03
    lateral_regularization: float = 0.20
    max_reflection_amplitude: float = 0.001
    dynamic_range_db: float = 51.0

    def validate(self) -> None:
        if min(self.axial_regularization, self.lateral_regularization) <= 0:
            raise ValueError("inverse regularization must be positive")
        if not 0 < self.max_reflection_amplitude <= 1:
            raise ValueError("max_reflection_amplitude must be in (0, 1]")
        if self.dynamic_range_db <= 0:
            raise ValueError("dynamic_range_db must be positive")


def _tikhonov_pseudoinverse(matrix: np.ndarray, regularization: float) -> np.ndarray:
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    weights = singular_values / (singular_values**2 + regularization**2)
    return (vh.conj().T * weights) @ u.conj().T


@lru_cache(maxsize=8)
def _inverse_operators(
    n_depth: int,
    n_lateral: int,
    pixel_size_z: float,
    pixel_size_x: float,
    wavelength: float,
    beam_radius: float,
    axial_regularization: float,
    lateral_regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    scanner = ExperimentConfig(
        n_depth=n_depth,
        n_lateral=n_lateral,
        pixel_size_z=pixel_size_z,
        pixel_size_x=pixel_size_x,
        wavelength=wavelength,
        beam_radius=beam_radius,
    )
    k = scanner_wavenumbers(scanner)
    window = ifftshift(np.hanning(n_depth))
    z_centers = (np.arange(n_depth, dtype=np.float64) + 0.5) * pixel_size_z
    axial = np.stack([ifft(window * np.exp(-2j * k * z)) for z in z_centers], axis=1)

    x_centers = (np.arange(n_lateral, dtype=np.float64) - n_lateral / 2.0) * pixel_size_x
    distance_sq = (x_centers[:, None] - x_centers[None, :]) ** 2
    lateral = np.exp(-distance_sq / beam_radius**2)
    lateral[distance_sq >= 3.0 * beam_radius**2] = 0.0
    return (
        _tikhonov_pseudoinverse(axial, axial_regularization),
        _tikhonov_pseudoinverse(lateral, lateral_regularization),
    )


def holographic_inverse_phantom(
    reference_path: str | Path,
    output_path: str | Path,
    *,
    seed: int = 7,
    scatterers_count: int = 300_000,
    inverse: HolographicInverseConfig | None = None,
    diagnostics_path: str | Path | None = None,
    scanner: ExperimentConfig | None = None,
) -> Path:
    """Invert the fixed coherent scanner into a challenge-format phantom.

    A complex coefficient is solved at every axial/lateral voxel. Its magnitude
    becomes reflection energy and its phase becomes a sub-wavelength z offset.
    Remaining rows are zero-energy filler scatterers so the official 300k-row
    contract is preserved without changing the rendered signal.
    """
    inverse = inverse or HolographicInverseConfig()
    inverse.validate()
    scanner = scanner or ExperimentConfig(scatterers_count=scatterers_count)
    active_count = scanner.n_depth * scanner.n_lateral
    if scatterers_count < active_count:
        raise ValueError(f"holographic inversion requires at least {active_count} scatterers")

    scan = np.asarray(load_scan(reference_path), dtype=np.float64)
    shape = (scanner.n_depth, scanner.n_lateral)
    if scan.shape != shape:
        scan = np.asarray(resize(scan, shape, anti_aliasing=True, preserve_range=True), dtype=np.float64)
    target_magnitude = np.power(10.0, (scan * inverse.dynamic_range_db - inverse.dynamic_range_db) / 20.0)
    target_field = target_magnitude.astype(np.complex128)

    axial_inverse, lateral_inverse = _inverse_operators(
        scanner.n_depth,
        scanner.n_lateral,
        scanner.pixel_size_z,
        scanner.pixel_size_x,
        scanner.wavelength,
        scanner.beam_radius,
        inverse.axial_regularization,
        inverse.lateral_regularization,
    )
    coefficients = axial_inverse @ target_field @ lateral_inverse.T
    coefficient_magnitude = np.abs(coefficients)
    coefficient_phase = np.angle(coefficients)
    peak_coefficient = max(float(coefficient_magnitude.max()), 1e-15)
    amplitude_scale = inverse.max_reflection_amplitude / peak_coefficient
    energies = 100.0 * (coefficient_magnitude * amplitude_scale) ** 2

    x_centers = (
        np.arange(scanner.n_lateral, dtype=np.float64) - scanner.n_lateral / 2.0
    ) * scanner.pixel_size_x
    z_centers = (np.arange(scanner.n_depth, dtype=np.float64) + 0.5) * scanner.pixel_size_z
    phase_offsets = -coefficient_phase * scanner.wavelength / (4.0 * np.pi)
    active = np.column_stack(
        (
            np.tile(x_centers, scanner.n_depth),
            np.zeros(active_count, dtype=np.float64),
            (z_centers[:, None] + phase_offsets).ravel(),
            energies.ravel(),
        )
    )

    filler_count = scatterers_count - active_count
    if filler_count:
        rng = np.random.default_rng(seed)
        filler = np.column_stack(
            (
                rng.uniform(-scanner.x_max / 2.0, scanner.x_max / 2.0, filler_count),
                rng.uniform(-scanner.beam_radius, scanner.beam_radius, filler_count),
                rng.uniform(0.0, scanner.z_max, filler_count),
                np.zeros(filler_count, dtype=np.float64),
            )
        )
        phantom = np.vstack((active, filler))
    else:
        phantom = active

    output_scanner = ExperimentConfig(
        n_depth=scanner.n_depth,
        n_lateral=scanner.n_lateral,
        pixel_size_z=scanner.pixel_size_z,
        pixel_size_x=scanner.pixel_size_x,
        wavelength=scanner.wavelength,
        beam_radius=scanner.beam_radius,
        b_scans_count=scanner.b_scans_count,
        scatterers_count=scatterers_count,
        config_filename=scanner.config_filename,
        scan_filename=scanner.scan_filename,
    )
    result = save_phantom(phantom, output_path, config=output_scanner)
    if diagnostics_path is not None:
        diagnostics_path = Path(diagnostics_path)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generator": "holographic-inverse-v1",
            "scatterers_count": scatterers_count,
            "active_scatterers": active_count,
            "zero_energy_fillers": filler_count,
            "inverse": {
                "axial_regularization": inverse.axial_regularization,
                "lateral_regularization": inverse.lateral_regularization,
                "max_reflection_amplitude": inverse.max_reflection_amplitude,
                "dynamic_range_db": inverse.dynamic_range_db,
            },
            "energy_percent_quantiles": np.quantile(energies, [0.0, 0.5, 0.9, 0.99, 1.0]).tolist(),
            "coefficient_magnitude_quantiles": np.quantile(
                coefficient_magnitude, [0.0, 0.5, 0.9, 0.99, 1.0]
            ).tolist(),
        }
        diagnostics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
