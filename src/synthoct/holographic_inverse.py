from __future__ import annotations

import hashlib
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

    axial_regularization: float = 0.02
    lateral_regularization: float = 0.05
    max_reflection_amplitude: float = 0.001
    dynamic_range_db: float = 51.0
    phase_iterations: int = 200
    phase_momentum: float = 1.0
    phase_encoding: str = "dispersion-canceling-pair"

    def validate(self) -> None:
        numeric_controls = (
            self.axial_regularization,
            self.lateral_regularization,
            self.max_reflection_amplitude,
            self.dynamic_range_db,
            self.phase_momentum,
        )
        if not np.isfinite(numeric_controls).all():
            raise ValueError("inverse controls must be finite")
        if min(self.axial_regularization, self.lateral_regularization) <= 0:
            raise ValueError("inverse regularization must be positive")
        if not 0 < self.max_reflection_amplitude <= 1:
            raise ValueError("max_reflection_amplitude must be in (0, 1]")
        if self.dynamic_range_db <= 0:
            raise ValueError("dynamic_range_db must be positive")
        if isinstance(self.phase_iterations, bool) or not isinstance(self.phase_iterations, int):
            raise ValueError("phase_iterations must be an integer")
        if self.phase_iterations < 0:
            raise ValueError("phase_iterations must be non-negative")
        if not 0.0 <= self.phase_momentum <= 1.0:
            raise ValueError("phase_momentum must be in [0, 1]")
        if self.phase_encoding not in {"single", "dispersion-canceling-pair"}:
            raise ValueError("phase_encoding must be 'single' or 'dispersion-canceling-pair'")


def _tikhonov_pseudoinverse(matrix: np.ndarray, regularization: float) -> np.ndarray:
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    weights = singular_values / (singular_values**2 + regularization**2)
    return (vh.conj().T * weights) @ u.conj().T


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _forward_operators(
    n_depth: int,
    n_lateral: int,
    pixel_size_z: float,
    pixel_size_x: float,
    wavelength: float,
    beam_radius: float,
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
    return axial, lateral


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
    axial, lateral = _forward_operators(
        n_depth,
        n_lateral,
        pixel_size_z,
        pixel_size_x,
        wavelength,
        beam_radius,
    )
    return (
        _tikhonov_pseudoinverse(axial, axial_regularization),
        _tikhonov_pseudoinverse(lateral, lateral_regularization),
    )


def _solve_complex_coefficients(
    target_magnitude: np.ndarray,
    scanner: ExperimentConfig,
    inverse: HolographicInverseConfig,
) -> tuple[np.ndarray, float]:
    """Find a phase favored by the regularized local forward model.

    OCT intensity fixes field magnitude but leaves its complex phase free.  The
    original inverse forced every target pixel to zero phase, which needlessly
    amplified the two axial null modes and produced ringing.  Alternating
    projections choose a phase that survives the regularized scanner operator.
    Circular phase momentum accelerates the otherwise slow Gerchberg-Saxton
    iteration without changing the target magnitude.
    """
    axial, lateral = _forward_operators(
        scanner.n_depth,
        scanner.n_lateral,
        scanner.pixel_size_z,
        scanner.pixel_size_x,
        scanner.wavelength,
        scanner.beam_radius,
    )
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

    target_field = target_magnitude.astype(np.complex128)
    previous_projected_phase: np.ndarray | None = None
    for _ in range(inverse.phase_iterations):
        coefficients = axial_inverse @ target_field @ lateral_inverse.T
        projected_field = axial @ coefficients @ lateral.T
        projected_phase = np.angle(projected_field)
        if previous_projected_phase is not None and inverse.phase_momentum:
            phase_delta = np.angle(
                np.exp(1j * (projected_phase - previous_projected_phase))
            )
            target_phase = projected_phase + inverse.phase_momentum * phase_delta
        else:
            target_phase = projected_phase
        previous_projected_phase = projected_phase
        target_field = target_magnitude * np.exp(1j * target_phase)

    coefficients = axial_inverse @ target_field @ lateral_inverse.T
    projected_magnitude = np.abs(axial @ coefficients @ lateral.T)
    magnitude_mae = float(np.mean(np.abs(projected_magnitude - target_magnitude)))
    return coefficients, magnitude_mae


def _encode_coefficients(
    coefficients: np.ndarray,
    scanner: ExperimentConfig,
    inverse: HolographicInverseConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode complex coefficients as nonnegative physical scatterers.

    The paired encoding uses two carrier-equivalent depths separated by half a
    wavelength.  Nonnegative amplitude weights preserve the coefficient at the
    center wavenumber while making their weighted depth offset zero.  This
    cancels first-order phase dispersion across the scanner bandwidth and fits
    in the 300k contract with two rows per 256x512 voxel.
    """
    magnitude = np.abs(coefficients).ravel()
    phase = np.angle(coefficients).ravel()
    x_centers = (
        np.arange(scanner.n_lateral, dtype=np.float64) - scanner.n_lateral / 2.0
    ) * scanner.pixel_size_x
    z_centers = (np.arange(scanner.n_depth, dtype=np.float64) + 0.5) * scanner.pixel_size_z
    base_x = np.tile(x_centers, scanner.n_depth)
    base_z = np.repeat(z_centers, scanner.n_lateral)
    primary_offset = -phase * scanner.wavelength / (4.0 * np.pi)

    if inverse.phase_encoding == "single":
        peak_amplitude = max(float(magnitude.max()), 1e-15)
        amplitude_scale = inverse.max_reflection_amplitude / peak_amplitude
        energies = 100.0 * (magnitude * amplitude_scale) ** 2
        active = np.column_stack(
            (
                base_x,
                np.zeros(len(magnitude), dtype=np.float64),
                base_z + primary_offset,
                energies,
            )
        )
        return active, energies

    offset_sign = np.where(primary_offset >= 0.0, 1.0, -1.0)
    companion_offset = primary_offset - offset_sign * scanner.wavelength / 2.0
    offset_separation = primary_offset - companion_offset
    primary_weight = -companion_offset / offset_separation
    companion_weight = primary_offset / offset_separation
    primary_amplitude = magnitude * primary_weight
    companion_amplitude = magnitude * companion_weight
    peak_amplitude = max(
        float(primary_amplitude.max()),
        float(companion_amplitude.max()),
        1e-15,
    )
    amplitude_scale = inverse.max_reflection_amplitude / peak_amplitude
    primary_energy = 100.0 * (primary_amplitude * amplitude_scale) ** 2
    companion_energy = 100.0 * (companion_amplitude * amplitude_scale) ** 2

    active = np.empty((2 * len(magnitude), 4), dtype=np.float64)
    active[0::2, 0] = base_x
    active[1::2, 0] = base_x
    active[:, 1] = 0.0
    active[0::2, 2] = base_z + primary_offset
    active[1::2, 2] = base_z + companion_offset
    active[0::2, 3] = primary_energy
    active[1::2, 3] = companion_energy
    energies = np.column_stack((primary_energy, companion_energy)).ravel()
    return active, energies


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
    rows_per_voxel = 2 if inverse.phase_encoding == "dispersion-canceling-pair" else 1
    active_count = rows_per_voxel * scanner.n_depth * scanner.n_lateral
    if scatterers_count < active_count:
        raise ValueError(f"holographic inversion requires at least {active_count} scatterers")

    scan = np.asarray(load_scan(reference_path), dtype=np.float64)
    shape = (scanner.n_depth, scanner.n_lateral)
    if scan.shape != shape:
        scan = np.asarray(resize(scan, shape, anti_aliasing=True, preserve_range=True), dtype=np.float64)
    target_magnitude = np.power(10.0, (scan * inverse.dynamic_range_db - inverse.dynamic_range_db) / 20.0)
    coefficients, projected_magnitude_mae = _solve_complex_coefficients(
        target_magnitude,
        scanner,
        inverse,
    )
    coefficient_magnitude = np.abs(coefficients)
    active, energies = _encode_coefficients(coefficients, scanner, inverse)

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
            "schema_version": 2,
            "generator": (
                "holographic-inverse-v3-phase-pair"
                if inverse.phase_encoding == "dispersion-canceling-pair"
                else (
                    "holographic-inverse-v2-phase-retrieval"
                    if inverse.phase_iterations
                    else "holographic-inverse-v1"
                )
            ),
            "scatterers_count": scatterers_count,
            "active_scatterers": active_count,
            "zero_energy_fillers": filler_count,
            "reference": {
                "path": str(Path(reference_path)),
                "sha256": _sha256(Path(reference_path)),
                "size_bytes": Path(reference_path).stat().st_size,
            },
            "phantom": {
                "path": str(Path(result)),
                "sha256": _sha256(Path(result)),
                "size_bytes": Path(result).stat().st_size,
            },
            "inverse": {
                "axial_regularization": inverse.axial_regularization,
                "lateral_regularization": inverse.lateral_regularization,
                "max_reflection_amplitude": inverse.max_reflection_amplitude,
                "dynamic_range_db": inverse.dynamic_range_db,
                "phase_iterations": inverse.phase_iterations,
                "phase_momentum": inverse.phase_momentum,
                "phase_encoding": inverse.phase_encoding,
            },
            "projected_magnitude_mae": projected_magnitude_mae,
            "energy_percent_quantiles": np.quantile(
                energies, [0.0, 0.5, 0.9, 0.99, 1.0]
            ).tolist(),
            "coefficient_magnitude_quantiles": np.quantile(
                coefficient_magnitude, [0.0, 0.5, 0.9, 0.99, 1.0]
            ).tolist(),
        }
        diagnostics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
