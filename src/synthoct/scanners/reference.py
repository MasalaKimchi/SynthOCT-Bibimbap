from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit, prange
from scipy.fft import ifft, ifftshift
from skimage import io

from synthoct.phantom import ExperimentConfig, load_phantom


@njit(parallel=True, cache=True)
def _accumulate_all_ascans(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    amplitudes: np.ndarray,
    z_order: np.ndarray,
    x_ascans: np.ndarray,
    w0: float,
    k_samples: np.ndarray,
) -> np.ndarray:
    """Reproduce the fixed scanner's coherent spectral accumulation."""
    ascans = len(x_ascans)
    n_scatterers = len(xs)
    n_k = len(k_samples)
    spectra = np.zeros((ascans, n_k), dtype=np.complex128)
    limit_sq = 3.0 * w0**2

    for ix in prange(ascans):
        beam = np.empty(n_scatterers, dtype=np.float64)
        distance_sq = np.empty(n_scatterers, dtype=np.float64)
        x_a = x_ascans[ix]
        for j in range(n_scatterers):
            d2 = (x_a - xs[j]) ** 2 + ys[j] ** 2
            distance_sq[j] = d2
            beam[j] = np.exp(-d2 / w0**2)

        transmission = np.empty(n_scatterers, dtype=np.float64)
        transmission[0] = 1.0
        for rank in range(1, n_scatterers):
            previous = z_order[rank - 1]
            term = (beam[previous] * amplitudes[previous]) ** 2
            if term > 1.0:
                term = 1.0
            transmission[rank] = transmission[rank - 1] * (1.0 - term)

        spectrum = np.zeros(n_k, dtype=np.complex128)
        for rank in range(n_scatterers):
            idx = z_order[rank]
            if distance_sq[idx] >= limit_sq:
                continue
            coefficient = transmission[rank] * amplitudes[idx] * beam[idx]
            z = zs[idx]
            for ik in range(n_k):
                phase = -2.0 * k_samples[ik] * z
                spectrum[ik] += coefficient * (np.cos(phase) + 1j * np.sin(phase))
        spectra[ix] = spectrum
    return spectra


def scanner_wavenumbers(config: ExperimentConfig) -> np.ndarray:
    """Return the exact ifft-shifted wavenumber samples used by Part2."""
    n = config.n_depth
    height = config.pixel_size_z * (n - 1)
    dk = np.pi / height * n / (n + 1)
    k0 = 2.0 * np.pi / config.wavelength
    half_n = n // 2
    k = np.empty(n, dtype=np.float64)
    k[: half_n + 1] = k0 + np.arange(half_n + 1, dtype=np.float64) * dk
    k[half_n + 1 :] = k0 - np.arange(half_n - 1, 0, -1, dtype=np.float64) * dk
    return k


def render_reference_array(
    phantom: np.ndarray,
    *,
    config: ExperimentConfig | None = None,
) -> np.ndarray:
    """Render a phantom with the local implementation of the published Part2 model."""
    config = config or ExperimentConfig(scatterers_count=len(phantom))
    data = np.asarray(phantom[: config.scatterers_count], dtype=np.float64)
    xs, ys, zs, energy = data.T
    amplitudes = np.sqrt(energy / 100.0)
    z_order = np.argsort(zs).astype(np.int64)
    dx = config.x_max / config.n_lateral
    x_ascans = (np.arange(config.n_lateral, dtype=np.float64) - config.n_lateral / 2.0) * dx
    spectra = _accumulate_all_ascans(
        xs,
        ys,
        zs,
        amplitudes,
        z_order,
        x_ascans,
        float(config.beam_radius),
        scanner_wavenumbers(config),
    )
    window = ifftshift(np.hanning(config.n_depth))
    signal = ifft(spectra * window[None, :], axis=1)
    magnitude = np.abs(signal)
    magnitude[magnitude == 0] = 1e-10
    db = 20.0 * np.log10(magnitude).T
    display = np.clip(db - float(db.max()) + 51.0, 0.0, None)
    peak = float(display.max()) or 1.0
    return np.asarray(display / peak, dtype=np.float64)


def render_reference_scanner(
    phantom_path: str | Path,
    output_png: str | Path,
    *,
    config: ExperimentConfig | None = None,
) -> Path:
    """Render and save a grayscale PNG with the recovered fixed scanner model."""
    phantom = load_phantom(phantom_path)
    image = render_reference_array(phantom, config=config)
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(output_png, np.round(image * 255.0).astype(np.uint8), check_contrast=False)
    return output_png
