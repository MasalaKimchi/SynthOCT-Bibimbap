from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import io

from .phantom import ExperimentConfig, load_phantom


def phantom_to_surrogate_scan(
    phantom_path: str | Path,
    output_path: str | Path | None = None,
    config: ExperimentConfig = ExperimentConfig(),
    seed: int = 0,
) -> np.ndarray:
    """Render a deterministic, OCT-like surrogate B-scan from a phantom.

    This is not the official scanner. It is a fast internal proxy that rewards
    physically plausible depth attenuation, lateral coherence, and speckle.
    """
    data = load_phantom(phantom_path)
    x_idx = np.clip(((data[:, 0] + config.x_max / 2) / config.x_max * config.n_lateral).astype(int), 0, config.n_lateral - 1)
    z_idx = np.clip((data[:, 2] / config.z_max * config.n_depth).astype(int), 0, config.n_depth - 1)
    energy = np.clip(data[:, 3], 0, 100) / 100.0

    backscatter = np.zeros((config.n_depth, config.n_lateral), dtype=np.float64)
    np.add.at(backscatter, (z_idx, x_idx), np.sqrt(energy))
    backscatter = gaussian_filter(backscatter, sigma=(1.1, 0.7), mode="reflect")

    mu = gaussian_filter(backscatter, sigma=(5.0, 3.0), mode="reflect")
    mu = mu / (np.percentile(mu, 99) + 1e-8)
    optical_depth = np.cumsum(mu, axis=0) * 0.018
    attenuation = np.exp(-2.0 * optical_depth)

    rng = np.random.default_rng(seed)
    speckle = rng.rayleigh(scale=1.0, size=backscatter.shape)
    speckle = gaussian_filter(speckle, sigma=(0.35, 0.35), mode="reflect")
    signal = backscatter * attenuation * speckle
    signal = gaussian_filter(signal, sigma=(0.6, 0.35), mode="reflect")

    db = 20 * np.log10(signal / (np.percentile(signal, 99.7) + 1e-10) + 1e-4)
    scan = np.clip((db + 51.0) / 51.0, 0.0, 1.0).astype(np.float32)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        io.imsave(output_path, (scan * 255).astype(np.uint8))
    return scan
