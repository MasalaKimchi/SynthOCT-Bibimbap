from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage import io

HPX = 6.0
WINDOW_SIZE = 20


def load_scan(path: str | Path) -> np.ndarray:
    """Load a real or rendered OCT B-scan as a normalized grayscale image."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        arr = io.imread(path, as_gray=True)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.max(initial=0) > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def load_and_linearize_image(path: str | Path) -> np.ndarray:
    """Convert display-space OCT intensity into the linearized scale used by OAC maps."""
    img = load_scan(path)
    return 10.0 ** (img * 4.0)


def calculate_oac(intensity: np.ndarray, h_px: float = HPX) -> np.ndarray:
    """Estimate the optical attenuation coefficient map from a reference scan."""
    epsilon = 1e-10
    cumsum_from_bottom = np.cumsum(intensity[::-1, :], axis=0)[::-1, :]
    denom = cumsum_from_bottom - 0.5 * intensity
    return intensity / (2 * h_px * (denom + epsilon))


def calculate_speckle_contrast_map(data: np.ndarray, window_size: int = WINDOW_SIZE) -> np.ndarray:
    """Estimate local speckle contrast over a sliding window."""
    mean_val = uniform_filter(data, size=window_size, mode="reflect")
    mean_sq_val = uniform_filter(data**2, size=window_size, mode="reflect")
    var_val = np.maximum(mean_sq_val - mean_val**2, 0)
    sc_map = np.sqrt(var_val) / (mean_val + 1e-10)
    border = window_size // 2
    if min(sc_map.shape) <= 2 * border:
        return sc_map
    cropped = sc_map[border:-border, border:-border]
    return np.pad(cropped, border, mode="edge")


def estimate_layer_boundary(scan: np.ndarray) -> int:
    """Locate the strongest smoothed depth-gradient boundary in a reference B-scan."""
    profile = scan.mean(axis=1)
    smooth = gaussian_filter(profile, sigma=max(1, len(profile) // 80))
    grad = np.abs(np.gradient(smooth))
    return int(np.argmax(grad))


def normalize_map(data: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    if vmin is None:
        vmin = float(np.min(data))
    if vmax is None:
        vmax = float(np.max(data))
    return np.clip((data - vmin) / (vmax - vmin + 1e-10), 0, 1)


def generate_maps(input_image_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Path]:
    """Write challenge-style Struct/OAC/SC/RSC PNG maps for evaluation."""
    input_image_path = Path(input_image_path)
    output_dir = Path(output_dir) if output_dir else input_image_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / input_image_path.stem

    intensity = load_and_linearize_image(input_image_path)
    mu_map = calculate_oac(intensity)
    sc_map = calculate_speckle_contrast_map(intensity)
    rsc_map = calculate_speckle_contrast_map(mu_map)

    paths = {
        "Struct": input_image_path,
        "OAC": base.with_name(base.name + "_OAC.png"),
        "SC": base.with_name(base.name + "_SC.png"),
        "RSC": base.with_name(base.name + "_RSC.png"),
    }
    io.imsave(paths["OAC"], (normalize_map(mu_map, vmax=np.percentile(mu_map, 99)) * 255).astype(np.uint8))
    io.imsave(paths["SC"], (normalize_map(sc_map, vmin=0.5, vmax=5.0) * 255).astype(np.uint8))
    io.imsave(paths["RSC"], (normalize_map(rsc_map, vmin=0.5, vmax=5.0) * 255).astype(np.uint8))
    return paths
