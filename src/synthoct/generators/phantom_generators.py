from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

from synthoct.features import (
    calculate_oac,
    calculate_speckle_contrast_map,
    estimate_layer_boundary,
    load_and_linearize_image,
    load_scan,
)
from synthoct.phantom import ExperimentConfig, generate_two_layers, generate_uniform, save_phantom


PhantomConfig = dict[str, float | bool]

HYPOTHESIS_CONFIGS: dict[str, PhantomConfig] = {
    "H11_low_depth_comp": {
        "density_power": 1.0,
        "depth_compensation": 1.45,
        "oac_weight": 2.0,
        "texture_weight": 0.35,
        "energy_oac_scale": 35.0,
        "band_boost": 0.0,
        "lateral_smooth": 0.8,
        "oac_percentile": 70.0,
        "base_energy_mix": 0.55,
        "texture_sigma_scale": 1.0,
    },
    "H61_api_low_depth_prelim": {
        "density_power": 0.90,
        "depth_compensation": -0.80,
        "oac_weight": 1.20,
        "texture_weight": 0.22,
        "energy_oac_scale": 10.0,
        "band_boost": 0.00,
        "lateral_smooth": 0.90,
        "oac_percentile": 70.0,
        "base_energy_mix": 0.40,
        "texture_sigma_scale": 0.70,
    },
    "H67_coarse_to_fine_crisp": {
        "density_power": 1.32,
        "depth_compensation": 1.16,
        "oac_weight": 2.28,
        "texture_weight": 0.32,
        "energy_oac_scale": 27.0,
        "band_boost": 0.26,
        "lateral_smooth": 0.28,
        "oac_percentile": 82.0,
        "base_energy_mix": 0.76,
        "texture_sigma_scale": 1.10,
    },
    "H68_layer_map_prior": {
        "density_power": 1.08,
        "depth_compensation": 1.34,
        "oac_weight": 2.05,
        "texture_weight": 0.30,
        "energy_oac_scale": 34.0,
        "band_boost": 0.48,
        "lateral_smooth": 0.82,
        "oac_percentile": 74.0,
        "base_energy_mix": 0.58,
        "texture_sigma_scale": 0.90,
        "multi_layer": True,
    },
}

FINAL_CONFIG_NAME = "H61_api_low_depth_prelim"

HYPOTHESIS_WAVES: dict[str, tuple[str, ...]] = {
    "api-candidates": ("H61_api_low_depth_prelim", "H67_coarse_to_fine_crisp", "H68_layer_map_prior"),
    "api-with-backup": ("H61_api_low_depth_prelim", "H67_coarse_to_fine_crisp", "H68_layer_map_prior", "H11_low_depth_comp"),
}


def official_baseline_phantom(
    output_path: str | Path,
    method: str = "two-layer",
    seed: int = 7,
    scatterers_count: int = 300_000,
) -> Path:
    """Generate an unconditioned official-style digital phantom scatterer file."""
    config = ExperimentConfig(scatterers_count=scatterers_count)
    if method == "uniform":
        data = generate_uniform(config, seed=seed, amp=1.0)
    elif method == "two-layer":
        data = generate_two_layers(config, seed=seed)
    else:
        raise ValueError(f"Unknown official baseline method: {method}")
    return save_phantom(data, output_path, config=config)


def estimate_layer_params(input_path: str | Path) -> tuple[float, float, float]:
    """Estimate simple layer parameters from a real OCT reference scan."""
    img = load_scan(input_path)
    boundary_idx = estimate_layer_boundary(img)
    config = ExperimentConfig()
    boundary_z = boundary_idx / max(1, img.shape[0] - 1) * config.z_max
    top = img[: max(1, boundary_idx)].mean() if boundary_idx > 0 else img.mean()
    bottom = img[boundary_idx:].mean() if boundary_idx < img.shape[0] else img.mean()
    amp_top = float(np.clip(0.1 + 1.2 * top, 0.03, 3.0))
    amp_bottom = float(np.clip(0.3 + 3.5 * bottom, 0.08, 8.0))
    return boundary_z, amp_top, amp_bottom


def heuristic_layer_phantom(
    input_path: str | Path,
    output_path: str | Path,
    seed: int = 7,
    scatterers_count: int = 300_000,
) -> Path:
    """Generate a layer-conditioned phantom from a real OCT reference scan."""
    config = ExperimentConfig(scatterers_count=scatterers_count)
    boundary_z, amp_top, amp_bottom = estimate_layer_params(input_path)
    data = generate_two_layers(
        config,
        seed=seed,
        boundary_z_mcm=boundary_z,
        amp_top=amp_top,
        amp_bottom=amp_bottom,
    )
    return save_phantom(data, output_path, config=config)


def _resized_feature(arr: np.ndarray, shape: tuple[int, int], sigma: float | tuple[float, float] = 0.0) -> np.ndarray:
    if np.any(np.asarray(sigma, dtype=float) > 0):
        arr = gaussian_filter(arr, sigma=sigma)
    return np.asarray(resize(arr, shape, anti_aliasing=True, preserve_range=True), dtype=np.float64)


def _normalize(arr: np.ndarray, floor: float = 0.0) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr - np.nanmin(arr)
    peak = float(np.nanmax(arr))
    if peak > 1e-12:
        arr = arr / peak
    return np.maximum(arr, floor)


def physics_guided_phantom(
    input_path: str | Path,
    output_path: str | Path,
    seed: int = 7,
    scatterers_count: int = 300_000,
    lateral_bins: int = 64,
    depth_bins: int = 64,
    density_power: float = 1.0,
    depth_compensation: float = 1.45,
    oac_weight: float = 2.0,
    texture_weight: float = 0.35,
    energy_oac_scale: float = 35.0,
    void_fraction: float = 0.0,
    band_boost: float = 0.0,
    lateral_smooth: float = 0.8,
    multi_layer: bool = False,
    log_energy: bool = False,
    oac_percentile: float = 70.0,
    base_energy_mix: float = 0.55,
    texture_sigma_scale: float = 1.0,
) -> Path:
    """Generate a scanner-compatible phantom using features from a real reference scan."""
    config = ExperimentConfig(scatterers_count=scatterers_count)
    rng = np.random.default_rng(seed)
    img = load_scan(input_path)
    linear = load_and_linearize_image(input_path)
    oac = calculate_oac(linear)
    speckle = calculate_speckle_contrast_map(linear)

    shape = (depth_bins, lateral_bins)
    smooth_sigma = max(0.0, lateral_smooth)
    img_bins = _normalize(_resized_feature(img, shape, sigma=(0.0, smooth_sigma)))
    oac_bins = _normalize(_resized_feature(oac, shape, sigma=(0.0, smooth_sigma)))
    speckle_bins = _normalize(_resized_feature(speckle, shape, sigma=texture_sigma_scale))
    texture = _normalize(np.abs(img - gaussian_filter(img, sigma=1.0)))
    texture_bins = _normalize(_resized_feature(texture, shape, sigma=texture_sigma_scale))

    base = np.maximum(img_bins, 1e-4) ** max(0.05, density_power)
    density = base + oac_weight * oac_bins + texture_weight * (0.75 * texture_bins + 0.25 * speckle_bins)
    depth_prior = np.exp(np.linspace(0.0, depth_compensation, depth_bins, dtype=np.float64))[:, None]
    density *= depth_prior

    if band_boost:
        boundary = estimate_layer_boundary(img)
        boundary_bin = int(np.clip(round(boundary / max(1, img.shape[0] - 1) * (depth_bins - 1)), 0, depth_bins - 1))
        band = np.exp(-0.5 * ((np.arange(depth_bins) - boundary_bin) / max(1.0, depth_bins / 20.0)) ** 2)[:, None]
        density *= 1.0 + band_boost * band

    if multi_layer:
        profile = _normalize(_resized_feature(img.mean(axis=1)[:, None], (depth_bins, 1))).ravel()
        layer_edges = np.argsort(np.abs(np.gradient(profile)))[-3:]
        for edge in layer_edges:
            band = np.exp(-0.5 * ((np.arange(depth_bins) - edge) / max(1.0, depth_bins / 24.0)) ** 2)[:, None]
            density *= 1.0 + 0.18 * band

    if void_fraction > 0:
        void_mask = rng.random(shape) < void_fraction
        density[void_mask] *= 0.18

    density = np.maximum(density, 1e-8)
    density = density / density.sum()
    flat_choice = rng.choice(density.size, size=scatterers_count, replace=True, p=density.ravel())
    z_bin, x_bin = np.divmod(flat_choice, lateral_bins)

    xs = ((x_bin + rng.random(scatterers_count)) / lateral_bins - 0.5) * config.x_max
    ys = (rng.random(scatterers_count) - 0.5) * (2 * config.beam_radius)
    zs = ((z_bin + rng.random(scatterers_count)) / depth_bins) * config.z_max

    oac_ref = np.percentile(oac_bins, np.clip(oac_percentile, 1.0, 99.0))
    energy_map = base_energy_mix + energy_oac_scale * np.maximum(oac_bins, oac_ref * 0.15) + 1.5 * speckle_bins
    if log_energy:
        energy_map = np.log1p(energy_map)
    energies = energy_map[z_bin, x_bin]
    energies *= rng.lognormal(mean=0.0, sigma=0.18 + 0.10 * texture_weight, size=scatterers_count)
    energies = np.clip(energies, 0.01, 100.0)

    data = np.column_stack((xs, ys, zs, energies))
    return save_phantom(data, output_path, config=config)


def hypothesis_phantom(
    input_path: str | Path,
    output_path: str | Path,
    name: str,
    seed: int = 7,
    scatterers_count: int = 300_000,
) -> Path:
    """Generate one named H-series phantom hypothesis, preserving existing H61/H67/H68 behavior."""
    if name == "H0_official":
        return official_baseline_phantom(output_path, seed=seed, scatterers_count=scatterers_count)
    if name not in HYPOTHESIS_CONFIGS:
        raise ValueError(f"Unknown hypothesis config: {name}")
    return physics_guided_phantom(
        input_path,
        output_path,
        seed=seed,
        scatterers_count=scatterers_count,
        **HYPOTHESIS_CONFIGS[name],
    )


def final_phantom(
    input_path: str | Path,
    output_path: str | Path,
    seed: int = 7,
    scatterers_count: int = 300_000,
) -> Path:
    """Generate the current final challenge phantom: H61_api_low_depth_prelim."""
    return hypothesis_phantom(input_path, output_path, FINAL_CONFIG_NAME, seed=seed, scatterers_count=scatterers_count)
