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

PROMISING_PIPELINE_CONFIGS: dict[str, PhantomConfig] = {
    "P01_simulator_constrained_prior": {
        "density_power": 0.95,
        "depth_compensation": -0.65,
        "oac_weight": 1.35,
        "texture_weight": 0.26,
        "energy_oac_scale": 13.0,
        "band_boost": 0.12,
        "lateral_smooth": 0.75,
        "oac_percentile": 68.0,
        "base_energy_mix": 0.44,
        "texture_sigma_scale": 0.72,
        "multi_layer": True,
    },
    "P02_unrolled_feature_consistency": {
        "density_power": 1.18,
        "depth_compensation": 0.55,
        "oac_weight": 1.85,
        "texture_weight": 0.30,
        "energy_oac_scale": 22.0,
        "band_boost": 0.34,
        "lateral_smooth": 0.42,
        "oac_percentile": 76.0,
        "base_energy_mix": 0.58,
        "texture_sigma_scale": 0.86,
        "multi_layer": True,
    },
    "P03_speckle_preserving_texture": {
        "density_power": 0.88,
        "depth_compensation": -0.35,
        "oac_weight": 0.95,
        "texture_weight": 0.48,
        "energy_oac_scale": 9.5,
        "band_boost": 0.08,
        "lateral_smooth": 0.26,
        "oac_percentile": 62.0,
        "base_energy_mix": 0.36,
        "texture_sigma_scale": 0.52,
    },
    "P04_bayesian_posterior_sample": {
        "density_power": 1.02,
        "depth_compensation": 0.15,
        "oac_weight": 1.45,
        "texture_weight": 0.36,
        "energy_oac_scale": 16.5,
        "band_boost": 0.22,
        "lateral_smooth": 0.68,
        "oac_percentile": 70.0,
        "base_energy_mix": 0.50,
        "texture_sigma_scale": 1.05,
        "void_fraction": 0.035,
        "log_energy": True,
        "multi_layer": True,
    },
    "P05_attenuation_layer_map": {
        "density_power": 1.26,
        "depth_compensation": 1.05,
        "oac_weight": 2.35,
        "texture_weight": 0.24,
        "energy_oac_scale": 30.0,
        "band_boost": 0.52,
        "lateral_smooth": 0.96,
        "oac_percentile": 80.0,
        "base_energy_mix": 0.64,
        "texture_sigma_scale": 0.92,
        "multi_layer": True,
    },
}

VISUAL_PIPELINE_CONFIGS: dict[str, dict[str, str]] = {
    "P06_visual_surface_dark_body": {"recipe": "broad_surface_dark_body"},
    "P07_surface_cutoff_broad_mix": {"recipe": "surface_cutoff_broad_mix"},
    "P08_sparse_top_texture_ssim": {"recipe": "sparse_top_texture_ssim"},
    "P09_gamma_sparse_lowfloor_ssim": {"recipe": "gamma_sparse_lowfloor_ssim"},
}

PROMISING_PIPELINE_WAVES: dict[str, tuple[str, ...]] = {
    "promising-pipelines": (*PROMISING_PIPELINE_CONFIGS.keys(), *VISUAL_PIPELINE_CONFIGS.keys()),
    "promising-fast-triad": (
        "P01_simulator_constrained_prior",
        "P03_speckle_preserving_texture",
        "P05_attenuation_layer_map",
    ),
    "visual-pipelines": tuple(VISUAL_PIPELINE_CONFIGS.keys()),
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


def _sample_density_energy_fields(
    density: np.ndarray,
    energy_map: np.ndarray,
    output_path: str | Path,
    seed: int,
    scatterers_count: int,
    energy_sigma: float = 0.16,
) -> Path:
    config = ExperimentConfig(scatterers_count=scatterers_count)
    rng = np.random.default_rng(seed)
    density = np.maximum(np.asarray(density, dtype=np.float64), 1e-12)
    density = density / density.sum()
    flat_choice = rng.choice(density.size, size=scatterers_count, replace=True, p=density.ravel())
    z_bin, x_bin = np.divmod(flat_choice, density.shape[1])

    xs = ((x_bin + rng.random(scatterers_count)) / density.shape[1] - 0.5) * config.x_max
    ys = (rng.random(scatterers_count) - 0.5) * (2 * config.beam_radius)
    zs = ((z_bin + rng.random(scatterers_count)) / density.shape[0]) * config.z_max

    energy_map = np.broadcast_to(np.asarray(energy_map, dtype=np.float64), density.shape)
    energies = energy_map[z_bin, x_bin] * rng.lognormal(mean=0.0, sigma=energy_sigma, size=scatterers_count)
    data = np.column_stack((xs, ys, zs, np.clip(energies, 0.001, 100.0)))
    return save_phantom(data, output_path, config=config)


def visual_inversion_phantom(
    input_path: str | Path,
    output_path: str | Path,
    recipe: str = "broad_surface_dark_body",
    seed: int = 7,
    scatterers_count: int = 300_000,
) -> Path:
    """Generate a scanner-calibrated visual inverse phantom from a reference B-scan.

    These recipes came from hosted-scanner sweeps: they trade the earlier
    physics-only priors for explicit control of the tissue surface, dark body,
    and superficial scattering band that dominate the rendered PNG metrics.
    """
    config = ExperimentConfig(scatterers_count=scatterers_count)
    target = load_scan(input_path)
    target = np.asarray(resize(target, (config.n_depth, config.n_lateral), anti_aliasing=True, preserve_range=True), dtype=np.float64)
    smooth = gaussian_filter(target, sigma=(1.5, 3.0))

    if recipe == "broad_surface_dark_body":
        z_norm = np.linspace(0.0, 1.0, config.n_depth, dtype=np.float64)[:, None]
        hi = np.clip((smooth - 0.07) / 0.33, 0.0, 1.0)
        mid = np.clip((smooth - 0.025) / 0.20, 0.0, 1.0)
        gate = np.exp(-0.5 * ((np.arange(config.n_depth)[:, None] - 72.0) / 55.0) ** 2)
        body_decay = np.exp(-3.0 * z_norm)
        density = 0.0015 + 2.8 * hi * gate + 0.010 * mid * body_decay
        energy = 0.035 + 0.050 * hi
        return _sample_density_energy_fields(density, energy, output_path, seed=seed, scatterers_count=scatterers_count)

    if recipe == "surface_cutoff_broad_mix":
        rows = np.arange(config.n_depth, dtype=np.float64)[:, None]
        surface_mask = smooth > 0.045
        surface = np.argmax(surface_mask, axis=0)
        surface[~surface_mask.any(axis=0)] = 24
        surface = gaussian_filter(surface.astype(np.float64), sigma=5.0).astype(int)
        rel_depth = rows - surface[None, :]
        tissue = (rel_depth >= 2).astype(np.float64)
        near = np.exp(-0.5 * ((rel_depth - 34.0) / 26.0) ** 2) * tissue
        broad = np.exp(-0.5 * ((rel_depth - 52.0) / 44.0) ** 2) * tissue
        body = np.exp(-np.maximum(rel_depth, 0.0) / 72.0) * tissue
        hi = np.clip((smooth - 0.07) / 0.33, 0.0, 1.0) * tissue
        density = 0.0005 * tissue + 1.2 * hi * near + 2.0 * hi * broad + 0.005 * body
        energy = 0.022 + 0.052 * hi
        return _sample_density_energy_fields(density, energy, output_path, seed=seed, scatterers_count=scatterers_count)

    if recipe == "sparse_top_texture_ssim":
        sparse = np.clip((target - 0.06) / 0.40, 0.0, 1.0)
        gate = np.exp(-0.5 * ((np.arange(config.n_depth)[:, None] - 72.0) / 55.0) ** 2)
        density = 0.0005 + 4.0 * sparse**1.1 * gate
        energy_scale = min(1.0, 300_000.0 / max(1, scatterers_count))
        energy = (0.040 + 0.060 * sparse) * energy_scale
        energy_sigma = 0.06 if scatterers_count >= 600_000 else 0.14
        return _sample_density_energy_fields(
            density,
            energy,
            output_path,
            seed=seed,
            scatterers_count=scatterers_count,
            energy_sigma=energy_sigma,
        )

    if recipe == "gamma_sparse_lowfloor_ssim":
        z = np.arange(config.n_depth, dtype=np.float64)[:, None]
        sparse = np.clip((target - 0.06) / 0.40, 0.0, 1.0)
        gate = np.exp(-0.5 * ((z - 72.0) / 55.0) ** 2)
        body_gate = np.exp(-np.maximum(z - 45.0, 0.0) / 95.0)
        density = 0.00035 + 5.8 * sparse**2.0 * gate + 0.0006 * body_gate
        energy = 0.010 + 0.035 * sparse**1.6
        energy_sigma = 0.055 if scatterers_count >= 600_000 else 0.12
        return _sample_density_energy_fields(
            density,
            energy,
            output_path,
            seed=seed,
            scatterers_count=scatterers_count,
            energy_sigma=energy_sigma,
        )

    raise ValueError(f"Unknown visual inversion recipe: {recipe}")


def pipeline_phantom(
    input_path: str | Path,
    output_path: str | Path,
    name: str,
    seed: int = 7,
    scatterers_count: int = 300_000,
) -> Path:
    """Generate one P-series promising pipeline phantom from a real reference scan."""
    if name in VISUAL_PIPELINE_CONFIGS:
        return visual_inversion_phantom(
            input_path,
            output_path,
            recipe=VISUAL_PIPELINE_CONFIGS[name]["recipe"],
            seed=seed,
            scatterers_count=scatterers_count,
        )
    if name not in PROMISING_PIPELINE_CONFIGS:
        raise ValueError(f"Unknown promising pipeline: {name}")
    return physics_guided_phantom(
        input_path,
        output_path,
        seed=seed,
        scatterers_count=scatterers_count,
        **PROMISING_PIPELINE_CONFIGS[name],
    )


def final_phantom(
    input_path: str | Path,
    output_path: str | Path,
    seed: int = 7,
    scatterers_count: int = 300_000,
) -> Path:
    """Generate the current final challenge phantom: H61_api_low_depth_prelim."""
    return hypothesis_phantom(input_path, output_path, FINAL_CONFIG_NAME, seed=seed, scatterers_count=scatterers_count)
