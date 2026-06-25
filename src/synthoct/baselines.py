from __future__ import annotations

from pathlib import Path

import numpy as np

from .cnn import cnn_embedding
from .phantom import ExperimentConfig, generate_two_layers, generate_uniform, save_phantom
from .processor import calculate_oac, calculate_speckle_contrast_map, load_and_linearize_image, load_scan


def official_baseline(
    output_path: str | Path,
    method: str = "two-layer",
    seed: int = 7,
    scatterers_count: int = 300_000,
) -> Path:
    config = ExperimentConfig(scatterers_count=scatterers_count)
    if method == "uniform":
        data = generate_uniform(config, seed=seed, amp=1.0)
    elif method == "two-layer":
        data = generate_two_layers(config, seed=seed, boundary_z_mcm=400.0, amp_top=0.05, amp_bottom=2.5)
    else:
        raise ValueError("method must be 'uniform' or 'two-layer'.")
    return save_phantom(data, output_path, config=config)


def estimate_layer_params(scan_path: str | Path) -> dict[str, float]:
    img = load_scan(scan_path)
    intensity = load_and_linearize_image(scan_path)
    oac = calculate_oac(intensity)
    sc = calculate_speckle_contrast_map(intensity)

    profile = img.mean(axis=1)
    smooth = np.convolve(profile, np.ones(15) / 15, mode="same")
    grad = np.abs(np.gradient(smooth))
    start = max(10, int(0.08 * len(profile)))
    stop = min(len(profile) - 10, int(0.75 * len(profile)))
    boundary_idx = int(np.argmax(grad[start:stop]) + start)

    shallow = slice(0, max(boundary_idx, 1))
    deep = slice(boundary_idx, None)
    shallow_energy = float(np.clip(np.percentile(oac[shallow], 75) * 30, 0.01, 8.0))
    deep_energy = float(np.clip(np.percentile(oac[deep], 75) * 60, 0.01, 12.0))
    heterogeneity = float(np.clip(np.nanmean(sc), 0.2, 5.0))
    return {
        "boundary_z_mcm": boundary_idx * ExperimentConfig().pixel_size_z,
        "amp_top": shallow_energy,
        "amp_bottom": deep_energy,
        "heterogeneity": heterogeneity,
    }


def heuristic_baseline(
    input_path: str | Path,
    output_path: str | Path,
    seed: int = 7,
    scatterers_count: int = 300_000,
) -> Path:
    config = ExperimentConfig(scatterers_count=scatterers_count)
    params = estimate_layer_params(input_path)
    data = generate_two_layers(
        config,
        seed=seed,
        boundary_z_mcm=params["boundary_z_mcm"],
        amp_top=params["amp_top"],
        amp_bottom=params["amp_bottom"],
    )
    rng = np.random.default_rng(seed + 1009)
    jitter = rng.lognormal(mean=0.0, sigma=0.12 * params["heterogeneity"], size=len(data))
    data[:, 3] = np.clip(data[:, 3] * jitter, 0, 100)
    return save_phantom(data, output_path, config=config)


def pretrained_cnn_baseline(
    input_path: str | Path,
    output_path: str | Path,
    backbone: str = "resnet50",
    seed: int = 7,
    scatterers_count: int = 300_000,
    pretrained: bool = True,
) -> Path:
    emb = cnn_embedding(input_path, backbone=backbone, pretrained=pretrained)
    config = ExperimentConfig(scatterers_count=scatterers_count)
    rng = np.random.default_rng(seed)
    digest = np.tanh(emb[:16] if emb.size >= 16 else np.pad(emb, (0, 16 - emb.size)))
    boundary = float(np.clip(500 + 220 * digest[0], 180, 1100))
    amp_top = float(np.clip(0.8 + 1.5 * abs(digest[1]), 0.02, 5.0))
    amp_bottom = float(np.clip(1.5 + 4.0 * abs(digest[2]), 0.02, 12.0))
    data = generate_two_layers(config, seed=seed, boundary_z_mcm=boundary, amp_top=amp_top, amp_bottom=amp_bottom)

    # Use frozen features as a deterministic parameter source, while keeping output a valid phantom.
    texture_sigma = float(np.clip(0.1 + 0.35 * abs(digest[3]), 0.05, 0.6))
    data[:, 3] = np.clip(data[:, 3] * rng.lognormal(0, texture_sigma, len(data)), 0, 100)
    return save_phantom(data, output_path, config=config)


def parameter_search_baseline(
    input_path: str | Path,
    output_path: str | Path,
    seed: int = 7,
    scatterers_count: int = 300_000,
) -> Path:
    # Scanner-in-the-loop scoring is unavailable on macOS by default, so this initializes
    # search from image-derived parameters and leaves real scoring to the Windows adapter.
    params = estimate_layer_params(input_path)
    grid_boundaries = [params["boundary_z_mcm"] - 60, params["boundary_z_mcm"], params["boundary_z_mcm"] + 60]
    grid_top = [params["amp_top"] * 0.75, params["amp_top"], params["amp_top"] * 1.25]
    grid_bottom = [params["amp_bottom"] * 0.75, params["amp_bottom"], params["amp_bottom"] * 1.25]
    config = ExperimentConfig(scatterers_count=scatterers_count)
    best = generate_two_layers(
        config,
        seed=seed,
        boundary_z_mcm=float(np.clip(grid_boundaries[1], 0, config.z_max)),
        amp_top=float(np.clip(grid_top[1], 0.01, 100)),
        amp_bottom=float(np.clip(grid_bottom[1], 0.01, 100)),
    )
    return save_phantom(best, output_path, config=config)
