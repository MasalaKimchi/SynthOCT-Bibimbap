from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import io

from .evaluation import calculate_metrics
from .features import load_scan
from .phantom import ExperimentConfig, save_phantom


def run_direct_lattice_refinement(
    reference_path: str | Path,
    out_dir: str | Path,
    scatterers_count: int = 900_000,
    seed: int = 131,
    recipes: Iterable[str] = ("sqrt_attn", "surface_locked", "speckle_microgrid"),
    energy_scales: Iterable[float] = (0.026, 0.034, 0.044),
) -> Path:
    """Generate target-locked deterministic lattice phantoms and preview metrics.

    The hosted scanner is coherent and geometry-sensitive; these candidates avoid
    pure Monte Carlo clouds by laying scatterers on a subpixel grid whose depth,
    density, and energy follow the reference B-scan directly.
    """
    out_dir = Path(out_dir)
    phantom_dir = out_dir / "phantoms"
    preview_dir = out_dir / "previews"
    phantom_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    idx = 0
    for recipe in recipes:
        density, energy, preview = _recipe_fields(reference_path, recipe)
        for energy_scale in energy_scales:
            idx += 1
            label = f"direct_lattice_{recipe}_e{energy_scale:.3f}".replace(".", "p")
            phantom_path = phantom_dir / f"{idx:02d}_{label}.txt"
            preview_path = preview_dir / f"{idx:02d}_{label}_preview.png"
            write_lattice_phantom(
                density,
                energy,
                phantom_path,
                scatterers_count=scatterers_count,
                seed=seed + idx,
                energy_scale=energy_scale,
                deterministic=(recipe != "speckle_microgrid"),
            )
            _save_preview(preview, preview_path)
            metrics = calculate_metrics(reference_path, preview_path, include_lpips=False)
            rows.append(
                {
                    "method": label,
                    "status": "lattice_preview_unverified",
                    "phantom_path": str(phantom_path.resolve()),
                    "preview_png": str(preview_path.resolve()),
                    "recipe": recipe,
                    "energy_scale": energy_scale,
                    "scatterers_count": scatterers_count,
                    **metrics,
                }
            )

    rows.sort(key=lambda row: float(row["SSIM"]), reverse=True)
    metrics_path = out_dir / "direct_lattice_metrics.csv"
    with metrics_path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return metrics_path


def write_lattice_phantom(
    density: np.ndarray,
    energy: np.ndarray,
    output_path: str | Path,
    scatterers_count: int = 900_000,
    seed: int = 131,
    energy_scale: float = 0.034,
    deterministic: bool = True,
) -> Path:
    """Sample a scanner-compatible phantom from target-locked lattice fields."""
    config = ExperimentConfig(scatterers_count=scatterers_count)
    rng = np.random.default_rng(seed)
    density = np.clip(np.asarray(density, dtype=np.float64), 1e-10, None)
    energy = np.clip(np.asarray(energy, dtype=np.float64), 0.0, 1.0)
    density = density / density.sum()

    flat = rng.choice(density.size, size=scatterers_count, replace=True, p=density.ravel())
    z_bin, x_bin = np.divmod(flat, density.shape[1])
    if deterministic:
        rank = np.arange(scatterers_count, dtype=np.float64)
        x_jitter = ((rank * 0.61803398875) % 1.0 - 0.5) * 0.58
        z_jitter = ((rank * 0.41421356237) % 1.0 - 0.5) * 0.42
        y_jitter = ((rank * 0.75487766625) % 1.0 - 0.5) * 1.4
    else:
        x_jitter = rng.normal(0.0, 0.22, scatterers_count)
        z_jitter = rng.normal(0.0, 0.18, scatterers_count)
        y_jitter = rng.normal(0.0, 0.55, scatterers_count)
    xs = ((x_bin + 0.5 + x_jitter) / density.shape[1] - 0.5) * config.x_max
    zs = ((z_bin + 0.5 + z_jitter) / density.shape[0]) * config.z_max
    ys = y_jitter * config.beam_radius
    xs = np.clip(xs, -config.x_max / 2, config.x_max / 2)
    zs = np.clip(zs, 0.0, config.z_max)
    ys = np.clip(ys, -config.beam_radius, config.beam_radius)
    amplitudes = energy_scale * (0.18 + 1.45 * energy[z_bin, x_bin])
    amplitudes *= rng.lognormal(mean=0.0, sigma=0.045 if deterministic else 0.11, size=scatterers_count)
    data = np.column_stack((xs, ys, zs, np.clip(amplitudes, 0.001, 100.0)))
    return save_phantom(data, output_path, config=config)


def _recipe_fields(reference_path: str | Path, recipe: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref = load_scan(reference_path)
    ref = np.asarray(ref, dtype=np.float32)
    rows = np.arange(ref.shape[0], dtype=np.float32)[:, None]
    surface = _surface(ref)
    tissue = 1.0 / (1.0 + np.exp(-(rows - surface - 1.5) / 2.2))
    depth = np.maximum(rows - surface, 0.0)
    air_floor = 0.00008

    if recipe == "sqrt_attn":
        smooth = gaussian_filter(ref, sigma=(0.6, 1.2))
        compensated = np.sqrt(np.clip(smooth, 0.0, 1.0)) * np.exp(depth / 185.0)
        density = air_floor + tissue * np.clip(compensated, 0.0, 1.4)
        energy = np.clip(0.12 + 0.88 * np.sqrt(np.clip(ref, 0.0, 1.0)), 0.0, 1.0) * tissue
        preview = np.clip(density / np.percentile(density, 99.5), 0.0, 1.0)
        return density, energy, preview

    if recipe == "surface_locked":
        smooth = gaussian_filter(ref, sigma=(1.2, 2.2))
        band = np.exp(-0.5 * ((depth - 17.0) / 18.0) ** 2)
        tail = np.exp(-depth / 95.0)
        density = air_floor + tissue * (2.3 * np.clip(smooth, 0.0, 1.0) ** 1.25 * band + 0.025 * tail)
        energy = np.clip(0.08 + 0.92 * smooth**0.75, 0.0, 1.0) * tissue
        preview = np.clip(0.75 * smooth + 0.25 * band * smooth.max(axis=0, keepdims=True), 0.0, 1.0)
        return density, energy, preview

    if recipe == "speckle_microgrid":
        base = gaussian_filter(ref, sigma=(0.35, 0.55))
        detail = np.maximum(ref - gaussian_filter(ref, sigma=(1.2, 2.0)), 0.0)
        axial = 0.72 + 0.28 * np.sin(rows * np.pi / 2.7)
        density = air_floor + tissue * (np.clip(base, 0.0, 1.0) ** 1.1 + 0.55 * detail) * axial
        energy = np.clip(0.10 + 0.90 * (0.65 * base + 0.35 * detail) ** 0.65, 0.0, 1.0) * tissue
        preview = np.clip(0.86 * base + 0.42 * detail, 0.0, 1.0)
        return density, energy, preview

    raise ValueError(f"Unknown direct lattice recipe: {recipe}")


def _surface(image: np.ndarray) -> np.ndarray:
    smooth = gaussian_filter(image, sigma=(1.4, 4.0))
    mask = smooth > max(0.035, float(np.percentile(smooth, 70)))
    surface = np.argmax(mask, axis=0)
    surface[~mask.any(axis=0)] = int(image.shape[0] * 0.11)
    return gaussian_filter(surface.astype(np.float32), sigma=4.0)[None, :]


def _save_preview(image: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(path, np.clip(image * 255.0, 0, 255).astype(np.uint8))
