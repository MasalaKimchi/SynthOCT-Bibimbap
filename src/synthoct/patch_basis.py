from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import nnls
from skimage import io
from skimage.transform import resize

from .evaluation import calculate_metrics
from .features import load_scan
from .learned_surrogate import discover_scanner_pairs, field_to_phantom, phantom_to_field
from .phantom import load_phantom


def run_patch_basis_refinement(
    reference_path: str | Path,
    out_dir: str | Path,
    outputs_dir: str | Path = "outputs",
    shape: tuple[int, int] = (128, 256),
    basis_count: int = 32,
    tile_shape: tuple[int, int] = (24, 32),
    scatterers_count: int = 900_000,
    seed: int = 101,
    residual_exponents: Iterable[float] = (0.0, 0.18),
    texture_strengths: Iterable[float] = (0.0, 0.25),
    energy_ratios: Iterable[float] = (0.82, 0.94, 1.06),
) -> Path:
    """Patchwise nonnegative scanner-basis inverse with phantom-field stitching."""
    out_dir = Path(out_dir)
    phantom_dir = out_dir / "phantoms"
    preview_dir = out_dir / "patch_preview"
    phantom_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_scanner_pairs(outputs_dir, limit=basis_count)
    if not pairs:
        raise RuntimeError(f"No scanner-rendered training pairs found under {outputs_dir}.")
    target = _load_resized(reference_path, shape)
    basis_images = np.stack([_load_resized(pair.rendered_gray_path, shape) for pair in pairs], axis=0)
    low_fields = np.stack([phantom_to_field(pair.phantom_path, shape=shape) for pair in pairs], axis=0)

    preview, density_low, energy_low, weight_summary = _solve_patch_fields(
        target,
        basis_images,
        low_fields,
        tile_shape=tile_shape,
    )
    preview_full = _resize01(preview, (256, 512))
    density_full = _resize01(density_low, (256, 512))
    energy_full = _resize01(energy_low, (256, 512))
    texture = _reference_texture(reference_path, (256, 512))
    residual = _energy_residual(reference_path, preview_full)
    base_totals = np.array([load_phantom(pair.phantom_path)[:, 3].sum() for pair in pairs], dtype=np.float64)
    base_total = float(np.dot(weight_summary, base_totals))
    if base_total <= 0:
        base_total = float(base_totals[0])

    preview_path = preview_dir / "patch_basis_preview.png"
    _save_gray(preview_full, preview_path)
    preview_metrics = calculate_metrics(reference_path, preview_path, include_lpips=False)

    rows = []
    idx = 0
    for residual_exponent in residual_exponents:
        corrected_density = np.clip(density_full * np.power(residual, residual_exponent * 0.5), 0.0, 1.0)
        corrected_energy = np.clip(energy_full * np.power(residual, residual_exponent), 0.0, 1.0)
        for texture_strength in texture_strengths:
            density = _blend(corrected_density, texture, texture_strength)
            energy = _blend(corrected_energy, texture, texture_strength * 0.45)
            candidate_preview = _blend(preview_full, texture, texture_strength * 0.18)
            for energy_ratio in energy_ratios:
                idx += 1
                label = f"patch_basis_res{residual_exponent:.2f}_tex{texture_strength:.2f}_er{energy_ratio:.2f}".replace(".", "p")
                phantom_path = phantom_dir / f"{idx:02d}_{label}.txt"
                candidate_preview_path = preview_dir / f"{idx:02d}_{label}_preview.png"
                field_to_phantom(
                    density,
                    energy,
                    phantom_path,
                    scatterers_count=scatterers_count,
                    seed=seed + idx,
                    total_energy=base_total * energy_ratio,
                    energy_floor=0.002,
                    energy_ceiling=0.09,
                )
                _save_gray(candidate_preview, candidate_preview_path)
                metrics = calculate_metrics(reference_path, candidate_preview_path, include_lpips=False)
                rows.append(
                    {
                        "method": label,
                        "status": "patch_preview_unverified",
                        "phantom_path": str(phantom_path.resolve()),
                        "patch_preview_png": str(candidate_preview_path.resolve()),
                        "linear_patch_preview_png": str(preview_path.resolve()),
                        "basis_count": len(pairs),
                        "tile_rows": tile_shape[0],
                        "tile_cols": tile_shape[1],
                        "preview_SSIM": preview_metrics["SSIM"],
                        "top_basis_label": pairs[int(np.argmax(weight_summary))].label,
                        "top_basis_weight": float(weight_summary.max()),
                        "residual_exponent": residual_exponent,
                        "texture_strength": texture_strength,
                        "energy_ratio": energy_ratio,
                        **metrics,
                    }
                )

    rows.sort(key=lambda row: float(row["MS-SSIM"]), reverse=True)
    metrics_path = out_dir / "patch_basis_metrics.csv"
    with metrics_path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return metrics_path


def _solve_patch_fields(
    target: np.ndarray,
    basis_images: np.ndarray,
    fields: np.ndarray,
    tile_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = target.shape
    tile_rows, tile_cols = tile_shape
    preview = np.zeros_like(target, dtype=np.float32)
    density = np.zeros_like(target, dtype=np.float32)
    energy = np.zeros_like(target, dtype=np.float32)
    weight_accum = np.zeros(basis_images.shape[0], dtype=np.float64)
    tile_count = 0
    for r0 in range(0, rows, tile_rows):
        r1 = min(rows, r0 + tile_rows)
        for c0 in range(0, cols, tile_cols):
            c1 = min(cols, c0 + tile_cols)
            matrix = basis_images[:, r0:r1, c0:c1].reshape(basis_images.shape[0], -1).T
            target_vec = target[r0:r1, c0:c1].reshape(-1)
            try:
                weights, _ = nnls(matrix, target_vec, maxiter=matrix.shape[1] * 32)
            except Exception:
                weights = np.clip(np.linalg.lstsq(matrix, target_vec, rcond=1e-4)[0], 0.0, None)
            if weights.sum() <= 1e-12:
                weights = np.ones(basis_images.shape[0], dtype=np.float64)
            normalized = weights / weights.sum()
            weight_accum += normalized
            tile_count += 1
            preview[r0:r1, c0:c1] = np.tensordot(normalized, basis_images[:, r0:r1, c0:c1], axes=(0, 0))
            density[r0:r1, c0:c1] = np.tensordot(normalized, fields[:, 0, r0:r1, c0:c1], axes=(0, 0))
            energy[r0:r1, c0:c1] = np.tensordot(normalized, fields[:, 1, r0:r1, c0:c1], axes=(0, 0))
    weight_summary = weight_accum / max(1, tile_count)
    weight_summary /= max(weight_summary.sum(), 1e-12)
    preview = np.clip(gaussian_filter(preview, sigma=(0.35, 0.55)), 0.0, 1.0)
    density = np.clip(gaussian_filter(density, sigma=(0.7, 1.0)), 0.0, 1.0)
    energy = np.clip(gaussian_filter(energy, sigma=(0.7, 1.0)), 0.0, 1.0)
    return preview, density, energy, weight_summary


def _load_resized(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    image = load_scan(path)
    return np.asarray(resize(image, shape, anti_aliasing=True, preserve_range=True), dtype=np.float32)


def _resize01(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(resize(np.clip(image, 0.0, 1.0), shape, anti_aliasing=True, preserve_range=True), dtype=np.float32)


def _energy_residual(reference_path: str | Path, preview: np.ndarray) -> np.ndarray:
    reference = _load_resized(reference_path, preview.shape)
    ratio = (gaussian_filter(reference, sigma=(2.5, 5.0)) + 0.035) / (gaussian_filter(preview, sigma=(2.5, 5.0)) + 0.035)
    rows = np.arange(preview.shape[0], dtype=np.float32)[:, None]
    air_gate = 1.0 / (1.0 + np.exp(-(rows - 28.0) / 5.0))
    ratio = 1.0 + (np.clip(ratio, 0.45, 1.35) - 1.0) * air_gate
    return ratio.astype(np.float32)


def _reference_texture(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    reference = _load_resized(path, shape)
    smooth = gaussian_filter(reference, sigma=(1.6, 3.2))
    detail = np.maximum(reference - gaussian_filter(reference, sigma=(0.55, 0.9)), 0.0)
    rows = np.arange(shape[0], dtype=np.float32)[:, None]
    surface_mask = smooth > max(0.035, float(np.percentile(smooth, 70)))
    surface = np.argmax(surface_mask, axis=0)
    surface[~surface_mask.any(axis=0)] = int(shape[0] * 0.11)
    surface = gaussian_filter(surface.astype(np.float32), sigma=4.0)[None, :]
    tissue = 1.0 / (1.0 + np.exp(-(rows - surface - 1.5) / 2.0))
    depth_decay = np.exp(-np.maximum(rows - surface, 0.0) / max(18.0, shape[0] * 0.26))
    texture = (0.68 * _normalize01(smooth) + 0.32 * _normalize01(detail)) * tissue * depth_decay
    return np.clip(texture, 0.0, 1.0).astype(np.float32)


def _normalize01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr - float(arr.min())
    peak = float(arr.max())
    if peak > 1e-8:
        arr = arr / peak
    return arr


def _blend(base: np.ndarray, texture: np.ndarray, strength: float) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    return np.clip((1.0 - strength) * base + strength * texture, 0.0, 1.0).astype(np.float32)


def _save_gray(image: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(path, np.clip(image * 255.0, 0, 255).astype(np.uint8))
