from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter

from .evaluation import calculate_metrics
from .features import load_scan
from .phantom import ExperimentConfig, load_phantom, save_phantom
from .scanners import render_with_api, write_api_config
from .submission import to_gray_png


DEFAULT_TEXTURE_VARIANTS = (
    (0.08, 0.45, 1.4),
    (0.12, 0.75, 1.8),
    (0.16, 1.05, 2.2),
    (0.20, 1.35, 2.8),
)


def _scatterer_pixels(data: np.ndarray, config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    xb = np.clip(((data[:, 0] / config.x_max) + 0.5) * config.n_lateral, 0, config.n_lateral - 1).astype(int)
    zb = np.clip((data[:, 2] / config.z_max) * config.n_depth, 0, config.n_depth - 1).astype(int)
    return zb, xb


def _local_std(image: np.ndarray, sigma: float) -> np.ndarray:
    mean = gaussian_filter(image, sigma)
    second = gaussian_filter(image * image, sigma)
    return np.sqrt(np.maximum(second - mean * mean, 0.0))


def write_texture_matched_phantom(
    reference_path: str | Path,
    phantom_path: str | Path,
    rendered_gray_path: str | Path,
    output_path: str | Path,
    mean_exponent: float,
    texture_exponent: float,
    deep_exponent: float,
    mean_sigma: float = 1.35,
    texture_sigma: float = 2.4,
    stabilizer: float = 0.004,
    texture_stabilizer: float = 0.010,
    ratio_low: float = 0.55,
    ratio_high: float = 1.35,
    texture_low: float = 0.45,
    texture_high: float = 1.20,
    deep_start: float = 76.0,
    deep_width: float = 18.0,
    clip_low: float = 0.45,
    clip_high: float = 1.25,
    match_total_energy: bool = False,
) -> Path:
    """Suppress scanner-observed excess speckle variance while preserving coordinates."""
    data = load_phantom(phantom_path)
    config = ExperimentConfig(scatterers_count=len(data))
    ref = load_scan(reference_path)
    pred = load_scan(rendered_gray_path)

    mean_ratio = (gaussian_filter(ref, mean_sigma) + stabilizer) / (gaussian_filter(pred, mean_sigma) + stabilizer)
    mean_ratio = np.clip(mean_ratio, ratio_low, ratio_high)

    ref_std = _local_std(ref, texture_sigma)
    pred_std = _local_std(pred, texture_sigma)
    texture_ratio = (ref_std + texture_stabilizer) / (pred_std + texture_stabilizer)
    texture_ratio = np.clip(texture_ratio, texture_low, texture_high)

    rows = np.arange(config.n_depth, dtype=np.float64)[:, None]
    deep_gate = 1.0 / (1.0 + np.exp(-(rows - deep_start) / deep_width))
    texture_power = texture_exponent + deep_gate * deep_exponent

    zb, xb = _scatterer_pixels(data, config)
    scale = mean_ratio[zb, xb] ** mean_exponent
    scale *= texture_ratio[zb, xb] ** texture_power[zb, 0]
    scale = np.clip(scale, clip_low, clip_high)

    shaped = data.copy()
    shaped[:, 3] = np.clip(shaped[:, 3] * scale, 0.001, 100.0)
    if match_total_energy:
        shaped[:, 3] *= data[:, 3].sum() / (shaped[:, 3].sum() + 1e-12)
        shaped[:, 3] = np.clip(shaped[:, 3], 0.001, 100.0)
    return save_phantom(shaped, output_path, config=config)


def run_texture_refinement(
    reference_path: str | Path,
    phantom_path: str | Path,
    rendered_gray_path: str | Path,
    out_dir: str | Path,
    variants: Iterable[tuple[float, float, float]] = DEFAULT_TEXTURE_VARIANTS,
    api_key_file: str | Path | None = None,
    poll_interval_seconds: float = 10.0,
    max_polls: int = 90,
    skip_existing: bool = True,
) -> Path:
    """Render and rank texture-aware amplitude refinement candidates."""
    reference_path = Path(reference_path)
    phantom_path = Path(phantom_path)
    rendered_gray_path = Path(rendered_gray_path)
    out_dir = Path(out_dir)
    phantom_dir = out_dir / "phantoms"
    synthetic_dir = out_dir / "synthetic"
    gray_dir = out_dir / "synthetic_gray"
    for path in (phantom_dir, synthetic_dir, gray_dir):
        path.mkdir(parents=True, exist_ok=True)

    base = load_phantom(phantom_path)
    config_path = write_api_config(out_dir / "Configuration_api.ini", scatterers_count=len(base))
    rows: list[dict[str, float | int | str]] = []
    for idx, (mean_exp, texture_exp, deep_exp) in enumerate(variants, start=1):
        label = f"texture_m{mean_exp:.3f}_t{texture_exp:.3f}_d{deep_exp:.3f}".replace(".", "p")
        shaped_path = phantom_dir / f"{idx:02d}_{label}.txt"
        synthetic_path = synthetic_dir / f"{idx:02d}_{label}.png"
        gray_path = gray_dir / f"{idx:02d}_{label}_gray.png"
        if not shaped_path.exists() or not skip_existing:
            write_texture_matched_phantom(
                reference_path,
                phantom_path,
                rendered_gray_path,
                shaped_path,
                mean_exponent=mean_exp,
                texture_exponent=texture_exp,
                deep_exponent=deep_exp,
            )

        try:
            if skip_existing and gray_path.exists():
                request_id = "existing"
                render_seconds = 0.0
                poll_count = 0
            else:
                request_id, rendered_path, render_seconds, poll_count = render_with_api(
                    shaped_path,
                    config_path,
                    synthetic_path,
                    api_key_file=api_key_file,
                    poll_interval_seconds=poll_interval_seconds,
                    max_polls=max_polls,
                )
                to_gray_png(rendered_path, gray_path)
            metrics = calculate_metrics(reference_path, gray_path, include_lpips=False)
            status = "ok"
            error = ""
        except Exception as exc:
            request_id = "failed"
            render_seconds = 0.0
            poll_count = 0
            metrics = {
                "MSE": float("nan"),
                "PSNR": float("nan"),
                "SSIM": float("nan"),
                "MS-SSIM": float("nan"),
                "VIF": float("nan"),
                "LPIPS": float("nan"),
                "LPIPS_PROXY": float("nan"),
            }
            status = "failed"
            error = str(exc)

        rows.append(
            {
                "status": status,
                "method": label,
                "mean_exponent": mean_exp,
                "texture_exponent": texture_exp,
                "deep_exponent": deep_exp,
                "request_id": request_id,
                "phantom_path": str(shaped_path.resolve()),
                "synthetic_png": str(synthetic_path.resolve()),
                "synthetic_gray_png": str(gray_path.resolve()),
                "render_seconds": render_seconds,
                "poll_count": poll_count,
                "error": error,
                **metrics,
            }
        )

    rows.sort(key=lambda row: float(row["MS-SSIM"]) if str(row["MS-SSIM"]) != "nan" else -1.0, reverse=True)
    metrics_path = out_dir / "texture_refinement_metrics.csv"
    with metrics_path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return metrics_path
