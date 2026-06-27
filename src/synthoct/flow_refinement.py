from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.registration import optical_flow_tvl1

from .evaluation import calculate_metrics
from .features import load_scan
from .phantom import ExperimentConfig, load_phantom, save_phantom
from .scanners import render_with_api, write_api_config
from .submission import to_gray_png


DEFAULT_FLOW_VARIANTS = (
    (0.15, 1.0, 5.0),
    (0.30, 1.0, 5.0),
    (-0.15, 1.0, 5.0),
    (-0.30, 1.0, 5.0),
)


def _scatterer_pixels(data: np.ndarray, config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    xb = np.clip(((data[:, 0] / config.x_max) + 0.5) * config.n_lateral, 0, config.n_lateral - 1).astype(int)
    zb = np.clip((data[:, 2] / config.z_max) * config.n_depth, 0, config.n_depth - 1).astype(int)
    return zb, xb


def _sample_field(field: np.ndarray, z: np.ndarray, x: np.ndarray) -> np.ndarray:
    z0 = np.floor(z).astype(int)
    x0 = np.floor(x).astype(int)
    z1 = np.clip(z0 + 1, 0, field.shape[0] - 1)
    x1 = np.clip(x0 + 1, 0, field.shape[1] - 1)
    z0 = np.clip(z0, 0, field.shape[0] - 1)
    x0 = np.clip(x0, 0, field.shape[1] - 1)
    wz = np.clip(z - z0, 0.0, 1.0)
    wx = np.clip(x - x0, 0.0, 1.0)
    top = field[z0, x0] * (1.0 - wx) + field[z0, x1] * wx
    bottom = field[z1, x0] * (1.0 - wx) + field[z1, x1] * wx
    return top * (1.0 - wz) + bottom * wz


def _flow_fields(
    reference_path: str | Path,
    rendered_gray_path: str | Path,
    smooth_sigma: float,
    attachment: float,
) -> tuple[np.ndarray, np.ndarray]:
    ref = gaussian_filter(load_scan(reference_path), smooth_sigma)
    pred = gaussian_filter(load_scan(rendered_gray_path), smooth_sigma)
    ref = (ref - ref.mean()) / (ref.std() + 1e-8)
    pred = (pred - pred.mean()) / (pred.std() + 1e-8)
    v_flow, u_flow = optical_flow_tvl1(ref, pred, attachment=attachment, tightness=0.3, num_warp=8, num_iter=12)
    return np.asarray(v_flow, dtype=np.float64), np.asarray(u_flow, dtype=np.float64)


def write_flow_transport_phantom(
    reference_path: str | Path,
    phantom_path: str | Path,
    rendered_gray_path: str | Path,
    output_path: str | Path,
    strength: float,
    smooth_sigma: float = 1.0,
    attachment: float = 5.0,
    max_lateral_pixels: float = 12.0,
    max_depth_pixels: float = 8.0,
    tissue_start: float = 20.0,
    tissue_width: float = 8.0,
) -> Path:
    """Move scatterers by a scanner-observed optical-flow field."""
    data = load_phantom(phantom_path)
    config = ExperimentConfig(scatterers_count=len(data))
    v_flow, u_flow = _flow_fields(reference_path, rendered_gray_path, smooth_sigma=smooth_sigma, attachment=attachment)
    zb, xb = _scatterer_pixels(data, config)

    dz_px = np.clip(v_flow[zb, xb], -max_depth_pixels, max_depth_pixels)
    dx_px = np.clip(u_flow[zb, xb], -max_lateral_pixels, max_lateral_pixels)
    gate = 1.0 / (1.0 + np.exp(-(zb.astype(np.float64) - tissue_start) / tissue_width))

    shaped = data.copy()
    shaped[:, 0] = np.clip(
        shaped[:, 0] - strength * gate * dx_px * config.pixel_size_x,
        -config.x_max / 2,
        config.x_max / 2,
    )
    shaped[:, 2] = np.clip(
        shaped[:, 2] - strength * gate * dz_px * config.pixel_size_z,
        0.0,
        config.z_max,
    )
    return save_phantom(shaped, output_path, config=config)


def run_flow_refinement(
    reference_path: str | Path,
    phantom_path: str | Path,
    rendered_gray_path: str | Path,
    out_dir: str | Path,
    variants: Iterable[tuple[float, float, float]] = DEFAULT_FLOW_VARIANTS,
    api_key_file: str | Path | None = None,
    poll_interval_seconds: float = 10.0,
    max_polls: int = 90,
    skip_existing: bool = True,
) -> Path:
    """Render and rank coordinate-transport candidates through the hosted scanner."""
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
    for idx, (strength, smooth_sigma, attachment) in enumerate(variants, start=1):
        label = f"flow_s{strength:.3f}_sig{smooth_sigma:.2f}_att{attachment:.1f}".replace(".", "p").replace("-", "n")
        shaped_path = phantom_dir / f"{idx:02d}_{label}.txt"
        synthetic_path = synthetic_dir / f"{idx:02d}_{label}.png"
        gray_path = gray_dir / f"{idx:02d}_{label}_gray.png"
        if not shaped_path.exists() or not skip_existing:
            write_flow_transport_phantom(
                reference_path,
                phantom_path,
                rendered_gray_path,
                shaped_path,
                strength=strength,
                smooth_sigma=smooth_sigma,
                attachment=attachment,
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
                "strength": strength,
                "smooth_sigma": smooth_sigma,
                "attachment": attachment,
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
    metrics_path = out_dir / "flow_refinement_metrics.csv"
    with metrics_path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return metrics_path
