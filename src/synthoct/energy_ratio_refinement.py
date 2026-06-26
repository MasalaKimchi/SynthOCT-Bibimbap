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


DEFAULT_ENERGY_RATIO_EXPONENTS = (0.008, 0.012, 0.024, 0.04, 0.08)


def _scatterer_pixels(data: np.ndarray, config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    xb = np.clip(((data[:, 0] / config.x_max) + 0.5) * config.n_lateral, 0, config.n_lateral - 1).astype(int)
    zb = np.clip((data[:, 2] / config.z_max) * config.n_depth, 0, config.n_depth - 1).astype(int)
    return zb, xb


def write_energy_ratio_phantom(
    reference_path: str | Path,
    phantom_path: str | Path,
    rendered_gray_path: str | Path,
    output_path: str | Path,
    exponent: float,
    stabilizer: float = 0.025,
    sigma: float = 2.0,
    ratio_low: float = 0.65,
    ratio_high: float = 1.32,
    clip_low: float = 0.78,
    clip_high: float = 1.22,
    air_boundary: float = 28.0,
    match_total_energy: bool = True,
) -> Path:
    """Preserve coordinates while applying smoothed reference/render energy feedback."""
    data = load_phantom(phantom_path)
    config = ExperimentConfig(scatterers_count=len(data))
    ref = load_scan(reference_path)
    pred = load_scan(rendered_gray_path)
    ratio = (gaussian_filter(ref, sigma) + stabilizer) / (gaussian_filter(pred, sigma) + stabilizer)
    ratio = np.clip(ratio, ratio_low, ratio_high)
    z = np.arange(config.n_depth, dtype=np.float64)[:, None]
    tissue_gate = 1.0 / (1.0 + np.exp(-(z - air_boundary) / 5.0))
    ratio = 1.0 + (ratio - 1.0) * tissue_gate

    zb, xb = _scatterer_pixels(data, config)
    scale = np.clip(ratio[zb, xb] ** exponent, clip_low, clip_high)
    shaped = data.copy()
    shaped[:, 3] = np.clip(shaped[:, 3] * scale, 0.001, 100.0)
    if match_total_energy:
        shaped[:, 3] *= data[:, 3].sum() / (shaped[:, 3].sum() + 1e-12)
        shaped[:, 3] = np.clip(shaped[:, 3], 0.001, 100.0)
    return save_phantom(shaped, output_path, config=config)


def run_energy_ratio_refinement(
    reference_path: str | Path,
    phantom_path: str | Path,
    rendered_gray_path: str | Path,
    out_dir: str | Path,
    exponents: Iterable[float] = DEFAULT_ENERGY_RATIO_EXPONENTS,
    api_key_file: str | Path | None = None,
    poll_interval_seconds: float = 10.0,
    max_polls: int = 90,
    skip_existing: bool = True,
    stabilizer: float = 0.025,
    sigma: float = 2.0,
    ratio_low: float = 0.65,
    ratio_high: float = 1.32,
    clip_low: float = 0.78,
    clip_high: float = 1.22,
    air_boundary: float = 28.0,
    match_total_energy: bool = True,
) -> Path:
    """Render and rank coordinate-preserving energy-ratio feedback candidates."""
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
    rows: list[dict[str, float | int | str | bool]] = []
    for idx, exponent in enumerate(exponents, start=1):
        label = f"energy_ratio_e{exponent:.3f}".replace(".", "p")
        shaped_path = phantom_dir / f"{idx:02d}_{label}.txt"
        synthetic_path = synthetic_dir / f"{idx:02d}_{label}.png"
        gray_path = gray_dir / f"{idx:02d}_{label}_gray.png"
        if not shaped_path.exists() or not skip_existing:
            write_energy_ratio_phantom(
                reference_path,
                phantom_path,
                rendered_gray_path,
                shaped_path,
                exponent=exponent,
                stabilizer=stabilizer,
                sigma=sigma,
                ratio_low=ratio_low,
                ratio_high=ratio_high,
                clip_low=clip_low,
                clip_high=clip_high,
                air_boundary=air_boundary,
                match_total_energy=match_total_energy,
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
                "exponent": exponent,
                "stabilizer": stabilizer,
                "sigma": sigma,
                "ratio_low": ratio_low,
                "ratio_high": ratio_high,
                "clip_low": clip_low,
                "clip_high": clip_high,
                "air_boundary": air_boundary,
                "match_total_energy": match_total_energy,
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

    rows.sort(key=lambda row: float(row["SSIM"]) if str(row["SSIM"]) != "nan" else -1.0, reverse=True)
    metrics_path = out_dir / "energy_ratio_metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return metrics_path
