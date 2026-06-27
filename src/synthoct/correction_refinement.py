from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

from .evaluation import calculate_metrics
from .features import load_scan
from .phantom import ExperimentConfig, save_phantom
from .scanners import render_with_api, write_api_config
from .submission import to_gray_png


DEFAULT_CORRECTION_EXPONENTS = (0.70, 0.55, 0.70, 0.90)
DEFAULT_RATIO_HIGHS = (3.0, 2.4, 2.0, 1.9)


def _p09_base_fields(input_path: str | Path, scatterers_count: int) -> tuple[np.ndarray, np.ndarray, ExperimentConfig]:
    config = ExperimentConfig(scatterers_count=scatterers_count)
    target = load_scan(input_path)
    target = np.asarray(resize(target, (config.n_depth, config.n_lateral), anti_aliasing=True, preserve_range=True), dtype=np.float64)
    z = np.arange(config.n_depth, dtype=np.float64)[:, None]
    sparse = np.clip((target - 0.06) / 0.40, 0.0, 1.0)
    gate = np.exp(-0.5 * ((z - 72.0) / 55.0) ** 2)
    body_gate = np.exp(-np.maximum(z - 45.0, 0.0) / 95.0)
    density = 0.00035 + 5.8 * sparse**2.0 * gate + 0.0006 * body_gate
    energy = 0.010 + 0.035 * sparse**1.6
    return density, energy, config


def _write_phantom_from_fields(
    density: np.ndarray,
    energy: np.ndarray,
    out_path: str | Path,
    config: ExperimentConfig,
    seed: int,
    energy_sigma: float = 0.055,
) -> Path:
    rng = np.random.default_rng(seed)
    density = np.maximum(np.asarray(density, dtype=np.float64), 1e-12)
    density = density / density.sum()
    count = config.scatterers_count
    flat_choice = rng.choice(density.size, size=count, replace=True, p=density.ravel())
    z_bin, x_bin = np.divmod(flat_choice, density.shape[1])

    xs = ((x_bin + rng.random(count)) / density.shape[1] - 0.5) * config.x_max
    ys = (rng.random(count) - 0.5) * (2 * config.beam_radius)
    zs = ((z_bin + rng.random(count)) / density.shape[0]) * config.z_max
    energies = np.broadcast_to(energy, density.shape)[z_bin, x_bin]
    energies *= rng.lognormal(mean=0.0, sigma=energy_sigma, size=count)
    data = np.column_stack((xs, ys, zs, np.clip(energies, 0.001, 100.0)))
    return save_phantom(data, out_path, config=config)


def _ratio_field(
    reference_path: str | Path,
    rendered_path: str | Path,
    shape: tuple[int, int],
    ratio_high: float,
    ratio_low: float = 0.35,
    sigma: float = 2.0,
    stabilizer: float = 0.020,
) -> np.ndarray:
    ref = load_scan(reference_path)
    pred = load_scan(rendered_path)
    ratio = (gaussian_filter(ref, sigma) + stabilizer) / (gaussian_filter(pred, sigma) + stabilizer)
    ratio = np.clip(ratio, ratio_low, ratio_high)
    return np.asarray(resize(ratio, shape, anti_aliasing=True, preserve_range=True), dtype=np.float64)


def run_density_correction_refinement(
    input_path: str | Path,
    out_dir: str | Path,
    scatterers_count: int = 900_000,
    seed: int = 7,
    correction_exponents: Iterable[float] = DEFAULT_CORRECTION_EXPONENTS,
    ratio_highs: Iterable[float] = DEFAULT_RATIO_HIGHS,
    api_key_file: str | Path | None = None,
    poll_interval_seconds: float = 10.0,
    max_polls: int = 90,
    skip_existing: bool = True,
) -> Path:
    """Iteratively refine a P09-like phantom using scanner-rendered error fields.

    Each iteration renders the current phantom, computes a smoothed
    ``reference / rendered`` correction field, updates density only, and renders
    the next phantom. The default schedule mirrors the best hosted-API sweep
    observed for the first reference scan.
    """
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    phantom_dir = out_dir / "phantoms"
    synthetic_dir = out_dir / "synthetic"
    gray_dir = out_dir / "synthetic_gray"
    for path in (phantom_dir, synthetic_dir, gray_dir):
        path.mkdir(parents=True, exist_ok=True)

    density, energy, config = _p09_base_fields(input_path, scatterers_count=scatterers_count)
    config_path = write_api_config(out_dir / "Configuration_api.ini", scatterers_count=scatterers_count)
    rows: list[dict[str, float | int | str]] = []
    prev_gray: Path | None = None
    exponent_list = list(correction_exponents)
    ratio_high_list = list(ratio_highs)

    labels = ["base", *[f"iter{idx}" for idx, _ in enumerate(exponent_list, start=1)]]
    for idx, label in enumerate(labels):
        if idx > 0:
            assert prev_gray is not None
            high = ratio_high_list[min(idx - 1, len(ratio_high_list) - 1)]
            exponent = exponent_list[idx - 1]
            ratio = _ratio_field(input_path, prev_gray, density.shape, ratio_high=high)
            density = density * ratio**exponent

        phantom_path = phantom_dir / f"{label}.txt"
        synthetic_path = synthetic_dir / f"{label}.png"
        gray_path = gray_dir / f"{label}_gray.png"
        if not phantom_path.exists() or not skip_existing:
            _write_phantom_from_fields(density, energy, phantom_path, config=config, seed=seed + idx)

        try:
            if skip_existing and gray_path.exists():
                request_id = "existing"
                render_seconds = 0.0
                poll_count = 0
            else:
                request_id, rendered_path, render_seconds, poll_count = render_with_api(
                    phantom_path,
                    config_path,
                    synthetic_path,
                    api_key_file=api_key_file,
                    poll_interval_seconds=poll_interval_seconds,
                    max_polls=max_polls,
                )
                to_gray_png(rendered_path, gray_path)
            metrics = calculate_metrics(input_path, gray_path, include_lpips=False)
            status = "ok"
            error = ""
            prev_gray = gray_path
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
                "iteration": idx,
                "label": label,
                "scatterers_count": scatterers_count,
                "request_id": request_id,
                "phantom_path": str(phantom_path.resolve()),
                "synthetic_png": str(synthetic_path.resolve()),
                "synthetic_gray_png": str(gray_path.resolve()),
                "render_seconds": render_seconds,
                "poll_count": poll_count,
                "error": error,
                **metrics,
            }
        )
        if status != "ok":
            break

    rows.sort(key=lambda row: float(row["MS-SSIM"]) if str(row["MS-SSIM"]) != "nan" else -1.0, reverse=True)
    metrics_path = out_dir / "correction_refinement_metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return metrics_path
