from __future__ import annotations

import csv
from pathlib import Path

from .evaluation import calculate_metrics
from .scanners import render_with_api, write_api_config
from .submission import to_gray_png
from .validation import run_method


def run_api_seed_search(
    input_path: str | Path,
    out_dir: str | Path,
    method: str,
    seeds: list[int],
    scatterers_count: int = 300_000,
    api_key_file: str | Path | None = None,
    poll_interval_seconds: float = 10.0,
    max_polls: int = 60,
    skip_existing: bool = True,
) -> Path:
    """Render one named method across seeds and rank by scanner-rendered SSIM.

    The loop is intentionally resumable because hosted scanner calls can be slow
    or temporarily unavailable. Existing grayscale renders are rescored unless
    ``skip_existing`` is false.
    """
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    phantom_dir = out_dir / "phantoms"
    synthetic_dir = out_dir / "synthetic"
    gray_dir = out_dir / "synthetic_gray"
    for path in (phantom_dir, synthetic_dir, gray_dir):
        path.mkdir(parents=True, exist_ok=True)

    config_path = write_api_config(out_dir / "Configuration_api.ini", scatterers_count=scatterers_count)
    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        label = f"{method}_seed{seed}"
        phantom_path = phantom_dir / f"{label}.txt"
        synthetic_path = synthetic_dir / f"{label}.png"
        gray_path = gray_dir / f"{label}_gray.png"

        if not phantom_path.exists() or not skip_existing:
            run_method(method, input_path, phantom_path, scatterers_count=scatterers_count, seed=seed)

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
        except Exception as exc:
            request_id = "failed"
            render_seconds = 0.0
            poll_count = 0
            metrics = {"MSE": float("nan"), "PSNR": float("nan"), "SSIM": float("nan"), "MS-SSIM": float("nan"), "VIF": float("nan"), "LPIPS": float("nan"), "LPIPS_PROXY": float("nan")}
            status = "failed"
            error = str(exc)

        rows.append(
            {
                "status": status,
                "method": method,
                "seed": seed,
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

    rows.sort(key=lambda row: float(row["SSIM"]) if str(row["SSIM"]) != "nan" else -1.0, reverse=True)
    metrics_path = out_dir / "seed_search_metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return metrics_path
