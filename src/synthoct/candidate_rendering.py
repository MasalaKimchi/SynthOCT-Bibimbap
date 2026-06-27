from __future__ import annotations

import csv
from pathlib import Path

from .evaluation import calculate_metrics
from .phantom import load_phantom
from .scanners import render_with_api, write_api_config
from .submission import to_gray_png


def render_candidate_queue(
    queue_csv: str | Path,
    reference_path: str | Path,
    out_dir: str | Path,
    api_key_file: str | Path | None = None,
    poll_interval_seconds: float = 10.0,
    max_polls: int = 90,
    max_candidates: int | None = None,
    skip_existing: bool = True,
) -> Path:
    """Render queued phantom candidates with the hosted scanner and rank metrics."""
    queue_csv = Path(queue_csv)
    reference_path = Path(reference_path)
    out_dir = Path(out_dir)
    synthetic_dir = out_dir / "synthetic"
    gray_dir = out_dir / "synthetic_gray"
    config_dir = out_dir / "configs"
    for path in (synthetic_dir, gray_dir, config_dir):
        path.mkdir(parents=True, exist_ok=True)

    with queue_csv.open() as fobj:
        queued = list(csv.DictReader(fobj))
    if max_candidates is not None:
        queued = queued[:max_candidates]
    if not queued:
        raise RuntimeError(f"No candidates found in {queue_csv}.")

    rows = []
    for idx, row in enumerate(queued, start=1):
        method = row.get("method") or f"candidate_{idx:03d}"
        phantom_path = Path(row["phantom_path"])
        synthetic_path = synthetic_dir / f"{idx:02d}_{method}.png"
        gray_path = gray_dir / f"{idx:02d}_{method}_gray.png"
        scatterers_count = len(load_phantom(phantom_path))
        config_path = write_api_config(config_dir / f"{idx:02d}_{method}_Configuration.ini", scatterers_count=scatterers_count)
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
                "evidence_source": "hosted_api_true_scanner",
                "evidence_scope": "single_reference_candidate_queue",
                "priority": row.get("priority", idx),
                "method": method,
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

    rows.sort(key=lambda item: float(item["MS-SSIM"]) if str(item["MS-SSIM"]) != "nan" else -1.0, reverse=True)
    metrics_path = out_dir / "candidate_queue_metrics.csv"
    with metrics_path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return metrics_path
