from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .evaluation import calculate_metrics, metric_evaluation_metadata
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
    api_concurrency: int = 1,
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
    api_concurrency = max(1, int(api_concurrency))

    metrics_path = out_dir / "candidate_queue_metrics.csv"

    def write_metrics(rows_to_write: list[dict[str, float | int | str]]) -> None:
        if not rows_to_write:
            return
        ordered = sorted(
            rows_to_write,
            key=lambda item: float(item["MS-SSIM"]) if str(item["MS-SSIM"]) != "nan" else -1.0,
            reverse=True,
        )
        with metrics_path.open("w", newline="") as fobj:
            writer = csv.DictWriter(fobj, fieldnames=list(ordered[0].keys()))
            writer.writeheader()
            writer.writerows(ordered)

    def render_row(idx: int, row: dict[str, str]) -> dict[str, float | int | str]:
        method = row.get("method") or f"candidate_{idx:03d}"
        phantom_path = Path(row["phantom_path"])
        row_reference_path = Path(row.get("reference_png") or reference_path)
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
            metrics = calculate_metrics(row_reference_path, gray_path, include_lpips=False)
            metrics.update(metric_evaluation_metadata(row_reference_path, gray_path))
            status = "ok"
            error = ""
        except Exception as exc:
            request_id = str(getattr(exc, "request_id", "failed"))
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
                "evaluation_region": "",
                "reference_shape": "",
                "prediction_shape": "",
                "evaluated_shape": "",
                "prediction_resized_to_reference": "",
            }
            status = "failed"
            error = str(exc)
        return {
            "status": status,
            "evidence_source": "hosted_api_true_scanner",
            "evidence_scope": "single_reference_candidate_queue",
            "priority": row.get("priority", idx),
            "method": method,
            "request_id": request_id,
            "reference_png": str(row_reference_path.resolve()),
            "phantom_path": str(phantom_path.resolve()),
            "synthetic_png": str(synthetic_path.resolve()),
            "synthetic_gray_png": str(gray_path.resolve()),
            "render_seconds": render_seconds,
            "poll_count": poll_count,
            "error": error,
            **metrics,
        }

    rows = []
    if api_concurrency == 1:
        for idx, row in enumerate(queued, start=1):
            rows.append(render_row(idx, row))
            write_metrics(rows)
    else:
        executor = ThreadPoolExecutor(max_workers=api_concurrency)
        futures = [executor.submit(render_row, idx, row) for idx, row in enumerate(queued, start=1)]
        try:
            for future in as_completed(futures):
                rows.append(future.result())
                write_metrics(rows)
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            write_metrics(rows)
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown()

    write_metrics(rows)
    return metrics_path
