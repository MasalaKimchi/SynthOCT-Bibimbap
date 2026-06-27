from __future__ import annotations

import csv
import re
from pathlib import Path

import requests

from .evaluation import calculate_metrics
from .submission import to_gray_png


RESULT_ID_PATTERN = re.compile(r"result_([A-Za-z0-9_-]+)\.png")


def recover_api_results(
    metrics_csv: str | Path,
    reference_path: str | Path,
    result_base_url: str = "https://synthoct.com/results",
    timeout_seconds: tuple[float, float] = (15, 120),
) -> Path:
    """Recover late hosted-scanner PNGs referenced by failed metrics rows."""
    metrics_csv = Path(metrics_csv)
    with metrics_csv.open() as fobj:
        reader = csv.DictReader(fobj)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise RuntimeError(f"No rows found in {metrics_csv}.")

    for row in rows:
        if row.get("status") == "ok":
            continue
        result_id = _result_id_from_row(row)
        if not result_id:
            continue
        url = f"{result_base_url.rstrip('/')}/result_{result_id}.png"
        response = requests.get(url, timeout=timeout_seconds)
        if response.status_code != 200 or not response.content.startswith(b"\x89PNG"):
            row["request_id"] = result_id
            continue
        synthetic_path = Path(row["synthetic_png"])
        gray_path = Path(row["synthetic_gray_png"])
        synthetic_path.parent.mkdir(parents=True, exist_ok=True)
        gray_path.parent.mkdir(parents=True, exist_ok=True)
        synthetic_path.write_bytes(response.content)
        to_gray_png(synthetic_path, gray_path)
        metrics = calculate_metrics(reference_path, gray_path, include_lpips=False)
        updates = {
            "status": "ok",
            "request_id": result_id,
            "error": "",
            **{key: str(value) for key, value in metrics.items()},
        }
        if "render_seconds" in fieldnames:
            updates["render_seconds"] = row.get("render_seconds") or "0.0"
        if "poll_count" in fieldnames:
            updates["poll_count"] = row.get("poll_count") or "0"
        row.update(updates)

    with metrics_csv.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return metrics_csv


def _result_id_from_row(row: dict[str, str]) -> str | None:
    request_id = row.get("request_id", "")
    if request_id and request_id not in {"failed", "existing"}:
        return request_id
    error = row.get("error", "")
    match = RESULT_ID_PATTERN.search(error)
    if match:
        return match.group(1)
    return None
