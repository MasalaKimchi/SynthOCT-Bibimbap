from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path

from .candidate_rendering import render_candidate_queue
from .energy_ratio_refinement import write_energy_ratio_phantom
from .flow_refinement import write_flow_transport_phantom


def _slug(text: str, max_length: int = 80) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", Path(text).stem).strip("_")[:max_length]


def _write_rows(path: Path, rows: list[dict[str, object]]) -> Path:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def run_flow_energy_rank_batch(
    base_api_metrics: str | Path,
    out_dir: str | Path,
    rank_start: int,
    rank_end: int,
    api_key_file: str | Path | None = None,
    api_concurrency: int = 2,
    poll_interval_seconds: float = 3.0,
    max_polls: int = 80,
    flow_strength: float = 0.25,
    flow_smooth_sigma: float = 1.2,
    flow_attachment: float = 6.0,
    energy_exponent: float = 0.8,
) -> Path:
    """Run a fixed flow plus energy-residual rescue batch over base MS-SSIM ranks."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flow_dir = out_dir / "flow_phantoms"
    energy_dir = out_dir / "energy_phantoms"
    flow_dir.mkdir(parents=True, exist_ok=True)
    energy_dir.mkdir(parents=True, exist_ok=True)

    base_rows = _read_rows(base_api_metrics)
    ranked = sorted(enumerate(base_rows), key=lambda item: float(item[1]["MS-SSIM"]))
    selected = [(rank, idx, row) for rank, (idx, row) in enumerate(ranked, start=1) if rank_start <= rank <= rank_end]
    if not selected:
        raise ValueError(f"No rows selected for ranks {rank_start}-{rank_end}")

    manifest_rows = []
    for rank, idx, row in selected:
        manifest_rows.append(
            {
                "rank": rank,
                "row_index": idx,
                "source_archive_path": row["source_archive_path"],
                "base_ms_ssim": row["MS-SSIM"],
                "base_lpips": row["LPIPS"],
                "reference_png": row["reference_png"],
                "base_phantom_path": row["phantom_path"],
                "base_synthetic_gray_png": row["synthetic_gray_png"],
            }
        )
    _write_rows(out_dir / f"rank{rank_start}_{rank_end}_manifest.csv", manifest_rows)

    flow_rows = []
    flow_label = (
        f"s{flow_strength:.3f}_sig{flow_smooth_sigma:.2f}_att{flow_attachment:.1f}"
        .replace(".", "p")
        .replace("-", "n")
    )
    for priority, (rank, _idx, row) in enumerate(selected, start=1):
        source_slug = _slug(row["source_archive_path"])
        method = f"flowfix_{flow_label}_rank{rank}_{source_slug}"
        flow_path = flow_dir / f"rank{rank}_{source_slug}_flowfix.txt"
        if not flow_path.exists():
            write_flow_transport_phantom(
                row["reference_png"],
                row["phantom_path"],
                row["synthetic_gray_png"],
                flow_path,
                strength=flow_strength,
                smooth_sigma=flow_smooth_sigma,
                attachment=flow_attachment,
            )
        flow_rows.append(
            {
                "priority": priority,
                "rank": rank,
                "method": method,
                "source_archive_path": row["source_archive_path"],
                "base_ms_ssim": row["MS-SSIM"],
                "base_lpips": row["LPIPS"],
                "reference_png": row["reference_png"],
                "phantom_path": str(flow_path),
                "base_phantom_path": row["phantom_path"],
                "base_synthetic_gray_png": row["synthetic_gray_png"],
            }
        )
    flow_queue = _write_rows(out_dir / "flow_queue_with_refs.csv", flow_rows)
    flow_metrics = render_candidate_queue(
        flow_queue,
        selected[0][2]["reference_png"],
        out_dir / "flow_rendered",
        api_key_file=api_key_file,
        poll_interval_seconds=poll_interval_seconds,
        max_polls=max_polls,
        api_concurrency=api_concurrency,
    )

    flow_by_priority = {str(row["priority"]): row for row in _read_rows(flow_metrics)}
    flow_correct = []
    for row in flow_rows:
        metrics = flow_by_priority[str(row["priority"])]
        flow_correct.append(
            {
                **row,
                "flow_ms_ssim": metrics["MS-SSIM"],
                "flow_lpips_proxy": metrics["LPIPS_PROXY"],
                "delta_ms_ssim": float(metrics["MS-SSIM"]) - float(row["base_ms_ssim"]),
                "flow_synthetic_gray_png": metrics["synthetic_gray_png"],
            }
        )
    _write_rows(out_dir / "flow_correct_metrics.csv", flow_correct)

    energy_rows = []
    energy_label = f"e{energy_exponent:.3f}".replace(".", "p")
    for row in flow_correct:
        rank = int(row["rank"])
        source_slug = _slug(str(row["source_archive_path"]))
        method = f"flowfix_energy_{energy_label}_{flow_label}_rank{rank}_{source_slug}"
        energy_path = energy_dir / f"rank{rank}_{source_slug}_flow_energy_{energy_label}.txt"
        if not energy_path.exists():
            write_energy_ratio_phantom(
                row["reference_png"],
                row["phantom_path"],
                row["flow_synthetic_gray_png"],
                energy_path,
                exponent=energy_exponent,
            )
        energy_rows.append(
            {
                "priority": row["priority"],
                "rank": row["rank"],
                "method": method,
                "source_archive_path": row["source_archive_path"],
                "base_ms_ssim": row["base_ms_ssim"],
                "base_lpips": row["base_lpips"],
                "flow_ms_ssim": row["flow_ms_ssim"],
                "reference_png": row["reference_png"],
                "phantom_path": str(energy_path),
                "flow_phantom_path": row["phantom_path"],
                "flow_synthetic_gray_png": row["flow_synthetic_gray_png"],
            }
        )
    energy_queue = _write_rows(out_dir / "energy_queue_with_refs.csv", energy_rows)
    energy_metrics = render_candidate_queue(
        energy_queue,
        selected[0][2]["reference_png"],
        out_dir / "energy_rendered",
        api_key_file=api_key_file,
        poll_interval_seconds=poll_interval_seconds,
        max_polls=max_polls,
        api_concurrency=api_concurrency,
    )

    energy_by_priority = {str(row["priority"]): row for row in _read_rows(energy_metrics)}
    combined = []
    for row in energy_rows:
        metrics = energy_by_priority[str(row["priority"])]
        base_ms = float(row["base_ms_ssim"])
        flow_ms = float(row["flow_ms_ssim"])
        energy_ms = float(metrics["MS-SSIM"])
        combined.append(
            {
                "priority": row["priority"],
                "rank": row["rank"],
                "method": metrics["method"],
                "source_archive_path": row["source_archive_path"],
                "base_ms_ssim": row["base_ms_ssim"],
                "base_lpips": row["base_lpips"],
                "flow_ms_ssim": row["flow_ms_ssim"],
                "flow_energy_ms_ssim": metrics["MS-SSIM"],
                "flow_delta_ms_ssim": flow_ms - base_ms,
                "flow_energy_delta_ms_ssim": energy_ms - base_ms,
                "best_stage": "flow_energy"
                if energy_ms >= max(base_ms, flow_ms)
                else "flow"
                if flow_ms >= base_ms
                else "base",
                "reference_png": row["reference_png"],
                "flow_phantom_path": row["flow_phantom_path"],
                "flow_synthetic_gray_png": row["flow_synthetic_gray_png"],
                "flow_energy_phantom_path": metrics["phantom_path"],
                "flow_energy_synthetic_gray_png": metrics["synthetic_gray_png"],
                "flow_energy_lpips_proxy": metrics["LPIPS_PROXY"],
                "flow_energy_request_id": metrics["request_id"],
            }
        )
    combined_path = _write_rows(out_dir / "flow_energy_correct_metrics.csv", combined)

    deltas = [float(row["flow_energy_delta_ms_ssim"]) for row in combined]
    summary = [
        {
            "rank_start": rank_start,
            "rank_end": rank_end,
            "n": len(combined),
            "base_ms_ssim_mean": statistics.mean(float(row["base_ms_ssim"]) for row in combined),
            "flow_ms_ssim_mean": statistics.mean(float(row["flow_ms_ssim"]) for row in combined),
            "flow_energy_ms_ssim_mean": statistics.mean(float(row["flow_energy_ms_ssim"]) for row in combined),
            "flow_energy_delta_mean": statistics.mean(deltas),
            "flow_energy_delta_min": min(deltas),
            "flow_energy_delta_max": max(deltas),
            "wins_flow_energy": sum(row["best_stage"] == "flow_energy" for row in combined),
            "wins_flow": sum(row["best_stage"] == "flow" for row in combined),
            "wins_base": sum(row["best_stage"] == "base" for row in combined),
        }
    ]
    _write_rows(out_dir / "batch_summary.csv", summary)
    return combined_path
