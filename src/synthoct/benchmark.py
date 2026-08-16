from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .evaluation import (
    calculate_metrics,
    competition_formula_estimate,
    evaluate_feature_map_metrics,
    profile_scores,
)
from .features import ORGANIZER_MAP_MODE
from .holographic_inverse import HolographicInverseConfig, holographic_inverse_phantom
from .phantom import ExperimentConfig
from .provenance import dependency_versions, git_provenance
from .scanners import render_reference_scanner


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_manifest_sha256(rows: list[dict[str, float | str]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["reference"]).encode())
        digest.update(b"\0")
        digest.update(str(row["reference_sha256"]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _write_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metric_summary(rows: list[dict[str, float | str]]) -> dict[str, dict[str, float]]:
    result = {}
    metric_keys = [
        key
        for key in rows[0]
        if key.endswith("MS-SSIM") or key.endswith("LPIPS")
    ]
    for key in metric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        if np.isfinite(values).all():
            result[key] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    return result


def _method_name(inverse: HolographicInverseConfig) -> str:
    if inverse.phase_encoding == "dispersion-canceling-pair":
        return "holographic-inverse-v3-phase-pair"
    if inverse.phase_iterations:
        return "holographic-inverse-v2-phase-retrieval"
    return "holographic-inverse-v1"


def run_local_benchmark(
    input_root: str | Path,
    output_dir: str | Path,
    *,
    inverse: HolographicInverseConfig | None = None,
    scatterers_count: int = 300_000,
    include_maps: bool = True,
    include_lpips: bool = False,
    limit: int | None = None,
    map_mode: str = ORGANIZER_MAP_MODE,
) -> tuple[Path, Path]:
    """Evaluate references with the local implementation of the published forward model."""
    input_root = Path(input_root)
    output_dir = Path(output_dir)
    inverse = inverse or HolographicInverseConfig()
    inverse.validate()
    if map_mode != ORGANIZER_MAP_MODE and include_lpips:
        raise ValueError("LPIPS/competition scoring requires organizer-compatible map mode")
    method = _method_name(inverse)
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    references = [
        path
        for path in sorted(input_root.rglob("*.png"))
        if not path.stem.endswith(("_OAC", "_SC", "_RSC"))
    ]
    if limit is not None:
        references = references[:limit]
    if not references:
        raise ValueError(f"no reference PNGs found below {input_root}")

    detail_path = output_dir / "detail.csv"
    partial_detail_path = output_dir / "detail.partial.csv"
    summary_path = output_dir / "summary.json"
    partial_summary_path = output_dir / "summary.partial.json"
    rows: list[dict[str, float | str]] = []
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    # A rerun must never leave a new partial detail file beside an old summary.
    # Keep the previous detail for inspection, but invalidate its summary until
    # this run atomically promotes both replacement artifacts.
    summary_path.unlink(missing_ok=True)
    partial_detail_path.unlink(missing_ok=True)
    partial_summary_path.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(prefix="synthoct-benchmark-") as temp:
        work = Path(temp)
        for index, reference in enumerate(references, start=1):
            phantom = work / "phantom.txt"
            rendered = work / "rendered.png"
            generation_start = time.perf_counter()
            holographic_inverse_phantom(
                reference,
                phantom,
                scatterers_count=scatterers_count,
                inverse=inverse,
            )
            generation_seconds = time.perf_counter() - generation_start
            render_start = time.perf_counter()
            render_reference_scanner(phantom, rendered)
            render_seconds = time.perf_counter() - render_start

            struct = calculate_metrics(reference, rendered, include_lpips=include_lpips)
            row: dict[str, float | str] = {
                "index": index,
                "reference": str(reference.relative_to(input_root)),
                "reference_sha256": _sha256(reference),
                "method": method,
                "evidence_source": "source_equivalent_local_scanner",
                "generation_seconds": generation_seconds,
                "render_seconds": render_seconds,
            }
            row.update({f"Struct_{key}": value for key, value in struct.items()})
            row.update(profile_scores(reference, rendered, map_mode=map_mode))
            if include_maps:
                row.update(
                    evaluate_feature_map_metrics(
                        reference,
                        rendered,
                        work / "reference_maps",
                        work / "prediction_maps",
                        include_lpips=include_lpips,
                        map_mode=map_mode,
                    )
                )
            rows.append(row)
            _write_rows(partial_detail_path, rows)
            print(
                f"[{index}/{len(references)}] {row['reference']} "
                f"Struct_MS-SSIM={float(row['Struct_MS-SSIM']):.6f}",
                flush=True,
            )

    summary: dict[str, object] = {
        "method": method,
        "evidence_source": "source_equivalent_local_scanner",
        "official_or_hidden_score": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "n": len(rows),
        "input_root": str(input_root),
        "dataset_manifest_sha256": _dataset_manifest_sha256(rows),
        "parameters": asdict(inverse),
        "benchmark_controls": {
            "scatterers_count": scatterers_count,
            "include_maps": include_maps,
            "include_lpips": include_lpips,
            "limit": limit,
            "map_mode": map_mode,
        },
        "scanner": asdict(ExperimentConfig(scatterers_count=scatterers_count)),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "dependencies": dependency_versions(),
            "git": git_provenance(),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": _metric_summary(rows),
    }
    if include_maps and include_lpips:
        score, medians = competition_formula_estimate(rows, map_mode=map_mode)
        summary["competition_formula_estimate"] = score
        summary["competition_medians"] = medians
    partial_detail_path.replace(detail_path)
    partial_summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial_summary_path.replace(summary_path)
    return detail_path, summary_path
