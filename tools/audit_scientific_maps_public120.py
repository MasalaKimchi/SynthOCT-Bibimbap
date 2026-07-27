#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from synthoct.evaluation import evaluate_feature_map_metrics, profile_scores
from synthoct.features import SCIENTIFIC_MAP_MODE, ScientificMapConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute masked 51 dB scientific-v1 maps for a retained hosted run."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/hosted_api_public120_v3_200"),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("DATASET/DATASET_PNG"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/hosted_api_public120_v3_200/audit"),
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    run_dir = args.run_dir.resolve()
    reference_root = args.reference_root.resolve()
    output_dir = args.out_dir.resolve()
    detail_path = run_dir / "detail.csv"
    with detail_path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    if len(source_rows) != 120:
        raise ValueError(f"expected 120 retained rows, found {len(source_rows)}")

    rows: list[dict[str, object]] = []
    for index, source in enumerate(source_rows, start=1):
        reference = reference_root / source["reference"]
        prediction = run_dir / "gray" / f"{source['case_id']}.png"
        if not reference.is_file() or not prediction.is_file():
            raise ValueError(f"missing pair for {source['case_id']}")
        with tempfile.TemporaryDirectory(prefix="synthoct-scientific-maps-") as temp:
            temp = Path(temp)
            metrics = evaluate_feature_map_metrics(
                reference,
                prediction,
                temp / "reference",
                temp / "prediction",
                map_mode=SCIENTIFIC_MAP_MODE,
            )
        profiles = profile_scores(
            reference,
            prediction,
            map_mode=SCIENTIFIC_MAP_MODE,
        )
        rows.append(
            {
                "index": index,
                "case_id": source["case_id"],
                "reference": source["reference"],
                **profiles,
                **metrics,
            }
        )
        print(f"[{index}/120] {source['case_id']}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_output = output_dir / "scientific_v1_metrics.csv"
    temporary_detail = detail_output.with_suffix(".csv.tmp")
    with temporary_detail.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary_detail.replace(detail_output)

    numeric_keys = [
        key
        for key, value in rows[0].items()
        if key not in {"index", "case_id", "reference", "Map_Mode"}
        and isinstance(value, (int, float, np.number))
    ]
    summaries: dict[str, dict[str, float]] = {}
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite scientific metric: {key}")
        summaries[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "mode": SCIENTIFIC_MAP_MODE,
        "official_or_hidden_score": False,
        "competition_formula_applicable": False,
        "n": len(rows),
        "source_detail": str(detail_path),
        "source_detail_sha256": _sha256(detail_path),
        "config": asdict(ScientificMapConfig()),
        "metrics": summaries,
        "interpretation": (
            "Reference-defined masked comparisons of derived 51 dB float maps; "
            "not independent optical-property ground truth."
        ),
    }
    summary_output = output_dir / "scientific_v1_summary.json"
    temporary_summary = summary_output.with_suffix(".json.tmp")
    temporary_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_summary.replace(summary_output)
    print(detail_output)
    print(summary_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
