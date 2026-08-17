#!/usr/bin/env python3
"""Run a resumable hosted SynthOCT evaluation over a public reference set.

The API key is taken only from the process environment.  Request metadata,
raw PNGs, grayscale metric inputs, map images, per-case metrics, and summary
statistics are retained below ``--out-dir``; the key is never serialized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from skimage import io

from synthoct.evaluation import (
    calculate_metrics,
    competition_formula_estimate,
    evaluate_feature_map_metrics,
    profile_scores,
)
from synthoct.features import ORGANIZER_MAP_MODE, load_scan
from synthoct.scanners.api import poll_api_result, submit_api_render
from synthoct.scanners.config import write_api_config
from synthoct.submission import flatten_reference, sha256_file


RUNNER_SCHEMA_VERSION = 2


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_jobs(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "map_mode": ORGANIZER_MAP_MODE,
            "cases": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in {1, RUNNER_SCHEMA_VERSION} or not isinstance(
        payload.get("cases"), dict
    ):
        raise ValueError(f"unsupported jobs state: {path}")
    return payload


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["case_id"]: row for row in csv.DictReader(handle)}


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def case_fingerprint(reference: Path, phantom: Path, config: Path) -> tuple[str, dict[str, str]]:
    inputs = {
        "reference_sha256": sha256_file(reference),
        "phantom_sha256": sha256_file(phantom),
        "config_sha256": sha256_file(config),
        "map_mode": ORGANIZER_MAP_MODE,
        "runner_schema": str(RUNNER_SCHEMA_VERSION),
    }
    payload = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest(), inputs


def write_gray(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = np.rint(load_scan(source) * 255.0).astype(np.uint8)
    io.imsave(destination, image, check_contrast=False)
    return destination


def summary(rows: list[dict[str, object]]) -> dict[str, object]:
    metric_keys = [key for key in rows[0] if key.endswith(("MS-SSIM", "LPIPS"))]
    aggregates: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        if np.isfinite(values).all():
            aggregates[key] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    score, medians = competition_formula_estimate(rows)
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "n": len(rows),
        "evidence_source": "organizer_hosted_challenge_service",
        "official_or_hidden_score": False,
        "map_mode": ORGANIZER_MAP_MODE,
        "competition_formula_estimate": score,
        "competition_medians": medians,
        "metrics": aggregates,
    }


def run(args: argparse.Namespace) -> None:
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"set {args.api_key_env} in the environment")

    reference_root = args.reference_root.resolve()
    phantom_dir = args.phantom_dir.resolve()
    out_dir = args.out_dir.resolve()
    raw_dir = out_dir / "raw"
    gray_dir = out_dir / "gray"
    maps_dir = out_dir / "maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = write_api_config(out_dir / "Configuration.ini")
    jobs_path = out_dir / "jobs.json"
    detail_path = out_dir / "detail.partial.csv"
    jobs = load_jobs(jobs_path)
    cases = jobs["cases"]
    assert isinstance(cases, dict)

    references = sorted(reference_root.rglob("*.png"))
    if not references:
        raise ValueError(f"no PNG references below {reference_root}")
    existing_rows = load_rows(detail_path)
    rows: list[dict[str, object]] = []

    for index, reference in enumerate(references, start=1):
        phantom_name = flatten_reference(reference, reference_root)
        phantom = phantom_dir / phantom_name
        if not phantom.is_file():
            raise FileNotFoundError(f"missing phantom for {reference}: {phantom}")
        case_id = phantom.stem
        raw = raw_dir / f"{case_id}.png"
        gray = gray_dir / f"{case_id}.png"
        fingerprint, fingerprint_inputs = case_fingerprint(reference, phantom, config)
        record = cases.get(case_id)
        if record is None:
            record = {
                "reference": str(reference.relative_to(reference_root)),
                "phantom": str(phantom.relative_to(phantom_dir)),
                "case_fingerprint": fingerprint,
                **fingerprint_inputs,
                "state": "new",
            }
            cases[case_id] = record
        if not isinstance(record, dict):
            raise ValueError(f"invalid job record for {case_id}")
        if record.get("case_fingerprint") != fingerprint:
            reason = (
                "legacy state has no content fingerprint"
                if "case_fingerprint" not in record
                else "reference, phantom, configuration, or map mode changed"
            )
            raise RuntimeError(
                f"refusing unsafe resume for {case_id}: {reason}; use a fresh --out-dir"
            )
        if raw.exists() and not record.get("request_id"):
            raise RuntimeError(
                f"refusing orphan raw PNG for {case_id}; use a fresh --out-dir"
            )
        if raw.exists() and record.get("raw_png_sha256") not in {
            None,
            sha256_file(raw),
        }:
            raise RuntimeError(f"raw PNG hash changed for {case_id}")
        if gray.exists() and record.get("gray_png_sha256") not in {
            None,
            sha256_file(gray),
        }:
            raise RuntimeError(f"gray PNG hash changed for {case_id}")

        if not raw.exists():
            request_id = record.get("request_id")
            if not request_id:
                request_id, submit_seconds = submit_api_render(phantom, config, api_key=api_key)
                record.update(
                    {
                        "request_id": request_id,
                        "submitted_at_utc": utc_now(),
                        "submit_seconds": submit_seconds,
                        "state": "submitted",
                    }
                )
                atomic_json(jobs_path, jobs)
            rendered, poll_seconds, poll_count = poll_api_result(
                str(request_id),
                raw,
                poll_interval_seconds=args.poll_interval_seconds,
                max_polls=args.max_polls,
            )
            record.update(
                {
                    "state": "downloaded",
                    "poll_seconds": poll_seconds,
                    "poll_count": poll_count,
                    "raw_png": str(rendered.relative_to(out_dir)),
                    "downloaded_at_utc": utc_now(),
                }
            )
            atomic_json(jobs_path, jobs)

        if not gray.exists():
            write_gray(raw, gray)
        record["raw_png_sha256"] = sha256_file(raw)
        record["gray_png_sha256"] = sha256_file(gray)
        atomic_json(jobs_path, jobs)

        if case_id in existing_rows:
            if existing_rows[case_id].get("case_fingerprint") != fingerprint:
                raise RuntimeError(
                    f"refusing stale metric row for {case_id}; use a fresh --out-dir"
                )
            rows.append({key: value for key, value in existing_rows[case_id].items()})
            print(f"[{index}/{len(references)}] reuse metrics {case_id}", flush=True)
            continue

        row: dict[str, object] = {
            "index": index,
            "case_id": case_id,
            "reference": str(reference.relative_to(reference_root)),
            "phantom": phantom.name,
            "request_id": record["request_id"],
            "case_fingerprint": fingerprint,
            **fingerprint_inputs,
            "api_submit_seconds": record.get("submit_seconds", ""),
            "api_poll_seconds": record.get("poll_seconds", ""),
            "api_poll_count": record.get("poll_count", ""),
        }
        row.update({f"Struct_{key}": value for key, value in calculate_metrics(reference, gray, include_lpips=True).items()})
        row.update(profile_scores(reference, gray))
        row.update(
            evaluate_feature_map_metrics(
                reference,
                gray,
                maps_dir / "reference" / case_id,
                maps_dir / "prediction" / case_id,
                include_lpips=True,
                map_mode=ORGANIZER_MAP_MODE,
            )
        )
        rows.append(row)
        rows.sort(key=lambda item: int(item["index"]))
        write_rows(detail_path, rows)
        record["state"] = "evaluated"
        record["evaluated_at_utc"] = utc_now()
        atomic_json(jobs_path, jobs)
        print(
            f"[{index}/{len(references)}] hosted {case_id} "
            f"Struct_MS-SSIM={float(row['Struct_MS-SSIM']):.6f}",
            flush=True,
        )

    final_detail = out_dir / "detail.csv"
    write_rows(final_detail, rows)
    atomic_json(out_dir / "summary.json", summary(rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--phantom-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--api-key-env", default="SYNTHOCT_API_KEY")
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    parser.add_argument("--max-polls", type=int, default=120)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
