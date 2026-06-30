from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

from .candidate_rendering import render_candidate_queue
from .energy_ratio_refinement import write_energy_ratio_phantom
from .evaluation import calculate_metrics, competition_proxy_score, evaluate_feature_map_metrics, profile_scores
from .flow_refinement import write_flow_transport_phantom
from .texture_refinement import write_texture_matched_phantom


STAGE2_CONTROL_FIELDS = [
    "strength_policy",
    "evaluated_strengths",
    "strength_candidate_count",
    "residual_control_policy",
    "flow_smooth_sigma",
    "flow_attachment",
    "energy_base_policy",
    "energy_base_flow_delta_threshold",
    "energy_exponent",
    "energy_sigma",
    "energy_ratio_low",
    "energy_ratio_high",
    "energy_clip_low",
    "energy_clip_high",
    "texture_mean_exponent",
    "texture_exponent",
    "texture_deep_exponent",
]

ENERGY_GATE_SWEEP_FIELDS = [
    "min_energy_flow_delta",
    "energy_calls",
    "skipped_energy_calls",
    "negative_energy_calls",
    "skipped_positive_energy_calls",
    "best_delta_sum",
    "retained_best_delta_fraction",
    "delta_loss_vs_full_energy",
    "delta_per_energy_call",
    "energy_win_rate",
    "map_objective_examples",
    "negative_map_calls",
    "skipped_positive_map_calls",
    "map_delta_sum",
    "retained_map_delta_fraction",
    "map_delta_loss_vs_full_energy",
    "map_delta_per_energy_call",
    "map_win_rate",
]

ENERGY_GATE_CONTEXT_FIELDS = [
    "energy_gate_summary",
    "energy_gate_min_flow_delta",
    "energy_gate_expected_status",
    "energy_gate_score_multiplier",
    "energy_gate_retained_delta_fraction",
    "energy_gate_retained_map_delta_fraction",
]

ENERGY_MAP_OBJECTIVE_FIELDS = [
    "flow_energy_map_objective_score",
    "flow_energy_map_delta_vs_current",
    "flow_energy_map_delta_vs_flow",
    "flow_energy_map_objective_status",
    "flow_energy_map_objective_error",
]

EXACT_PROBE_FEEDBACK_FIELDS = [
    "exact_probe_feedback_source_archive_path",
    "exact_probe_feedback_n",
    "exact_probe_feedback_energy_n",
    "exact_probe_feedback_best_delta_mean",
    "exact_probe_feedback_flow_delta_mean",
    "exact_probe_feedback_energy_delta_mean",
    "exact_probe_feedback_status",
    "exact_probe_feedback_score_multiplier",
]


def _slug(text: str, max_length: int = 80) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", Path(text).stem).strip("_")[:max_length]


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames:
            with path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()
            return path
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _ordered_union(rows: list[dict[str, object]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    return fields


def calibrate_energy_followup_gate(
    probe_metrics_paths: tuple[str | Path, ...],
    out_path: str | Path,
    *,
    min_retained_delta_fraction: float = 0.995,
    min_retained_map_delta_fraction: float | None = None,
    min_energy_calls: int = 1,
    max_negative_energy_calls: int | None = 0,
    max_negative_map_calls: int | None = None,
) -> Path:
    """Learn a flow-delta gate for spending stage-2 energy calls from true-scanner probes."""
    examples = _energy_gate_examples(probe_metrics_paths)
    if not examples:
        raise ValueError("No probe rows with finite flow and energy MS-SSIM metrics were found.")

    full_delta_sum = sum(row["best_delta_with_energy"] for row in examples)
    full_map_delta_sum = sum(
        row["best_map_delta_with_energy"]
        for row in examples
        if math.isfinite(row["best_map_delta_with_energy"])
    )
    has_map_objective = any(math.isfinite(row["flow_energy_map_delta_vs_current"]) for row in examples)
    thresholds = _energy_gate_thresholds(examples)
    sweep_rows = [
        _energy_gate_threshold_row(
            examples,
            threshold,
            full_delta_sum=full_delta_sum,
            full_map_delta_sum=full_map_delta_sum,
        )
        for threshold in thresholds
    ]

    eligible = [
        row
        for row in sweep_rows
        if int(row["energy_calls"]) >= min_energy_calls
        and float(row["retained_best_delta_fraction"]) >= min_retained_delta_fraction
        and (
            not has_map_objective
            or min_retained_map_delta_fraction is None
            or float(row["retained_map_delta_fraction"]) >= min_retained_map_delta_fraction
        )
        and (
            max_negative_energy_calls is None
            or int(row["negative_energy_calls"]) <= max_negative_energy_calls
        )
        and (
            not has_map_objective
            or max_negative_map_calls is None
            or int(row["negative_map_calls"]) <= max_negative_map_calls
        )
    ]
    selection_status = "selected_with_negative_limit"
    if not eligible:
        selection_status = "selected_without_negative_limit"
        eligible = [
            row
            for row in sweep_rows
            if int(row["energy_calls"]) >= min_energy_calls
            and float(row["retained_best_delta_fraction"]) >= min_retained_delta_fraction
            and (
                not has_map_objective
                or min_retained_map_delta_fraction is None
                or float(row["retained_map_delta_fraction"]) >= min_retained_map_delta_fraction
            )
        ]
    if not eligible:
        selection_status = "selected_best_available"
        eligible = sweep_rows

    selected = max(
        eligible,
        key=lambda row: (
            -int(row["energy_calls"]),
            float(row["best_delta_sum"]),
            -int(row["negative_energy_calls"]),
            -int(row["negative_map_calls"]),
            -float(row["min_energy_flow_delta"]),
        ),
    )
    out = Path(out_path)
    sweep_path = out.with_name(f"{out.stem}_threshold_sweep.csv")
    _write_rows(sweep_path, sweep_rows, fieldnames=ENERGY_GATE_SWEEP_FIELDS)

    summary = {
        "model_type": "flow_delta_energy_followup_gate",
        "probe_metrics_paths": [str(path) for path in probe_metrics_paths],
        "observed_energy_examples": len(examples),
        "observed_map_objective_examples": sum(
            1 for row in examples if math.isfinite(row["flow_energy_map_delta_vs_current"])
        ),
        "full_energy_best_delta_sum": full_delta_sum,
        "full_energy_map_delta_sum": full_map_delta_sum if has_map_objective else "",
        "min_retained_delta_fraction": min_retained_delta_fraction,
        "min_retained_map_delta_fraction": min_retained_map_delta_fraction if has_map_objective else "",
        "min_energy_calls": min_energy_calls,
        "max_negative_energy_calls": max_negative_energy_calls,
        "max_negative_map_calls": max_negative_map_calls if has_map_objective else "",
        "selection_status": selection_status,
        "recommended_min_energy_flow_delta": selected["min_energy_flow_delta"],
        "selected": selected,
        "sweep_csv": str(sweep_path),
        "evidence_scope": "true_scanner_probe_calibration_not_hidden_holdout",
        "promotion_allowed_without_true_scanner": False,
        "hidden_holdout_final_score": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def enrich_residual_probe_energy_maps(
    probe_metrics_paths: tuple[str | Path, ...],
    out_path: str | Path,
    *,
    maps_dir: str | Path | None = None,
) -> Path:
    """Backfill flow-energy map objective scores from rendered probe PNGs."""
    out = Path(out_path)
    maps_root = Path(maps_dir) if maps_dir is not None else out.with_name(f"{out.stem}_maps")
    enriched: list[dict[str, object]] = []
    for metrics_path in probe_metrics_paths:
        path = Path(metrics_path)
        for row_index, row in enumerate(_read_rows(path), start=1):
            enriched_row: dict[str, object] = {**row, "probe_metrics_path": str(path)}
            energy_score, energy_status, energy_error = _stage_map_objective(
                row.get("reference_png", ""),
                row.get("flow_energy_synthetic_gray_png", ""),
                maps_root / f"{path.stem}_row_{row_index:04d}" / "flow_energy",
            )
            current_score = _finite(row.get("current_map_objective_score"))
            flow_score = _finite(row.get("flow_map_objective_score"))
            enriched_row.update(
                {
                    "flow_energy_map_objective_score": energy_score,
                    "flow_energy_map_delta_vs_current": _delta(energy_score, current_score),
                    "flow_energy_map_delta_vs_flow": _delta(energy_score, flow_score),
                    "flow_energy_map_objective_status": energy_status,
                    "flow_energy_map_objective_error": energy_error,
                }
            )
            enriched.append(enriched_row)
    fieldnames = _ordered_union(enriched)
    _write_rows(out, enriched, fieldnames=fieldnames)
    rows_with_energy_map = sum(
        1 for row in enriched if math.isfinite(_finite(row.get("flow_energy_map_objective_score")))
    )
    summary = {
        "probe_metrics_paths": [str(Path(path)) for path in probe_metrics_paths],
        "enriched_metrics": str(out),
        "row_count": len(enriched),
        "rows_with_flow_energy_map_objective": rows_with_energy_map,
        "maps_dir": str(maps_root),
        "evidence_scope": "residual_probe_map_enrichment_not_challenge_evidence",
        "promotion_allowed_without_true_scanner": False,
        "hidden_holdout_final_score": False,
    }
    out.with_name(f"{out.stem}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out


def run_adaptive_flow_strength_batch(
    current_api_metrics: str | Path,
    out_dir: str | Path,
    rank_start: int,
    rank_end: int,
    strengths: tuple[float, ...] = (0.12, 0.38),
    api_key_file: str | Path | None = None,
    api_concurrency: int = 2,
    poll_interval_seconds: float = 3.0,
    max_polls: int = 80,
    flow_smooth_sigma: float = 1.2,
    flow_attachment: float = 6.0,
    energy_exponent: float = 0.8,
) -> Path:
    """Sweep flow strengths over weak current API rows and render energy follow-ups."""
    out_dir = Path(out_dir)
    flow_dir = out_dir / "flow_phantoms"
    energy_dir = out_dir / "energy_phantoms"
    flow_dir.mkdir(parents=True, exist_ok=True)
    energy_dir.mkdir(parents=True, exist_ok=True)

    current_rows = _read_rows(current_api_metrics)
    ranked = sorted(enumerate(current_rows), key=lambda item: float(item[1]["MS-SSIM"]))
    selected = [(rank, idx, row) for rank, (idx, row) in enumerate(ranked, start=1) if rank_start <= rank <= rank_end]
    if not selected:
        raise ValueError(f"No rows selected for ranks {rank_start}-{rank_end}")
    if not strengths:
        raise ValueError("At least one flow strength is required.")

    manifest_rows = [
        {
            "current_rank": rank,
            "row_index": idx,
            "source_archive_path": row["source_archive_path"],
            "current_ms_ssim": row["MS-SSIM"],
            "current_lpips": row["LPIPS"],
            "reference_png": row["reference_png"],
            "current_phantom_path": row["phantom_path"],
            "current_synthetic_gray_png": row["synthetic_gray_png"],
        }
        for rank, idx, row in selected
    ]
    _write_rows(out_dir / f"adaptive_rank{rank_start}_{rank_end}_manifest.csv", manifest_rows)

    flow_rows: list[dict[str, object]] = []
    priority = 1
    for rank, idx, row in selected:
        source_slug = _slug(row["source_archive_path"])
        for strength in strengths:
            strength_label = f"s{strength:.3f}".replace(".", "p").replace("-", "n")
            method = f"adapt_flow_{strength_label}_rank{rank}_row{idx}_{source_slug}"
            flow_path = flow_dir / f"rank{rank}_row{idx}_{source_slug}_{strength_label}_flow.txt"
            if not flow_path.exists():
                write_flow_transport_phantom(
                    row["reference_png"],
                    row["phantom_path"],
                    row["synthetic_gray_png"],
                    flow_path,
                    strength=strength,
                    smooth_sigma=flow_smooth_sigma,
                    attachment=flow_attachment,
                )
            flow_rows.append(
                {
                    "priority": priority,
                    "current_rank": rank,
                    "row_index": idx,
                    "strength": strength,
                    "method": method,
                    "source_archive_path": row["source_archive_path"],
                    "current_ms_ssim": row["MS-SSIM"],
                    "current_lpips": row["LPIPS"],
                    "reference_png": row["reference_png"],
                    "phantom_path": str(flow_path),
                    "current_phantom_path": row["phantom_path"],
                    "current_synthetic_gray_png": row["synthetic_gray_png"],
                }
            )
            priority += 1
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

    energy_rows: list[dict[str, object]] = []
    for row in flow_rows:
        flow_metric = flow_by_priority[str(row["priority"])]
        strength_label = f"s{float(row['strength']):.3f}".replace(".", "p").replace("-", "n")
        energy_label = f"e{energy_exponent:.3f}".replace(".", "p")
        source_slug = _slug(str(row["source_archive_path"]))
        method = (
            f"adapt_energy_{energy_label}_{strength_label}_"
            f"rank{row['current_rank']}_row{row['row_index']}_{source_slug}"
        )
        energy_path = (
            energy_dir
            / f"rank{row['current_rank']}_row{row['row_index']}_{source_slug}_{strength_label}_{energy_label}.txt"
        )
        if not energy_path.exists():
            write_energy_ratio_phantom(
                str(row["reference_png"]),
                str(row["phantom_path"]),
                flow_metric["synthetic_gray_png"],
                energy_path,
                exponent=energy_exponent,
            )
        energy_rows.append(
            {
                "priority": row["priority"],
                "current_rank": row["current_rank"],
                "row_index": row["row_index"],
                "strength": row["strength"],
                "method": method,
                "source_archive_path": row["source_archive_path"],
                "current_ms_ssim": row["current_ms_ssim"],
                "current_lpips": row["current_lpips"],
                "flow_ms_ssim": flow_metric["MS-SSIM"],
                "reference_png": row["reference_png"],
                "phantom_path": str(energy_path),
                "flow_phantom_path": row["phantom_path"],
                "flow_synthetic_gray_png": flow_metric["synthetic_gray_png"],
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
        energy_metric = energy_by_priority[str(row["priority"])]
        current_ms = float(row["current_ms_ssim"])
        flow_ms = float(row["flow_ms_ssim"])
        energy_ms = float(energy_metric["MS-SSIM"])
        best_stage = "flow_energy" if energy_ms >= max(current_ms, flow_ms) else "flow" if flow_ms >= current_ms else "current"
        best_ms = max(current_ms, flow_ms, energy_ms)
        combined.append(
            {
                "priority": row["priority"],
                "current_rank": row["current_rank"],
                "row_index": row["row_index"],
                "strength": row["strength"],
                "method": energy_metric["method"],
                "source_archive_path": row["source_archive_path"],
                "current_ms_ssim": row["current_ms_ssim"],
                "current_lpips": row["current_lpips"],
                "flow_ms_ssim": row["flow_ms_ssim"],
                "flow_energy_ms_ssim": energy_metric["MS-SSIM"],
                "flow_delta_vs_current": flow_ms - current_ms,
                "flow_energy_delta_vs_current": energy_ms - current_ms,
                "best_stage": best_stage,
                "best_ms_ssim": best_ms,
                "best_delta_vs_current": best_ms - current_ms,
                "reference_png": row["reference_png"],
                "flow_phantom_path": row["flow_phantom_path"],
                "flow_synthetic_gray_png": row["flow_synthetic_gray_png"],
                "flow_energy_phantom_path": energy_metric["phantom_path"],
                "flow_energy_synthetic_gray_png": energy_metric["synthetic_gray_png"],
                "flow_energy_lpips_proxy": energy_metric["LPIPS_PROXY"],
                "flow_energy_request_id": energy_metric["request_id"],
            }
        )
    combined_path = _write_rows(out_dir / "adaptive_strength_metrics.csv", combined)

    winners: list[dict[str, object]] = []
    selected_keys = sorted({(int(row["current_rank"]), int(row["row_index"])) for row in combined})
    for current_rank, row_index in selected_keys:
        candidates = [row for row in combined if int(row["current_rank"]) == current_rank and int(row["row_index"]) == row_index]
        winner = max(candidates, key=lambda row: float(row["best_ms_ssim"]))
        if float(winner["best_delta_vs_current"]) > 0.0:
            winners.append(winner)
    if winners:
        _write_rows(out_dir / "adaptive_winners.csv", winners)
    return combined_path


def run_residual_selector_flow_batch(
    selector_queue: str | Path,
    out_dir: str | Path,
    *,
    max_candidates: int | None = None,
    max_energy_followups: int | None = None,
    min_energy_flow_delta: float | None = None,
    api_key_file: str | Path | None = None,
    api_concurrency: int = 2,
    poll_interval_seconds: float = 3.0,
    max_polls: int = 80,
    flow_smooth_sigma: float = 1.2,
    flow_attachment: float = 6.0,
    energy_exponent: float = 0.8,
) -> Path:
    """Render selector-recommended row residual probes through flow then energy stages."""
    out_dir = Path(out_dir)
    flow_dir = out_dir / "flow_phantoms"
    energy_dir = out_dir / "energy_phantoms"
    flow_dir.mkdir(parents=True, exist_ok=True)
    energy_dir.mkdir(parents=True, exist_ok=True)

    recommendations = _read_rows(selector_queue)
    if max_candidates is not None:
        recommendations = recommendations[:max(0, int(max_candidates))]
    metrics_path = out_dir / "residual_selector_probe_metrics.csv"
    if not recommendations:
        _write_rows(out_dir / "residual_selector_manifest.csv", [], fieldnames=_selector_manifest_fieldnames())
        return _write_rows(metrics_path, [], fieldnames=_selector_probe_fieldnames())

    manifest_rows = [_selector_manifest_row(row, idx) for idx, row in enumerate(recommendations, start=1)]
    _write_rows(out_dir / "residual_selector_manifest.csv", manifest_rows, fieldnames=_selector_manifest_fieldnames())

    flow_rows: list[dict[str, object]] = []
    for idx, row in enumerate(recommendations, start=1):
        priority = _priority(row, idx)
        strength = _finite(row.get("selected_strength"))
        if not math.isfinite(strength):
            raise ValueError(f"Selector row {idx} is missing selected_strength.")
        reference_png = _required(row, "reference_png", idx)
        current_phantom_path = _required(row, "current_phantom_path", idx)
        current_synthetic_gray_png = _required(row, "current_synthetic_gray_png", idx)
        source_slug = _slug(row.get("source_archive_path", f"row{idx}"))
        strength_label = f"s{strength:.3f}".replace(".", "p").replace("-", "n")
        row_flow_smooth_sigma = _row_float(row, "flow_smooth_sigma", flow_smooth_sigma)
        row_flow_attachment = _row_float(row, "flow_attachment", flow_attachment)
        flow_param_label = (
            f"sig{row_flow_smooth_sigma:.2f}_att{row_flow_attachment:.1f}".replace(".", "p").replace("-", "n")
        )
        method = (
            f"selector_flow_{strength_label}_{flow_param_label}_p{priority}_"
            f"rank{row.get('current_rank', '')}_row{row.get('row_index', idx)}_{source_slug}"
        )
        flow_path = flow_dir / f"p{priority}_rank{row.get('current_rank', '')}_row{row.get('row_index', idx)}_{source_slug}_{strength_label}_flow.txt"
        if not flow_path.exists():
            write_flow_transport_phantom(
                reference_png,
                current_phantom_path,
                current_synthetic_gray_png,
                flow_path,
                strength=strength,
                smooth_sigma=row_flow_smooth_sigma,
                attachment=row_flow_attachment,
            )
        context = _selector_context(row, idx)
        context["flow_smooth_sigma"] = row_flow_smooth_sigma
        context["flow_attachment"] = row_flow_attachment
        flow_rows.append(
            {
                **context,
                "method": method,
                "reference_png": reference_png,
                "phantom_path": str(flow_path),
                "current_phantom_path": current_phantom_path,
                "current_synthetic_gray_png": current_synthetic_gray_png,
            }
        )

    flow_queue = _write_rows(out_dir / "selector_flow_queue_with_refs.csv", flow_rows)
    flow_metrics = render_candidate_queue(
        flow_queue,
        str(flow_rows[0]["reference_png"]),
        out_dir / "flow_rendered",
        api_key_file=api_key_file,
        poll_interval_seconds=poll_interval_seconds,
        max_polls=max_polls,
        api_concurrency=api_concurrency,
    )
    flow_by_priority = {str(row["priority"]): row for row in _read_rows(flow_metrics)}

    combined: list[dict[str, object]] = []
    energy_rows: list[dict[str, object]] = []
    for flow_row in flow_rows:
        flow_metric = flow_by_priority.get(str(flow_row["priority"]), {})
        flow_status = flow_metric.get("status", "ok" if flow_metric else "failed")
        flow_ms = _finite(flow_metric.get("MS-SSIM"))
        current_ms = _finite(flow_row.get("current_ms_ssim"))
        if flow_status != "ok" or not math.isfinite(flow_ms):
            combined.append(
                _combined_probe_row(
                    flow_row,
                    flow_metric,
                    {},
                    flow_status=flow_status,
                    energy_status="not_run",
                    best_stage="current",
                    best_ms_ssim=current_ms,
                )
            )
            continue

        current_map_score, current_map_status, current_map_error = _stage_map_objective(
            flow_row.get("reference_png", ""),
            flow_row.get("current_synthetic_gray_png", ""),
            out_dir / "energy_followup_objectives" / f"p{flow_row['priority']}_current",
        )
        flow_map_score, flow_map_status, flow_map_error = _stage_map_objective(
            flow_row.get("reference_png", ""),
            flow_metric.get("synthetic_gray_png", ""),
            out_dir / "energy_followup_objectives" / f"p{flow_row['priority']}_flow",
        )
        flow_row.update(
            {
                "current_map_objective_score": current_map_score,
                "flow_map_objective_score": flow_map_score,
                "flow_map_delta_vs_current": _delta(flow_map_score, current_map_score),
                "flow_map_objective_status": _combine_map_status(current_map_status, flow_map_status),
                "flow_map_objective_error": "; ".join(
                    error for error in (current_map_error, flow_map_error) if error
                ),
            }
        )

        flow_delta = _delta(flow_ms, current_ms)
        if (
            min_energy_flow_delta is not None
            and math.isfinite(flow_delta)
            and flow_delta < float(min_energy_flow_delta)
        ):
            best_stage = "flow" if flow_ms >= current_ms else "current"
            combined.append(
                _combined_probe_row(
                    flow_row,
                    flow_metric,
                    {},
                    flow_status="ok",
                    energy_status="deferred_flow_regression",
                    best_stage=best_stage,
                    best_ms_ssim=max(_finite_or_floor(current_ms), _finite_or_floor(flow_ms)),
                )
            )
            continue

        strength_label = f"s{float(flow_row['selected_strength']):.3f}".replace(".", "p").replace("-", "n")
        row_energy_exponent = _finite(flow_row.get("energy_exponent"))
        if not math.isfinite(row_energy_exponent):
            row_energy_exponent = float(energy_exponent)
        row_energy_sigma = _row_float(flow_row, "energy_sigma", 2.0)
        row_energy_ratio_low = _row_float(flow_row, "energy_ratio_low", 0.65)
        row_energy_ratio_high = _row_float(flow_row, "energy_ratio_high", 1.32)
        row_energy_clip_low = _row_float(flow_row, "energy_clip_low", 0.78)
        row_energy_clip_high = _row_float(flow_row, "energy_clip_high", 1.22)
        texture_mean_exponent = _row_float(flow_row, "texture_mean_exponent", 0.0)
        texture_exponent = _row_float(flow_row, "texture_exponent", 0.0)
        texture_deep_exponent = _row_float(flow_row, "texture_deep_exponent", 0.0)
        texture_enabled = any(
            abs(value) > 1e-12 for value in (texture_mean_exponent, texture_exponent, texture_deep_exponent)
        )
        energy_base_stage, energy_base_phantom, energy_base_render = _select_energy_base(
            flow_row,
            flow_metric,
            flow_delta=flow_delta,
        )
        flow_row.update(
            {
                "energy_base_stage": energy_base_stage,
                "energy_base_phantom_path": energy_base_phantom,
                "energy_base_synthetic_gray_png": energy_base_render,
                "energy_exponent": row_energy_exponent,
                "energy_sigma": row_energy_sigma,
                "energy_ratio_low": row_energy_ratio_low,
                "energy_ratio_high": row_energy_ratio_high,
                "energy_clip_low": row_energy_clip_low,
                "energy_clip_high": row_energy_clip_high,
                "texture_mean_exponent": texture_mean_exponent,
                "texture_exponent": texture_exponent,
                "texture_deep_exponent": texture_deep_exponent,
            }
        )
        energy_label = f"e{row_energy_exponent:.3f}_sig{row_energy_sigma:.2f}".replace(".", "p")
        if energy_base_stage != "flow":
            energy_label = f"base{energy_base_stage}_{energy_label}"
        if texture_enabled:
            texture_label = f"tex{texture_exponent:.2f}_deep{texture_deep_exponent:.2f}".replace(".", "p")
            energy_label = f"{energy_label}_{texture_label}"
        source_slug = _slug(str(flow_row["source_archive_path"]))
        method = (
            f"selector_energy_{energy_label}_{strength_label}_p{flow_row['priority']}_"
            f"rank{flow_row.get('current_rank', '')}_row{flow_row.get('row_index', '')}_{source_slug}"
        )
        energy_path = (
            energy_dir
            / f"p{flow_row['priority']}_rank{flow_row.get('current_rank', '')}_row{flow_row.get('row_index', '')}_{source_slug}_{strength_label}_{energy_label}.txt"
        )
        try:
            if not energy_path.exists():
                ratio_path = energy_path
                if texture_enabled:
                    ratio_dir = energy_dir / "ratio_inputs"
                    ratio_dir.mkdir(parents=True, exist_ok=True)
                    ratio_path = ratio_dir / f"{energy_path.stem}_ratio.txt"
                write_energy_ratio_phantom(
                    str(flow_row["reference_png"]),
                    energy_base_phantom,
                    energy_base_render,
                    ratio_path,
                    exponent=row_energy_exponent,
                    sigma=row_energy_sigma,
                    ratio_low=row_energy_ratio_low,
                    ratio_high=row_energy_ratio_high,
                    clip_low=row_energy_clip_low,
                    clip_high=row_energy_clip_high,
                )
                if texture_enabled:
                    write_texture_matched_phantom(
                        str(flow_row["reference_png"]),
                        ratio_path,
                        energy_base_render,
                        energy_path,
                        mean_exponent=texture_mean_exponent,
                        texture_exponent=texture_exponent,
                        deep_exponent=texture_deep_exponent,
                    )
        except Exception as exc:
            failed_energy = {"status": "failed", "error": str(exc)}
            best_stage = "flow" if flow_ms >= current_ms else "current"
            combined.append(
                _combined_probe_row(
                    flow_row,
                    flow_metric,
                    failed_energy,
                    flow_status=flow_status,
                    energy_status="failed",
                    best_stage=best_stage,
                    best_ms_ssim=max(_finite_or_floor(current_ms), _finite_or_floor(flow_ms)),
                )
            )
            continue
        energy_rows.append(
            {
                **flow_row,
                "method": method,
                "phantom_path": str(energy_path),
                "flow_phantom_path": flow_row["phantom_path"],
                "flow_synthetic_gray_png": flow_metric["synthetic_gray_png"],
                "energy_base_stage": energy_base_stage,
                "energy_base_phantom_path": energy_base_phantom,
                "energy_base_synthetic_gray_png": energy_base_render,
                "flow_ms_ssim": flow_metric["MS-SSIM"],
                "flow_request_id": flow_metric.get("request_id", ""),
                "flow_error": flow_metric.get("error", ""),
            }
        )

    if max_energy_followups is not None:
        energy_cap = max(0, int(max_energy_followups))
        ranked_energy = sorted(energy_rows, key=_energy_followup_score, reverse=True)
        promoted_priorities = {str(row["priority"]) for row in ranked_energy[:energy_cap]}
        deferred_rows = [row for row in energy_rows if str(row["priority"]) not in promoted_priorities]
        energy_rows = [row for row in energy_rows if str(row["priority"]) in promoted_priorities]
        for deferred in deferred_rows:
            flow_metric = _flow_metric_from_energy_row(deferred)
            current_ms = _finite(deferred.get("current_ms_ssim"))
            flow_ms = _finite(deferred.get("flow_ms_ssim"))
            best_stage = "flow" if flow_ms >= current_ms else "current"
            combined.append(
                _combined_probe_row(
                    deferred,
                    flow_metric,
                    {},
                    flow_status="ok",
                    energy_status="deferred_budget",
                    best_stage=best_stage,
                    best_ms_ssim=max(_finite_or_floor(current_ms), _finite_or_floor(flow_ms)),
                )
            )

    if energy_rows:
        energy_queue = _write_rows(out_dir / "selector_energy_queue_with_refs.csv", energy_rows)
        energy_metrics = render_candidate_queue(
            energy_queue,
            str(energy_rows[0]["reference_png"]),
            out_dir / "energy_rendered",
            api_key_file=api_key_file,
            poll_interval_seconds=poll_interval_seconds,
            max_polls=max_polls,
            api_concurrency=api_concurrency,
        )
        energy_by_priority = {str(row["priority"]): row for row in _read_rows(energy_metrics)}
        for energy_row in energy_rows:
            flow_metric = _flow_metric_from_energy_row(energy_row)
            energy_metric = energy_by_priority.get(str(energy_row["priority"]), {})
            current_ms = _finite(energy_row.get("current_ms_ssim"))
            flow_ms = _finite(energy_row.get("flow_ms_ssim"))
            energy_ms = _finite(energy_metric.get("MS-SSIM"))
            energy_map_score, energy_map_status, energy_map_error = _stage_map_objective(
                energy_row.get("reference_png", ""),
                energy_metric.get("synthetic_gray_png", ""),
                out_dir / "energy_followup_objectives" / f"p{energy_row['priority']}_flow_energy",
            )
            current_map_score = _finite(energy_row.get("current_map_objective_score"))
            flow_map_score = _finite(energy_row.get("flow_map_objective_score"))
            energy_row.update(
                {
                    "flow_energy_map_objective_score": energy_map_score,
                    "flow_energy_map_delta_vs_current": _delta(energy_map_score, current_map_score),
                    "flow_energy_map_delta_vs_flow": _delta(energy_map_score, flow_map_score),
                    "flow_energy_map_objective_status": energy_map_status,
                    "flow_energy_map_objective_error": energy_map_error,
                }
            )
            best_stage, best_ms = _best_stage(current_ms, flow_ms, energy_ms)
            combined.append(
                _combined_probe_row(
                    energy_row,
                    flow_metric,
                    energy_metric,
                    flow_status="ok",
                    energy_status=energy_metric.get("status", "ok" if energy_metric else "failed"),
                    best_stage=best_stage,
                    best_ms_ssim=best_ms,
                )
            )
    else:
        _write_rows(out_dir / "selector_energy_queue_with_refs.csv", [], fieldnames=_selector_energy_queue_fieldnames())

    combined.sort(key=lambda row: int(row["priority"]))
    return _write_rows(metrics_path, combined, fieldnames=_selector_probe_fieldnames())


def _selector_context(row: dict[str, str], idx: int) -> dict[str, object]:
    return {
        "priority": _priority(row, idx),
        "budget_slot": row.get("budget_slot", ""),
        "selector_priority": row.get("selector_priority", ""),
        "row_index": row.get("row_index", idx - 1),
        "current_rank": row.get("current_rank", ""),
        "source_archive_path": row.get("source_archive_path", ""),
        "selected_strength": row.get("selected_strength", ""),
        "strength_policy": row.get("strength_policy", ""),
        "evaluated_strengths": row.get("evaluated_strengths", ""),
        "strength_candidate_count": row.get("strength_candidate_count", ""),
        "selector_holdout_group_key": row.get("selector_holdout_group_key", ""),
        "selector_holdout_group_n": row.get("selector_holdout_group_n", ""),
        "selector_holdout_group_win_rate": row.get("selector_holdout_group_win_rate", ""),
        "selector_holdout_group_delta_mean": row.get("selector_holdout_group_delta_mean", ""),
        "selector_holdout_group_status": row.get("selector_holdout_group_status", ""),
        "selector_holdout_score_multiplier": row.get("selector_holdout_score_multiplier", ""),
        "surrogate_calibration_status": row.get("surrogate_calibration_status", ""),
        "surrogate_calibration_support_n": row.get("surrogate_calibration_support_n", ""),
        "surrogate_calibration_distance": row.get("surrogate_calibration_distance", ""),
        "surrogate_calibration_score_multiplier": row.get("surrogate_calibration_score_multiplier", ""),
        "surrogate_calibration_training_ssim_min": row.get("surrogate_calibration_training_ssim_min", ""),
        "surrogate_calibration_training_ssim_max": row.get("surrogate_calibration_training_ssim_max", ""),
        "candidate_kind": row.get("candidate_kind", ""),
        "flow_strength_multiplier": row.get("flow_strength_multiplier", ""),
        "diversity_stratum": row.get("diversity_stratum", ""),
        "true_probe_feedback_group_key": row.get("true_probe_feedback_group_key", ""),
        "true_probe_feedback_n": row.get("true_probe_feedback_n", ""),
        "true_probe_feedback_energy_n": row.get("true_probe_feedback_energy_n", ""),
        "true_probe_feedback_best_delta_mean": row.get("true_probe_feedback_best_delta_mean", ""),
        "true_probe_feedback_energy_delta_mean": row.get("true_probe_feedback_energy_delta_mean", ""),
        "true_probe_feedback_flow_delta_mean": row.get("true_probe_feedback_flow_delta_mean", ""),
        "true_probe_feedback_flow_negative_rate": row.get("true_probe_feedback_flow_negative_rate", ""),
        "true_probe_feedback_flow_regression_rate": row.get("true_probe_feedback_flow_regression_rate", ""),
        "true_probe_feedback_energy_win_rate": row.get("true_probe_feedback_energy_win_rate", ""),
        "true_probe_feedback_status": row.get("true_probe_feedback_status", ""),
        "true_probe_feedback_score_multiplier": row.get("true_probe_feedback_score_multiplier", ""),
        "exact_probe_feedback_source_archive_path": row.get("exact_probe_feedback_source_archive_path", ""),
        "exact_probe_feedback_n": row.get("exact_probe_feedback_n", ""),
        "exact_probe_feedback_energy_n": row.get("exact_probe_feedback_energy_n", ""),
        "exact_probe_feedback_best_delta_mean": row.get("exact_probe_feedback_best_delta_mean", ""),
        "exact_probe_feedback_flow_delta_mean": row.get("exact_probe_feedback_flow_delta_mean", ""),
        "exact_probe_feedback_energy_delta_mean": row.get("exact_probe_feedback_energy_delta_mean", ""),
        "exact_probe_feedback_status": row.get("exact_probe_feedback_status", ""),
        "exact_probe_feedback_score_multiplier": row.get("exact_probe_feedback_score_multiplier", ""),
        "energy_gate_summary": row.get("energy_gate_summary", ""),
        "energy_gate_min_flow_delta": row.get("energy_gate_min_flow_delta", ""),
        "energy_gate_expected_status": row.get("energy_gate_expected_status", ""),
        "energy_gate_score_multiplier": row.get("energy_gate_score_multiplier", ""),
        "energy_gate_retained_delta_fraction": row.get("energy_gate_retained_delta_fraction", ""),
        "energy_gate_retained_map_delta_fraction": row.get("energy_gate_retained_map_delta_fraction", ""),
        "residual_control_policy": row.get("residual_control_policy", ""),
        "flow_smooth_sigma": row.get("flow_smooth_sigma", ""),
        "flow_attachment": row.get("flow_attachment", ""),
        "energy_base_policy": row.get("energy_base_policy", ""),
        "energy_base_flow_delta_threshold": row.get("energy_base_flow_delta_threshold", ""),
        "energy_exponent": row.get("energy_exponent", ""),
        "energy_sigma": row.get("energy_sigma", ""),
        "energy_ratio_low": row.get("energy_ratio_low", ""),
        "energy_ratio_high": row.get("energy_ratio_high", ""),
        "energy_clip_low": row.get("energy_clip_low", ""),
        "energy_clip_high": row.get("energy_clip_high", ""),
        "texture_mean_exponent": row.get("texture_mean_exponent", ""),
        "texture_exponent": row.get("texture_exponent", ""),
        "texture_deep_exponent": row.get("texture_deep_exponent", ""),
        "budget_score": row.get("budget_score", ""),
        "planned_api_stage": row.get("planned_api_stage", ""),
        "total_api_budget": row.get("total_api_budget", ""),
        "flow_budget": row.get("flow_budget", ""),
        "reserved_energy_followups": row.get("reserved_energy_followups", ""),
        "promotion_rule": row.get("promotion_rule", ""),
        "current_ms_ssim": row.get("current_ms_ssim", ""),
        "current_ssim": row.get("current_ssim", ""),
        "current_lpips": row.get("current_lpips", ""),
        "expected_delta_mean": row.get("expected_delta_mean", ""),
        "expected_delta_lcb": row.get("expected_delta_lcb", ""),
        "expected_delta_uncertainty": row.get("expected_delta_uncertainty", ""),
        "acquisition_score": row.get("acquisition_score", ""),
        "expected_flow_delta_mean": row.get("expected_flow_delta_mean", ""),
        "expected_energy_delta_mean": row.get("expected_energy_delta_mean", ""),
        "expected_energy_extra_mean": row.get("expected_energy_extra_mean", ""),
        "nearest_flow_win_rate": row.get("nearest_flow_win_rate", ""),
        "nearest_energy_win_rate": row.get("nearest_energy_win_rate", ""),
        "nearest_teacher_count": row.get("nearest_teacher_count", ""),
        "nearest_teacher_sources": row.get("nearest_teacher_sources", ""),
        "surrogate_gate_status": row.get("surrogate_gate_status", ""),
        "evidence_scope": row.get("evidence_scope", "selector_planning_not_challenge_evidence"),
    }


def _selector_manifest_row(row: dict[str, str], idx: int) -> dict[str, object]:
    return {
        **_selector_context(row, idx),
        "reference_png": row.get("reference_png", ""),
        "current_phantom_path": row.get("current_phantom_path", ""),
        "current_synthetic_gray_png": row.get("current_synthetic_gray_png", ""),
    }


def _combined_probe_row(
    row: dict[str, object],
    flow_metric: dict[str, str],
    energy_metric: dict[str, str],
    *,
    flow_status: str,
    energy_status: str,
    best_stage: str,
    best_ms_ssim: float,
) -> dict[str, object]:
    current_ms = _finite(row.get("current_ms_ssim"))
    flow_ms = _finite(flow_metric.get("MS-SSIM") or row.get("flow_ms_ssim"))
    energy_ms = _finite(energy_metric.get("MS-SSIM"))
    return {
        "priority": row["priority"],
        "budget_slot": row.get("budget_slot", ""),
        "selector_priority": row.get("selector_priority", ""),
        "row_index": row.get("row_index", ""),
        "current_rank": row.get("current_rank", ""),
        "source_archive_path": row.get("source_archive_path", ""),
        "selected_strength": row.get("selected_strength", ""),
        "strength_policy": row.get("strength_policy", ""),
        "evaluated_strengths": row.get("evaluated_strengths", ""),
        "strength_candidate_count": row.get("strength_candidate_count", ""),
        "selector_holdout_group_key": row.get("selector_holdout_group_key", ""),
        "selector_holdout_group_n": row.get("selector_holdout_group_n", ""),
        "selector_holdout_group_win_rate": row.get("selector_holdout_group_win_rate", ""),
        "selector_holdout_group_delta_mean": row.get("selector_holdout_group_delta_mean", ""),
        "selector_holdout_group_status": row.get("selector_holdout_group_status", ""),
        "selector_holdout_score_multiplier": row.get("selector_holdout_score_multiplier", ""),
        "surrogate_calibration_status": row.get("surrogate_calibration_status", ""),
        "surrogate_calibration_support_n": row.get("surrogate_calibration_support_n", ""),
        "surrogate_calibration_distance": row.get("surrogate_calibration_distance", ""),
        "surrogate_calibration_score_multiplier": row.get("surrogate_calibration_score_multiplier", ""),
        "surrogate_calibration_training_ssim_min": row.get("surrogate_calibration_training_ssim_min", ""),
        "surrogate_calibration_training_ssim_max": row.get("surrogate_calibration_training_ssim_max", ""),
        "candidate_kind": row.get("candidate_kind", ""),
        "flow_strength_multiplier": row.get("flow_strength_multiplier", ""),
        "diversity_stratum": row.get("diversity_stratum", ""),
        "true_probe_feedback_group_key": row.get("true_probe_feedback_group_key", ""),
        "true_probe_feedback_n": row.get("true_probe_feedback_n", ""),
        "true_probe_feedback_energy_n": row.get("true_probe_feedback_energy_n", ""),
        "true_probe_feedback_best_delta_mean": row.get("true_probe_feedback_best_delta_mean", ""),
        "true_probe_feedback_energy_delta_mean": row.get("true_probe_feedback_energy_delta_mean", ""),
        "true_probe_feedback_flow_delta_mean": row.get("true_probe_feedback_flow_delta_mean", ""),
        "true_probe_feedback_flow_negative_rate": row.get("true_probe_feedback_flow_negative_rate", ""),
        "true_probe_feedback_flow_regression_rate": row.get("true_probe_feedback_flow_regression_rate", ""),
        "true_probe_feedback_energy_win_rate": row.get("true_probe_feedback_energy_win_rate", ""),
        "true_probe_feedback_status": row.get("true_probe_feedback_status", ""),
        "true_probe_feedback_score_multiplier": row.get("true_probe_feedback_score_multiplier", ""),
        "exact_probe_feedback_source_archive_path": row.get("exact_probe_feedback_source_archive_path", ""),
        "exact_probe_feedback_n": row.get("exact_probe_feedback_n", ""),
        "exact_probe_feedback_energy_n": row.get("exact_probe_feedback_energy_n", ""),
        "exact_probe_feedback_best_delta_mean": row.get("exact_probe_feedback_best_delta_mean", ""),
        "exact_probe_feedback_flow_delta_mean": row.get("exact_probe_feedback_flow_delta_mean", ""),
        "exact_probe_feedback_energy_delta_mean": row.get("exact_probe_feedback_energy_delta_mean", ""),
        "exact_probe_feedback_status": row.get("exact_probe_feedback_status", ""),
        "exact_probe_feedback_score_multiplier": row.get("exact_probe_feedback_score_multiplier", ""),
        "residual_control_policy": row.get("residual_control_policy", ""),
        "flow_smooth_sigma": row.get("flow_smooth_sigma", ""),
        "flow_attachment": row.get("flow_attachment", ""),
        "energy_base_policy": row.get("energy_base_policy", ""),
        "energy_base_flow_delta_threshold": row.get("energy_base_flow_delta_threshold", ""),
        "energy_base_stage": row.get("energy_base_stage", ""),
        "energy_base_phantom_path": row.get("energy_base_phantom_path", ""),
        "energy_base_synthetic_gray_png": row.get("energy_base_synthetic_gray_png", ""),
        "energy_exponent": row.get("energy_exponent", ""),
        "energy_sigma": row.get("energy_sigma", ""),
        "energy_ratio_low": row.get("energy_ratio_low", ""),
        "energy_ratio_high": row.get("energy_ratio_high", ""),
        "energy_clip_low": row.get("energy_clip_low", ""),
        "energy_clip_high": row.get("energy_clip_high", ""),
        "texture_mean_exponent": row.get("texture_mean_exponent", ""),
        "texture_exponent": row.get("texture_exponent", ""),
        "texture_deep_exponent": row.get("texture_deep_exponent", ""),
        "budget_score": row.get("budget_score", ""),
        "planned_api_stage": row.get("planned_api_stage", ""),
        "total_api_budget": row.get("total_api_budget", ""),
        "flow_budget": row.get("flow_budget", ""),
        "reserved_energy_followups": row.get("reserved_energy_followups", ""),
        "promotion_rule": row.get("promotion_rule", ""),
        "current_ms_ssim": row.get("current_ms_ssim", ""),
        "current_ssim": row.get("current_ssim", ""),
        "current_lpips": row.get("current_lpips", ""),
        "expected_delta_mean": row.get("expected_delta_mean", ""),
        "expected_delta_lcb": row.get("expected_delta_lcb", ""),
        "expected_delta_uncertainty": row.get("expected_delta_uncertainty", ""),
        "acquisition_score": row.get("acquisition_score", ""),
        "expected_flow_delta_mean": row.get("expected_flow_delta_mean", ""),
        "expected_energy_delta_mean": row.get("expected_energy_delta_mean", ""),
        "expected_energy_extra_mean": row.get("expected_energy_extra_mean", ""),
        "nearest_flow_win_rate": row.get("nearest_flow_win_rate", ""),
        "nearest_energy_win_rate": row.get("nearest_energy_win_rate", ""),
        "nearest_teacher_count": row.get("nearest_teacher_count", ""),
        "nearest_teacher_sources": row.get("nearest_teacher_sources", ""),
        "surrogate_gate_status": row.get("surrogate_gate_status", ""),
        "flow_status": flow_status,
        "energy_status": energy_status,
        "best_stage": best_stage,
        "best_ms_ssim": best_ms_ssim,
        "flow_ms_ssim": flow_ms,
        "flow_energy_ms_ssim": energy_ms,
        "flow_delta_vs_current": _delta(flow_ms, current_ms),
        "flow_energy_delta_vs_current": _delta(energy_ms, current_ms),
        "current_map_objective_score": row.get("current_map_objective_score", ""),
        "flow_map_objective_score": row.get("flow_map_objective_score", ""),
        "flow_map_delta_vs_current": row.get("flow_map_delta_vs_current", ""),
        "flow_map_objective_status": row.get("flow_map_objective_status", ""),
        "flow_map_objective_error": row.get("flow_map_objective_error", ""),
        "flow_energy_map_objective_score": row.get("flow_energy_map_objective_score", ""),
        "flow_energy_map_delta_vs_current": row.get("flow_energy_map_delta_vs_current", ""),
        "flow_energy_map_delta_vs_flow": row.get("flow_energy_map_delta_vs_flow", ""),
        "flow_energy_map_objective_status": row.get("flow_energy_map_objective_status", ""),
        "flow_energy_map_objective_error": row.get("flow_energy_map_objective_error", ""),
        "method": energy_metric.get("method", row.get("method", "")),
        "reference_png": row.get("reference_png", ""),
        "current_phantom_path": row.get("current_phantom_path", ""),
        "current_synthetic_gray_png": row.get("current_synthetic_gray_png", ""),
        "flow_phantom_path": flow_metric.get("phantom_path", row.get("flow_phantom_path", row.get("phantom_path", ""))),
        "flow_synthetic_gray_png": flow_metric.get("synthetic_gray_png", row.get("flow_synthetic_gray_png", "")),
        "flow_request_id": flow_metric.get("request_id", row.get("flow_request_id", "")),
        "flow_error": flow_metric.get("error", row.get("flow_error", "")),
        "flow_energy_phantom_path": energy_metric.get("phantom_path", row.get("phantom_path", "")) if energy_metric else "",
        "flow_energy_synthetic_gray_png": energy_metric.get("synthetic_gray_png", ""),
        "flow_energy_request_id": energy_metric.get("request_id", ""),
        "flow_energy_error": energy_metric.get("error", ""),
        "flow_energy_lpips_proxy": energy_metric.get("LPIPS_PROXY", ""),
        "evidence_source": "hosted_api_true_scanner",
        "evidence_scope": "selector_residual_probe_not_challenge_evidence",
        "promotion_allowed_without_true_scanner": False,
        "hidden_holdout_final_score": False,
    }


def _best_stage(current_ms: float, flow_ms: float, energy_ms: float) -> tuple[str, float]:
    scored = [
        ("current", _finite_or_floor(current_ms)),
        ("flow", _finite_or_floor(flow_ms)),
        ("flow_energy", _finite_or_floor(energy_ms)),
    ]
    stage, score = max(scored, key=lambda item: item[1])
    return stage, score if math.isfinite(score) else float("nan")


def _energy_followup_score(row: dict[str, object]) -> tuple[float, float, float]:
    flow_delta = _delta(_finite(row.get("flow_ms_ssim")), _finite(row.get("current_ms_ssim")))
    map_delta = _finite(row.get("flow_map_delta_vs_current"))
    acquisition = _finite(row.get("acquisition_score"))
    primary = map_delta if math.isfinite(map_delta) else flow_delta
    return (_finite_or_floor(primary), _finite_or_floor(flow_delta), _finite_or_floor(acquisition))


def _stage_map_objective(
    reference_path: object,
    prediction_path: object,
    maps_dir: Path,
) -> tuple[float, str, str]:
    if reference_path in (None, "") or prediction_path in (None, ""):
        return float("nan"), "missing_png", ""
    try:
        struct = calculate_metrics(reference_path, prediction_path, include_lpips=False)
        maps = evaluate_feature_map_metrics(
            reference_path,
            prediction_path,
            maps_dir / "ref_maps",
            maps_dir / "pred_maps",
            include_lpips=False,
        )
        profiles = profile_scores(reference_path, prediction_path)
    except Exception as exc:
        return float("nan"), "failed", str(exc)
    row: dict[str, float] = {f"Struct_{key}": value for key, value in struct.items()}
    row.update(maps)
    row.update(profiles)
    return competition_proxy_score(row), "ok", ""


def _combine_map_status(current_status: str, flow_status: str) -> str:
    if current_status == flow_status == "ok":
        return "ok"
    if "failed" in {current_status, flow_status}:
        return "failed"
    return "missing_png"


def _flow_metric_from_energy_row(row: dict[str, object]) -> dict[str, str]:
    return {
        "status": "ok",
        "MS-SSIM": str(row.get("flow_ms_ssim", "")),
        "request_id": str(row.get("flow_request_id", "")),
        "error": str(row.get("flow_error", "")),
        "phantom_path": str(row.get("flow_phantom_path", "")),
        "synthetic_gray_png": str(row.get("flow_synthetic_gray_png", "")),
    }


def _select_energy_base(
    row: dict[str, object],
    flow_metric: dict[str, str],
    *,
    flow_delta: float,
) -> tuple[str, str, str]:
    policy = str(row.get("energy_base_policy", "") or "flow")
    threshold = _row_float(row, "energy_base_flow_delta_threshold", 0.0)
    use_current = policy == "current_on_flow_regression" and math.isfinite(flow_delta) and flow_delta < threshold
    if use_current:
        return (
            "current",
            str(row.get("current_phantom_path", "")),
            str(row.get("current_synthetic_gray_png", "")),
        )
    return (
        "flow",
        str(row.get("phantom_path", "")),
        str(flow_metric.get("synthetic_gray_png", "")),
    )


def _delta(value: float, baseline: float) -> float:
    return value - baseline if math.isfinite(value) and math.isfinite(baseline) else float("nan")


def _finite(value: object) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _row_float(row: dict[str, object], key: str, default: float) -> float:
    value = _finite(row.get(key))
    return value if math.isfinite(value) else float(default)


def _finite_or_floor(value: float) -> float:
    return value if math.isfinite(value) else -1.0


def _energy_gate_examples(probe_metrics_paths: tuple[str | Path, ...]) -> list[dict[str, float]]:
    examples: list[dict[str, float]] = []
    for path in probe_metrics_paths:
        for row in _read_rows(path):
            current_ms = _finite(row.get("current_ms_ssim"))
            flow_ms = _finite(row.get("flow_ms_ssim"))
            energy_ms = _finite(row.get("flow_energy_ms_ssim"))
            if not all(math.isfinite(value) for value in (current_ms, flow_ms, energy_ms)):
                continue
            flow_delta = _delta(flow_ms, current_ms)
            energy_delta = _delta(energy_ms, current_ms)
            flow_map_delta = _finite(row.get("flow_map_delta_vs_current"))
            energy_map_delta = _finite(row.get("flow_energy_map_delta_vs_current"))
            best_map_with_energy = (
                max(0.0, _finite_or_floor(flow_map_delta), _finite_or_floor(energy_map_delta))
                if math.isfinite(energy_map_delta)
                else float("nan")
            )
            best_map_without_energy = (
                max(0.0, _finite_or_floor(flow_map_delta)) if math.isfinite(energy_map_delta) else float("nan")
            )
            examples.append(
                {
                    "current_ms_ssim": current_ms,
                    "flow_ms_ssim": flow_ms,
                    "flow_energy_ms_ssim": energy_ms,
                    "flow_delta_vs_current": flow_delta,
                    "flow_energy_delta_vs_current": energy_delta,
                    "flow_map_delta_vs_current": flow_map_delta,
                    "flow_energy_map_delta_vs_current": energy_map_delta,
                    "best_delta_with_energy": max(current_ms, flow_ms, energy_ms) - current_ms,
                    "best_delta_without_energy": max(current_ms, flow_ms) - current_ms,
                    "best_map_delta_with_energy": best_map_with_energy,
                    "best_map_delta_without_energy": best_map_without_energy,
                }
            )
    return examples


def _energy_gate_thresholds(examples: list[dict[str, float]]) -> list[float]:
    flow_deltas = sorted({row["flow_delta_vs_current"] for row in examples})
    if not flow_deltas:
        return []
    return [flow_deltas[0] - 1e-6, *flow_deltas, flow_deltas[-1] + 1e-6]


def _energy_gate_threshold_row(
    examples: list[dict[str, float]],
    threshold: float,
    *,
    full_delta_sum: float,
    full_map_delta_sum: float,
) -> dict[str, object]:
    energy_calls = 0
    negative_energy_calls = 0
    skipped_positive_energy_calls = 0
    best_delta_sum = 0.0
    energy_wins = 0
    map_objective_examples = 0
    negative_map_calls = 0
    skipped_positive_map_calls = 0
    map_delta_sum = 0.0
    map_wins = 0
    for row in examples:
        has_map = math.isfinite(row["flow_energy_map_delta_vs_current"])
        if has_map:
            map_objective_examples += 1
        if row["flow_delta_vs_current"] >= threshold:
            energy_calls += 1
            negative_energy_calls += int(row["flow_energy_delta_vs_current"] <= 0.0)
            energy_wins += int(row["flow_energy_delta_vs_current"] > 0.0)
            best_delta_sum += row["best_delta_with_energy"]
            if has_map:
                negative_map_calls += int(row["flow_energy_map_delta_vs_current"] <= 0.0)
                map_wins += int(row["flow_energy_map_delta_vs_current"] > 0.0)
                map_delta_sum += row["best_map_delta_with_energy"]
        else:
            skipped_positive_energy_calls += int(row["flow_energy_delta_vs_current"] > 0.0)
            best_delta_sum += row["best_delta_without_energy"]
            if has_map:
                skipped_positive_map_calls += int(row["flow_energy_map_delta_vs_current"] > 0.0)
                map_delta_sum += row["best_map_delta_without_energy"]
    skipped_energy_calls = len(examples) - energy_calls
    retained = best_delta_sum / full_delta_sum if full_delta_sum > 0.0 else 1.0
    retained_map = map_delta_sum / full_map_delta_sum if full_map_delta_sum > 0.0 else 1.0
    return {
        "min_energy_flow_delta": threshold,
        "energy_calls": energy_calls,
        "skipped_energy_calls": skipped_energy_calls,
        "negative_energy_calls": negative_energy_calls,
        "skipped_positive_energy_calls": skipped_positive_energy_calls,
        "best_delta_sum": best_delta_sum,
        "retained_best_delta_fraction": retained,
        "delta_loss_vs_full_energy": full_delta_sum - best_delta_sum,
        "delta_per_energy_call": best_delta_sum / energy_calls if energy_calls else 0.0,
        "energy_win_rate": energy_wins / energy_calls if energy_calls else 0.0,
        "map_objective_examples": map_objective_examples,
        "negative_map_calls": negative_map_calls,
        "skipped_positive_map_calls": skipped_positive_map_calls,
        "map_delta_sum": map_delta_sum,
        "retained_map_delta_fraction": retained_map,
        "map_delta_loss_vs_full_energy": full_map_delta_sum - map_delta_sum if map_objective_examples else "",
        "map_delta_per_energy_call": map_delta_sum / energy_calls if energy_calls and map_objective_examples else "",
        "map_win_rate": map_wins / energy_calls if energy_calls and map_objective_examples else "",
    }


def _priority(row: dict[str, str], idx: int) -> int:
    try:
        return int(float(row.get("priority", idx)))
    except ValueError:
        return idx


def _required(row: dict[str, str], key: str, idx: int) -> str:
    value = row.get(key, "")
    if not value:
        raise ValueError(f"Selector row {idx} is missing {key}.")
    return value


def _selector_manifest_fieldnames() -> list[str]:
    return [
        "priority",
        "budget_slot",
        "selector_priority",
        "row_index",
        "current_rank",
        "source_archive_path",
        "selected_strength",
        "strength_policy",
        "evaluated_strengths",
        "strength_candidate_count",
        "selector_holdout_group_key",
        "selector_holdout_group_n",
        "selector_holdout_group_win_rate",
        "selector_holdout_group_delta_mean",
        "selector_holdout_group_status",
        "selector_holdout_score_multiplier",
        "surrogate_calibration_status",
        "surrogate_calibration_support_n",
        "surrogate_calibration_distance",
        "surrogate_calibration_score_multiplier",
        "surrogate_calibration_training_ssim_min",
        "surrogate_calibration_training_ssim_max",
        "candidate_kind",
        "flow_strength_multiplier",
        "diversity_stratum",
        "true_probe_feedback_group_key",
        "true_probe_feedback_n",
        "true_probe_feedback_energy_n",
        "true_probe_feedback_best_delta_mean",
        "true_probe_feedback_energy_delta_mean",
        "true_probe_feedback_flow_delta_mean",
        "true_probe_feedback_flow_negative_rate",
        "true_probe_feedback_flow_regression_rate",
        "true_probe_feedback_energy_win_rate",
        "true_probe_feedback_status",
        "true_probe_feedback_score_multiplier",
        *EXACT_PROBE_FEEDBACK_FIELDS,
        *ENERGY_GATE_CONTEXT_FIELDS,
        "residual_control_policy",
        "flow_smooth_sigma",
        "flow_attachment",
        "energy_base_policy",
        "energy_base_flow_delta_threshold",
        "energy_exponent",
        "energy_sigma",
        "energy_ratio_low",
        "energy_ratio_high",
        "energy_clip_low",
        "energy_clip_high",
        "texture_mean_exponent",
        "texture_exponent",
        "texture_deep_exponent",
        "budget_score",
        "planned_api_stage",
        "total_api_budget",
        "flow_budget",
        "reserved_energy_followups",
        "promotion_rule",
        "current_ms_ssim",
        "current_ssim",
        "current_lpips",
        "expected_delta_mean",
        "expected_delta_lcb",
        "expected_delta_uncertainty",
        "acquisition_score",
        "expected_flow_delta_mean",
        "expected_energy_delta_mean",
        "expected_energy_extra_mean",
        "nearest_flow_win_rate",
        "nearest_energy_win_rate",
        "nearest_teacher_count",
        "nearest_teacher_sources",
        "surrogate_gate_status",
        "evidence_scope",
        "reference_png",
        "current_phantom_path",
        "current_synthetic_gray_png",
    ]


def _selector_energy_queue_fieldnames() -> list[str]:
    return [
        *_selector_manifest_fieldnames(),
        "method",
        "phantom_path",
        "flow_phantom_path",
        "flow_synthetic_gray_png",
        "energy_base_stage",
        "energy_base_phantom_path",
        "energy_base_synthetic_gray_png",
        "flow_ms_ssim",
        "flow_request_id",
        "flow_error",
    ]


def _selector_probe_fieldnames() -> list[str]:
    return [
        "priority",
        "budget_slot",
        "selector_priority",
        "row_index",
        "current_rank",
        "source_archive_path",
        "selected_strength",
        "strength_policy",
        "evaluated_strengths",
        "strength_candidate_count",
        "selector_holdout_group_key",
        "selector_holdout_group_n",
        "selector_holdout_group_win_rate",
        "selector_holdout_group_delta_mean",
        "selector_holdout_group_status",
        "selector_holdout_score_multiplier",
        "surrogate_calibration_status",
        "surrogate_calibration_support_n",
        "surrogate_calibration_distance",
        "surrogate_calibration_score_multiplier",
        "surrogate_calibration_training_ssim_min",
        "surrogate_calibration_training_ssim_max",
        "candidate_kind",
        "flow_strength_multiplier",
        "diversity_stratum",
        "true_probe_feedback_group_key",
        "true_probe_feedback_n",
        "true_probe_feedback_energy_n",
        "true_probe_feedback_best_delta_mean",
        "true_probe_feedback_energy_delta_mean",
        "true_probe_feedback_flow_delta_mean",
        "true_probe_feedback_flow_negative_rate",
        "true_probe_feedback_flow_regression_rate",
        "true_probe_feedback_energy_win_rate",
        "true_probe_feedback_status",
        "true_probe_feedback_score_multiplier",
        *EXACT_PROBE_FEEDBACK_FIELDS,
        *ENERGY_GATE_CONTEXT_FIELDS,
        "residual_control_policy",
        "flow_smooth_sigma",
        "flow_attachment",
        "energy_base_policy",
        "energy_base_flow_delta_threshold",
        "energy_base_stage",
        "energy_base_phantom_path",
        "energy_base_synthetic_gray_png",
        "energy_exponent",
        "energy_sigma",
        "energy_ratio_low",
        "energy_ratio_high",
        "energy_clip_low",
        "energy_clip_high",
        "texture_mean_exponent",
        "texture_exponent",
        "texture_deep_exponent",
        "budget_score",
        "planned_api_stage",
        "total_api_budget",
        "flow_budget",
        "reserved_energy_followups",
        "promotion_rule",
        "current_ms_ssim",
        "current_ssim",
        "current_lpips",
        "expected_delta_mean",
        "expected_delta_lcb",
        "expected_delta_uncertainty",
        "acquisition_score",
        "expected_flow_delta_mean",
        "expected_energy_delta_mean",
        "expected_energy_extra_mean",
        "nearest_flow_win_rate",
        "nearest_energy_win_rate",
        "nearest_teacher_count",
        "nearest_teacher_sources",
        "surrogate_gate_status",
        "flow_status",
        "energy_status",
        "best_stage",
        "best_ms_ssim",
        "flow_ms_ssim",
        "flow_energy_ms_ssim",
        "flow_delta_vs_current",
        "flow_energy_delta_vs_current",
        "current_map_objective_score",
        "flow_map_objective_score",
        "flow_map_delta_vs_current",
        "flow_map_objective_status",
        "flow_map_objective_error",
        *ENERGY_MAP_OBJECTIVE_FIELDS,
        "method",
        "reference_png",
        "current_phantom_path",
        "current_synthetic_gray_png",
        "flow_phantom_path",
        "flow_synthetic_gray_png",
        "flow_request_id",
        "flow_error",
        "flow_energy_phantom_path",
        "flow_energy_synthetic_gray_png",
        "flow_energy_request_id",
        "flow_energy_error",
        "flow_energy_lpips_proxy",
        "evidence_source",
        "evidence_scope",
        "promotion_allowed_without_true_scanner",
        "hidden_holdout_final_score",
    ]
