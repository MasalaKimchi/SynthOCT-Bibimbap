from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .evaluation import calculate_metrics, evaluate_feature_map_metrics, profile_scores


STAGE2_CONTROL_FIELDS = [
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

STAGE2_NUMERIC_CONTROL_FIELDS = [
    "flow_smooth_sigma",
    "flow_attachment",
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

SURROGATE_CALIBRATION_FIELDS = [
    "surrogate_calibration_status",
    "surrogate_calibration_support_n",
    "surrogate_calibration_distance",
    "surrogate_calibration_score_multiplier",
    "surrogate_calibration_training_ssim_min",
    "surrogate_calibration_training_ssim_max",
]

TRUE_PROBE_FEEDBACK_FIELDS = [
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

ENERGY_GATE_FIELDS = [
    "energy_gate_summary",
    "energy_gate_min_flow_delta",
    "energy_gate_expected_status",
    "energy_gate_score_multiplier",
    "energy_gate_retained_delta_fraction",
    "energy_gate_retained_map_delta_fraction",
]

MAP_COMPONENTS = [
    ("struct", "Struct_MS-SSIM"),
    ("oac", "OAC_MS-SSIM"),
    ("sc", "SC_MS-SSIM"),
    ("rsc", "RSC_MS-SSIM"),
]

MAP_SAFETY_FIELDS = [
    "expected_struct_delta_mean",
    "expected_struct_delta_lcb",
    "expected_oac_delta_mean",
    "expected_oac_delta_lcb",
    "expected_sc_delta_mean",
    "expected_sc_delta_lcb",
    "expected_rsc_delta_mean",
    "expected_rsc_delta_lcb",
    "expected_map_delta_lcb_min",
    "nearest_map_safe_win_rate",
    "map_safe_support_n",
    "map_safety_status",
    "map_safety_score_multiplier",
]

TOPOLOGY_CONTROL_FEATURE_FIELDS = [
    "intercept",
    "base_ms_ssim",
    "base_lpips",
    "rank_norm",
    "frame_norm",
    "sex_female",
    "sex_male",
    "age_1950_1960",
    "age_1990_2000",
    "site_cheek",
    "site_eye_corner",
    "ms_gap",
    "lpips_load",
]

TOPOLOGY_CONTROL_TARGET_FIELDS = [
    "selected_strength",
    *STAGE2_NUMERIC_CONTROL_FIELDS,
    "expected_delta_mean",
    "expected_flow_delta_mean",
    "expected_energy_delta_mean",
    "expected_energy_extra_mean",
    "expected_ms_ssim_delta_mean",
    "expected_struct_delta_mean",
    "expected_oac_delta_mean",
    "expected_sc_delta_mean",
    "expected_rsc_delta_mean",
]


@dataclass(frozen=True)
class ResidualTeacherExample:
    source_archive_path: str
    row_index: int
    rank: int
    strength: float
    base_ms_ssim: float
    base_lpips: float
    flow_ms_ssim: float
    flow_energy_ms_ssim: float
    best_stage: str
    best_ms_ssim: float
    best_delta: float
    base_objective_score: float
    flow_objective_score: float
    flow_energy_objective_score: float
    best_objective_score: float
    best_objective_delta: float
    objective_source: str
    reference_png: str
    base_phantom_path: str
    base_synthetic_gray_png: str
    flow_phantom_path: str
    flow_energy_phantom_path: str
    teacher_metrics_path: str
    residual_control_policy: str = ""
    flow_smooth_sigma: float = float("nan")
    flow_attachment: float = float("nan")
    energy_exponent: float = float("nan")
    energy_sigma: float = float("nan")
    energy_ratio_low: float = float("nan")
    energy_ratio_high: float = float("nan")
    energy_clip_low: float = float("nan")
    energy_clip_high: float = float("nan")
    texture_mean_exponent: float = float("nan")
    texture_exponent: float = float("nan")
    texture_deep_exponent: float = float("nan")
    flow_struct_delta: float = float("nan")
    flow_oac_delta: float = float("nan")
    flow_sc_delta: float = float("nan")
    flow_rsc_delta: float = float("nan")
    flow_map_objective_delta: float = float("nan")
    flow_energy_struct_delta: float = float("nan")
    flow_energy_oac_delta: float = float("nan")
    flow_energy_sc_delta: float = float("nan")
    flow_energy_rsc_delta: float = float("nan")
    flow_energy_map_objective_delta: float = float("nan")


def train_residual_parameter_selector(
    base_api_metrics: str | Path,
    teacher_metrics_paths: list[str | Path],
    out_path: str | Path,
    *,
    default_strength: float = 0.25,
    holdout_fraction: float = 0.25,
    min_delta: float = 0.0,
    uncertainty_z: float = 1.0,
    neighbor_count: int = 12,
    exploration_weight: float = 0.25,
    surrogate_calibration_metrics: list[str | Path] | None = None,
    min_surrogate_ms_ssim: float = 0.90,
    max_surrogate_lpips_proxy: float = 0.03,
    holdout_group_fields: tuple[str, ...] = ("sex", "age_band", "body_site"),
) -> Path:
    """Distill flow/energy public rescue rows into a conservative selector artifact.

    The artifact is a planning aid, not challenge evidence. It intentionally
    learns compact residual controls around an existing topology-preserving
    generator instead of replacing the scanner-compatible phantom contract.
    """
    base_rows = _read_rows(base_api_metrics)
    examples = load_residual_teacher_examples(teacher_metrics_paths, default_strength=default_strength)
    if not examples:
        raise RuntimeError("No usable residual teacher examples found.")

    train_examples, holdout_examples, holdout_groups = _split_examples(
        examples,
        holdout_fraction=holdout_fraction,
        group_fields=holdout_group_fields,
    )
    if not train_examples:
        train_examples = examples
        holdout_examples = []
        holdout_groups = []

    strength_stats = _strength_stats(train_examples, min_delta=min_delta, uncertainty_z=uncertainty_z)
    selected = _select_strength(strength_stats)
    holdout = _evaluate_strength(holdout_examples, selected["strength"], min_delta=min_delta) if holdout_examples else {}
    holdout_by_group = (
        _evaluate_strength_by_group(
            holdout_examples,
            selected["strength"],
            min_delta=min_delta,
            group_fields=holdout_group_fields,
        )
        if holdout_examples
        else []
    )
    surrogate_gate = _surrogate_gate(
        surrogate_calibration_metrics or [],
        min_ms_ssim=min_surrogate_ms_ssim,
        max_lpips_proxy=max_surrogate_lpips_proxy,
        uncertainty_z=uncertainty_z,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    teacher_csv = out_path.with_name(f"{out_path.stem}_teacher_examples.csv")
    _write_rows(teacher_csv, [asdict(example) for example in examples])
    artifact = {
        "model_type": "conservative_residual_strength_lcb_selector",
        "evidence_source": "public_flow_energy_teacher_rows",
        "evidence_scope": "not_challenge_evidence",
        "promotion_allowed_without_true_scanner": False,
        "surrogate_scanner_is_true_scanner": False,
        "hidden_holdout_final_score": False,
        "official_final_ranking_proven": False,
        "base_api_metrics": str(Path(base_api_metrics)),
        "teacher_metrics_paths": [str(Path(path)) for path in teacher_metrics_paths],
        "teacher_examples_csv": str(teacher_csv),
        "base_row_count": len(base_rows),
        "teacher_example_count": len(examples),
        "train_example_count": len(train_examples),
        "holdout_example_count": len(holdout_examples),
        "holdout_group_fields": list(holdout_group_fields),
        "holdout_group_count": len(holdout_groups),
        "holdout_groups": holdout_groups,
        "min_delta": min_delta,
        "uncertainty_z": uncertainty_z,
        "neighbor_count": max(1, int(neighbor_count)),
        "exploration_weight": max(0.0, float(exploration_weight)),
        "selected_strength": selected["strength"],
        "selected_strength_stats": selected,
        "strength_stats": strength_stats,
        "holdout": holdout,
        "holdout_by_group": holdout_by_group,
        "surrogate_gate": surrogate_gate,
        "objective": {
            "primary": "Struct/OAC/SC/RSC MS-SSIM plus inverted LPIPS when present",
            "fallback": "plain MS-SSIM delta",
            "stage": "best of base, flow, and flow_energy",
        },
        "interpretation": (
            "Selector distilled from public flow+energy rescue rows. It ranks row-wise "
            "hosted-API probes from similar teacher examples; promotion still requires "
            "grouped true-scanner evidence."
        ),
    }
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def train_topology_residual_control_model(
    base_api_metrics: str | Path,
    teacher_metrics_paths: list[str | Path],
    out_path: str | Path,
    *,
    default_strength: float = 0.25,
    holdout_fraction: float = 0.25,
    ridge_lambda: float = 1e-3,
    uncertainty_z: float = 1.0,
    holdout_group_fields: tuple[str, ...] = ("sex", "age_band", "body_site"),
) -> Path:
    """Train a bounded residual-control model around the existing p140-t32 topology.

    This is deliberately a control model, not a phantom replacement model. It
    learns row-wise flow/geometry, energy, and texture knobs from true-scanner
    teacher rows and leaves scatterer topology generation anchored to the
    current base phantom.
    """
    base_rows = _read_rows(base_api_metrics)
    examples = load_residual_teacher_examples(teacher_metrics_paths, default_strength=default_strength)
    if not examples:
        raise RuntimeError("No usable residual teacher examples found.")

    train_examples, holdout_examples, holdout_groups = _split_examples(
        examples,
        holdout_fraction=holdout_fraction,
        group_fields=holdout_group_fields,
    )
    if not train_examples:
        train_examples = examples
        holdout_examples = []
        holdout_groups = []

    defaults = _control_model_feature_defaults(train_examples)
    coefficients, residual_std, train_metrics = _fit_control_model(
        train_examples,
        defaults=defaults,
        ridge_lambda=ridge_lambda,
    )
    holdout_metrics = (
        _evaluate_control_model(holdout_examples, coefficients, defaults=defaults, residual_std=residual_std)
        if holdout_examples
        else {"n": 0}
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_type": "topology_preserving_residual_control_model_v1",
        "evidence_source": "true_scanner_residual_teacher_rows",
        "evidence_scope": "residual_control_model_not_challenge_evidence",
        "promotion_allowed_without_true_scanner": False,
        "surrogate_scanner_is_true_scanner": False,
        "hidden_holdout_final_score": False,
        "official_final_ranking_proven": False,
        "base_api_metrics": str(Path(base_api_metrics)),
        "base_row_count": len(base_rows),
        "teacher_metrics_paths": [str(Path(path)) for path in teacher_metrics_paths],
        "teacher_example_count": len(examples),
        "train_example_count": len(train_examples),
        "holdout_example_count": len(holdout_examples),
        "holdout_group_fields": list(holdout_group_fields),
        "holdout_group_count": len(holdout_groups),
        "holdout_groups": holdout_groups,
        "feature_fields": TOPOLOGY_CONTROL_FEATURE_FIELDS,
        "target_fields": TOPOLOGY_CONTROL_TARGET_FIELDS,
        "feature_defaults": defaults,
        "coefficients": coefficients,
        "residual_std": residual_std,
        "ridge_lambda": ridge_lambda,
        "uncertainty_z": uncertainty_z,
        "train_metrics": train_metrics,
        "holdout_metrics": holdout_metrics,
        "map_objective_target_observed": _control_model_has_aggregate_map_objective(train_examples),
        "map_target_observed": _control_model_has_map_targets(train_examples),
        "control_bounds": {
            "selected_strength": [0.05, 0.55],
            **{field: list(bounds) for field, bounds in _stage2_control_bounds().items()},
        },
        "topology_contract": {
            "base_topology_preserved": True,
            "predicts_full_density_replacement": False,
            "predicted_controls": [
                "flow_strength",
                "flow_smooth_sigma",
                "flow_attachment",
                "energy_exponent",
                "energy_sigma",
                "energy_ratio_low",
                "energy_ratio_high",
                "energy_clip_low",
                "energy_clip_high",
                "texture_mean_exponent",
                "texture_exponent",
                "texture_deep_exponent",
            ],
        },
        "interpretation": (
            "Linear ridge model for bounded residual controls around p140-t32-like "
            "phantoms. It can plan hosted-scanner probes, but promotion still "
            "requires grouped true-scanner validation."
        ),
    }
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def plan_topology_residual_control_queue(
    base_api_metrics: str | Path,
    control_model: str | Path,
    out_path: str | Path,
    *,
    limit: int | None = None,
    min_expected_delta_lcb: float | None = None,
    min_map_delta_lcb: float = -0.0005,
    require_map_safe: bool = False,
    uncertainty_z: float | None = None,
) -> Path:
    """Predict a 120-row residual-control queue from a trained control model."""
    base_rows = _read_rows(base_api_metrics)
    artifact = json.loads(Path(control_model).read_text(encoding="utf-8"))
    if artifact.get("model_type") != "topology_preserving_residual_control_model_v1":
        raise ValueError(f"Unsupported residual control model: {artifact.get('model_type')}")

    z = float(artifact.get("uncertainty_z", 1.0) if uncertainty_z is None else uncertainty_z)
    defaults = artifact.get("feature_defaults", {})
    coefficients = artifact.get("coefficients", {})
    residual_std = artifact.get("residual_std", {})
    ranked_base = sorted(enumerate(base_rows), key=lambda item: _finite(item[1].get("MS-SSIM")))
    predictions: list[dict[str, object]] = []

    for current_rank, (idx, row) in enumerate(ranked_base, start=1):
        target_values = _predict_control_model_targets(
            row,
            rank=current_rank,
            coefficients=coefficients,
            defaults=defaults,
        )
        expected_delta = _finite(target_values.get("expected_delta_mean"))
        delta_std = max(0.0, _finite(residual_std.get("expected_delta_mean")))
        lcb = expected_delta - z * delta_std if math.isfinite(expected_delta) else float("nan")
        uncertainty = delta_std if math.isfinite(delta_std) else float("nan")
        map_safety = _control_model_map_safety(
            target_values,
            residual_std=residual_std,
            uncertainty_z=z,
            min_map_delta_lcb=min_map_delta_lcb,
            map_target_observed=bool(artifact.get("map_target_observed", False)),
        )
        acquisition = _apply_score_multiplier(
            lcb + 0.20 * uncertainty if math.isfinite(lcb) and math.isfinite(uncertainty) else lcb,
            float(map_safety["map_safety_score_multiplier"]),
        )
        controls = _control_model_stage2_controls(target_values)
        predictions.append(
            {
                "priority": 0,
                "row_index": idx,
                "current_rank": current_rank,
                "source_archive_path": row.get("source_archive_path", ""),
                "current_ms_ssim": _finite(row.get("MS-SSIM")),
                "current_ssim": _finite(row.get("SSIM") or row.get("Struct_SSIM")),
                "current_lpips": _finite(row.get("LPIPS") or row.get("LPIPS_PROXY")),
                "selected_strength": target_values["selected_strength"],
                "strength_policy": "topology_residual_control_model",
                "evaluated_strengths": "model_predicted",
                "strength_candidate_count": 1,
                "selector_holdout_group_key": _diversity_stratum(row, tuple(artifact.get("holdout_group_fields", ()))),
                "selector_holdout_group_n": artifact.get("holdout_example_count", ""),
                "selector_holdout_group_win_rate": artifact.get("holdout_metrics", {}).get("expected_delta_mean_positive_rate", ""),
                "selector_holdout_group_delta_mean": artifact.get("holdout_metrics", {}).get("expected_delta_mean_error_mean", ""),
                "selector_holdout_group_status": "control_model_holdout_summary",
                "selector_holdout_score_multiplier": 1.0,
                **_empty_surrogate_fields("not_configured"),
                **controls,
                "expected_delta_mean": expected_delta,
                "expected_delta_lcb": lcb,
                "expected_delta_uncertainty": uncertainty,
                "acquisition_score": acquisition,
                **map_safety,
                "expected_flow_delta_mean": target_values.get("expected_flow_delta_mean", ""),
                "expected_energy_delta_mean": target_values.get("expected_energy_delta_mean", ""),
                "expected_energy_extra_mean": target_values.get("expected_energy_extra_mean", ""),
                "nearest_flow_win_rate": "",
                "nearest_energy_win_rate": "",
                "nearest_teacher_count": artifact.get("train_example_count", ""),
                "nearest_teacher_sources": "topology_residual_control_model_v1",
                "selector_train_examples": artifact.get("train_example_count", ""),
                "reference_png": row.get("reference_png", ""),
                "current_phantom_path": row.get("phantom_path", ""),
                "current_synthetic_gray_png": row.get("synthetic_gray_png", ""),
                "surrogate_gate_status": "not_configured",
                "evidence_scope": "topology_residual_control_planning_not_challenge_evidence",
            }
        )

    if min_expected_delta_lcb is not None:
        predictions = [
            row for row in predictions if _finite(row.get("expected_delta_lcb")) >= float(min_expected_delta_lcb)
        ]
    if require_map_safe:
        predictions = [row for row in predictions if row.get("map_safety_status") == "map_safe_pass"]
    predictions = sorted(
        predictions,
        key=lambda row: (_finite(row.get("acquisition_score")), -_finite(row.get("current_ms_ssim"))),
        reverse=True,
    )
    if limit is not None:
        predictions = predictions[: max(0, int(limit))]
    for priority, row in enumerate(predictions, start=1):
        row["priority"] = priority

    out_path = Path(out_path)
    _write_rows(out_path, predictions, fieldnames=_recommendation_fieldnames())
    summary = {
        "control_model": str(Path(control_model)),
        "base_api_metrics": str(Path(base_api_metrics)),
        "queue": str(out_path),
        "base_row_count": len(base_rows),
        "planned_row_count": len(predictions),
        "require_map_safe": require_map_safe,
        "min_expected_delta_lcb": min_expected_delta_lcb if min_expected_delta_lcb is not None else "",
        "min_map_delta_lcb": min_map_delta_lcb,
        "evidence_scope": "topology_residual_control_planning_not_challenge_evidence",
        "promotion_allowed_without_true_scanner": False,
        "hidden_holdout_final_score": False,
    }
    out_path.with_name(f"{out_path.stem}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


def plan_residual_selector_queue(
    base_api_metrics: str | Path,
    selector_artifact: str | Path,
    out_path: str | Path,
    *,
    limit: int | None = None,
    min_expected_delta_lcb: float | None = 0.0,
    row_wise_strengths: bool = False,
    min_map_delta_lcb: float = -0.0005,
    min_map_safe_win_rate: float = 0.5,
    require_map_safe: bool = False,
) -> Path:
    """Write row/strength probe recommendations for the next adaptive API batch."""
    base_rows = _read_rows(base_api_metrics)
    artifact = json.loads(Path(selector_artifact).read_text(encoding="utf-8"))
    surrogate_gate = artifact.get("surrogate_gate", {"status": "not_configured"})
    if surrogate_gate.get("status") == "fail":
        recommendations: list[dict[str, object]] = []
    else:
        examples = _load_saved_teacher_examples(artifact["teacher_examples_csv"])
        predictions = _predict_base_rows(
            base_rows,
            examples,
            selected_strength=float(artifact["selected_strength"]),
            neighbor_count=int(artifact.get("neighbor_count", 12)),
            uncertainty_z=float(artifact.get("uncertainty_z", 1.0)),
            exploration_weight=float(artifact.get("exploration_weight", 0.25)),
            row_wise_strengths=row_wise_strengths,
            min_map_delta_lcb=min_map_delta_lcb,
            min_map_safe_win_rate=min_map_safe_win_rate,
        )
        if min_expected_delta_lcb is not None:
            predictions = [
                row for row in predictions if float(row["expected_delta_lcb"]) >= float(min_expected_delta_lcb)
            ]
        if require_map_safe:
            predictions = [row for row in predictions if row.get("map_safety_status") == "map_safe_pass"]
        ranked = sorted(
            predictions,
            key=lambda item: (float(item["acquisition_score"]), -float(item["current_ms_ssim"])),
            reverse=True,
        )
        if limit is not None:
            ranked = ranked[:limit]
        holdout_lookup = _holdout_group_lookup(artifact)
        holdout_group_fields = tuple(artifact.get("holdout_group_fields", ()))
        recommendations = []
        for priority, row in enumerate(ranked, start=1):
            controls = _row_residual_controls(row)
            holdout_context = _row_holdout_context(row, holdout_group_fields, holdout_lookup)
            surrogate_context = _surrogate_calibration_context(row, surrogate_gate)
            recommendations.append(
                {
                    "priority": priority,
                    "row_index": row["row_index"],
                    "current_rank": row["current_rank"],
                    "source_archive_path": row.get("source_archive_path", ""),
                    "current_ms_ssim": row.get("current_ms_ssim", ""),
                    "current_ssim": row.get("current_ssim", ""),
                    "current_lpips": row.get("current_lpips", ""),
                    "selected_strength": row.get("selected_strength", artifact["selected_strength"]),
                    "strength_policy": row.get("strength_policy", "global_strength"),
                    "evaluated_strengths": row.get("evaluated_strengths", ""),
                    "strength_candidate_count": row.get("strength_candidate_count", ""),
                    **holdout_context,
                    **surrogate_context,
                    **controls,
                    "expected_delta_mean": row["expected_delta_mean"],
                    "expected_delta_lcb": row["expected_delta_lcb"],
                    "expected_delta_uncertainty": row["expected_delta_uncertainty"],
                    "acquisition_score": row["acquisition_score"],
                    **_map_safety_context(row),
                    "expected_flow_delta_mean": row.get("expected_flow_delta_mean", ""),
                    "expected_energy_delta_mean": row.get("expected_energy_delta_mean", ""),
                    "expected_energy_extra_mean": row.get("expected_energy_extra_mean", ""),
                    "nearest_flow_win_rate": row.get("nearest_flow_win_rate", ""),
                    "nearest_energy_win_rate": row.get("nearest_energy_win_rate", ""),
                    "nearest_teacher_count": row["nearest_teacher_count"],
                    "nearest_teacher_sources": row["nearest_teacher_sources"],
                    "selector_train_examples": artifact["train_example_count"],
                    "reference_png": row.get("reference_png", ""),
                    "current_phantom_path": row.get("phantom_path", ""),
                    "current_synthetic_gray_png": row.get("synthetic_gray_png", ""),
                    "surrogate_gate_status": surrogate_gate.get("status", "not_configured"),
                    "evidence_scope": "selector_planning_not_challenge_evidence",
                }
            )

    out_path = Path(out_path)
    _write_rows(out_path, recommendations, fieldnames=_recommendation_fieldnames())
    return out_path


def plan_residual_api_budget(
    selector_queue: str | Path,
    out_path: str | Path,
    *,
    total_api_calls: int = 120,
    energy_followup_fraction: float = 1.0 / 3.0,
    strength_multipliers: tuple[float, ...] = (1.0, 0.72, 1.38),
    energy_exponents: tuple[float, ...] = (0.80, 0.72, 0.92),
    min_strength: float = 0.05,
    max_strength: float = 0.55,
    diversity_fields: tuple[str, ...] = (),
    max_per_stratum: int | None = None,
    probe_feedback_metrics: tuple[str | Path, ...] = (),
    feedback_group_fields: tuple[str, ...] = ("sex", "age_band", "body_site"),
    energy_gate_summary: str | Path | None = None,
    exclude_exact_probed: bool = False,
) -> Path:
    """Expand selector recommendations into a bounded two-stage API call plan."""
    rows = _read_rows(selector_queue)
    total = max(0, int(total_api_calls))
    reserved_energy = min(total, max(0, int(round(total * max(0.0, energy_followup_fraction)))))
    flow_budget = max(0, total - reserved_energy)
    feedback_lookup = _probe_feedback_lookup(probe_feedback_metrics, feedback_group_fields)
    exact_feedback_lookup = _probe_feedback_lookup(probe_feedback_metrics, ("source_archive_path",))
    energy_gate = _energy_gate_config(energy_gate_summary)
    candidates: list[dict[str, object]] = []

    for selector_priority, row in enumerate(rows, start=1):
        base_strength = _finite(row.get("selected_strength"))
        if not math.isfinite(base_strength):
            continue
        seen_strengths: set[float] = set()
        for variant_index, multiplier in enumerate(strength_multipliers):
            strength = min(max(base_strength * float(multiplier), float(min_strength)), float(max_strength))
            strength = round(strength, 6)
            if strength in seen_strengths:
                continue
            seen_strengths.add(strength)
            kind = _candidate_kind(multiplier, variant_index)
            controls = _budget_variant_controls(row, variant_index, energy_exponents=energy_exponents)
            diversity_stratum = _diversity_stratum(row, diversity_fields) if diversity_fields else ""
            holdout_multiplier = _row_float(row, "selector_holdout_score_multiplier", 1.0)
            surrogate_multiplier = _row_float(row, "surrogate_calibration_score_multiplier", 1.0)
            feedback_context = _row_probe_feedback_context(row, feedback_group_fields, feedback_lookup)
            feedback_multiplier = _row_float(feedback_context, "true_probe_feedback_score_multiplier", 1.0)
            exact_feedback_context = _row_exact_probe_feedback_context(row, exact_feedback_lookup)
            exact_feedback_multiplier = _row_float(
                exact_feedback_context,
                "exact_probe_feedback_score_multiplier",
                1.0,
            )
            energy_gate_context = _row_energy_gate_context(row, energy_gate)
            energy_gate_multiplier = _row_float(energy_gate_context, "energy_gate_score_multiplier", 1.0)
            map_safety_multiplier = _row_float(row, "map_safety_score_multiplier", 1.0)
            uncertainty = _finite(row.get("expected_delta_uncertainty"))
            acquisition = _finite(row.get("acquisition_score"))
            lcb = _finite(row.get("expected_delta_lcb"))
            base_score = acquisition if math.isfinite(acquisition) else lcb if math.isfinite(lcb) else 0.0
            explore_bonus = max(0.0, uncertainty if math.isfinite(uncertainty) else 0.0) * (0.0 if kind == "exploit" else 0.25)
            budget_score = (
                base_score
                * holdout_multiplier
                * surrogate_multiplier
                * feedback_multiplier
                * exact_feedback_multiplier
                * energy_gate_multiplier
                * map_safety_multiplier
                + explore_bonus
                - 0.0005 * variant_index
            )
            candidates.append(
                {
                    **row,
                    "selector_priority": row.get("priority", selector_priority),
                    "selected_strength": strength,
                    "candidate_kind": kind,
                    "flow_strength_multiplier": multiplier,
                    "diversity_stratum": diversity_stratum,
                    **feedback_context,
                    **exact_feedback_context,
                    **energy_gate_context,
                    **controls,
                    "budget_score": budget_score,
                    "planned_api_stage": "flow_then_energy_if_promoted",
                    "total_api_budget": total,
                    "flow_budget": flow_budget,
                    "reserved_energy_followups": reserved_energy,
                    "promotion_rule": _energy_promotion_rule(energy_gate),
                    "evidence_scope": "selector_budget_planning_not_challenge_evidence",
                    "_variant_index": variant_index,
                    "_selector_order": selector_priority,
                }
            )

    if exclude_exact_probed:
        candidates = [
            row
            for row in candidates
            if str(row.get("exact_probe_feedback_status", "")) in {"", "not_probed", "missing_source"}
        ]

    planned = _select_budget_candidates(
        candidates,
        flow_budget,
        diversity_fields=diversity_fields,
        max_per_stratum=max_per_stratum,
    )
    for priority, row in enumerate(planned, start=1):
        row["priority"] = priority
        row["budget_slot"] = priority
        row.pop("_variant_index", None)
        row.pop("_selector_order", None)

    out_path = Path(out_path)
    fieldnames = _budget_queue_fieldnames()
    planned_rows = [{field: row.get(field, "") for field in fieldnames} for row in planned]
    _write_rows(out_path, planned_rows, fieldnames=fieldnames)
    summary = {
        "selector_queue": str(Path(selector_queue)),
        "budget_queue": str(out_path),
        "total_api_budget": total,
        "planned_flow_calls": len(planned),
        "reserved_energy_followups": reserved_energy,
        "max_possible_api_calls": len(planned) + reserved_energy,
        "source_recommendations": len(rows),
        "candidate_pool_size": len(candidates),
        "exclude_exact_probed": bool(exclude_exact_probed),
        "candidate_energy_gate_expected_status": _count_values(candidates, "energy_gate_expected_status"),
        "planned_candidate_kinds": _count_values(planned, "candidate_kind"),
        "planned_surrogate_calibration_status": _count_values(planned, "surrogate_calibration_status"),
        "planned_true_probe_feedback_status": _count_values(planned, "true_probe_feedback_status"),
        "planned_exact_probe_feedback_status": _count_values(planned, "exact_probe_feedback_status"),
        "planned_energy_gate_expected_status": _count_values(planned, "energy_gate_expected_status"),
        "planned_energy_gate_eligible_followups": _energy_gate_eligible_count(planned, reserved_energy),
        "diversity_fields": list(diversity_fields),
        "max_per_stratum": max_per_stratum if max_per_stratum is not None else "",
        "planned_diversity_strata": _count_values(planned, "diversity_stratum") if diversity_fields else {},
        "probe_feedback_metrics": [str(Path(path)) for path in probe_feedback_metrics],
        "feedback_group_fields": list(feedback_group_fields),
        "energy_gate_summary": str(Path(energy_gate_summary)) if energy_gate_summary else "",
        "energy_gate_min_flow_delta": energy_gate.get("min_flow_delta", ""),
        "energy_gate_retained_delta_fraction": energy_gate.get("retained_delta_fraction", ""),
        "energy_gate_retained_map_delta_fraction": energy_gate.get("retained_map_delta_fraction", ""),
        "strength_multipliers": list(strength_multipliers),
        "energy_exponents": list(energy_exponents),
        "evidence_scope": "selector_budget_planning_not_challenge_evidence",
        "promotion_allowed_without_true_scanner": False,
        "hidden_holdout_final_score": False,
    }
    summary_path = out_path.with_name(f"{out_path.stem}_budget_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def enrich_residual_teacher_maps(
    base_api_metrics: str | Path,
    teacher_metrics: str | Path,
    out_path: str | Path,
    *,
    maps_dir: str | Path | None = None,
    include_lpips: bool = False,
    max_rows: int | None = None,
) -> Path:
    """Add challenge-map objective metrics to residual teacher rows when PNGs exist."""
    base_rows = _read_rows(base_api_metrics)
    base_by_source = {row.get("source_archive_path", ""): row for row in base_rows}
    teacher_rows = _read_rows(teacher_metrics)
    sibling_flow_by_key = _sibling_flow_lookup(Path(teacher_metrics))
    if max_rows is not None:
        teacher_rows = teacher_rows[: max(0, int(max_rows))]

    out_path = Path(out_path)
    maps_root = Path(maps_dir) if maps_dir is not None else out_path.with_name(f"{out_path.stem}_maps")
    enriched: list[dict[str, object]] = []
    for row_index, row in enumerate(teacher_rows, start=1):
        source = row.get("source_archive_path", "")
        base_row = base_by_source.get(source, {})
        sibling_flow = sibling_flow_by_key.get(_stage_join_key(row), {})
        reference_png = _first_existing_path(row.get("reference_png"), sibling_flow.get("reference_png"), base_row.get("reference_png"))
        current_png = _first_existing_path(
            row.get("current_synthetic_gray_png"),
            row.get("base_synthetic_gray_png"),
            row.get("rank120_synthetic_gray_png"),
            base_row.get("synthetic_gray_png"),
        )
        flow_png = _first_existing_path(row.get("flow_synthetic_gray_png"), sibling_flow.get("flow_synthetic_gray_png"))
        energy_png = _first_existing_path(row.get("flow_energy_synthetic_gray_png"), row.get("energy_synthetic_gray_png"))

        enriched_row: dict[str, object] = dict(row)
        if reference_png:
            enriched_row["reference_png"] = str(reference_png)
        if current_png:
            enriched_row["current_synthetic_gray_png"] = enriched_row.get("current_synthetic_gray_png") or str(current_png)
        if flow_png:
            enriched_row["flow_synthetic_gray_png"] = enriched_row.get("flow_synthetic_gray_png") or str(flow_png)
        if energy_png:
            enriched_row["flow_energy_synthetic_gray_png"] = enriched_row.get("flow_energy_synthetic_gray_png") or str(energy_png)

        stages = (
            ("current", current_png),
            ("flow", flow_png),
            ("flow_energy", energy_png),
        )
        for prefix, pred_png in stages:
            enriched_row.update(
                _prefixed_stage_map_metrics(
                    prefix,
                    reference_png,
                    pred_png,
                    maps_root / f"row_{row_index:04d}" / prefix,
                    include_lpips=include_lpips,
                )
            )
        enriched_row["objective_source"] = (
            "multi_objective_maps_proxy_lpips" if not include_lpips else "multi_objective_maps_real_lpips"
        )
        enriched_row["promotion_allowed_without_true_scanner"] = False
        enriched_row["hidden_holdout_final_score"] = False
        enriched_row["evidence_scope"] = "residual_teacher_map_enrichment_not_challenge_evidence"
        enriched.append(enriched_row)

    fieldnames = _ordered_union(enriched)
    _write_rows(out_path, enriched, fieldnames=fieldnames)
    summary = {
        "base_api_metrics": str(Path(base_api_metrics)),
        "teacher_metrics": str(Path(teacher_metrics)),
        "enriched_metrics": str(out_path),
        "row_count": len(enriched),
        "include_lpips": include_lpips,
        "maps_dir": str(maps_root),
        "rows_with_current_struct": sum(1 for row in enriched if math.isfinite(_finite(row.get("current_Struct_MS-SSIM")))),
        "rows_with_flow_struct": sum(1 for row in enriched if math.isfinite(_finite(row.get("flow_Struct_MS-SSIM")))),
        "rows_with_flow_energy_struct": sum(
            1 for row in enriched if math.isfinite(_finite(row.get("flow_energy_Struct_MS-SSIM")))
        ),
        "objective_source": "multi_objective_maps_proxy_lpips" if not include_lpips else "multi_objective_maps_real_lpips",
        "evidence_scope": "residual_teacher_map_enrichment_not_challenge_evidence",
        "promotion_allowed_without_true_scanner": False,
        "hidden_holdout_final_score": False,
    }
    out_path.with_name(f"{out_path.stem}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return out_path


def load_residual_teacher_examples(
    teacher_metrics_paths: list[str | Path],
    *,
    default_strength: float = 0.25,
) -> list[ResidualTeacherExample]:
    examples: list[ResidualTeacherExample] = []
    for metrics_path in teacher_metrics_paths:
        path = Path(metrics_path)
        for row in _read_rows(path):
            example = _teacher_example_from_row(row, path, default_strength=default_strength)
            if example is not None:
                examples.append(example)
    return examples


def _teacher_example_from_row(
    row: dict[str, str],
    metrics_path: Path,
    *,
    default_strength: float,
) -> ResidualTeacherExample | None:
    base_ms = _first_finite(row, "current_ms_ssim", "rank120_ms_ssim", "base_ms_ssim")
    flow_ms = _first_finite(row, "flow_ms_ssim")
    energy_ms = _first_finite(row, "flow_energy_ms_ssim")
    if not all(math.isfinite(value) for value in (base_ms, flow_ms, energy_ms)):
        return None
    strength = _first_finite(row, "strength", "selected_strength")
    if not math.isfinite(strength):
        strength = float(default_strength)

    best_stage = row.get("best_stage", "")
    best_ms = _first_finite(row, "best_ms_ssim")
    if not math.isfinite(best_ms):
        best_ms = max(base_ms, flow_ms, energy_ms)
    if not best_stage:
        best_stage = "flow_energy" if energy_ms >= max(base_ms, flow_ms) else "flow" if flow_ms >= base_ms else "base"
    base_objective = _stage_objective(row, ("current", "base", "rank120"), base_ms)
    flow_objective = _stage_objective(row, ("flow",), flow_ms)
    flow_energy_objective = _stage_objective(row, ("flow_energy", "energy"), energy_ms)
    best_objective = max(base_objective, flow_objective, flow_energy_objective)
    flow_map_deltas = _stage_map_deltas(row, ("flow",), ("current", "base", "rank120"))
    flow_energy_map_deltas = _stage_map_deltas(row, ("flow_energy", "energy"), ("current", "base", "rank120"))

    return ResidualTeacherExample(
        source_archive_path=row.get("source_archive_path", ""),
        row_index=_first_int(row, "row_index", "priority"),
        rank=_first_int(row, "current_rank", "rank", "priority"),
        strength=float(strength),
        base_ms_ssim=float(base_ms),
        base_lpips=_first_finite(row, "current_lpips", "base_lpips"),
        flow_ms_ssim=float(flow_ms),
        flow_energy_ms_ssim=float(energy_ms),
        best_stage=best_stage,
        best_ms_ssim=float(best_ms),
        best_delta=float(best_ms - base_ms),
        base_objective_score=base_objective,
        flow_objective_score=flow_objective,
        flow_energy_objective_score=flow_energy_objective,
        best_objective_score=best_objective,
        best_objective_delta=best_objective - base_objective,
        objective_source=_objective_source(row),
        reference_png=row.get("reference_png", ""),
        base_phantom_path=row.get("current_phantom_path", row.get("base_phantom_path", "")),
        base_synthetic_gray_png=row.get("current_synthetic_gray_png", row.get("base_synthetic_gray_png", "")),
        flow_phantom_path=row.get("flow_phantom_path", row.get("phantom_path", "")),
        flow_energy_phantom_path=row.get("flow_energy_phantom_path", row.get("energy_phantom_path", "")),
        teacher_metrics_path=str(metrics_path),
        residual_control_policy=row.get("residual_control_policy", ""),
        flow_smooth_sigma=_first_finite(row, "flow_smooth_sigma"),
        flow_attachment=_first_finite(row, "flow_attachment"),
        energy_exponent=_first_finite(row, "energy_exponent"),
        energy_sigma=_first_finite(row, "energy_sigma"),
        energy_ratio_low=_first_finite(row, "energy_ratio_low"),
        energy_ratio_high=_first_finite(row, "energy_ratio_high"),
        energy_clip_low=_first_finite(row, "energy_clip_low"),
        energy_clip_high=_first_finite(row, "energy_clip_high"),
        texture_mean_exponent=_first_finite(row, "texture_mean_exponent"),
        texture_exponent=_first_finite(row, "texture_exponent"),
        texture_deep_exponent=_first_finite(row, "texture_deep_exponent"),
        flow_struct_delta=flow_map_deltas["struct"],
        flow_oac_delta=flow_map_deltas["oac"],
        flow_sc_delta=flow_map_deltas["sc"],
        flow_rsc_delta=flow_map_deltas["rsc"],
        flow_map_objective_delta=flow_objective - base_objective,
        flow_energy_struct_delta=flow_energy_map_deltas["struct"],
        flow_energy_oac_delta=flow_energy_map_deltas["oac"],
        flow_energy_sc_delta=flow_energy_map_deltas["sc"],
        flow_energy_rsc_delta=flow_energy_map_deltas["rsc"],
        flow_energy_map_objective_delta=flow_energy_objective - base_objective,
    )


def _strength_stats(
    examples: list[ResidualTeacherExample],
    *,
    min_delta: float,
    uncertainty_z: float,
) -> list[dict[str, float | int]]:
    grouped: dict[float, list[ResidualTeacherExample]] = {}
    for example in examples:
        grouped.setdefault(round(example.strength, 6), []).append(example)

    rows: list[dict[str, float | int]] = []
    for strength, group in sorted(grouped.items()):
        deltas = [example.best_objective_delta for example in group]
        wins = [delta > min_delta for delta in deltas]
        std = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
        se = std / math.sqrt(len(deltas)) if deltas else 0.0
        mean = statistics.mean(deltas) if deltas else 0.0
        rows.append(
            {
                "strength": strength,
                "n": len(group),
                "win_rate": sum(wins) / max(1, len(wins)),
                "delta_mean": mean,
                "delta_median": statistics.median(deltas) if deltas else 0.0,
                "delta_min": min(deltas) if deltas else 0.0,
                "delta_max": max(deltas) if deltas else 0.0,
                "delta_std": std,
                "delta_se": se,
                "delta_lcb": mean - float(uncertainty_z) * se,
            }
        )
    return rows


def _select_strength(strength_stats: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not strength_stats:
        raise RuntimeError("Cannot select a residual strength without teacher statistics.")
    return max(
        strength_stats,
        key=lambda row: (float(row["delta_lcb"]), float(row["win_rate"]), int(row["n"])),
    )


def _evaluate_strength(
    examples: list[ResidualTeacherExample],
    strength: float,
    *,
    min_delta: float,
) -> dict[str, float | int]:
    selected = [example for example in examples if abs(example.strength - strength) <= 1e-6]
    if not selected:
        return {"n": 0}
    deltas = [example.best_objective_delta for example in selected]
    return {
        "n": len(selected),
        "delta_mean": statistics.mean(deltas),
        "delta_median": statistics.median(deltas),
        "delta_min": min(deltas),
        "delta_max": max(deltas),
        "win_rate": sum(delta > min_delta for delta in deltas) / len(deltas),
    }


def _evaluate_strength_by_group(
    examples: list[ResidualTeacherExample],
    strength: float,
    *,
    min_delta: float,
    group_fields: tuple[str, ...],
) -> list[dict[str, float | int | str]]:
    grouped: dict[str, list[ResidualTeacherExample]] = {}
    for example in examples:
        grouped.setdefault(_example_group_key(example, group_fields), []).append(example)
    rows: list[dict[str, float | int | str]] = []
    for group, group_examples in sorted(grouped.items()):
        stats = _evaluate_strength(group_examples, strength, min_delta=min_delta)
        rows.append(
            {
                "group": group,
                "strength": strength,
                **stats,
            }
        )
    return rows


def _holdout_group_lookup(artifact: dict[str, object]) -> dict[str, dict[str, object]]:
    rows = artifact.get("holdout_by_group", [])
    if not isinstance(rows, list):
        return {}
    lookup: dict[str, dict[str, object]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("group"):
            lookup[str(row["group"])] = row
    return lookup


def _row_holdout_context(
    row: dict[str, object],
    group_fields: tuple[str, ...],
    holdout_lookup: dict[str, dict[str, object]],
) -> dict[str, object]:
    group = _diversity_stratum(row, group_fields) if group_fields else ""
    stats = holdout_lookup.get(group)
    if not group_fields:
        return {
            "selector_holdout_group_key": "",
            "selector_holdout_group_n": "",
            "selector_holdout_group_win_rate": "",
            "selector_holdout_group_delta_mean": "",
            "selector_holdout_group_status": "not_configured",
            "selector_holdout_score_multiplier": 1.0,
        }
    if not stats:
        return {
            "selector_holdout_group_key": group,
            "selector_holdout_group_n": "",
            "selector_holdout_group_win_rate": "",
            "selector_holdout_group_delta_mean": "",
            "selector_holdout_group_status": "not_held_out",
            "selector_holdout_score_multiplier": 1.0,
        }

    n = _finite(stats.get("n"))  # type: ignore[arg-type]
    win_rate = _finite(stats.get("win_rate"))  # type: ignore[arg-type]
    delta_mean = _finite(stats.get("delta_mean"))  # type: ignore[arg-type]
    status, multiplier = _holdout_group_status(n=n, win_rate=win_rate, delta_mean=delta_mean)
    return {
        "selector_holdout_group_key": group,
        "selector_holdout_group_n": int(n) if math.isfinite(n) else "",
        "selector_holdout_group_win_rate": win_rate if math.isfinite(win_rate) else "",
        "selector_holdout_group_delta_mean": delta_mean if math.isfinite(delta_mean) else "",
        "selector_holdout_group_status": status,
        "selector_holdout_score_multiplier": multiplier,
    }


def _holdout_group_status(*, n: float, win_rate: float, delta_mean: float) -> tuple[str, float]:
    if not math.isfinite(n) or n <= 0:
        return "held_out_without_selected_strength", 0.75
    if math.isfinite(win_rate) and win_rate <= 0.0:
        return "holdout_group_no_wins", 0.50
    if math.isfinite(win_rate) and win_rate < 0.25:
        return "holdout_group_weak", 0.70
    if math.isfinite(delta_mean) and delta_mean <= 0.0:
        return "holdout_group_flat", 0.80
    return "holdout_group_supported", 1.0


def _probe_feedback_lookup(
    metrics_paths: tuple[str | Path, ...],
    group_fields: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    if not metrics_paths or not group_fields:
        return {}
    grouped: dict[str, list[dict[str, float]]] = {}
    for path in metrics_paths:
        for row in _read_rows(path):
            group = _diversity_stratum(row, group_fields)
            if not group:
                continue
            flow_delta = _finite(row.get("flow_delta_vs_current"))
            energy_delta = _finite(row.get("flow_energy_delta_vs_current"))
            current_ms = _finite(row.get("current_ms_ssim"))
            best_ms = _finite(row.get("best_ms_ssim"))
            if math.isfinite(best_ms) and math.isfinite(current_ms):
                best_delta = best_ms - current_ms
            else:
                finite_stage_deltas = [value for value in (flow_delta, energy_delta) if math.isfinite(value)]
                best_delta = max([0.0, *finite_stage_deltas]) if finite_stage_deltas else float("nan")
            energy_status = str(row.get("energy_status", ""))
            grouped.setdefault(group, []).append(
                {
                    "flow_delta": flow_delta,
                    "energy_delta": energy_delta,
                    "best_delta": best_delta,
                    "flow_negative": 1.0 if math.isfinite(flow_delta) and flow_delta < 0.0 else 0.0,
                    "flow_regression": 1.0 if math.isfinite(flow_delta) and flow_delta < -0.00389045250552833 else 0.0,
                    "energy_seen": 1.0 if energy_status == "ok" and math.isfinite(energy_delta) else 0.0,
                    "energy_win": 1.0
                    if energy_status == "ok" and math.isfinite(energy_delta) and energy_delta > max(0.0, _finite_or_zero(flow_delta))
                    else 0.0,
                }
            )

    lookup: dict[str, dict[str, object]] = {}
    for group, values in sorted(grouped.items()):
        flow_deltas = [value["flow_delta"] for value in values if math.isfinite(value["flow_delta"])]
        energy_deltas = [value["energy_delta"] for value in values if value["energy_seen"] > 0.0 and math.isfinite(value["energy_delta"])]
        best_deltas = [value["best_delta"] for value in values if math.isfinite(value["best_delta"])]
        energy_n = len(energy_deltas)
        energy_win_rate = sum(value["energy_win"] for value in values) / energy_n if energy_n else float("nan")
        flow_n = len(flow_deltas)
        flow_negative_rate = (
            sum(value["flow_negative"] for value in values if math.isfinite(value["flow_delta"])) / flow_n
            if flow_n
            else float("nan")
        )
        flow_regression_rate = (
            sum(value["flow_regression"] for value in values if math.isfinite(value["flow_delta"])) / flow_n
            if flow_n
            else float("nan")
        )
        best_delta_mean = statistics.mean(best_deltas) if best_deltas else float("nan")
        energy_delta_mean = statistics.mean(energy_deltas) if energy_deltas else float("nan")
        flow_delta_mean = statistics.mean(flow_deltas) if flow_deltas else float("nan")
        status, multiplier = _probe_feedback_status(
            energy_n=energy_n,
            best_delta_mean=best_delta_mean,
            energy_delta_mean=energy_delta_mean,
            flow_delta_mean=flow_delta_mean,
            flow_regression_rate=flow_regression_rate,
            energy_win_rate=energy_win_rate,
        )
        lookup[group] = {
            "group": group,
            "n": len(values),
            "energy_n": energy_n,
            "best_delta_mean": best_delta_mean,
            "energy_delta_mean": energy_delta_mean,
            "flow_delta_mean": flow_delta_mean,
            "flow_negative_rate": flow_negative_rate,
            "flow_regression_rate": flow_regression_rate,
            "energy_win_rate": energy_win_rate,
            "status": status,
            "score_multiplier": multiplier,
        }
    return lookup


def _row_probe_feedback_context(
    row: dict[str, object],
    group_fields: tuple[str, ...],
    feedback_lookup: dict[str, dict[str, object]],
) -> dict[str, object]:
    if not group_fields:
        return _empty_probe_feedback_context("not_configured", "")
    group = _diversity_stratum(row, group_fields)
    stats = feedback_lookup.get(group)
    if not stats:
        return _empty_probe_feedback_context("not_probed", group)
    return {
        "true_probe_feedback_group_key": group,
        "true_probe_feedback_n": int(_finite(stats.get("n"))),
        "true_probe_feedback_energy_n": int(_finite(stats.get("energy_n"))),
        "true_probe_feedback_best_delta_mean": _finite_or_blank(stats.get("best_delta_mean")),
        "true_probe_feedback_energy_delta_mean": _finite_or_blank(stats.get("energy_delta_mean")),
        "true_probe_feedback_flow_delta_mean": _finite_or_blank(stats.get("flow_delta_mean")),
        "true_probe_feedback_flow_negative_rate": _finite_or_blank(stats.get("flow_negative_rate")),
        "true_probe_feedback_flow_regression_rate": _finite_or_blank(stats.get("flow_regression_rate")),
        "true_probe_feedback_energy_win_rate": _finite_or_blank(stats.get("energy_win_rate")),
        "true_probe_feedback_status": stats.get("status", "not_probed"),
        "true_probe_feedback_score_multiplier": stats.get("score_multiplier", 1.0),
    }


def _row_exact_probe_feedback_context(
    row: dict[str, object],
    exact_feedback_lookup: dict[str, dict[str, object]],
) -> dict[str, object]:
    source = str(row.get("source_archive_path", ""))
    if not source:
        return _empty_exact_probe_feedback_context("missing_source", "", 1.0)
    stats = exact_feedback_lookup.get(source)
    if not stats:
        return _empty_exact_probe_feedback_context("not_probed", source, 1.0)
    n = _finite(stats.get("n"))
    energy_n = _finite(stats.get("energy_n"))
    best_delta_mean = _finite(stats.get("best_delta_mean"))
    flow_delta_mean = _finite(stats.get("flow_delta_mean"))
    energy_delta_mean = _finite(stats.get("energy_delta_mean"))
    status, multiplier = _exact_probe_feedback_status(
        n=n,
        energy_n=energy_n,
        best_delta_mean=best_delta_mean,
        flow_delta_mean=flow_delta_mean,
        energy_delta_mean=energy_delta_mean,
    )
    return {
        "exact_probe_feedback_source_archive_path": source,
        "exact_probe_feedback_n": int(n) if math.isfinite(n) else "",
        "exact_probe_feedback_energy_n": int(energy_n) if math.isfinite(energy_n) else "",
        "exact_probe_feedback_best_delta_mean": best_delta_mean if math.isfinite(best_delta_mean) else "",
        "exact_probe_feedback_flow_delta_mean": flow_delta_mean if math.isfinite(flow_delta_mean) else "",
        "exact_probe_feedback_energy_delta_mean": energy_delta_mean if math.isfinite(energy_delta_mean) else "",
        "exact_probe_feedback_status": status,
        "exact_probe_feedback_score_multiplier": multiplier,
    }


def _empty_exact_probe_feedback_context(status: str, source: str, multiplier: float) -> dict[str, object]:
    return {
        "exact_probe_feedback_source_archive_path": source,
        "exact_probe_feedback_n": "",
        "exact_probe_feedback_energy_n": "",
        "exact_probe_feedback_best_delta_mean": "",
        "exact_probe_feedback_flow_delta_mean": "",
        "exact_probe_feedback_energy_delta_mean": "",
        "exact_probe_feedback_status": status,
        "exact_probe_feedback_score_multiplier": multiplier,
    }


def _empty_probe_feedback_context(status: str, group: str) -> dict[str, object]:
    return {
        "true_probe_feedback_group_key": group,
        "true_probe_feedback_n": "",
        "true_probe_feedback_energy_n": "",
        "true_probe_feedback_best_delta_mean": "",
        "true_probe_feedback_energy_delta_mean": "",
        "true_probe_feedback_flow_delta_mean": "",
        "true_probe_feedback_flow_negative_rate": "",
        "true_probe_feedback_flow_regression_rate": "",
        "true_probe_feedback_energy_win_rate": "",
        "true_probe_feedback_status": status,
        "true_probe_feedback_score_multiplier": 1.0,
    }


def _energy_gate_config(summary_path: str | Path | None) -> dict[str, object]:
    if summary_path in (None, ""):
        return {}
    path = Path(summary_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    selected = data.get("selected", {})
    threshold = _finite(data.get("recommended_min_energy_flow_delta"))
    if not math.isfinite(threshold):
        threshold = _finite(selected.get("min_energy_flow_delta") if isinstance(selected, dict) else None)
    retained = _finite(selected.get("retained_best_delta_fraction") if isinstance(selected, dict) else None)
    retained_map = _finite(selected.get("retained_map_delta_fraction") if isinstance(selected, dict) else None)
    if not math.isfinite(threshold):
        raise ValueError(f"Energy gate summary is missing recommended_min_energy_flow_delta: {path}")
    return {
        "path": str(path),
        "min_flow_delta": threshold,
        "retained_delta_fraction": retained if math.isfinite(retained) else "",
        "retained_map_delta_fraction": retained_map if math.isfinite(retained_map) else "",
    }


def _row_energy_gate_context(row: dict[str, str], gate: dict[str, object]) -> dict[str, object]:
    if not gate:
        return _empty_energy_gate_context("not_configured", 1.0)
    threshold = _finite(gate.get("min_flow_delta"))
    if not math.isfinite(threshold):
        return _empty_energy_gate_context("not_configured", 1.0)
    expected_flow = _finite(row.get("expected_flow_delta_mean"))
    if not math.isfinite(expected_flow):
        return {
            **_empty_energy_gate_context("energy_gate_unknown", 1.0),
            "energy_gate_summary": gate.get("path", ""),
            "energy_gate_min_flow_delta": threshold,
            "energy_gate_retained_delta_fraction": gate.get("retained_delta_fraction", ""),
            "energy_gate_retained_map_delta_fraction": gate.get("retained_map_delta_fraction", ""),
        }

    expected_delta = _finite(row.get("expected_delta_mean"))
    status = "energy_gate_pass" if expected_flow >= threshold else "energy_gate_skip"
    if status == "energy_gate_pass" or not math.isfinite(expected_delta) or expected_delta <= 0.0:
        multiplier = 1.0
    else:
        flow_only_delta = max(0.0, expected_flow)
        multiplier = max(0.2, min(1.0, flow_only_delta / expected_delta))
    return {
        "energy_gate_summary": gate.get("path", ""),
        "energy_gate_min_flow_delta": threshold,
        "energy_gate_expected_status": status,
        "energy_gate_score_multiplier": multiplier,
        "energy_gate_retained_delta_fraction": gate.get("retained_delta_fraction", ""),
        "energy_gate_retained_map_delta_fraction": gate.get("retained_map_delta_fraction", ""),
    }


def _empty_energy_gate_context(status: str, multiplier: float) -> dict[str, object]:
    return {
        "energy_gate_summary": "",
        "energy_gate_min_flow_delta": "",
        "energy_gate_expected_status": status,
        "energy_gate_score_multiplier": multiplier,
        "energy_gate_retained_delta_fraction": "",
        "energy_gate_retained_map_delta_fraction": "",
    }


def _energy_promotion_rule(gate: dict[str, object]) -> str:
    threshold = _finite(gate.get("min_flow_delta")) if gate else float("nan")
    if math.isfinite(threshold):
        return (
            "Run energy only for successful flow probes ranked by true-scanner flow delta "
            f"and with flow_delta_vs_current >= {threshold:.9f}."
        )
    return "Run energy only for successful flow probes ranked by true-scanner flow delta."


def _energy_gate_eligible_count(planned: list[dict[str, object]], reserved_energy: int) -> int | str:
    statuses = [str(row.get("energy_gate_expected_status", "")) for row in planned]
    if not statuses or all(status in ("", "not_configured") for status in statuses):
        return ""
    eligible = sum(status in {"energy_gate_pass", "energy_gate_unknown", "not_configured"} for status in statuses)
    return min(max(0, int(reserved_energy)), eligible)


def _probe_feedback_status(
    *,
    energy_n: int,
    best_delta_mean: float,
    energy_delta_mean: float,
    flow_delta_mean: float,
    flow_regression_rate: float,
    energy_win_rate: float,
) -> tuple[str, float]:
    if math.isfinite(flow_regression_rate) and flow_regression_rate >= 0.5:
        return "flow_regression_risky", 0.70
    if energy_n >= 2 and math.isfinite(energy_delta_mean) and energy_delta_mean > 0.002 and energy_win_rate >= 0.5:
        return "energy_supported", 1.12
    if energy_n == 1 and math.isfinite(energy_delta_mean) and energy_delta_mean > 0.002 and energy_win_rate >= 1.0:
        return "energy_single_probe_positive", 1.06
    if math.isfinite(best_delta_mean) and best_delta_mean > 0.001:
        return "probe_supported", 1.05
    if energy_n > 0 and math.isfinite(energy_delta_mean) and energy_delta_mean <= 0.0:
        return "energy_not_supported", 0.82
    if math.isfinite(flow_delta_mean) and flow_delta_mean < -0.003:
        return "flow_negative_energy_unknown", 0.90
    return "probe_flat_or_mixed", 0.97


def _exact_probe_feedback_status(
    *,
    n: float,
    energy_n: float,
    best_delta_mean: float,
    flow_delta_mean: float,
    energy_delta_mean: float,
) -> tuple[str, float]:
    if not math.isfinite(n) or n <= 0:
        return "not_probed", 1.0
    if math.isfinite(best_delta_mean) and best_delta_mean <= 0.0 and math.isfinite(flow_delta_mean) and flow_delta_mean < -0.003:
        return "exact_flow_regression", 0.45
    if math.isfinite(energy_n) and energy_n > 0 and math.isfinite(energy_delta_mean) and energy_delta_mean <= 0.0:
        return "exact_energy_regression", 0.50
    if math.isfinite(best_delta_mean) and best_delta_mean <= 0.0:
        return "exact_no_gain", 0.62
    if math.isfinite(energy_n) and energy_n > 0 and math.isfinite(energy_delta_mean) and energy_delta_mean > 0.003:
        return "exact_energy_supported", 1.0
    if math.isfinite(flow_delta_mean) and flow_delta_mean > 0.003:
        return "exact_flow_supported", 1.0
    return "exact_mixed_or_flat", 0.90


def _surrogate_calibration_context(
    row: dict[str, object],
    surrogate_gate: dict[str, object],
) -> dict[str, object]:
    status = str(surrogate_gate.get("status", "not_configured"))
    base = {
        "surrogate_calibration_support_n": "",
        "surrogate_calibration_distance": "",
        "surrogate_calibration_training_ssim_min": "",
        "surrogate_calibration_training_ssim_max": "",
    }
    if status == "not_configured":
        return {
            **base,
            "surrogate_calibration_status": "not_configured",
            "surrogate_calibration_score_multiplier": 1.0,
        }
    if status == "fail":
        return {
            **base,
            "surrogate_calibration_status": "gate_failed",
            "surrogate_calibration_score_multiplier": 0.0,
        }

    support_n = _finite(surrogate_gate.get("training_row_ssim_n"))
    band_min = _finite(surrogate_gate.get("training_row_ssim_min"))
    band_max = _finite(surrogate_gate.get("training_row_ssim_max"))
    band_mean = _finite(surrogate_gate.get("training_row_ssim_mean"))
    band_std = _finite(surrogate_gate.get("training_row_ssim_std"))
    if not math.isfinite(support_n) or support_n <= 0:
        return {
            **base,
            "surrogate_calibration_status": "aggregate_only",
            "surrogate_calibration_score_multiplier": 1.0,
        }

    current_ssim = _finite(row.get("current_ssim") or row.get("SSIM") or row.get("Struct_SSIM"))
    calibration_base = {
        "surrogate_calibration_support_n": int(support_n),
        "surrogate_calibration_training_ssim_min": band_min if math.isfinite(band_min) else "",
        "surrogate_calibration_training_ssim_max": band_max if math.isfinite(band_max) else "",
    }
    if not math.isfinite(current_ssim):
        return {
            **base,
            **calibration_base,
            "surrogate_calibration_status": "missing_current_ssim",
            "surrogate_calibration_score_multiplier": 0.75,
        }

    scale = max(band_std if math.isfinite(band_std) else 0.0, 0.03)
    distance = 0.0
    if math.isfinite(band_min) and current_ssim < band_min:
        distance = 1.0 + (band_min - current_ssim) / scale
    elif math.isfinite(band_max) and current_ssim > band_max:
        distance = 1.0 + (current_ssim - band_max) / scale
    elif math.isfinite(band_mean):
        distance = max(0.0, abs(current_ssim - band_mean) / scale - 2.0)

    if distance <= 0.5:
        calibration_status = "calibration_supported"
        multiplier = 1.0
    elif distance <= 1.5:
        calibration_status = "calibration_edge"
        multiplier = 0.85
    else:
        calibration_status = "calibration_out_of_domain"
        multiplier = 0.65
    return {
        **base,
        **calibration_base,
        "surrogate_calibration_status": calibration_status,
        "surrogate_calibration_distance": round(distance, 6),
        "surrogate_calibration_score_multiplier": multiplier,
    }


def _predict_base_rows(
    base_rows: list[dict[str, str]],
    examples: list[ResidualTeacherExample],
    *,
    selected_strength: float,
    neighbor_count: int,
    uncertainty_z: float,
    exploration_weight: float,
    row_wise_strengths: bool = False,
    min_map_delta_lcb: float = -0.0005,
    min_map_safe_win_rate: float = 0.5,
) -> list[dict[str, object]]:
    if row_wise_strengths:
        return _predict_base_rows_row_wise_strengths(
            base_rows,
            examples,
            neighbor_count=neighbor_count,
            uncertainty_z=uncertainty_z,
            exploration_weight=exploration_weight,
            min_map_delta_lcb=min_map_delta_lcb,
            min_map_safe_win_rate=min_map_safe_win_rate,
        )

    candidates = [example for example in examples if abs(example.strength - selected_strength) <= 1e-6]
    if not candidates:
        candidates = examples
    ms_scale = max(statistics.pstdev([example.base_ms_ssim for example in candidates]), 0.02)
    lpips_values = [example.base_lpips for example in candidates if math.isfinite(example.base_lpips)]
    lpips_scale = max(statistics.pstdev(lpips_values), 0.02) if len(lpips_values) > 1 else 0.05

    ranked_base = sorted(enumerate(base_rows), key=lambda item: _finite(item[1].get("MS-SSIM")))
    predictions: list[dict[str, object]] = []
    for current_rank, (idx, row) in enumerate(ranked_base, start=1):
        row_ms = _finite(row.get("MS-SSIM"))
        row_ssim = _finite(row.get("SSIM") or row.get("Struct_SSIM"))
        row_lpips = _finite(row.get("LPIPS") or row.get("LPIPS_PROXY"))
        distances = [
            (_teacher_distance(row, example, ms_scale=ms_scale, lpips_scale=lpips_scale), example)
            for example in candidates
        ]
        distances.sort(key=lambda item: item[0])
        neighbors = [example for _distance, example in distances[: max(1, neighbor_count)]]
        deltas = [example.best_objective_delta for example in neighbors]
        stage_summary = _neighbor_stage_summary(neighbors)
        map_safety = _neighbor_map_safety_summary(
            neighbors,
            uncertainty_z=uncertainty_z,
            min_map_delta_lcb=min_map_delta_lcb,
            min_map_safe_win_rate=min_map_safe_win_rate,
        )
        mean = statistics.mean(deltas)
        std = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
        se = std / math.sqrt(len(deltas))
        lcb = mean - uncertainty_z * se
        acquisition = _apply_score_multiplier(
            lcb + max(0.0, exploration_weight) * std,
            float(map_safety["map_safety_score_multiplier"]),
        )
        predictions.append(
            {
                "row_index": idx,
                "current_rank": current_rank,
                "source_archive_path": row.get("source_archive_path", ""),
                "current_ms_ssim": row_ms,
                "current_ssim": row_ssim,
                "current_lpips": row_lpips,
                "selected_strength": selected_strength,
                "strength_policy": "global_strength",
                "evaluated_strengths": f"{selected_strength:g}",
                "strength_candidate_count": 1,
                "expected_delta_mean": mean,
                "expected_delta_lcb": lcb,
                "expected_delta_uncertainty": std,
                "acquisition_score": acquisition,
                **stage_summary,
                **map_safety,
                **_neighbor_control_summary(neighbors),
                "nearest_teacher_count": len(neighbors),
                "nearest_teacher_sources": ";".join(example.source_archive_path for example in neighbors[:5]),
                "reference_png": row.get("reference_png", ""),
                "phantom_path": row.get("phantom_path", ""),
                "synthetic_gray_png": row.get("synthetic_gray_png", ""),
            }
        )
    return predictions


def _predict_base_rows_row_wise_strengths(
    base_rows: list[dict[str, str]],
    examples: list[ResidualTeacherExample],
    *,
    neighbor_count: int,
    uncertainty_z: float,
    exploration_weight: float,
    min_map_delta_lcb: float,
    min_map_safe_win_rate: float,
) -> list[dict[str, object]]:
    grouped: dict[float, list[ResidualTeacherExample]] = {}
    for example in examples:
        if math.isfinite(example.strength):
            grouped.setdefault(round(example.strength, 6), []).append(example)
    if not grouped:
        return []

    all_candidates = [example for group in grouped.values() for example in group]
    ms_scale = max(statistics.pstdev([example.base_ms_ssim for example in all_candidates]), 0.02)
    lpips_values = [example.base_lpips for example in all_candidates if math.isfinite(example.base_lpips)]
    lpips_scale = max(statistics.pstdev(lpips_values), 0.02) if len(lpips_values) > 1 else 0.05

    ranked_base = sorted(enumerate(base_rows), key=lambda item: _finite(item[1].get("MS-SSIM")))
    predictions: list[dict[str, object]] = []
    evaluated_strengths = ";".join(f"{strength:g}" for strength in sorted(grouped))
    for current_rank, (idx, row) in enumerate(ranked_base, start=1):
        strength_predictions = [
            _predict_row_strength(
                row,
                strength,
                group,
                neighbor_count=neighbor_count,
                uncertainty_z=uncertainty_z,
                exploration_weight=exploration_weight,
                ms_scale=ms_scale,
                lpips_scale=lpips_scale,
                min_map_delta_lcb=min_map_delta_lcb,
                min_map_safe_win_rate=min_map_safe_win_rate,
            )
            for strength, group in sorted(grouped.items())
        ]
        best = max(
            strength_predictions,
            key=lambda item: (
                float(item["acquisition_score"]),
                float(item["expected_map_delta_lcb_min"]) if item["expected_map_delta_lcb_min"] != "" else -999.0,
                float(item["expected_delta_lcb"]),
                float(item["expected_delta_mean"]),
                int(item["nearest_teacher_count"]),
            ),
        )
        row_ms = _finite(row.get("MS-SSIM"))
        row_ssim = _finite(row.get("SSIM") or row.get("Struct_SSIM"))
        row_lpips = _finite(row.get("LPIPS") or row.get("LPIPS_PROXY"))
        predictions.append(
            {
                "row_index": idx,
                "current_rank": current_rank,
                "source_archive_path": row.get("source_archive_path", ""),
                "current_ms_ssim": row_ms,
                "current_ssim": row_ssim,
                "current_lpips": row_lpips,
                "selected_strength": best["selected_strength"],
                "strength_policy": "row_wise_neighbor_strength",
                "evaluated_strengths": evaluated_strengths,
                "strength_candidate_count": len(grouped),
                "expected_delta_mean": best["expected_delta_mean"],
                "expected_delta_lcb": best["expected_delta_lcb"],
                "expected_delta_uncertainty": best["expected_delta_uncertainty"],
                "acquisition_score": best["acquisition_score"],
                **_map_safety_context(best),
                "expected_flow_delta_mean": best["expected_flow_delta_mean"],
                "expected_energy_delta_mean": best["expected_energy_delta_mean"],
                "expected_energy_extra_mean": best["expected_energy_extra_mean"],
                "nearest_flow_win_rate": best["nearest_flow_win_rate"],
                "nearest_energy_win_rate": best["nearest_energy_win_rate"],
                **_teacher_control_context(best),
                "nearest_teacher_count": best["nearest_teacher_count"],
                "nearest_teacher_sources": best["nearest_teacher_sources"],
                "reference_png": row.get("reference_png", ""),
                "phantom_path": row.get("phantom_path", ""),
                "synthetic_gray_png": row.get("synthetic_gray_png", ""),
            }
        )
    return predictions


def _predict_row_strength(
    row: dict[str, str],
    strength: float,
    candidates: list[ResidualTeacherExample],
    *,
    neighbor_count: int,
    uncertainty_z: float,
    exploration_weight: float,
    ms_scale: float,
    lpips_scale: float,
    min_map_delta_lcb: float,
    min_map_safe_win_rate: float,
) -> dict[str, object]:
    distances = [
        (_teacher_distance(row, example, ms_scale=ms_scale, lpips_scale=lpips_scale), example)
        for example in candidates
    ]
    distances.sort(key=lambda item: item[0])
    neighbors = [example for _distance, example in distances[: max(1, neighbor_count)]]
    deltas = [example.best_objective_delta for example in neighbors]
    stage_summary = _neighbor_stage_summary(neighbors)
    map_safety = _neighbor_map_safety_summary(
        neighbors,
        uncertainty_z=uncertainty_z,
        min_map_delta_lcb=min_map_delta_lcb,
        min_map_safe_win_rate=min_map_safe_win_rate,
    )
    mean = statistics.mean(deltas)
    std = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
    se = std / math.sqrt(len(deltas))
    lcb = mean - uncertainty_z * se
    acquisition = _apply_score_multiplier(
        lcb + max(0.0, exploration_weight) * std,
        float(map_safety["map_safety_score_multiplier"]),
    )
    return {
        "selected_strength": strength,
        "expected_delta_mean": mean,
        "expected_delta_lcb": lcb,
        "expected_delta_uncertainty": std,
        "acquisition_score": acquisition,
        **stage_summary,
        **map_safety,
        **_neighbor_control_summary(neighbors),
        "nearest_teacher_count": len(neighbors),
        "nearest_teacher_sources": ";".join(example.source_archive_path for example in neighbors[:5]),
    }


def _teacher_distance(
    row: dict[str, str],
    example: ResidualTeacherExample,
    *,
    ms_scale: float,
    lpips_scale: float,
) -> float:
    row_ms = _finite(row.get("MS-SSIM"))
    row_lpips = _finite(row.get("LPIPS") or row.get("LPIPS_PROXY"))
    dist = ((_safe_value(row_ms, example.base_ms_ssim) - example.base_ms_ssim) / ms_scale) ** 2
    if math.isfinite(row_lpips) and math.isfinite(example.base_lpips):
        dist += ((row_lpips - example.base_lpips) / lpips_scale) ** 2
    row_traits = _source_traits(row.get("source_archive_path", ""))
    example_traits = _source_traits(example.source_archive_path)
    if row_traits["body_site"] and example_traits["body_site"] and row_traits["body_site"] != example_traits["body_site"]:
        dist += 0.35
    if row_traits["sex"] and example_traits["sex"] and row_traits["sex"] != example_traits["sex"]:
        dist += 0.12
    if row_traits["age_band"] and example_traits["age_band"] and row_traits["age_band"] != example_traits["age_band"]:
        dist += 0.08
    if row_traits["frame"] >= 0 and example_traits["frame"] >= 0:
        dist += min(abs(row_traits["frame"] - example_traits["frame"]) / 400.0, 1.0) * 0.08
    return float(dist)


def _safe_value(value: float, fallback: float) -> float:
    return value if math.isfinite(value) else fallback


def _source_traits(source_archive_path: str) -> dict[str, str | int]:
    parts = Path(source_archive_path).parts
    frame = -1
    match = re.search(r"frame(\d+)", source_archive_path)
    if match:
        frame = int(match.group(1))
    return {
        "sex": parts[1] if len(parts) > 1 else "",
        "age_band": parts[2] if len(parts) > 2 else "",
        "body_site": parts[3] if len(parts) > 3 else "",
        "frame": frame,
    }


def _load_saved_teacher_examples(path: str | Path) -> list[ResidualTeacherExample]:
    examples = []
    for row in _read_rows(path):
        examples.append(
            ResidualTeacherExample(
                source_archive_path=row["source_archive_path"],
                row_index=_first_int(row, "row_index"),
                rank=_first_int(row, "rank"),
                strength=_finite(row.get("strength")),
                base_ms_ssim=_finite(row.get("base_ms_ssim")),
                base_lpips=_finite(row.get("base_lpips")),
                flow_ms_ssim=_finite(row.get("flow_ms_ssim")),
                flow_energy_ms_ssim=_finite(row.get("flow_energy_ms_ssim")),
                best_stage=row.get("best_stage", ""),
                best_ms_ssim=_finite(row.get("best_ms_ssim")),
                best_delta=_finite(row.get("best_delta")),
                base_objective_score=_finite(row.get("base_objective_score")),
                flow_objective_score=_finite(row.get("flow_objective_score")),
                flow_energy_objective_score=_finite(row.get("flow_energy_objective_score")),
                best_objective_score=_finite(row.get("best_objective_score")),
                best_objective_delta=_finite(row.get("best_objective_delta")),
                objective_source=row.get("objective_source", "plain_ms_ssim"),
                reference_png=row.get("reference_png", ""),
                base_phantom_path=row.get("base_phantom_path", ""),
                base_synthetic_gray_png=row.get("base_synthetic_gray_png", ""),
                flow_phantom_path=row.get("flow_phantom_path", ""),
                flow_energy_phantom_path=row.get("flow_energy_phantom_path", ""),
                teacher_metrics_path=row.get("teacher_metrics_path", ""),
                residual_control_policy=row.get("residual_control_policy", ""),
                flow_smooth_sigma=_finite(row.get("flow_smooth_sigma")),
                flow_attachment=_finite(row.get("flow_attachment")),
                energy_exponent=_finite(row.get("energy_exponent")),
                energy_sigma=_finite(row.get("energy_sigma")),
                energy_ratio_low=_finite(row.get("energy_ratio_low")),
                energy_ratio_high=_finite(row.get("energy_ratio_high")),
                energy_clip_low=_finite(row.get("energy_clip_low")),
                energy_clip_high=_finite(row.get("energy_clip_high")),
                texture_mean_exponent=_finite(row.get("texture_mean_exponent")),
                texture_exponent=_finite(row.get("texture_exponent")),
                texture_deep_exponent=_finite(row.get("texture_deep_exponent")),
                flow_struct_delta=_finite(row.get("flow_struct_delta")),
                flow_oac_delta=_finite(row.get("flow_oac_delta")),
                flow_sc_delta=_finite(row.get("flow_sc_delta")),
                flow_rsc_delta=_finite(row.get("flow_rsc_delta")),
                flow_map_objective_delta=_finite(row.get("flow_map_objective_delta")),
                flow_energy_struct_delta=_finite(row.get("flow_energy_struct_delta")),
                flow_energy_oac_delta=_finite(row.get("flow_energy_oac_delta")),
                flow_energy_sc_delta=_finite(row.get("flow_energy_sc_delta")),
                flow_energy_rsc_delta=_finite(row.get("flow_energy_rsc_delta")),
                flow_energy_map_objective_delta=_finite(row.get("flow_energy_map_objective_delta")),
            )
        )
    return examples


def _surrogate_gate(
    metrics_paths: list[str | Path],
    *,
    min_ms_ssim: float,
    max_lpips_proxy: float,
    uncertainty_z: float,
) -> dict[str, float | int | str | list[str]]:
    if not metrics_paths:
        return {"status": "not_configured"}
    ms_values: list[float] = []
    lpips_proxy_values: list[float] = []
    training_row_ssim_values: list[float] = []
    for path in metrics_paths:
        for row in _read_rows(path):
            ms = _finite(row.get("surrogate_MS-SSIM"))
            proxy = _finite(row.get("surrogate_LPIPS_PROXY"))
            training_row_ssim = _finite(row.get("training_row_ssim"))
            if math.isfinite(ms):
                ms_values.append(ms)
            if math.isfinite(proxy):
                lpips_proxy_values.append(proxy)
            if math.isfinite(training_row_ssim):
                training_row_ssim_values.append(training_row_ssim)
    if not ms_values:
        return {"status": "fail", "reason": "no_surrogate_ms_ssim", "metrics_paths": [str(Path(p)) for p in metrics_paths]}
    ms_mean = statistics.mean(ms_values)
    ms_std = statistics.pstdev(ms_values) if len(ms_values) > 1 else 0.0
    ms_lcb = ms_mean - float(uncertainty_z) * ms_std / math.sqrt(len(ms_values))
    lpips_proxy_max = max(lpips_proxy_values) if lpips_proxy_values else float("inf")
    status = "pass" if ms_lcb >= min_ms_ssim and lpips_proxy_max <= max_lpips_proxy else "fail"
    gate: dict[str, float | int | str | list[str]] = {
        "status": status,
        "n": len(ms_values),
        "surrogate_ms_ssim_mean": ms_mean,
        "surrogate_ms_ssim_lcb": ms_lcb,
        "surrogate_lpips_proxy_max": lpips_proxy_max,
        "min_surrogate_ms_ssim": min_ms_ssim,
        "max_surrogate_lpips_proxy": max_lpips_proxy,
        "metrics_paths": [str(Path(p)) for p in metrics_paths],
    }
    if training_row_ssim_values:
        gate.update(
            {
                "training_row_ssim_n": len(training_row_ssim_values),
                "training_row_ssim_mean": statistics.mean(training_row_ssim_values),
                "training_row_ssim_std": statistics.pstdev(training_row_ssim_values)
                if len(training_row_ssim_values) > 1
                else 0.0,
                "training_row_ssim_min": min(training_row_ssim_values),
                "training_row_ssim_max": max(training_row_ssim_values),
            }
        )
    return gate


def _split_examples(
    examples: list[ResidualTeacherExample],
    *,
    holdout_fraction: float,
    group_fields: tuple[str, ...] = (),
) -> tuple[list[ResidualTeacherExample], list[ResidualTeacherExample], list[str]]:
    fraction = min(max(float(holdout_fraction), 0.0), 0.8)
    if fraction <= 0:
        return examples, [], []
    train: list[ResidualTeacherExample] = []
    holdout: list[ResidualTeacherExample] = []
    holdout_groups: list[str] = []
    threshold = int(fraction * 10_000)
    if group_fields:
        grouped: dict[str, list[ResidualTeacherExample]] = {}
        for example in examples:
            grouped.setdefault(_example_group_key(example, group_fields), []).append(example)
        for group, group_examples in sorted(grouped.items()):
            digest = hashlib.sha1(group.encode("utf-8")).hexdigest()
            bucket = int(digest[:8], 16) % 10_000
            if bucket < threshold:
                holdout.extend(group_examples)
                holdout_groups.append(group)
            else:
                train.extend(group_examples)
        return train, holdout, holdout_groups

    for example in examples:
        group = example.source_archive_path
        digest = hashlib.sha1(group.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 10_000
        if bucket < threshold:
            holdout.append(example)
            holdout_groups.append(group)
        else:
            train.append(example)
    return train, holdout, holdout_groups


def _example_group_key(example: ResidualTeacherExample, fields: tuple[str, ...]) -> str:
    traits = _source_traits(example.source_archive_path)
    values: list[str] = []
    for field in fields:
        if field == "sex":
            values.append(str(traits.get("sex", "")) or "unknown_sex")
        elif field in {"age", "age_band"}:
            values.append(str(traits.get("age_band", "")) or "unknown_age")
        elif field in {"site", "body_site"}:
            values.append(str(traits.get("body_site", "")) or "unknown_site")
        elif field == "rank_bucket":
            values.append(_rank_bucket(example.rank))
        elif field == "source_archive_path":
            values.append(example.source_archive_path)
        else:
            values.append(getattr(example, field, "") or f"unknown_{field}")
    return "|".join(str(value) for value in values)


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames:
            with path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        else:
            path.write_text("", encoding="utf-8")
        return path
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _prefixed_stage_map_metrics(
    prefix: str,
    reference_png: Path | None,
    pred_png: Path | None,
    maps_dir: Path,
    *,
    include_lpips: bool,
) -> dict[str, object]:
    if reference_png is None or pred_png is None:
        return {f"{prefix}_map_enrichment_status": "missing_png"}
    try:
        struct_metrics = calculate_metrics(reference_png, pred_png, include_lpips=include_lpips)
        map_metrics = evaluate_feature_map_metrics(
            reference_png,
            pred_png,
            maps_dir / "ref_maps",
            maps_dir / "pred_maps",
            include_lpips=include_lpips,
        )
        profiles = profile_scores(reference_png, pred_png)
    except Exception as exc:
        return {f"{prefix}_map_enrichment_status": "failed", f"{prefix}_map_enrichment_error": str(exc)}

    out: dict[str, object] = {f"{prefix}_map_enrichment_status": "ok"}
    for key, value in struct_metrics.items():
        out[f"{prefix}_Struct_{key}"] = value
    for key, value in map_metrics.items():
        out[f"{prefix}_{key}"] = value
    for key, value in profiles.items():
        out[f"{prefix}_{key}"] = value
    return out


def _first_existing_path(*values: str | Path | None) -> Path | None:
    for value in values:
        if value in (None, ""):
            continue
        path = Path(value)
        if path.exists():
            return path.resolve()
    return None


def _sibling_flow_lookup(teacher_metrics: Path) -> dict[tuple[str, str], dict[str, str]]:
    sibling = teacher_metrics.with_name("flow_correct_metrics.csv")
    if not sibling.exists():
        return {}
    try:
        rows = _read_rows(sibling)
    except Exception:
        return {}
    return {_stage_join_key(row): row for row in rows}


def _stage_join_key(row: dict[str, str]) -> tuple[str, str]:
    return (str(row.get("priority", "")), str(row.get("source_archive_path", "")))


def _neighbor_stage_summary(neighbors: list[ResidualTeacherExample]) -> dict[str, float]:
    if not neighbors:
        return {
            "expected_flow_delta_mean": 0.0,
            "expected_energy_delta_mean": 0.0,
            "expected_energy_extra_mean": 0.0,
            "nearest_flow_win_rate": 0.0,
            "nearest_energy_win_rate": 0.0,
        }
    flow_deltas = [example.flow_objective_score - example.base_objective_score for example in neighbors]
    energy_deltas = [example.flow_energy_objective_score - example.base_objective_score for example in neighbors]
    energy_extras = [
        example.flow_energy_objective_score - max(example.base_objective_score, example.flow_objective_score)
        for example in neighbors
    ]
    flow_wins = [
        flow_delta > 0.0 and flow_delta >= energy_delta - 0.001
        for flow_delta, energy_delta in zip(flow_deltas, energy_deltas, strict=True)
    ]
    energy_wins = [energy_extra > 0.001 and energy_delta > 0.0 for energy_extra, energy_delta in zip(energy_extras, energy_deltas, strict=True)]
    return {
        "expected_flow_delta_mean": statistics.mean(flow_deltas),
        "expected_energy_delta_mean": statistics.mean(energy_deltas),
        "expected_energy_extra_mean": statistics.mean(energy_extras),
        "nearest_flow_win_rate": sum(flow_wins) / len(flow_wins),
        "nearest_energy_win_rate": sum(energy_wins) / len(energy_wins),
    }


def _neighbor_map_safety_summary(
    neighbors: list[ResidualTeacherExample],
    *,
    uncertainty_z: float,
    min_map_delta_lcb: float,
    min_map_safe_win_rate: float,
) -> dict[str, object]:
    if not neighbors:
        return _empty_map_safety_context("map_safety_no_neighbors", 0.35)

    summary: dict[str, object] = {}
    lcbs: list[float] = []
    for label, _metric in MAP_COMPONENTS:
        values = [
            delta
            for delta in (_example_best_stage_map_delta(example, label) for example in neighbors)
            if math.isfinite(delta)
        ]
        if values:
            mean = statistics.mean(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            lcb = mean - float(uncertainty_z) * std / math.sqrt(len(values))
            lcbs.append(lcb)
            summary[f"expected_{label}_delta_mean"] = mean
            summary[f"expected_{label}_delta_lcb"] = lcb
        else:
            summary[f"expected_{label}_delta_mean"] = ""
            summary[f"expected_{label}_delta_lcb"] = ""

    fully_observed = [
        _example_best_stage_map_deltas(example)
        for example in neighbors
        if all(math.isfinite(value) for value in _example_best_stage_map_deltas(example).values())
    ]
    safe_wins = [
        all(float(delta) >= float(min_map_delta_lcb) for delta in deltas.values())
        for deltas in fully_observed
    ]
    safe_win_rate = sum(safe_wins) / len(safe_wins) if safe_wins else float("nan")
    lcb_min = min(lcbs) if len(lcbs) == len(MAP_COMPONENTS) else float("nan")
    if not fully_observed or not math.isfinite(lcb_min):
        status = "map_safety_unknown"
        multiplier = 0.65
    elif lcb_min < float(min_map_delta_lcb):
        status = "map_safety_lcb_fail"
        multiplier = 0.35
    elif safe_win_rate < float(min_map_safe_win_rate):
        status = "map_safety_weak_support"
        multiplier = 0.55
    else:
        status = "map_safe_pass"
        multiplier = 1.0
    summary.update(
        {
            "expected_map_delta_lcb_min": lcb_min if math.isfinite(lcb_min) else "",
            "nearest_map_safe_win_rate": safe_win_rate if math.isfinite(safe_win_rate) else "",
            "map_safe_support_n": len(fully_observed),
            "map_safety_status": status,
            "map_safety_score_multiplier": multiplier,
        }
    )
    return summary


def _empty_map_safety_context(status: str, multiplier: float) -> dict[str, object]:
    return {
        "expected_struct_delta_mean": "",
        "expected_struct_delta_lcb": "",
        "expected_oac_delta_mean": "",
        "expected_oac_delta_lcb": "",
        "expected_sc_delta_mean": "",
        "expected_sc_delta_lcb": "",
        "expected_rsc_delta_mean": "",
        "expected_rsc_delta_lcb": "",
        "expected_map_delta_lcb_min": "",
        "nearest_map_safe_win_rate": "",
        "map_safe_support_n": 0,
        "map_safety_status": status,
        "map_safety_score_multiplier": multiplier,
    }


def _map_safety_context(row: dict[str, object]) -> dict[str, object]:
    empty = _empty_map_safety_context("map_safety_not_evaluated", 1.0)
    return {field: row.get(field, empty[field]) for field in MAP_SAFETY_FIELDS}


def _example_best_stage_map_deltas(example: ResidualTeacherExample) -> dict[str, float]:
    if example.best_stage == "flow":
        return {label: getattr(example, f"flow_{label}_delta") for label, _metric in MAP_COMPONENTS}
    if example.best_stage in {"flow_energy", "energy"}:
        return {label: getattr(example, f"flow_energy_{label}_delta") for label, _metric in MAP_COMPONENTS}
    return {label: 0.0 for label, _metric in MAP_COMPONENTS}


def _example_best_stage_map_delta(example: ResidualTeacherExample, label: str) -> float:
    return _example_best_stage_map_deltas(example).get(label, float("nan"))


def _apply_score_multiplier(score: float, multiplier: float) -> float:
    if not math.isfinite(score):
        return score
    multiplier = _clamp(multiplier if math.isfinite(multiplier) else 1.0, 0.01, 2.0)
    return score * multiplier if score >= 0.0 else score / multiplier


def _neighbor_control_summary(neighbors: list[ResidualTeacherExample]) -> dict[str, object]:
    controlled = [
        example
        for example in neighbors
        if example.best_objective_delta > 0.0
        and any(math.isfinite(getattr(example, field)) for field in STAGE2_NUMERIC_CONTROL_FIELDS)
    ]
    if not controlled:
        return {"teacher_control_support_n": 0}

    weights = [max(example.best_objective_delta, 1e-4) for example in controlled]
    total = sum(weights)
    summary: dict[str, object] = {"teacher_control_support_n": len(controlled)}
    for field in STAGE2_NUMERIC_CONTROL_FIELDS:
        values = [
            (getattr(example, field), weight)
            for example, weight in zip(controlled, weights, strict=True)
            if math.isfinite(getattr(example, field))
        ]
        if not values:
            continue
        weighted = sum(value * weight for value, weight in values) / max(sum(weight for _value, weight in values), 1e-12)
        summary[f"teacher_control_{field}"] = weighted
    summary["teacher_control_weight_sum"] = total
    return summary


def _teacher_control_context(row: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in row.items()
        if key == "teacher_control_support_n" or key == "teacher_control_weight_sum" or key.startswith("teacher_control_")
    }


def _row_residual_controls(row: dict[str, object]) -> dict[str, object]:
    current_ms = _finite(row.get("current_ms_ssim"))
    current_lpips = _finite(row.get("current_lpips"))
    selected_strength = _finite(row.get("selected_strength"))
    uncertainty = _finite(row.get("expected_delta_uncertainty"))
    flow_delta = _finite(row.get("expected_flow_delta_mean"))
    energy_delta = _finite(row.get("expected_energy_delta_mean"))
    energy_extra = _finite(row.get("expected_energy_extra_mean"))
    flow_win_rate = _finite(row.get("nearest_flow_win_rate"))
    energy_win_rate = _finite(row.get("nearest_energy_win_rate"))
    traits = _source_traits(str(row.get("source_archive_path", "")))
    body_site = str(traits.get("body_site", "")).lower()

    ms_gap = _clamp((0.78 - current_ms) / 0.20 if math.isfinite(current_ms) else 0.5, 0.0, 1.0)
    lpips_load = _clamp((current_lpips - 0.35) / 0.35 if math.isfinite(current_lpips) else 0.4, 0.0, 1.0)
    uncertainty_load = _clamp(uncertainty / 0.03 if math.isfinite(uncertainty) else 0.2, 0.0, 1.0)
    strength_load = _clamp(selected_strength / 0.38 if math.isfinite(selected_strength) else 0.7, 0.0, 1.4)
    flow_signal = _clamp(flow_delta / 0.05 if math.isfinite(flow_delta) else 0.0, -0.5, 1.0)
    energy_signal = _clamp(energy_delta / 0.05 if math.isfinite(energy_delta) else 0.0, -0.5, 1.0)
    energy_extra_signal = _clamp(energy_extra / 0.025 if math.isfinite(energy_extra) else 0.0, -0.5, 1.0)
    flow_win = _clamp(flow_win_rate if math.isfinite(flow_win_rate) else 0.0, 0.0, 1.0)
    energy_win = _clamp(energy_win_rate if math.isfinite(energy_win_rate) else 0.0, 0.0, 1.0)
    site_texture_bias = 0.10 if "cheek" in body_site else 0.07 if "eye" in body_site else 0.04

    geometry_load = _clamp(
        0.35 * ms_gap + 0.20 * lpips_load + 0.15 * uncertainty_load + 0.22 * max(flow_signal, 0.0) + 0.08 * flow_win,
        0.0,
        1.0,
    )
    energy_load = _clamp(
        0.30 * ms_gap + 0.12 * lpips_load + 0.10 * strength_load + 0.28 * max(energy_signal, 0.0) + 0.20 * max(energy_extra_signal, 0.0),
        0.0,
        1.0,
    )
    texture_load = _clamp(
        0.20 * ms_gap + 0.28 * lpips_load + 0.08 * uncertainty_load + 0.30 * max(energy_extra_signal, 0.0) + 0.10 * energy_win + site_texture_bias,
        0.0,
        1.0,
    )

    controls: dict[str, object] = {
        "residual_control_policy": "teacher_neighbor_physics_dl_v2",
        "flow_smooth_sigma": round(_clamp(1.00 + 0.62 * geometry_load + (0.08 if "cheek" in body_site else 0.0), 0.85, 1.90), 3),
        "flow_attachment": round(_clamp(5.10 + 1.30 * (1.0 - geometry_load) + 0.35 * uncertainty_load - 0.20 * flow_win, 4.7, 7.2), 3),
        "energy_base_policy": "current_on_flow_regression",
        "energy_base_flow_delta_threshold": 0.0,
        "energy_exponent": round(_clamp(0.66 + 0.28 * energy_load + 0.06 * strength_load, 0.58, 1.12), 3),
        "energy_sigma": round(_clamp(1.45 + 0.55 * geometry_load + 0.75 * energy_load, 1.25, 3.15), 3),
        "energy_ratio_low": round(_clamp(0.68 - 0.07 * energy_load - 0.02 * ms_gap, 0.56, 0.72), 3),
        "energy_ratio_high": round(_clamp(1.20 + 0.11 * energy_load + 0.04 * ms_gap, 1.16, 1.40), 3),
        "energy_clip_low": round(_clamp(0.82 - 0.07 * energy_load, 0.68, 0.86), 3),
        "energy_clip_high": round(_clamp(1.12 + 0.11 * energy_load, 1.08, 1.28), 3),
        "texture_mean_exponent": round(_clamp(0.01 + 0.03 * texture_load, 0.0, 0.06), 3),
        "texture_exponent": round(_clamp(0.08 + 0.46 * texture_load, 0.0, 0.62), 3),
        "texture_deep_exponent": round(_clamp(0.30 + 0.90 * texture_load, 0.0, 1.35), 3),
    }
    support_n = _finite(row.get("teacher_control_support_n"))
    if not math.isfinite(support_n) or support_n <= 0:
        return controls

    blend = _clamp(0.35 + 0.08 * support_n - 0.12 * uncertainty_load, 0.30, 0.72)
    bounds = _stage2_control_bounds()
    blended = dict(controls)
    for field in STAGE2_NUMERIC_CONTROL_FIELDS:
        teacher_value = _finite(row.get(f"teacher_control_{field}"))
        base_value = _finite(controls.get(field))
        if not math.isfinite(teacher_value) or not math.isfinite(base_value):
            continue
        low, high = bounds[field]
        blended[field] = round(_clamp((1.0 - blend) * base_value + blend * teacher_value, low, high), 3)
    blended["residual_control_policy"] = "teacher_neighbor_feedback_distilled_v3"
    return blended


def _stage2_control_bounds() -> dict[str, tuple[float, float]]:
    return {
        "flow_smooth_sigma": (0.85, 1.90),
        "flow_attachment": (4.7, 7.2),
        "energy_exponent": (0.45, 1.25),
        "energy_sigma": (1.25, 3.15),
        "energy_ratio_low": (0.56, 0.72),
        "energy_ratio_high": (1.16, 1.40),
        "energy_clip_low": (0.68, 0.86),
        "energy_clip_high": (1.08, 1.28),
        "texture_mean_exponent": (0.0, 0.06),
        "texture_exponent": (0.0, 0.62),
        "texture_deep_exponent": (0.0, 1.35),
    }


def _budget_variant_controls(
    row: dict[str, str],
    variant_index: int,
    *,
    energy_exponents: tuple[float, ...],
) -> dict[str, object]:
    controls = {
        field: row.get(field, "")
        for field in STAGE2_CONTROL_FIELDS
        if field != "energy_exponent"
    }
    base_exponent = _finite(row.get("energy_exponent"))
    if math.isfinite(base_exponent):
        reference = _variant_value(energy_exponents, 0, default=0.80)
        target = _variant_value(energy_exponents, variant_index, default=reference)
        scale = target / reference if abs(reference) > 1e-9 else 1.0
        controls["energy_exponent"] = round(_clamp(base_exponent * scale, 0.45, 1.25), 6)
    else:
        controls["energy_exponent"] = _variant_value(energy_exponents, variant_index, default=0.80)
    return controls


def _control_model_feature_defaults(examples: list[ResidualTeacherExample]) -> dict[str, float]:
    def median_or(values: list[float], fallback: float) -> float:
        finite = [value for value in values if math.isfinite(value)]
        return statistics.median(finite) if finite else fallback

    ranks = [float(example.rank) for example in examples if example.rank > 0]
    frames = [float(_source_traits(example.source_archive_path)["frame"]) for example in examples]
    return {
        "base_ms_ssim": median_or([example.base_ms_ssim for example in examples], 0.65),
        "base_lpips": median_or([example.base_lpips for example in examples], 0.58),
        "rank": median_or(ranks, 60.0),
        "frame": median_or(frames, 250.0),
    }


def _fit_control_model(
    examples: list[ResidualTeacherExample],
    *,
    defaults: dict[str, float],
    ridge_lambda: float,
) -> tuple[dict[str, list[float]], dict[str, float], dict[str, object]]:
    x = np.array([_control_model_features_for_example(example, defaults) for example in examples], dtype=float)
    y_rows = [_control_model_targets_for_example(example) for example in examples]
    y = np.array([[row[field] for field in TOPOLOGY_CONTROL_TARGET_FIELDS] for row in y_rows], dtype=float)
    weights = np.array(
        [0.25 + min(max(example.best_objective_delta, 0.0), 0.05) / 0.05 for example in examples],
        dtype=float,
    )
    sqrt_w = np.sqrt(weights)[:, None]
    xw = x * sqrt_w
    yw = y * sqrt_w
    penalty = np.eye(x.shape[1], dtype=float) * max(float(ridge_lambda), 0.0)
    penalty[0, 0] = 0.0
    lhs = xw.T @ xw + penalty
    rhs = xw.T @ yw
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ rhs

    pred = x @ beta
    residual = pred - y
    coefficients = {
        field: [float(value) for value in beta[:, index]]
        for index, field in enumerate(TOPOLOGY_CONTROL_TARGET_FIELDS)
    }
    residual_std = {
        field: float(np.std(residual[:, index])) if residual.shape[0] > 1 else 0.0
        for index, field in enumerate(TOPOLOGY_CONTROL_TARGET_FIELDS)
    }
    metrics = _control_model_error_metrics(y, pred)
    metrics["n"] = len(examples)
    metrics["weighted_training"] = True
    return coefficients, residual_std, metrics


def _evaluate_control_model(
    examples: list[ResidualTeacherExample],
    coefficients: dict[str, list[float]],
    *,
    defaults: dict[str, float],
    residual_std: dict[str, float],
) -> dict[str, object]:
    x = np.array([_control_model_features_for_example(example, defaults) for example in examples], dtype=float)
    y_rows = [_control_model_targets_for_example(example) for example in examples]
    y = np.array([[row[field] for field in TOPOLOGY_CONTROL_TARGET_FIELDS] for row in y_rows], dtype=float)
    beta = np.array([coefficients[field] for field in TOPOLOGY_CONTROL_TARGET_FIELDS], dtype=float).T
    pred = x @ beta
    metrics = _control_model_error_metrics(y, pred)
    metrics["n"] = len(examples)
    for field in ("expected_delta_mean", "expected_ms_ssim_delta_mean"):
        index = TOPOLOGY_CONTROL_TARGET_FIELDS.index(field)
        truth_positive = y[:, index] > 0.0
        pred_positive = pred[:, index] - float(residual_std.get(field, 0.0)) > 0.0
        metrics[f"{field}_positive_rate"] = float(np.mean(pred_positive)) if pred_positive.size else 0.0
        metrics[f"{field}_positive_precision"] = (
            float(np.mean(truth_positive[pred_positive])) if np.any(pred_positive) else 0.0
        )
    return metrics


def _control_model_error_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for index, field in enumerate(TOPOLOGY_CONTROL_TARGET_FIELDS):
        err = pred[:, index] - y[:, index]
        metrics[f"{field}_mae"] = float(np.mean(np.abs(err))) if err.size else 0.0
        metrics[f"{field}_rmse"] = float(np.sqrt(np.mean(err * err))) if err.size else 0.0
        metrics[f"{field}_error_mean"] = float(np.mean(err)) if err.size else 0.0
    return metrics


def _control_model_targets_for_example(example: ResidualTeacherExample) -> dict[str, float]:
    flow_delta = example.flow_objective_score - example.base_objective_score
    energy_delta = example.flow_energy_objective_score - example.base_objective_score
    energy_extra = example.flow_energy_objective_score - max(example.base_objective_score, example.flow_objective_score)
    map_deltas = _example_best_stage_map_deltas(example)
    targets = {
        "selected_strength": _clamp(example.strength if math.isfinite(example.strength) else 0.25, 0.05, 0.55),
        "flow_smooth_sigma": _finite_or_default(example.flow_smooth_sigma, 1.2),
        "flow_attachment": _finite_or_default(example.flow_attachment, 6.0),
        "energy_exponent": _finite_or_default(example.energy_exponent, 0.8),
        "energy_sigma": _finite_or_default(example.energy_sigma, 2.0),
        "energy_ratio_low": _finite_or_default(example.energy_ratio_low, 0.65),
        "energy_ratio_high": _finite_or_default(example.energy_ratio_high, 1.32),
        "energy_clip_low": _finite_or_default(example.energy_clip_low, 0.78),
        "energy_clip_high": _finite_or_default(example.energy_clip_high, 1.22),
        "texture_mean_exponent": _finite_or_default(example.texture_mean_exponent, 0.0),
        "texture_exponent": _finite_or_default(example.texture_exponent, 0.0),
        "texture_deep_exponent": _finite_or_default(example.texture_deep_exponent, 0.0),
        "expected_delta_mean": example.best_objective_delta,
        "expected_flow_delta_mean": flow_delta,
        "expected_energy_delta_mean": energy_delta,
        "expected_energy_extra_mean": energy_extra,
        "expected_ms_ssim_delta_mean": example.best_delta,
        "expected_struct_delta_mean": _finite_or_default(map_deltas["struct"], 0.0),
        "expected_oac_delta_mean": _finite_or_default(map_deltas["oac"], 0.0),
        "expected_sc_delta_mean": _finite_or_default(map_deltas["sc"], 0.0),
        "expected_rsc_delta_mean": _finite_or_default(map_deltas["rsc"], 0.0),
    }
    bounds = {"selected_strength": (0.05, 0.55), **_stage2_control_bounds()}
    for field, (low, high) in bounds.items():
        targets[field] = _clamp(targets[field], low, high)
    return targets


def _control_model_has_map_targets(examples: list[ResidualTeacherExample]) -> bool:
    return any(
        all(math.isfinite(value) for value in _example_best_stage_map_deltas(example).values())
        for example in examples
        if example.best_stage != "base"
    )


def _control_model_has_aggregate_map_objective(examples: list[ResidualTeacherExample]) -> bool:
    return any("map_objective" in example.objective_source for example in examples)


def _control_model_features_for_example(
    example: ResidualTeacherExample,
    defaults: dict[str, float],
) -> list[float]:
    return _control_model_features(
        example.source_archive_path,
        base_ms=example.base_ms_ssim,
        base_lpips=example.base_lpips,
        rank=example.rank,
        defaults=defaults,
    )


def _control_model_features_for_row(
    row: dict[str, str],
    *,
    rank: int,
    defaults: dict[str, float],
) -> list[float]:
    return _control_model_features(
        row.get("source_archive_path", ""),
        base_ms=_finite(row.get("MS-SSIM")),
        base_lpips=_finite(row.get("LPIPS") or row.get("LPIPS_PROXY")),
        rank=rank,
        defaults=defaults,
    )


def _control_model_features(
    source_archive_path: str,
    *,
    base_ms: float,
    base_lpips: float,
    rank: int | float,
    defaults: dict[str, float],
) -> list[float]:
    traits = _source_traits(source_archive_path)
    base_ms = _finite_or_default(base_ms, float(defaults.get("base_ms_ssim", 0.65)))
    base_lpips = _finite_or_default(base_lpips, float(defaults.get("base_lpips", 0.58)))
    rank_value = _finite_or_default(float(rank), float(defaults.get("rank", 60.0)))
    frame_value = _finite_or_default(float(traits.get("frame", -1)), float(defaults.get("frame", 250.0)))
    sex = str(traits.get("sex", "")).lower()
    age = str(traits.get("age_band", ""))
    site = str(traits.get("body_site", "")).lower()
    ms_gap = _clamp((0.78 - base_ms) / 0.20, 0.0, 1.0)
    lpips_load = _clamp((base_lpips - 0.35) / 0.35, 0.0, 1.0)
    return [
        1.0,
        base_ms,
        base_lpips,
        _clamp(rank_value / 120.0, 0.0, 1.25),
        _clamp(frame_value / 500.0, 0.0, 1.25),
        1.0 if sex == "female" else 0.0,
        1.0 if sex == "male" else 0.0,
        1.0 if age == "1950-1960" else 0.0,
        1.0 if age == "1990-2000" else 0.0,
        1.0 if "cheek" in site else 0.0,
        1.0 if "eye" in site else 0.0,
        ms_gap,
        lpips_load,
    ]


def _predict_control_model_targets(
    row: dict[str, str],
    *,
    rank: int,
    coefficients: dict[str, list[float]],
    defaults: dict[str, float],
) -> dict[str, float]:
    features = np.array(_control_model_features_for_row(row, rank=rank, defaults=defaults), dtype=float)
    out: dict[str, float] = {}
    for field in TOPOLOGY_CONTROL_TARGET_FIELDS:
        beta = np.array(coefficients.get(field, [0.0] * len(TOPOLOGY_CONTROL_FEATURE_FIELDS)), dtype=float)
        out[field] = float(features @ beta)
    bounds = {"selected_strength": (0.05, 0.55), **_stage2_control_bounds()}
    for field, (low, high) in bounds.items():
        out[field] = _clamp(out[field], low, high)
    return out


def _control_model_stage2_controls(targets: dict[str, float]) -> dict[str, object]:
    controls = {
        "residual_control_policy": "topology_residual_control_model_v1",
        "energy_base_policy": "current_on_flow_regression",
        "energy_base_flow_delta_threshold": 0.0,
    }
    for field in STAGE2_NUMERIC_CONTROL_FIELDS:
        controls[field] = round(_finite_or_default(targets.get(field), 0.0), 6)
    return controls


def _control_model_map_safety(
    targets: dict[str, float],
    *,
    residual_std: dict[str, float],
    uncertainty_z: float,
    min_map_delta_lcb: float,
    map_target_observed: bool,
) -> dict[str, object]:
    if not map_target_observed:
        return _empty_map_safety_context("map_safety_not_trained", 0.65)
    out: dict[str, object] = {}
    lcbs: list[float] = []
    for label, target_field in (
        ("struct", "expected_struct_delta_mean"),
        ("oac", "expected_oac_delta_mean"),
        ("sc", "expected_sc_delta_mean"),
        ("rsc", "expected_rsc_delta_mean"),
    ):
        mean = _finite(targets.get(target_field))
        std = max(0.0, _finite(residual_std.get(target_field)))
        lcb = mean - float(uncertainty_z) * std if math.isfinite(mean) else float("nan")
        out[f"expected_{label}_delta_mean"] = mean if math.isfinite(mean) else ""
        out[f"expected_{label}_delta_lcb"] = lcb if math.isfinite(lcb) else ""
        if math.isfinite(lcb):
            lcbs.append(lcb)
    lcb_min = min(lcbs) if len(lcbs) == 4 else float("nan")
    if not math.isfinite(lcb_min):
        status = "map_safety_unknown"
        multiplier = 0.65
        safe_win_rate: float | str = ""
    elif lcb_min < min_map_delta_lcb:
        status = "map_safety_lcb_fail"
        multiplier = 0.35
        safe_win_rate = 0.0
    else:
        status = "map_safe_pass"
        multiplier = 1.0
        safe_win_rate = 1.0
    out.update(
        {
            "expected_map_delta_lcb_min": lcb_min if math.isfinite(lcb_min) else "",
            "nearest_map_safe_win_rate": safe_win_rate,
            "map_safe_support_n": "",
            "map_safety_status": status,
            "map_safety_score_multiplier": multiplier,
        }
    )
    return out


def _empty_surrogate_fields(status: str) -> dict[str, object]:
    return {
        "surrogate_calibration_status": status,
        "surrogate_calibration_support_n": "",
        "surrogate_calibration_distance": "",
        "surrogate_calibration_score_multiplier": 1.0,
        "surrogate_calibration_training_ssim_min": "",
        "surrogate_calibration_training_ssim_max": "",
    }


def _finite_or_default(value: object, default: float) -> float:
    finite = _finite(value)
    return finite if math.isfinite(finite) else default


def _ordered_union(rows: list[dict[str, object]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    return fields


def _recommendation_fieldnames() -> list[str]:
    return [
        "priority",
        "row_index",
        "current_rank",
        "source_archive_path",
        "current_ms_ssim",
        "current_ssim",
        "current_lpips",
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
        *SURROGATE_CALIBRATION_FIELDS,
        *STAGE2_CONTROL_FIELDS,
        "expected_delta_mean",
        "expected_delta_lcb",
        "expected_delta_uncertainty",
        "acquisition_score",
        *MAP_SAFETY_FIELDS,
        "expected_flow_delta_mean",
        "expected_energy_delta_mean",
        "expected_energy_extra_mean",
        "nearest_flow_win_rate",
        "nearest_energy_win_rate",
        "nearest_teacher_count",
        "nearest_teacher_sources",
        "selector_train_examples",
        "reference_png",
        "current_phantom_path",
        "current_synthetic_gray_png",
        "surrogate_gate_status",
        "evidence_scope",
    ]


def _budget_queue_fieldnames() -> list[str]:
    fields = _recommendation_fieldnames()
    additions = [
        "budget_slot",
        "selector_priority",
        "candidate_kind",
        "flow_strength_multiplier",
        "diversity_stratum",
        *TRUE_PROBE_FEEDBACK_FIELDS,
        *EXACT_PROBE_FEEDBACK_FIELDS,
        *ENERGY_GATE_FIELDS,
        "budget_score",
        "planned_api_stage",
        "total_api_budget",
        "flow_budget",
        "reserved_energy_followups",
        "promotion_rule",
    ]
    ordered = fields[:1] + additions[:2] + fields[1:7] + additions[2:5] + fields[7:] + additions[5:]
    return _dedupe_fields(ordered)


def _select_budget_candidates(
    candidates: list[dict[str, object]],
    flow_budget: int,
    *,
    diversity_fields: tuple[str, ...] = (),
    max_per_stratum: int | None = None,
) -> list[dict[str, object]]:
    if flow_budget <= 0 or not candidates:
        return []

    def score_key(row: dict[str, object]) -> tuple[float, int]:
        return (-float(row["budget_score"]), int(float(row["selector_priority"])))

    stratum_counts: dict[str, int] = {}
    planned: list[dict[str, object]] = []

    exploit = sorted([row for row in candidates if int(row["_variant_index"]) == 0], key=score_key)
    for row in exploit:
        if len(planned) >= flow_budget:
            break
        if _append_budget_row(
            planned,
            row,
            stratum_counts,
            diversity_fields=diversity_fields,
            max_per_stratum=max_per_stratum,
        ):
            continue
    remaining = flow_budget - len(planned)
    if remaining <= 0:
        return planned

    selected_ids = {id(row) for row in planned}
    explore_groups: dict[int, list[dict[str, object]]] = {}
    for row in candidates:
        variant_index = int(row["_variant_index"])
        if variant_index == 0 or id(row) in selected_ids:
            continue
        explore_groups.setdefault(variant_index, []).append(row)
    for rows in explore_groups.values():
        rows.sort(key=score_key)

    variant_order = sorted(explore_groups)
    while remaining > 0 and any(explore_groups.values()):
        made_progress = False
        for variant_index in variant_order:
            rows = explore_groups.get(variant_index, [])
            if not rows:
                continue
            appended = False
            while rows:
                row = rows.pop(0)
                if _append_budget_row(
                    planned,
                    row,
                    stratum_counts,
                    diversity_fields=diversity_fields,
                    max_per_stratum=max_per_stratum,
                ):
                    appended = True
                    break
            if not appended:
                continue
            remaining -= 1
            made_progress = True
            if remaining <= 0:
                break
        if not made_progress:
            break

    if remaining > 0:
        overflow = sorted([row for row in candidates if id(row) not in {id(item) for item in planned}], key=score_key)
        planned.extend(overflow[:remaining])
    return planned


def _append_budget_row(
    planned: list[dict[str, object]],
    row: dict[str, object],
    stratum_counts: dict[str, int],
    *,
    diversity_fields: tuple[str, ...],
    max_per_stratum: int | None,
) -> bool:
    if diversity_fields and max_per_stratum is not None:
        stratum = str(row.get("diversity_stratum") or _diversity_stratum(row, diversity_fields))
        if stratum_counts.get(stratum, 0) >= max(0, int(max_per_stratum)):
            return False
        stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1
    planned.append(row)
    return True


def _diversity_stratum(row: dict[str, object], fields: tuple[str, ...]) -> str:
    traits = _source_traits(str(row.get("source_archive_path", "")))
    values: list[str] = []
    for field in fields:
        if field == "sex":
            values.append(str(traits.get("sex", "")) or "unknown_sex")
        elif field in {"age", "age_band"}:
            values.append(str(traits.get("age_band", "")) or "unknown_age")
        elif field in {"site", "body_site"}:
            values.append(str(traits.get("body_site", "")) or "unknown_site")
        elif field == "rank_bucket":
            values.append(_rank_bucket(row.get("current_rank")))
        else:
            values.append(str(row.get(field, "")) or f"unknown_{field}")
    return "|".join(values)


def _rank_bucket(value: object, width: int = 12) -> str:
    rank = _finite(value)
    if not math.isfinite(rank):
        return "rank_unknown"
    start = int((max(1, int(rank)) - 1) // width) * width + 1
    end = start + width - 1
    return f"rank_{start:03d}_{end:03d}"


def _count_values(rows: list[dict[str, object]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _candidate_kind(multiplier: float, variant_index: int) -> str:
    if variant_index == 0 or abs(float(multiplier) - 1.0) <= 1e-9:
        return "exploit"
    return "explore_low_strength" if float(multiplier) < 1.0 else "explore_high_strength"


def _variant_value(values: tuple[float, ...], variant_index: int, *, default: float) -> float:
    if not values:
        return default
    return float(values[min(variant_index, len(values) - 1)])


def _dedupe_fields(fields: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for field in fields:
        if field in seen:
            continue
        seen.add(field)
        out.append(field)
    return out


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _first_finite(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        value = _finite(row.get(key))
        if math.isfinite(value):
            return value
    return float("nan")


def _finite(value: str | float | None) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _finite_or_zero(value: object) -> float:
    number = _finite(value)  # type: ignore[arg-type]
    return number if math.isfinite(number) else 0.0


def _finite_or_blank(value: object) -> float | str:
    number = _finite(value)  # type: ignore[arg-type]
    return number if math.isfinite(number) else ""


def _row_float(row: dict[str, str], key: str, default: float) -> float:
    value = _finite(row.get(key))
    return value if math.isfinite(value) else float(default)


def _first_int(row: dict[str, str], *keys: str) -> int:
    for key in keys:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            return int(float(value))
        except ValueError:
            continue
    return -1


def _objective_source(row: dict[str, str]) -> str:
    for prefix in ("flow_energy", "flow", "current", "base", "rank120"):
        if any(math.isfinite(_finite(row.get(_metric_key(prefix, metric)))) for metric in ("Struct_MS-SSIM", "OAC_MS-SSIM", "SC_MS-SSIM", "RSC_MS-SSIM")):
            return "multi_objective_maps"
        if math.isfinite(_finite(row.get(_aggregate_objective_key(prefix)))):
            return "multi_objective_map_objective"
    return "plain_ms_ssim"


def _stage_objective(row: dict[str, str], prefixes: tuple[str, ...], fallback_ms_ssim: float) -> float:
    ms_values: list[float] = []
    lpips_values: list[float] = []
    for metric in ("Struct_MS-SSIM", "OAC_MS-SSIM", "SC_MS-SSIM", "RSC_MS-SSIM"):
        value = _first_prefixed_finite(row, prefixes, metric)
        if math.isfinite(value):
            ms_values.append(value)
    for metric in ("Struct_LPIPS", "OAC_LPIPS", "SC_LPIPS", "RSC_LPIPS"):
        value = _first_prefixed_finite(row, prefixes, metric)
        if math.isfinite(value):
            lpips_values.append(1.0 - value)
    if ms_values or lpips_values:
        return statistics.mean(ms_values + lpips_values)
    aggregate = _first_aggregate_objective(row, prefixes)
    if math.isfinite(aggregate):
        return aggregate
    return fallback_ms_ssim


def _stage_map_deltas(
    row: dict[str, str],
    stage_prefixes: tuple[str, ...],
    base_prefixes: tuple[str, ...],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for label, metric in MAP_COMPONENTS:
        stage_value = _first_prefixed_finite(row, stage_prefixes, metric)
        base_value = _first_prefixed_finite(row, base_prefixes, metric)
        out[label] = stage_value - base_value if math.isfinite(stage_value) and math.isfinite(base_value) else float("nan")
    return out


def _first_aggregate_objective(row: dict[str, str], prefixes: tuple[str, ...]) -> float:
    for prefix in prefixes:
        value = _finite(row.get(_aggregate_objective_key(prefix)))
        if math.isfinite(value):
            return value
    return float("nan")


def _aggregate_objective_key(prefix: str) -> str:
    if prefix in {"base", "rank120"}:
        prefix = "current"
    if prefix == "energy":
        prefix = "flow_energy"
    return f"{prefix}_map_objective_score"


def _first_prefixed_finite(row: dict[str, str], prefixes: tuple[str, ...], metric: str) -> float:
    for prefix in prefixes:
        value = _finite(row.get(_metric_key(prefix, metric)))
        if math.isfinite(value):
            return value
    value = _finite(row.get(metric))
    return value if math.isfinite(value) else float("nan")


def _metric_key(prefix: str, metric: str) -> str:
    return f"{prefix}_{metric}"
