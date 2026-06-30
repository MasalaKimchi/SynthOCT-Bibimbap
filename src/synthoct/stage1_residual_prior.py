from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from .evaluation.maps import evaluate_feature_map_metrics, profile_scores
from .evaluation.metrics import calculate_metrics
from .features import load_scan
from .generators.phantom_generators import learned_prior_fields
from .learned_surrogate import field_to_phantom, phantom_to_field
from .neural_prior import reference_feature_stack
from .phantom import ExperimentConfig, load_phantom, save_phantom


TRUE_SCANNER_SOURCES = {"hosted_api_true_scanner", "official_windows_true_scanner", ""}
MAP_METRIC_KEYS = ("Struct_MS-SSIM", "OAC_MS-SSIM", "SC_MS-SSIM", "RSC_MS-SSIM")


def train_stage1_residual_prior(
    teacher_metrics_paths: list[str | Path],
    out_path: str | Path,
    *,
    base_api_metrics: str | Path | None = None,
    shape: tuple[int, int] = (128, 256),
    epochs: int = 80,
    learning_rate: float = 2e-3,
    seed: int = 53,
    min_positive_flow_delta: float = 0.0,
    residual_clip: float = 0.55,
    residual_scale: float = 0.28,
    include_regression_anchors: bool = True,
) -> Path:
    """Train a bounded Stage 1 residual field model around the p140/t32 prior.

    The model predicts low-frequency density and energy log-residuals relative
    to a known base phantom field. It is training material only: generated
    phantoms still require true-scanner validation before promotion.
    """
    try:
        import torch
        from torch.nn import functional as F
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("Training a Stage 1 residual prior requires torch.") from exc

    torch.manual_seed(seed)
    np.random.seed(seed)
    base_lookup = _base_lookup(base_api_metrics)
    examples = _load_stage1_examples(
        teacher_metrics_paths,
        base_lookup=base_lookup,
        shape=shape,
        min_positive_flow_delta=min_positive_flow_delta,
        residual_clip=residual_clip,
        include_regression_anchors=include_regression_anchors,
    )
    if not examples:
        raise RuntimeError("No usable Stage 1 residual teacher examples found.")

    x_np = np.stack([example["features"] for example in examples], axis=0).astype(np.float32)
    y_np = np.stack([example["target"] for example in examples], axis=0).astype(np.float32)
    weights_np = np.array([example["weight"] for example in examples], dtype=np.float32)

    device = torch.device("cpu")
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    w = torch.from_numpy(weights_np).to(device).view(-1, 1, 1, 1)

    model = _Stage1ResidualNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    kernel = 9
    for _epoch in range(max(1, int(epochs))):
        pred = _lowpass(model(x), kernel=kernel, functional=F)
        loss = ((pred - y).abs() * w).mean() + 0.16 * _gradient_loss(pred, y) + 0.025 * _smoothness(pred)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = _lowpass(model(x), kernel=kernel, functional=F)
        residual_mae = float(((pred - y).abs() * w).mean().cpu().item())
        zero_mae = float((y.abs() * w).mean().cpu().item())
        shrinkage_ratio = residual_mae / max(zero_mae, 1e-8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_state": model.state_dict(),
        "model_name": "Stage1ResidualNet",
        "input_channels": [
            "reference_struct",
            "reference_detail",
            "reference_oac",
            "reference_speckle",
            "base_density",
            "base_energy",
            "base_mean_energy",
            "base_cumulative_energy",
        ],
        "output_channels": ["density_log_residual", "energy_log_residual"],
        "shape": tuple(int(v) for v in shape),
        "teacher_metrics_paths": [str(Path(path)) for path in teacher_metrics_paths],
        "base_api_metrics": str(Path(base_api_metrics)) if base_api_metrics else "",
        "example_count": len(examples),
        "positive_example_count": sum(1 for example in examples if example["example_kind"] == "positive_flow"),
        "regression_anchor_count": sum(1 for example in examples if example["example_kind"] == "regression_anchor"),
        "residual_clip": float(residual_clip),
        "residual_scale": float(residual_scale),
        "min_positive_flow_delta": float(min_positive_flow_delta),
        "final_weighted_residual_mae": residual_mae,
        "zero_residual_baseline_mae": zero_mae,
        "training_shrinkage_ratio": shrinkage_ratio,
        "training_rows": [
            {key: example[key] for key in _EXAMPLE_METADATA_FIELDS}
            for example in examples
        ],
        "evidence_source": "stage1_residual_prior_model_training",
        "evidence_scope": "not_challenge_evidence",
        "promotion_allowed_without_true_scanner": False,
        "hidden_holdout_final_score": False,
        "interpretation": (
            "Physics-constrained Stage 1 residual model around learned-prior-sparse-p140-t32. "
            "It predicts bounded low-frequency field residuals; generated phantoms require hosted true-scanner validation."
        ),
    }
    torch.save(artifact, out_path)
    metadata_path = out_path.with_suffix(".json")
    metadata = {key: value for key, value in artifact.items() if key != "model_state"}
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def stage1_residual_prior_phantom(
    input_path: str | Path,
    output_path: str | Path,
    learned_artifact_path: str | Path | None,
    residual_artifact_path: str | Path,
    *,
    base_phantom_path: str | Path | None = None,
    preserve_scatterers: bool = False,
    preserve_depth_profile: bool = False,
    preserve_local_profile: bool = False,
    seed: int = 7,
    scatterers_count: int = 300_000,
    target_blend: float = 0.62,
    prior_blend: float = 0.38,
    texture_weight: float = 0.32,
    density_power: float = 14.0,
    residual_blend: float = 1.0,
    energy_sigma: float = 0.10,
) -> Path:
    """Generate a scanner-compatible phantom with a trained Stage 1 residual."""
    if preserve_scatterers:
        if not base_phantom_path:
            raise ValueError("preserve_scatterers requires base_phantom_path.")
        return _stage1_residual_preserve_scatterers(
            input_path,
            base_phantom_path,
            output_path,
            residual_artifact_path,
            residual_blend=residual_blend,
            preserve_depth_profile=preserve_depth_profile,
            preserve_local_profile=preserve_local_profile,
        )
    density, energy = stage1_residual_prior_fields(
        input_path,
        learned_artifact_path,
        residual_artifact_path,
        base_phantom_path=base_phantom_path,
        target_blend=target_blend,
        prior_blend=prior_blend,
        texture_weight=texture_weight,
        density_power=density_power,
        residual_blend=residual_blend,
    )
    return field_to_phantom(
        density,
        energy,
        output_path,
        scatterers_count=scatterers_count,
        seed=seed,
        energy_floor=0.002,
        energy_ceiling=0.075,
    )


def enrich_stage1_residual_smoke_metrics(
    candidate_metrics: str | Path,
    candidate_queue: str | Path,
    base_api_metrics: str | Path,
    out_path: str | Path,
    *,
    maps_dir: str | Path | None = None,
    include_lpips: bool = False,
) -> Path:
    """Compare Stage 1 residual smoke rows against their current p140/t32 rows.

    This is an evidence report, not promotion logic. It exists to keep Stage 1
    residual experiments aligned with Struct/OAC/SC/RSC guardrails before any
    larger API tranche is considered.
    """
    metrics_rows = _read_rows(candidate_metrics)
    queue_rows = {row["method"]: row for row in _read_rows(candidate_queue)}
    base_rows = {row["source_archive_path"]: row for row in _read_rows(base_api_metrics)}
    out_path = Path(out_path)
    maps_root = Path(maps_dir) if maps_dir else out_path.with_suffix("").with_name(f"{out_path.stem}_maps")
    enriched: list[dict[str, object]] = []
    for idx, metric_row in enumerate(metrics_rows, start=1):
        method = metric_row.get("method", "")
        queue_row = queue_rows.get(method, {})
        source = queue_row.get("source_archive_path") or metric_row.get("source_archive_path", "")
        base_row = base_rows.get(source, {})
        reference_png = _existing_path(metric_row.get("reference_png") or queue_row.get("reference_png") or base_row.get("reference_png"))
        candidate_png = _existing_path(metric_row.get("synthetic_gray_png"))
        current_png = _existing_path(base_row.get("synthetic_gray_png"))
        current_metrics = _stage_map_metrics("current", reference_png, current_png, maps_root / f"row_{idx:04d}" / "current", include_lpips=include_lpips)
        candidate_stage_metrics = _stage_map_metrics("stage1", reference_png, candidate_png, maps_root / f"row_{idx:04d}" / "stage1", include_lpips=include_lpips)
        delta_metrics = _stage_delta_metrics(current_metrics, candidate_stage_metrics)
        enriched.append(
            {
                **metric_row,
                **queue_row,
                "source_archive_path": source,
                "current_ms_ssim": base_row.get("MS-SSIM", ""),
                "stage1_ms_ssim": metric_row.get("MS-SSIM", ""),
                "stage1_delta_vs_current": _delta(_finite(metric_row.get("MS-SSIM")), _finite(base_row.get("MS-SSIM"))),
                **current_metrics,
                **candidate_stage_metrics,
                **delta_metrics,
                "guardrail_pass": _guardrail_pass(delta_metrics),
                "promotion_allowed_without_true_scanner": False,
                "hidden_holdout_final_score": False,
                "evidence_scope": "stage1_residual_smoke_map_enrichment_not_hidden_holdout",
            }
        )
    _write_rows(out_path, enriched)
    _write_stage1_summary(out_path.with_suffix(".json"), enriched, candidate_metrics, candidate_queue, base_api_metrics)
    return out_path


def select_stage1_residual_variants(
    enriched_metrics_paths: list[str | Path],
    out_queue_path: str | Path,
    *,
    out_summary_path: str | Path | None = None,
    max_rows: int | None = None,
    min_struct_delta: float = 0.0,
    min_oac_delta: float = -0.001,
    min_sc_delta: float = -0.00025,
    min_rsc_delta: float = -0.00025,
    struct_weight: float = 1.0,
    oac_weight: float = 1.6,
    sc_weight: float = 0.7,
    rsc_weight: float = 0.7,
    lpips_weight: float = 0.2,
    require_true_scanner: bool = True,
) -> Path:
    """Select one Stage 1 residual variant per source under map guardrails.

    The selected queue is intentionally conservative evidence plumbing. It can
    feed a follow-up scanner run, but it is not a hidden-holdout promotion.
    """
    candidates: list[dict[str, object]] = []
    for path in enriched_metrics_paths:
        variant_set = Path(path).parent.name or Path(path).stem
        for row in _read_rows(path):
            enriched = dict(row)
            enriched["variant_set"] = variant_set
            enriched["enriched_metrics_path"] = str(Path(path))
            enriched["stage1_multi_objective_score"] = _stage1_variant_score(
                enriched,
                struct_weight=struct_weight,
                oac_weight=oac_weight,
                sc_weight=sc_weight,
                rsc_weight=rsc_weight,
                lpips_weight=lpips_weight,
            )
            enriched["stage1_selection_status"] = _stage1_selection_status(
                enriched,
                min_struct_delta=min_struct_delta,
                min_oac_delta=min_oac_delta,
                min_sc_delta=min_sc_delta,
                min_rsc_delta=min_rsc_delta,
                require_true_scanner=require_true_scanner,
            )
            candidates.append(enriched)

    eligible = [row for row in candidates if row["stage1_selection_status"] == "selected_candidate"]
    by_source: dict[str, dict[str, object]] = {}
    for row in eligible:
        source = str(row.get("source_archive_path", ""))
        if not source:
            continue
        current = by_source.get(source)
        if current is None or _finite(row.get("stage1_multi_objective_score")) > _finite(current.get("stage1_multi_objective_score")):
            by_source[source] = row

    selected = sorted(
        by_source.values(),
        key=lambda row: (
            _finite(row.get("stage1_multi_objective_score")),
            _finite(row.get("delta_Struct_MS-SSIM")),
            _finite(row.get("stage1_ms_ssim") or row.get("MS-SSIM")),
        ),
        reverse=True,
    )
    if max_rows is not None:
        selected = selected[: max(0, int(max_rows))]

    selected_rows = [_stage1_selected_queue_row(row, idx) for idx, row in enumerate(selected, start=1)]
    out_queue_path = _write_rows(out_queue_path, selected_rows)
    summary_path = Path(out_summary_path) if out_summary_path else Path(out_queue_path).with_suffix(".json")
    _write_stage1_selection_summary(
        summary_path,
        candidates,
        selected_rows,
        enriched_metrics_paths,
        thresholds={
            "min_struct_delta": min_struct_delta,
            "min_oac_delta": min_oac_delta,
            "min_sc_delta": min_sc_delta,
            "min_rsc_delta": min_rsc_delta,
            "require_true_scanner": require_true_scanner,
        },
        weights={
            "struct_weight": struct_weight,
            "oac_weight": oac_weight,
            "sc_weight": sc_weight,
            "rsc_weight": rsc_weight,
            "lpips_weight": lpips_weight,
        },
    )
    return out_queue_path


def plan_stage1_residual_validation_queue(
    base_api_metrics: str | Path,
    residual_artifact_path: str | Path,
    out_queue_path: str | Path,
    *,
    exclude_metrics_paths: list[str | Path] | None = None,
    max_sources: int = 8,
    residual_blends: list[float] | tuple[float, ...] = (1.0,),
    include_depth_profile_variant: bool = True,
    include_local_profile_variant: bool = False,
    variant_modes: list[str] | tuple[str, ...] | None = None,
    exclude_residual_training_sources: bool = True,
    min_base_ms_ssim: float | None = None,
) -> Path:
    """Build held-out Stage 1 residual phantoms from base p140/t32 rows."""
    exclude_sources = _collect_sources(exclude_metrics_paths or [])
    if exclude_residual_training_sources:
        exclude_sources.update(_collect_residual_artifact_training_sources(residual_artifact_path))
    rows = []
    for row in _read_rows(base_api_metrics):
        source = row.get("source_archive_path", "")
        if not source or source in exclude_sources:
            continue
        base_ms = _finite(row.get("MS-SSIM"))
        if min_base_ms_ssim is not None and (not math.isfinite(base_ms) or base_ms < min_base_ms_ssim):
            continue
        reference_png = _existing_path(row.get("reference_png"))
        base_phantom = _existing_path(row.get("phantom_path"))
        if not reference_png or not base_phantom:
            continue
        rows.append({**row, "base_ms_ssim": base_ms, "reference_png": str(reference_png), "base_phantom_path": str(base_phantom)})
    rows.sort(key=lambda row: _finite(row.get("base_ms_ssim")), reverse=True)
    rows = rows[: max(0, int(max_sources))]

    out_queue_path = Path(out_queue_path)
    phantom_dir = out_queue_path.with_suffix("").with_name(f"{out_queue_path.stem}_phantoms")
    queue_rows: list[dict[str, object]] = []
    priority = 1
    for source_idx, row in enumerate(rows, start=1):
        modes = _stage1_variant_modes(
            variant_modes,
            include_depth_profile_variant=include_depth_profile_variant,
            include_local_profile_variant=include_local_profile_variant,
        )
        for preserve_depth, preserve_local in modes:
            for blend in residual_blends:
                method = _stage1_validation_method(source_idx, blend, preserve_depth, preserve_local)
                phantom_path = phantom_dir / f"{method}.txt"
                stage1_residual_prior_phantom(
                    row["reference_png"],
                    phantom_path,
                    None,
                    residual_artifact_path,
                    base_phantom_path=row["base_phantom_path"],
                    preserve_scatterers=True,
                    preserve_depth_profile=preserve_depth,
                    preserve_local_profile=preserve_local,
                    residual_blend=float(blend),
                )
                queue_rows.append(
                    {
                        "priority": priority,
                        "method": method,
                        "phantom_path": str(phantom_path.resolve()),
                        "reference_png": row["reference_png"],
                        "source_archive_path": row.get("source_archive_path", ""),
                        "base_phantom_path": row["base_phantom_path"],
                        "base_ms_ssim": row.get("base_ms_ssim", ""),
                        "residual_blend": float(blend),
                        "preserve_depth_profile": preserve_depth,
                        "preserve_local_profile": preserve_local,
                        "excluded_source_count": len(exclude_sources),
                        "evidence_scope": "stage1_residual_validation_queue_requires_true_scanner",
                        "promotion_allowed_without_true_scanner": False,
                        "hidden_holdout_final_score": False,
                    }
                )
                priority += 1

    queue_path = _write_rows(out_queue_path, queue_rows)
    _write_stage1_validation_queue_summary(
        queue_path.with_suffix(".json"),
        queue_rows,
        base_api_metrics,
        residual_artifact_path,
        exclude_metrics_paths or [],
        max_sources=max_sources,
        residual_blends=residual_blends,
        include_depth_profile_variant=include_depth_profile_variant,
        include_local_profile_variant=include_local_profile_variant,
        variant_modes=variant_modes,
        exclude_residual_training_sources=exclude_residual_training_sources,
        min_base_ms_ssim=min_base_ms_ssim,
    )
    return queue_path


def stage2_texture_overlay_phantom(
    input_path: str | Path,
    base_phantom_path: str | Path,
    output_path: str | Path,
    *,
    replace_fraction: float = 0.015,
    energy_quantile: float = 0.72,
    texture_weight: float = 0.55,
    seed: int = 71,
    match_depth_profile: bool = True,
) -> Path:
    """Preserve p140/t32 topology while replacing a small low-energy tail.

    This Stage 2 probe injects target-guided speckle texture as a residual on
    top of a stable base phantom. It keeps scatterer count and total energy
    fixed, then optionally matches the base depth-energy profile.
    """
    config = ExperimentConfig()
    base = load_phantom(base_phantom_path)
    count = len(base)
    replace_count = int(np.clip(round(count * float(replace_fraction)), 1, max(1, count // 5)))
    rng = np.random.default_rng(seed)
    replace_pool = np.argsort(base[:, 3])[: max(replace_count * 4, replace_count)]
    replace_idx = rng.choice(replace_pool, size=replace_count, replace=False)
    overlay = _target_texture_scatterers(
        input_path,
        base,
        replace_count,
        config,
        energy_quantile=energy_quantile,
        texture_weight=texture_weight,
        rng=rng,
    )
    shaped = base.copy()
    shaped[replace_idx] = overlay
    if match_depth_profile:
        shaped = _match_depth_energy_profile(base, shaped, config, strict=True)
    shaped[:, 3] *= base[:, 3].sum() / max(float(shaped[:, 3].sum()), 1e-12)
    shaped[:, 3] = np.clip(shaped[:, 3], 0.001, 100.0)
    return save_phantom(shaped, output_path, config=ExperimentConfig(scatterers_count=len(shaped)))


def plan_stage2_texture_overlay_validation_queue(
    base_api_metrics: str | Path,
    out_queue_path: str | Path,
    *,
    max_sources: int = 6,
    replace_fractions: list[float] | tuple[float, ...] = (0.005, 0.015, 0.035),
    texture_weights: list[float] | tuple[float, ...] = (0.35, 0.65),
    energy_quantile: float = 0.72,
    seed: int = 71,
    min_base_ms_ssim: float | None = None,
) -> Path:
    """Build scanner validation rows for Stage 2 target-guided texture overlays."""
    rows = []
    for row in _read_rows(base_api_metrics):
        base_ms = _finite(row.get("MS-SSIM"))
        if min_base_ms_ssim is not None and (not math.isfinite(base_ms) or base_ms < min_base_ms_ssim):
            continue
        reference_png = _existing_path(row.get("reference_png"))
        base_phantom = _existing_path(row.get("phantom_path"))
        source = row.get("source_archive_path", "")
        if not reference_png or not base_phantom or not source:
            continue
        rows.append({**row, "base_ms_ssim": base_ms, "reference_png": str(reference_png), "base_phantom_path": str(base_phantom)})
    rows.sort(key=lambda row: _finite(row.get("base_ms_ssim")), reverse=True)
    rows = rows[: max(0, int(max_sources))]

    out_queue_path = Path(out_queue_path)
    phantom_dir = out_queue_path.with_suffix("").with_name(f"{out_queue_path.stem}_phantoms")
    queue_rows: list[dict[str, object]] = []
    priority = 1
    for source_idx, row in enumerate(rows, start=1):
        for replace_fraction in replace_fractions:
            for texture_weight in texture_weights:
                method = _stage2_overlay_method(source_idx, replace_fraction, texture_weight)
                phantom_path = phantom_dir / f"{method}.txt"
                stage2_texture_overlay_phantom(
                    row["reference_png"],
                    row["base_phantom_path"],
                    phantom_path,
                    replace_fraction=float(replace_fraction),
                    texture_weight=float(texture_weight),
                    energy_quantile=energy_quantile,
                    seed=seed + priority,
                    match_depth_profile=True,
                )
                queue_rows.append(
                    {
                        "priority": priority,
                        "method": method,
                        "phantom_path": str(phantom_path.resolve()),
                        "reference_png": row["reference_png"],
                        "source_archive_path": row.get("source_archive_path", ""),
                        "base_phantom_path": row["base_phantom_path"],
                        "base_ms_ssim": row.get("base_ms_ssim", ""),
                        "replace_fraction": float(replace_fraction),
                        "texture_weight": float(texture_weight),
                        "energy_quantile": float(energy_quantile),
                        "evidence_scope": "stage2_texture_overlay_validation_queue_requires_true_scanner",
                        "promotion_allowed_without_true_scanner": False,
                        "hidden_holdout_final_score": False,
                    }
                )
                priority += 1
    queue_path = _write_rows(out_queue_path, queue_rows)
    _write_stage2_overlay_queue_summary(
        queue_path.with_suffix(".json"),
        queue_rows,
        base_api_metrics,
        max_sources=max_sources,
        replace_fractions=replace_fractions,
        texture_weights=texture_weights,
        energy_quantile=energy_quantile,
        min_base_ms_ssim=min_base_ms_ssim,
    )
    return queue_path


def stage1_residual_prior_fields(
    input_path: str | Path,
    learned_artifact_path: str | Path | None,
    residual_artifact_path: str | Path,
    *,
    base_phantom_path: str | Path | None = None,
    target_blend: float = 0.62,
    prior_blend: float = 0.38,
    texture_weight: float = 0.32,
    density_power: float = 14.0,
    residual_blend: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return p140/t32 density and energy fields after bounded Stage 1 residuals."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("Applying a Stage 1 residual prior requires torch.") from exc

    artifact = torch.load(residual_artifact_path, map_location="cpu", weights_only=False)
    shape = tuple(int(v) for v in artifact["shape"])
    if base_phantom_path:
        base_field = phantom_to_field(base_phantom_path, shape=shape)
        base_density = base_field[0]
        base_energy = base_field[1]
    else:
        if learned_artifact_path is None:
            raise ValueError("Either learned_artifact_path or base_phantom_path is required.")
        base_density, base_energy = learned_prior_fields(
            input_path,
            learned_artifact_path,
            shape=shape,
            target_blend=target_blend,
            prior_blend=prior_blend,
            texture_weight=texture_weight,
            density_power=density_power,
        )
        base_field = _field_stack_from_density_energy(base_density, base_energy)
    features = np.concatenate([reference_feature_stack(input_path, shape=shape), base_field], axis=0)

    model = _Stage1ResidualNet()
    model.load_state_dict(artifact["model_state"])
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(features[None].astype(np.float32))).cpu().numpy()[0]
    pred = _lowpass_np(pred, kernel=9)
    residual_scale = float(artifact.get("residual_scale", 0.28)) * float(np.clip(residual_blend, 0.0, 1.5))
    residual_clip = float(artifact.get("residual_clip", 0.55))
    density = _normalize(base_density * np.exp(np.clip(pred[0], -residual_clip, residual_clip) * residual_scale), floor=1e-8)
    energy = np.clip(base_energy * np.exp(np.clip(pred[1], -residual_clip, residual_clip) * residual_scale), 0.001, 0.12)
    return density, energy


def _load_stage1_examples(
    teacher_metrics_paths: list[str | Path],
    *,
    base_lookup: dict[str, dict[str, str]],
    shape: tuple[int, int],
    min_positive_flow_delta: float,
    residual_clip: float,
    include_regression_anchors: bool,
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for metrics_path in teacher_metrics_paths:
        for row in _read_rows(metrics_path):
            if row.get("evidence_source", "") not in TRUE_SCANNER_SOURCES:
                continue
            source = row.get("source_archive_path", "")
            base_row = base_lookup.get(source, {})
            reference_path = _existing_path(row.get("reference_png") or base_row.get("reference_png"))
            base_phantom_path = _existing_path(
                row.get("current_phantom_path") or row.get("base_phantom_path") or base_row.get("phantom_path")
            )
            flow_phantom_path = _existing_path(row.get("flow_phantom_path") or row.get("phantom_path"))
            if not reference_path or not base_phantom_path or not flow_phantom_path:
                continue
            flow_delta = _first_finite(row, "flow_delta_vs_current", "flow_delta")
            if not math.isfinite(flow_delta):
                flow_ms = _first_finite(row, "flow_ms_ssim")
                base_ms = _first_finite(row, "current_ms_ssim", "base_ms_ssim", "MS-SSIM")
                flow_delta = flow_ms - base_ms if math.isfinite(flow_ms) and math.isfinite(base_ms) else float("nan")
            use_flow_target = math.isfinite(flow_delta) and flow_delta >= float(min_positive_flow_delta)
            if not use_flow_target and not include_regression_anchors:
                continue
            target_phantom_path = flow_phantom_path if use_flow_target else base_phantom_path
            key = (str(reference_path), str(base_phantom_path), str(target_phantom_path))
            if key in seen:
                continue
            seen.add(key)

            base_field = phantom_to_field(base_phantom_path, shape=shape)
            target_field = phantom_to_field(target_phantom_path, shape=shape)
            residual = _residual_target(base_field, target_field, residual_clip=residual_clip)
            features = np.concatenate([reference_feature_stack(reference_path, shape=shape), base_field], axis=0)
            example_kind = "positive_flow" if use_flow_target else "regression_anchor"
            weight = _example_weight(row, flow_delta=flow_delta, example_kind=example_kind)
            examples.append(
                {
                    "features": features,
                    "target": residual,
                    "weight": weight,
                    "example_kind": example_kind,
                    "source_archive_path": source,
                    "reference_png": str(reference_path),
                    "base_phantom_path": str(base_phantom_path),
                    "target_phantom_path": str(target_phantom_path),
                    "flow_delta": flow_delta if math.isfinite(flow_delta) else "",
                    "flow_map_delta": _first_finite(row, "flow_map_delta_vs_current"),
                    "teacher_metrics_path": str(Path(metrics_path)),
                }
            )
    return examples


def _base_lookup(path: str | Path | None) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    lookup: dict[str, dict[str, str]] = {}
    for row in _read_rows(path):
        source = row.get("source_archive_path", "")
        if source:
            lookup[source] = row
    return lookup


def _residual_target(base_field: np.ndarray, target_field: np.ndarray, *, residual_clip: float) -> np.ndarray:
    density = np.log(np.maximum(target_field[0], 1e-6) / np.maximum(base_field[0], 1e-6))
    energy = np.log(np.maximum(target_field[1], 1e-6) / np.maximum(base_field[1], 1e-6))
    residual = np.stack([density, energy], axis=0)
    residual = gaussian_filter(residual, sigma=(0.0, 2.0, 3.0))
    return np.clip(residual, -float(residual_clip), float(residual_clip)).astype(np.float32)


def _example_weight(row: dict[str, str], *, flow_delta: float, example_kind: str) -> float:
    flow_map_delta = _first_finite(row, "flow_map_delta_vs_current")
    if example_kind == "regression_anchor":
        severity = abs(flow_delta) if math.isfinite(flow_delta) and flow_delta < 0 else 0.0
        return float(np.clip(0.75 + 55.0 * severity, 0.75, 2.0))
    gain = max(0.0, flow_delta if math.isfinite(flow_delta) else 0.0)
    map_gain = max(0.0, flow_map_delta if math.isfinite(flow_map_delta) else 0.0)
    return float(np.clip(0.85 + 70.0 * gain + 25.0 * map_gain, 0.85, 2.5))


def _field_stack_from_density_energy(density: np.ndarray, energy: np.ndarray) -> np.ndarray:
    density = _normalize(density, floor=1e-8)
    energy = _normalize(energy, floor=1e-8)
    mean_energy = _normalize(energy / np.maximum(density, 1e-6), floor=1e-8)
    cumulative = np.cumsum(energy, axis=0)
    cumulative = _normalize(cumulative, floor=1e-8)
    return np.stack([density, energy, mean_energy, cumulative], axis=0).astype(np.float32)


def _stage1_residual_preserve_scatterers(
    input_path: str | Path,
    base_phantom_path: str | Path,
    output_path: str | Path,
    residual_artifact_path: str | Path,
    *,
    residual_blend: float,
    preserve_depth_profile: bool,
    preserve_local_profile: bool,
) -> Path:
    _density, energy = stage1_residual_prior_fields(
        input_path,
        None,
        residual_artifact_path,
        base_phantom_path=base_phantom_path,
        residual_blend=residual_blend,
    )
    config = ExperimentConfig()
    data = load_phantom(base_phantom_path)
    xb = np.clip(((data[:, 0] / config.x_max) + 0.5) * energy.shape[1], 0, energy.shape[1] - 1).astype(int)
    zb = np.clip((data[:, 2] / config.z_max) * energy.shape[0], 0, energy.shape[0] - 1).astype(int)
    base_field = phantom_to_field(base_phantom_path, shape=energy.shape)
    residual = np.log(np.maximum(energy, 1e-6) / np.maximum(base_field[1], 1e-6))
    rows = np.arange(energy.shape[0], dtype=np.float64)[:, None]
    tissue_gate = 1.0 / (1.0 + np.exp(-(rows - energy.shape[0] * 0.11) / max(1.0, energy.shape[0] * 0.02)))
    scale = np.exp(np.clip(residual * tissue_gate, -0.35, 0.35)[zb, xb])
    shaped = data.copy()
    shaped[:, 3] = np.clip(shaped[:, 3] * scale, 0.001, 100.0)
    if preserve_local_profile:
        shaped = _match_local_energy_profile(data, shaped, config)
    if preserve_depth_profile:
        shaped = _match_depth_energy_profile(data, shaped, config, strict=preserve_local_profile)
    shaped[:, 3] *= data[:, 3].sum() / max(float(shaped[:, 3].sum()), 1e-12)
    shaped[:, 3] = np.clip(shaped[:, 3], 0.001, 100.0)
    return save_phantom(shaped, output_path, config=ExperimentConfig(scatterers_count=len(shaped)))


def _match_depth_energy_profile(base: np.ndarray, shaped: np.ndarray, config: ExperimentConfig, *, strict: bool = False) -> np.ndarray:
    z_base = np.clip((base[:, 2] / config.z_max) * config.n_depth, 0, config.n_depth - 1).astype(int)
    z_shaped = np.clip((shaped[:, 2] / config.z_max) * config.n_depth, 0, config.n_depth - 1).astype(int)
    base_sum = np.bincount(z_base, weights=base[:, 3], minlength=config.n_depth)
    shaped_sum = np.bincount(z_shaped, weights=shaped[:, 3], minlength=config.n_depth)
    ratio = np.ones(config.n_depth, dtype=np.float64)
    mask = (base_sum > 1e-12) & (shaped_sum > 1e-12)
    ratio[mask] = base_sum[mask] / shaped_sum[mask]
    ratio = gaussian_filter(ratio, sigma=1.2)
    ratio = np.clip(ratio, 0.52, 1.85) if strict else np.clip(ratio, 0.72, 1.38)
    out = shaped.copy()
    out[:, 3] = np.clip(out[:, 3] * ratio[z_shaped], 0.001, 100.0)
    return out


def _match_local_energy_profile(base: np.ndarray, shaped: np.ndarray, config: ExperimentConfig) -> np.ndarray:
    z_bins = 48
    x_bins = 48
    z_base = np.clip((base[:, 2] / config.z_max) * z_bins, 0, z_bins - 1).astype(int)
    x_base = np.clip(((base[:, 0] / config.x_max) + 0.5) * x_bins, 0, x_bins - 1).astype(int)
    z_shaped = np.clip((shaped[:, 2] / config.z_max) * z_bins, 0, z_bins - 1).astype(int)
    x_shaped = np.clip(((shaped[:, 0] / config.x_max) + 0.5) * x_bins, 0, x_bins - 1).astype(int)
    base_sum = np.zeros((z_bins, x_bins), dtype=np.float64)
    shaped_sum = np.zeros((z_bins, x_bins), dtype=np.float64)
    np.add.at(base_sum, (z_base, x_base), base[:, 3])
    np.add.at(shaped_sum, (z_shaped, x_shaped), shaped[:, 3])
    base_sum = gaussian_filter(base_sum, sigma=(1.2, 1.0))
    shaped_sum = gaussian_filter(shaped_sum, sigma=(1.2, 1.0))
    ratio = np.ones_like(base_sum)
    mask = (base_sum > 1e-12) & (shaped_sum > 1e-12)
    ratio[mask] = base_sum[mask] / shaped_sum[mask]
    ratio = gaussian_filter(ratio, sigma=(0.8, 0.8))
    ratio = np.clip(ratio, 0.82, 1.22)
    out = shaped.copy()
    out[:, 3] = np.clip(out[:, 3] * ratio[z_shaped, x_shaped], 0.001, 100.0)
    return out


def _target_texture_scatterers(
    input_path: str | Path,
    base: np.ndarray,
    count: int,
    config: ExperimentConfig,
    *,
    energy_quantile: float,
    texture_weight: float,
    rng: np.random.Generator,
) -> np.ndarray:
    ref = load_scan(input_path)
    ref = np.asarray(ref, dtype=np.float64)
    smooth = gaussian_filter(ref, sigma=(1.0, 1.8))
    detail = np.maximum(ref - gaussian_filter(ref, sigma=(2.0, 3.2)), 0.0)
    rows = np.arange(ref.shape[0], dtype=np.float64)[:, None]
    surface = _surface_from_reference(ref)
    tissue = 1.0 / (1.0 + np.exp(-(rows - surface - 1.0) / 2.5))
    density = tissue * (
        (1.0 - float(texture_weight)) * np.clip(smooth, 0.0, 1.0) ** 1.15
        + float(texture_weight) * np.clip(detail / max(float(np.percentile(detail, 99.0)), 1e-6), 0.0, 1.0) ** 0.75
    )
    density += 1e-8
    density = density / density.sum()
    flat = rng.choice(density.size, size=count, replace=True, p=density.ravel())
    z_bin, x_bin = np.divmod(flat, density.shape[1])
    xs = ((x_bin + rng.random(count)) / density.shape[1] - 0.5) * config.x_max
    zs = ((z_bin + rng.random(count)) / density.shape[0]) * config.z_max
    ys = rng.normal(0.0, 0.42, count) * config.beam_radius
    ys = np.clip(ys, -config.beam_radius, config.beam_radius)
    base_energy = float(np.quantile(base[:, 3], np.clip(float(energy_quantile), 0.05, 0.95)))
    ref_amp = np.clip(0.45 + 0.85 * ref[z_bin, x_bin], 0.35, 1.65)
    amps = base_energy * ref_amp * rng.lognormal(mean=0.0, sigma=0.08, size=count)
    return np.column_stack((xs, ys, zs, np.clip(amps, 0.001, 100.0)))


def _surface_from_reference(image: np.ndarray) -> np.ndarray:
    smooth = gaussian_filter(image, sigma=(1.2, 4.0))
    mask = smooth > max(0.035, float(np.percentile(smooth, 70)))
    surface = np.argmax(mask, axis=0)
    surface[~mask.any(axis=0)] = int(image.shape[0] * 0.11)
    return gaussian_filter(surface.astype(np.float64), sigma=4.0)[None, :]


def _stage_map_metrics(
    prefix: str,
    reference_png: Path | None,
    prediction_png: Path | None,
    maps_dir: Path,
    *,
    include_lpips: bool,
) -> dict[str, object]:
    if reference_png is None or prediction_png is None:
        return {f"{prefix}_map_enrichment_status": "missing_png"}
    try:
        struct_metrics = calculate_metrics(reference_png, prediction_png, include_lpips=include_lpips)
        map_metrics = evaluate_feature_map_metrics(
            reference_png,
            prediction_png,
            maps_dir / "ref_maps",
            maps_dir / "pred_maps",
            include_lpips=include_lpips,
        )
        profiles = profile_scores(reference_png, prediction_png)
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


def _stage_delta_metrics(current: dict[str, object], stage1: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for metric in MAP_METRIC_KEYS:
        out[f"delta_{metric}"] = _delta(_finite(stage1.get(f"stage1_{metric}")), _finite(current.get(f"current_{metric}")))
    out["delta_Struct_LPIPS_PROXY"] = _delta(
        _finite(current.get("current_Struct_LPIPS_PROXY")),
        _finite(stage1.get("stage1_Struct_LPIPS_PROXY")),
    )
    out["delta_OACProfileCorr"] = _delta(
        _finite(stage1.get("stage1_OACProfileCorr")),
        _finite(current.get("current_OACProfileCorr")),
    )
    out["delta_SCMeanAbsErr"] = _delta(
        _finite(current.get("current_SCMeanAbsErr")),
        _finite(stage1.get("stage1_SCMeanAbsErr")),
    )
    return out


def _guardrail_pass(delta_metrics: dict[str, object]) -> bool:
    for metric in ("OAC_MS-SSIM", "SC_MS-SSIM", "RSC_MS-SSIM"):
        value = _finite(delta_metrics.get(f"delta_{metric}"))
        if math.isfinite(value) and value < 0.0:
            return False
    struct_delta = _finite(delta_metrics.get("delta_Struct_MS-SSIM"))
    return not math.isfinite(struct_delta) or struct_delta >= 0.0


def _write_stage1_summary(
    path: Path,
    rows: list[dict[str, object]],
    candidate_metrics: str | Path,
    candidate_queue: str | Path,
    base_api_metrics: str | Path,
) -> None:
    struct_deltas = [_finite(row.get("delta_Struct_MS-SSIM")) for row in rows]
    struct_deltas = [value for value in struct_deltas if math.isfinite(value)]
    guardrail_rows = [row for row in rows if row.get("guardrail_pass") is True]
    summary = {
        "candidate_metrics": str(Path(candidate_metrics)),
        "candidate_queue": str(Path(candidate_queue)),
        "base_api_metrics": str(Path(base_api_metrics)),
        "n": len(rows),
        "guardrail_pass_count": len(guardrail_rows),
        "struct_win_count": sum(1 for value in struct_deltas if value > 0.0),
        "mean_delta_Struct_MS_SSIM": float(np.mean(struct_deltas)) if struct_deltas else "",
        "min_delta_Struct_MS_SSIM": min(struct_deltas) if struct_deltas else "",
        "max_delta_Struct_MS_SSIM": max(struct_deltas) if struct_deltas else "",
        "mean_guardrail_delta_OAC_MS_SSIM": _mean_metric(rows, "delta_OAC_MS-SSIM"),
        "mean_guardrail_delta_SC_MS_SSIM": _mean_metric(rows, "delta_SC_MS-SSIM"),
        "mean_guardrail_delta_RSC_MS_SSIM": _mean_metric(rows, "delta_RSC_MS-SSIM"),
        "promotion_allowed_without_true_scanner": False,
        "hidden_holdout_final_score": False,
        "interpretation": "Stage 1 residual smoke map enrichment; use as known-source evidence only, not hidden-holdout promotion.",
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _stage1_variant_score(
    row: dict[str, object],
    *,
    struct_weight: float,
    oac_weight: float,
    sc_weight: float,
    rsc_weight: float,
    lpips_weight: float,
) -> float:
    struct_delta = _finite(row.get("delta_Struct_MS-SSIM"))
    oac_delta = _finite(row.get("delta_OAC_MS-SSIM"))
    sc_delta = _finite(row.get("delta_SC_MS-SSIM"))
    rsc_delta = _finite(row.get("delta_RSC_MS-SSIM"))
    lpips_delta = _finite(row.get("delta_Struct_LPIPS_PROXY"))
    values = {
        "struct": struct_delta if math.isfinite(struct_delta) else -1.0,
        "oac": oac_delta if math.isfinite(oac_delta) else -1.0,
        "sc": sc_delta if math.isfinite(sc_delta) else -1.0,
        "rsc": rsc_delta if math.isfinite(rsc_delta) else -1.0,
        "lpips": lpips_delta if math.isfinite(lpips_delta) else 0.0,
    }
    return float(
        struct_weight * values["struct"]
        + oac_weight * values["oac"]
        + sc_weight * values["sc"]
        + rsc_weight * values["rsc"]
        + lpips_weight * values["lpips"]
    )


def _stage1_selection_status(
    row: dict[str, object],
    *,
    min_struct_delta: float,
    min_oac_delta: float,
    min_sc_delta: float,
    min_rsc_delta: float,
    require_true_scanner: bool,
) -> str:
    if require_true_scanner and row.get("status", "") != "ok":
        return "blocked_not_true_scanner_ok"
    if row.get("stage1_map_enrichment_status", "") != "ok" or row.get("current_map_enrichment_status", "") != "ok":
        return "blocked_map_enrichment_missing"
    checks = (
        ("delta_Struct_MS-SSIM", min_struct_delta, "blocked_struct_delta"),
        ("delta_OAC_MS-SSIM", min_oac_delta, "blocked_oac_delta"),
        ("delta_SC_MS-SSIM", min_sc_delta, "blocked_sc_delta"),
        ("delta_RSC_MS-SSIM", min_rsc_delta, "blocked_rsc_delta"),
    )
    for field, threshold, status in checks:
        value = _finite(row.get(field))
        if not math.isfinite(value) or value < threshold:
            return status
    return "selected_candidate"


def _stage1_selected_queue_row(row: dict[str, object], priority: int) -> dict[str, object]:
    return {
        "priority": priority,
        "method": row.get("method", ""),
        "source_archive_path": row.get("source_archive_path", ""),
        "reference_png": row.get("reference_png", ""),
        "phantom_path": row.get("phantom_path", ""),
        "synthetic_gray_png": row.get("synthetic_gray_png", ""),
        "variant_set": row.get("variant_set", ""),
        "enriched_metrics_path": row.get("enriched_metrics_path", ""),
        "stage1_ms_ssim": row.get("stage1_ms_ssim") or row.get("MS-SSIM", ""),
        "current_ms_ssim": row.get("current_ms_ssim", ""),
        "delta_Struct_MS-SSIM": row.get("delta_Struct_MS-SSIM", ""),
        "delta_OAC_MS-SSIM": row.get("delta_OAC_MS-SSIM", ""),
        "delta_SC_MS-SSIM": row.get("delta_SC_MS-SSIM", ""),
        "delta_RSC_MS-SSIM": row.get("delta_RSC_MS-SSIM", ""),
        "delta_Struct_LPIPS_PROXY": row.get("delta_Struct_LPIPS_PROXY", ""),
        "stage1_multi_objective_score": row.get("stage1_multi_objective_score", ""),
        "stage1_selection_status": "selected",
        "promotion_allowed_without_true_scanner": False,
        "hidden_holdout_final_score": False,
        "evidence_scope": "stage1_residual_multi_objective_selected_known_source_not_hidden_holdout",
    }


def _write_stage1_selection_summary(
    path: Path,
    candidates: list[dict[str, object]],
    selected_rows: list[dict[str, object]],
    enriched_metrics_paths: list[str | Path],
    *,
    thresholds: dict[str, object],
    weights: dict[str, float],
) -> None:
    status_counts: dict[str, int] = {}
    for row in candidates:
        status = str(row.get("stage1_selection_status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "enriched_metrics_paths": [str(Path(path)) for path in enriched_metrics_paths],
        "candidate_count": len(candidates),
        "selected_count": len(selected_rows),
        "unique_source_count": len({row.get("source_archive_path", "") for row in candidates if row.get("source_archive_path", "")}),
        "status_counts": status_counts,
        "thresholds": thresholds,
        "weights": weights,
        "selected_mean_delta_Struct_MS_SSIM": _mean_metric(selected_rows, "delta_Struct_MS-SSIM"),
        "selected_mean_delta_OAC_MS_SSIM": _mean_metric(selected_rows, "delta_OAC_MS-SSIM"),
        "selected_mean_delta_SC_MS_SSIM": _mean_metric(selected_rows, "delta_SC_MS-SSIM"),
        "selected_mean_delta_RSC_MS_SSIM": _mean_metric(selected_rows, "delta_RSC_MS-SSIM"),
        "selected_min_delta_OAC_MS_SSIM": _min_metric(selected_rows, "delta_OAC_MS-SSIM"),
        "selected_min_delta_SC_MS_SSIM": _min_metric(selected_rows, "delta_SC_MS-SSIM"),
        "selected_min_delta_RSC_MS_SSIM": _min_metric(selected_rows, "delta_RSC_MS-SSIM"),
        "promotion_allowed_without_true_scanner": False,
        "hidden_holdout_final_score": False,
        "interpretation": (
            "Multi-objective Stage 1 residual variant selection. Rows are known-source smoke evidence "
            "chosen for follow-up validation, not hidden-holdout proof."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _collect_sources(paths: list[str | Path]) -> set[str]:
    sources: set[str] = set()
    for path in paths:
        for row in _read_rows(path):
            source = row.get("source_archive_path", "")
            if source:
                sources.add(source)
    return sources


def _collect_residual_artifact_training_sources(path: str | Path) -> set[str]:
    metadata_path = Path(path).with_suffix(".json")
    if not metadata_path.exists():
        return set()
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    sources: set[str] = set()
    for row in metadata.get("training_rows", []):
        if isinstance(row, dict) and row.get("source_archive_path"):
            sources.add(str(row["source_archive_path"]))
    return sources


def _stage1_variant_modes(
    variant_modes: list[str] | tuple[str, ...] | None,
    *,
    include_depth_profile_variant: bool,
    include_local_profile_variant: bool,
) -> list[tuple[bool, bool]]:
    if variant_modes:
        lookup = {
            "preserve": (False, False),
            "depth": (True, False),
            "local": (False, True),
            "depthlocal": (True, True),
        }
        modes = []
        for mode in variant_modes:
            if mode not in lookup:
                raise ValueError(f"Unknown Stage 1 validation variant mode: {mode}")
            if lookup[mode] not in modes:
                modes.append(lookup[mode])
        return modes
    modes = [(False, False)]
    if include_depth_profile_variant:
        modes.append((True, False))
    if include_local_profile_variant:
        modes.append((False, True))
        if include_depth_profile_variant:
            modes.append((True, True))
    return modes


def _stage1_validation_method(source_idx: int, residual_blend: float, preserve_depth: bool, preserve_local: bool) -> str:
    blend_tag = str(f"{residual_blend:.2f}").replace(".", "p")
    if preserve_depth and preserve_local:
        mode_tag = "depthlocal"
    elif preserve_depth:
        mode_tag = "depth"
    elif preserve_local:
        mode_tag = "local"
    else:
        mode_tag = "preserve"
    return f"stage1_holdout_{mode_tag}_b{blend_tag}_{source_idx:03d}"


def _stage2_overlay_method(source_idx: int, replace_fraction: float, texture_weight: float) -> str:
    fraction_tag = str(f"{replace_fraction:.3f}").replace(".", "p")
    texture_tag = str(f"{texture_weight:.2f}").replace(".", "p")
    return f"stage2_texture_overlay_f{fraction_tag}_tw{texture_tag}_{source_idx:03d}"


def _write_stage1_validation_queue_summary(
    path: Path,
    queue_rows: list[dict[str, object]],
    base_api_metrics: str | Path,
    residual_artifact_path: str | Path,
    exclude_metrics_paths: list[str | Path],
    *,
    max_sources: int,
    residual_blends: list[float] | tuple[float, ...],
    include_depth_profile_variant: bool,
    include_local_profile_variant: bool,
    variant_modes: list[str] | tuple[str, ...] | None,
    exclude_residual_training_sources: bool,
    min_base_ms_ssim: float | None,
) -> None:
    sources = {row.get("source_archive_path", "") for row in queue_rows if row.get("source_archive_path", "")}
    summary = {
        "base_api_metrics": str(Path(base_api_metrics)),
        "residual_artifact_path": str(Path(residual_artifact_path)),
        "exclude_metrics_paths": [str(Path(path)) for path in exclude_metrics_paths],
        "queue_count": len(queue_rows),
        "source_count": len(sources),
        "max_sources": int(max_sources),
        "residual_blends": [float(value) for value in residual_blends],
        "include_depth_profile_variant": include_depth_profile_variant,
        "include_local_profile_variant": include_local_profile_variant,
        "variant_modes": list(variant_modes) if variant_modes else [],
        "exclude_residual_training_sources": exclude_residual_training_sources,
        "min_base_ms_ssim": min_base_ms_ssim if min_base_ms_ssim is not None else "",
        "mean_base_ms_ssim": _mean_metric(queue_rows, "base_ms_ssim"),
        "min_base_ms_ssim_selected": _min_metric(queue_rows, "base_ms_ssim"),
        "promotion_allowed_without_true_scanner": False,
        "hidden_holdout_final_score": False,
        "interpretation": (
            "Stage 1 residual validation queue generated from base p140/t32 rows after source exclusions. "
            "Render with the hosted scanner before treating any row as evidence."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _write_stage2_overlay_queue_summary(
    path: Path,
    queue_rows: list[dict[str, object]],
    base_api_metrics: str | Path,
    *,
    max_sources: int,
    replace_fractions: list[float] | tuple[float, ...],
    texture_weights: list[float] | tuple[float, ...],
    energy_quantile: float,
    min_base_ms_ssim: float | None,
) -> None:
    sources = {row.get("source_archive_path", "") for row in queue_rows if row.get("source_archive_path", "")}
    summary = {
        "base_api_metrics": str(Path(base_api_metrics)),
        "queue_count": len(queue_rows),
        "source_count": len(sources),
        "max_sources": int(max_sources),
        "replace_fractions": [float(value) for value in replace_fractions],
        "texture_weights": [float(value) for value in texture_weights],
        "energy_quantile": float(energy_quantile),
        "min_base_ms_ssim": min_base_ms_ssim if min_base_ms_ssim is not None else "",
        "mean_base_ms_ssim": _mean_metric(queue_rows, "base_ms_ssim"),
        "promotion_allowed_without_true_scanner": False,
        "hidden_holdout_final_score": False,
        "interpretation": (
            "Stage 2 target-guided texture overlay queue. It preserves p140/t32 scatterer count "
            "and most coordinates; render with the hosted scanner before using as evidence."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _mean_metric(rows: list[dict[str, object]], field: str) -> float | str:
    values = [_finite(row.get(field)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else ""


def _min_metric(rows: list[dict[str, object]], field: str) -> float | str:
    values = [_finite(row.get(field)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return min(values) if values else ""


def _delta(new_value: float, old_value: float) -> float | str:
    if not math.isfinite(new_value) or not math.isfinite(old_value):
        return ""
    return new_value - old_value


def _write_rows(path: str | Path, rows: list[dict[str, object]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as fobj:
        return list(csv.DictReader(fobj))


def _existing_path(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value)
    if path.exists():
        return path.resolve()
    candidate = Path.cwd() / path
    if candidate.exists():
        return candidate.resolve()
    return None


def _first_finite(row: dict[str, str], *fields: str) -> float:
    for field in fields:
        value = _finite(row.get(field))
        if math.isfinite(value):
            return value
    return float("nan")


def _finite(value: str | float | None) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _normalize(arr: np.ndarray, floor: float = 0.0) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr - np.nanmin(arr)
    peak = float(np.nanmax(arr))
    if peak > 1e-8:
        arr = arr / peak
    return np.maximum(arr, floor).astype(np.float32)


def _lowpass(pred, *, kernel: int, functional):
    if kernel <= 1:
        return pred
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    return functional.avg_pool2d(functional.pad(pred, (pad, pad, pad, pad), mode="reflect"), kernel_size=kernel, stride=1)


def _lowpass_np(pred: np.ndarray, *, kernel: int) -> np.ndarray:
    if kernel <= 1:
        return pred
    sigma = max(0.5, kernel / 4.0)
    return gaussian_filter(pred, sigma=(0.0, sigma, sigma)).astype(np.float32)


def _gradient_loss(pred, target):
    return (pred[:, :, 1:, :] - target[:, :, 1:, :]).abs().mean() + (pred[:, :, :, 1:] - target[:, :, :, 1:]).abs().mean()


def _smoothness(value):
    return (value[:, :, 1:, :] - value[:, :, :-1, :]).abs().mean() + (value[:, :, :, 1:] - value[:, :, :, :-1]).abs().mean()


class _Stage1ResidualNet:
    def __new__(cls):
        from torch import nn

        return nn.Sequential(
            nn.Conv2d(8, 24, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(24, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 24, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(24, 2, kernel_size=1),
            nn.Tanh(),
        )


_EXAMPLE_METADATA_FIELDS = (
    "example_kind",
    "source_archive_path",
    "reference_png",
    "base_phantom_path",
    "target_phantom_path",
    "flow_delta",
    "flow_map_delta",
    "weight",
    "teacher_metrics_path",
)
