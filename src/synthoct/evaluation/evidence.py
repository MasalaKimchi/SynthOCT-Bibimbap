from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .summaries import HIGHER_IS_BETTER_GUARDRAILS, LOWER_IS_BETTER_GUARDRAILS, finite_float

TRUE_SCANNER_SOURCES = {"hosted_api_true_scanner", "official_windows_true_scanner"}
NON_CHALLENGE_SCANNER_SOURCES = {"learned_surrogate_preview", "learned_surrogate_holdout", "preview_renderer"}
GROUPED_VALIDATION_PREFIX = "grouped_validation_"


def audit_challenge_evidence(
    metrics_csv: str | Path,
    *,
    min_samples_per_method: int = 2,
    require_real_lpips: bool = False,
    max_generation_seconds: float = 600.0,
) -> dict[str, object]:
    """Classify whether a metrics CSV is strong enough to promote a candidate.

    This is a local evidence gate only. Even a passing grouped validation CSV is
    not an official final score because SynthOCT final ranking uses hidden data.
    """
    metrics_csv = Path(metrics_csv)
    with metrics_csv.open(newline="") as fobj:
        reader = csv.DictReader(fobj)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not rows:
        return {
            "metrics_csv": str(metrics_csv),
            "status": "not_challenge_evidence",
            "promotion_ready": False,
            "issues": ["metrics CSV has no rows"],
            "rows": 0,
        }

    issues: list[str] = []
    warnings: list[str] = []
    evidence_sources = sorted({str(row.get("evidence_source", "")) for row in rows if row.get("evidence_source", "")})
    evidence_scopes = sorted({str(row.get("evidence_scope", "")) for row in rows if row.get("evidence_scope", "")})
    evaluation_regions = sorted({_evaluation_region(row) for row in rows if _evaluation_region(row)})
    source_counts = Counter(str(row.get("evidence_source", "")) or "<missing>" for row in rows)
    scope_counts = Counter(str(row.get("evidence_scope", "")) or "<missing>" for row in rows)

    if not evidence_sources:
        issues.append("missing evidence_source labels")
    elif any(source not in TRUE_SCANNER_SOURCES for source in evidence_sources):
        issues.append(f"non-true-scanner evidence_source present: {', '.join(evidence_sources)}")

    if not evidence_scopes:
        issues.append("missing evidence_scope labels")
    grouped_validation = bool(evidence_scopes) and all(scope.startswith(GROUPED_VALIDATION_PREFIX) for scope in evidence_scopes)
    if not grouped_validation:
        warnings.append("evidence is not grouped validation; single-reference, preview, and submission-manifest renders are limited evidence")

    if not _has_metric(fieldnames, rows, ("MS-SSIM_mean", "Struct_MS-SSIM", "MS-SSIM")):
        issues.append("missing finite MS-SSIM metric")
    if not evaluation_regions:
        issues.append("missing evaluation_region labels")
    elif any(region != "full_frame" for region in evaluation_regions):
        issues.append(f"non-full-frame evaluation region present: {', '.join(evaluation_regions)}")

    has_real_lpips = _has_real_lpips(fieldnames, rows)
    has_lpips_proxy = _has_lpips_proxy(fieldnames, rows)
    official_missing_metrics = _official_ranking_missing_metrics(fieldnames, rows)
    official_metric_complete = not official_missing_metrics
    if require_real_lpips and not has_real_lpips:
        issues.append("missing finite real LPIPS metric")
    elif not has_real_lpips and has_lpips_proxy:
        warnings.append("LPIPS appears to be proxy/fallback rather than real LPIPS")
    if official_missing_metrics:
        warnings.append(
            "official ranking metric set is incomplete; missing "
            + ", ".join(official_missing_metrics)
            + ". Hidden-holdout ranking uses median MS-SSIM and 1-LPIPS over Struct/OAC/SC/RSC."
        )
    preliminary_failures = _preliminary_threshold_failures(fieldnames, rows)
    if preliminary_failures:
        warnings.append("preliminary threshold gate is not fully proven/passed: " + "; ".join(preliminary_failures))

    runtime_values = _generation_seconds_by_method(rows)
    if not runtime_values:
        issues.append("missing generation_seconds_max runtime evidence")
    else:
        over_budget = {method: seconds for method, seconds in runtime_values.items() if seconds > max_generation_seconds}
        if over_budget:
            issues.append(f"generation runtime exceeds {max_generation_seconds:.1f}s: {_format_float_counts(over_budget)}")

    method_counts = _method_counts(rows)
    undersampled = {method: count for method, count in method_counts.items() if count < min_samples_per_method}
    if undersampled:
        warnings.append(f"fewer than {min_samples_per_method} samples for method(s): {_format_counts(undersampled)}")

    true_scanner = bool(evidence_sources) and all(source in TRUE_SCANNER_SOURCES for source in evidence_sources)
    enough_samples = bool(method_counts) and not undersampled
    promotion_ready = true_scanner and grouped_validation and enough_samples and not issues
    status = "promotion_ready" if promotion_ready else "limited_true_scanner_evidence" if true_scanner and not issues else "not_challenge_evidence"

    return {
        "metrics_csv": str(metrics_csv),
        "status": status,
        "promotion_ready": promotion_ready,
        "hidden_holdout_final_score": False,
        "rows": len(rows),
        "methods": sorted(method_counts),
        "method_counts": dict(sorted(method_counts.items())),
        "evidence_sources": evidence_sources,
        "evidence_source_counts": dict(source_counts),
        "evidence_scopes": evidence_scopes,
        "evidence_scope_counts": dict(scope_counts),
        "evaluation_regions": evaluation_regions,
        "true_scanner_evidence": true_scanner,
        "surrogate_scanner_is_true_scanner": False,
        "true_scanner_sources": sorted(TRUE_SCANNER_SOURCES),
        "non_challenge_scanner_sources": sorted(NON_CHALLENGE_SCANNER_SOURCES),
        "grouped_validation_evidence": grouped_validation,
        "has_ms_ssim": _has_metric(fieldnames, rows, ("MS-SSIM_mean", "Struct_MS-SSIM", "MS-SSIM")),
        "has_real_lpips": has_real_lpips,
        "has_lpips_proxy": has_lpips_proxy,
        "official_ranking_metric_complete": official_metric_complete,
        "official_ranking_metric_missing": official_missing_metrics,
        "preliminary_threshold_failures": preliminary_failures,
        "max_generation_seconds": max_generation_seconds,
        "generation_seconds_max_by_method": dict(sorted(runtime_values.items())),
        "issues": issues,
        "warnings": warnings,
        "interpretation": _interpretation(status),
    }


def decide_candidate_promotion(
    metrics_csv: str | Path,
    *,
    candidate: str,
    baseline: str,
    min_samples_per_method: int = 2,
    require_real_lpips: bool = False,
    min_ms_ssim_delta: float = 0.0,
    max_lpips_delta: float = 0.0,
    max_generation_seconds: float = 600.0,
    max_guardrail_regression: float = 0.0,
) -> dict[str, object]:
    """Decide whether candidate evidence beats a baseline under challenge metrics."""
    metrics_csv = Path(metrics_csv)
    audit = audit_challenge_evidence(
        metrics_csv,
        min_samples_per_method=min_samples_per_method,
        require_real_lpips=require_real_lpips,
        max_generation_seconds=max_generation_seconds,
    )
    with metrics_csv.open(newline="") as fobj:
        rows = list(csv.DictReader(fobj))

    by_method = {str(row.get("method", "")): row for row in rows}
    issues = list(audit.get("issues", []))
    warnings = list(audit.get("warnings", []))
    if candidate not in by_method:
        issues.append(f"candidate method not found: {candidate}")
    if baseline not in by_method:
        issues.append(f"baseline method not found: {baseline}")
    if issues:
        return {
            "metrics_csv": str(metrics_csv),
            "candidate": candidate,
            "baseline": baseline,
            "promote": False,
            "status": "not_promoted",
            "issues": issues,
            "warnings": warnings,
            "evidence_audit": audit,
        }

    candidate_row = by_method[candidate]
    baseline_row = by_method[baseline]
    cand_ms = finite_float(candidate_row.get("MS-SSIM_mean"), -1.0)
    base_ms = finite_float(baseline_row.get("MS-SSIM_mean"), -1.0)
    cand_lpips = finite_float(candidate_row.get("LPIPS_or_proxy_mean"), float("inf"))
    base_lpips = finite_float(baseline_row.get("LPIPS_or_proxy_mean"), float("inf"))
    cand_ms_wins = int(finite_float(candidate_row.get("MS-SSIM_wins"), 0.0))
    base_ms_wins = int(finite_float(baseline_row.get("MS-SSIM_wins"), 0.0))
    cand_lpips_wins = int(finite_float(candidate_row.get("LPIPS_wins"), 0.0))
    base_lpips_wins = int(finite_float(baseline_row.get("LPIPS_wins"), 0.0))
    cand_generation_max = finite_float(candidate_row.get("generation_seconds_max"), float("inf"))
    base_generation_max = finite_float(baseline_row.get("generation_seconds_max"), float("inf"))
    cand_official = finite_float(candidate_row.get("official_score"))
    base_official = finite_float(baseline_row.get("official_score"))
    ms_delta = cand_ms - base_ms
    lpips_delta = cand_lpips - base_lpips
    generation_delta = cand_generation_max - base_generation_max
    official_delta = cand_official - base_official if cand_official == cand_official and base_official == base_official else float("nan")
    ms_pass = ms_delta > min_ms_ssim_delta
    lpips_pass = lpips_delta <= max_lpips_delta
    wins_pass = cand_ms_wins >= base_ms_wins and cand_lpips_wins >= base_lpips_wins
    runtime_pass = cand_generation_max <= max_generation_seconds
    official_metric_complete = bool(audit.get("official_ranking_metric_complete"))
    official_pass = not official_metric_complete or (official_delta == official_delta and official_delta > 0.0)
    guardrail_deltas = _guardrail_deltas(candidate_row, baseline_row)
    guardrail_failures = _guardrail_failures(guardrail_deltas, max_guardrail_regression=max_guardrail_regression)
    guardrails_pass = not guardrail_failures
    if official_metric_complete:
        promote = bool(audit.get("promotion_ready")) and runtime_pass and official_pass
    else:
        promote = bool(audit.get("promotion_ready")) and ms_pass and lpips_pass and wins_pass and runtime_pass and guardrails_pass
    if not official_metric_complete and not ms_pass:
        warnings.append(f"candidate MS-SSIM delta {ms_delta:.6g} does not exceed {min_ms_ssim_delta:.6g}")
    if not official_metric_complete and not lpips_pass:
        warnings.append(f"candidate LPIPS/proxy delta {lpips_delta:.6g} exceeds {max_lpips_delta:.6g}")
    if not official_metric_complete and not wins_pass:
        warnings.append("candidate does not match or exceed baseline per-sample MS-SSIM and LPIPS wins")
    if not runtime_pass:
        warnings.append(f"candidate generation runtime {cand_generation_max:.6g}s exceeds {max_generation_seconds:.6g}s")
    if not official_pass:
        warnings.append(f"candidate official score delta {official_delta:.6g} does not exceed 0")
    if not official_metric_complete and guardrail_failures:
        warnings.append(f"candidate regresses physical guardrail(s): {_format_guardrail_failures(guardrail_failures)}")

    return {
        "metrics_csv": str(metrics_csv),
        "candidate": candidate,
        "baseline": baseline,
        "promote": promote,
        "status": "promoted" if promote else "not_promoted",
        "hidden_holdout_final_score": False,
        "candidate_MS-SSIM_mean": cand_ms,
        "baseline_MS-SSIM_mean": base_ms,
        "MS-SSIM_delta": ms_delta,
        "candidate_LPIPS_or_proxy_mean": cand_lpips,
        "baseline_LPIPS_or_proxy_mean": base_lpips,
        "LPIPS_or_proxy_delta": lpips_delta,
        "candidate_MS-SSIM_wins": cand_ms_wins,
        "baseline_MS-SSIM_wins": base_ms_wins,
        "candidate_LPIPS_wins": cand_lpips_wins,
        "baseline_LPIPS_wins": base_lpips_wins,
        "candidate_generation_seconds_max": cand_generation_max,
        "baseline_generation_seconds_max": base_generation_max,
        "generation_seconds_delta": generation_delta,
        "candidate_official_score": cand_official,
        "baseline_official_score": base_official,
        "official_score_delta": official_delta,
        "official_ranking_metric_complete": official_metric_complete,
        "max_generation_seconds": max_generation_seconds,
        "guardrail_deltas": guardrail_deltas,
        "guardrail_failures": guardrail_failures,
        "max_guardrail_regression": max_guardrail_regression,
        "issues": [],
        "warnings": warnings,
        "evidence_audit": audit,
        "interpretation": _promotion_interpretation(promote),
    }


def select_best_candidate(
    metrics_csv: str | Path,
    *,
    baseline: str,
    min_samples_per_method: int = 2,
    require_real_lpips: bool = False,
    min_ms_ssim_delta: float = 0.0,
    max_lpips_delta: float = 0.0,
    max_generation_seconds: float = 600.0,
    max_guardrail_regression: float = 0.0,
) -> dict[str, object]:
    """Rank all methods and return the best candidate that beats the baseline."""
    metrics_csv = Path(metrics_csv)
    with metrics_csv.open(newline="") as fobj:
        rows = list(csv.DictReader(fobj))
    method_names = [str(row.get("method", "")) for row in rows if row.get("method", "")]
    decisions = [
        decide_candidate_promotion(
            metrics_csv,
            candidate=method,
            baseline=baseline,
            min_samples_per_method=min_samples_per_method,
            require_real_lpips=require_real_lpips,
            min_ms_ssim_delta=min_ms_ssim_delta,
            max_lpips_delta=max_lpips_delta,
            max_generation_seconds=max_generation_seconds,
            max_guardrail_regression=max_guardrail_regression,
        )
        for method in method_names
        if method != baseline
    ]
    decisions.sort(key=_promotion_rank_key, reverse=True)
    promoted = [decision for decision in decisions if decision.get("promote")]
    best = promoted[0] if promoted else (decisions[0] if decisions else None)
    return {
        "metrics_csv": str(metrics_csv),
        "baseline": baseline,
        "selected": best.get("candidate") if best and best.get("promote") else None,
        "promote": bool(best and best.get("promote")),
        "status": "promoted" if best and best.get("promote") else "not_promoted",
        "ranked_candidates": [
            {
                "candidate": decision.get("candidate"),
                "promote": decision.get("promote"),
                "MS-SSIM_delta": decision.get("MS-SSIM_delta"),
                "LPIPS_or_proxy_delta": decision.get("LPIPS_or_proxy_delta"),
                "candidate_MS-SSIM_wins": decision.get("candidate_MS-SSIM_wins"),
                "candidate_LPIPS_wins": decision.get("candidate_LPIPS_wins"),
                "candidate_generation_seconds_max": decision.get("candidate_generation_seconds_max"),
                "candidate_official_score": decision.get("candidate_official_score"),
                "official_score_delta": decision.get("official_score_delta"),
                "official_ranking_metric_complete": decision.get("official_ranking_metric_complete"),
                "guardrail_deltas": decision.get("guardrail_deltas", {}),
                "guardrail_failures": decision.get("guardrail_failures", {}),
                "warnings": decision.get("warnings", []),
                "issues": decision.get("issues", []),
            }
            for decision in decisions
        ],
        "best_decision": best,
        "interpretation": _selection_interpretation(bool(best and best.get("promote"))),
    }


def challenge_readiness_report(
    metrics_csv: str | Path,
    *,
    baseline: str,
    method: str | None = None,
    submission_dir: str | Path | None = None,
    min_samples_per_method: int = 2,
    require_real_lpips: bool = False,
    min_ms_ssim_delta: float = 0.0,
    max_lpips_delta: float = 0.0,
    max_generation_seconds: float = 600.0,
    max_guardrail_regression: float = 0.0,
) -> dict[str, object]:
    """Summarize whether local evidence is ready for challenge submission.

    This report intentionally separates local readiness from official victory:
    the hidden hold-out final score can only be established by organizer
    execution of the submitted generator/model.
    """
    metrics_csv = Path(metrics_csv)
    audit = audit_challenge_evidence(
        metrics_csv,
        min_samples_per_method=min_samples_per_method,
        require_real_lpips=require_real_lpips,
        max_generation_seconds=max_generation_seconds,
    )
    selection = select_best_candidate(
        metrics_csv,
        baseline=baseline,
        min_samples_per_method=min_samples_per_method,
        require_real_lpips=require_real_lpips,
        min_ms_ssim_delta=min_ms_ssim_delta,
        max_lpips_delta=max_lpips_delta,
        max_generation_seconds=max_generation_seconds,
        max_guardrail_regression=max_guardrail_regression,
    )
    selected = selection.get("selected")
    issues: list[str] = []
    warnings: list[str] = ["hidden hold-out final score is not locally proven"]
    issues.extend(str(issue) for issue in audit.get("issues", []))
    best_decision = selection.get("best_decision")
    if isinstance(best_decision, dict):
        issues.extend(str(issue) for issue in best_decision.get("issues", []))
        warnings.extend(str(warning) for warning in best_decision.get("warnings", []))

    method_matches_selection = None
    if method is not None:
        method_matches_selection = bool(selection.get("promote")) and selected == method
        if not method_matches_selection:
            issues.append(f"method {method!r} is not the selected promoted candidate")

    submission_report = None
    submission_validation_present = None
    submission_readiness_present = None
    if submission_dir is not None:
        submission_dir = Path(submission_dir)
        validation_path = submission_dir / "submission_validation.csv"
        readiness_path = submission_dir / "submission_readiness_report.json"
        submission_validation_present = validation_path.exists()
        submission_readiness_present = readiness_path.exists()
        if not submission_validation_present:
            issues.append(f"missing submission validation CSV: {validation_path}")
        if not submission_readiness_present:
            issues.append(f"missing submission readiness report: {readiness_path}")
        else:
            submission_report = _load_submission_readiness(readiness_path, issues)
            if isinstance(submission_report, dict):
                if submission_report.get("status") != "ready":
                    issues.append("submission readiness report is not ready")
                packaged_method = submission_report.get("packaged_method")
                if method is not None and packaged_method != method:
                    issues.append(f"submission packaged method {packaged_method!r} does not match requested method {method!r}")
                if selection.get("promote") and packaged_method != selected:
                    issues.append(f"submission packaged method {packaged_method!r} does not match selected method {selected!r}")

    local_candidate_ready = bool(selection.get("promote")) and (method_matches_selection is not False)
    if submission_dir is not None:
        local_candidate_ready = (
            local_candidate_ready
            and submission_validation_present is True
            and submission_readiness_present is True
            and isinstance(submission_report, dict)
            and submission_report.get("status") == "ready"
        )
    status = "local_candidate_ready" if local_candidate_ready and not issues else "not_ready"

    return {
        "status": status,
        "local_candidate_ready": status == "local_candidate_ready",
        "hidden_holdout_final_score": False,
        "official_final_ranking_proven": False,
        "metrics_csv": str(metrics_csv),
        "baseline": baseline,
        "method": method,
        "selected_method": selected,
        "submission_dir": str(submission_dir) if submission_dir is not None else None,
        "submission_validation_present": submission_validation_present,
        "submission_readiness_present": submission_readiness_present,
        "submission_readiness_report": submission_report,
        "gates": {
            "true_scanner_grouped_full_frame_evidence": bool(audit.get("promotion_ready")),
            "candidate_beats_baseline": bool(selection.get("promote")),
            "method_matches_selection": method_matches_selection,
            "submission_validation_present": submission_validation_present,
            "submission_readiness_present": submission_readiness_present,
            "official_ranking_metric_complete": bool(audit.get("official_ranking_metric_complete")),
            "hidden_holdout_final_score": False,
        },
        "issues": _dedupe(issues),
        "warnings": _dedupe(warnings),
        "evidence_audit": audit,
        "selection": selection,
        "interpretation": _readiness_interpretation(status),
    }


def _method_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        method = str(row.get("method", "")) or "<missing>"
        n_value = finite_float(row.get("n"))
        counts[method] += int(n_value) if n_value >= 1 else 1
    return dict(counts)


def _evaluation_region(row: dict[str, str]) -> str:
    return str(row.get("evaluation_region") or row.get("Struct_evaluation_region") or "")


def _generation_seconds_by_method(rows: list[dict[str, str]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        method = str(row.get("method", "")) or "<missing>"
        value = finite_float(row.get("generation_seconds_max"))
        if value != value:
            value = finite_float(row.get("generation_seconds"))
        if value == value:
            out[method] = max(out.get(method, 0.0), value)
    return out


def _guardrail_deltas(candidate_row: dict[str, str], baseline_row: dict[str, str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in HIGHER_IS_BETTER_GUARDRAILS:
        summary_key = f"{key}_mean"
        candidate = finite_float(candidate_row.get(summary_key))
        baseline = finite_float(baseline_row.get(summary_key))
        if candidate == candidate and baseline == baseline:
            out[summary_key] = candidate - baseline
    for key in LOWER_IS_BETTER_GUARDRAILS:
        summary_key = f"{key}_mean"
        candidate = finite_float(candidate_row.get(summary_key))
        baseline = finite_float(baseline_row.get(summary_key))
        if candidate == candidate and baseline == baseline:
            out[summary_key] = baseline - candidate
    return out


def _guardrail_failures(guardrail_deltas: dict[str, float], *, max_guardrail_regression: float) -> dict[str, float]:
    return {key: delta for key, delta in guardrail_deltas.items() if delta < -max_guardrail_regression}


def _has_metric(fieldnames: list[str], rows: list[dict[str, str]], keys: tuple[str, ...]) -> bool:
    present = [key for key in keys if key in fieldnames]
    return any(any(finite_float(row.get(key)) == finite_float(row.get(key)) for row in rows) for key in present)


def _has_real_lpips(fieldnames: list[str], rows: list[dict[str, str]]) -> bool:
    if "LPIPS_metric" in fieldnames:
        return any(str(row.get("LPIPS_metric", "")).upper() == "LPIPS" and finite_float(row.get("LPIPS_or_proxy_mean")) == finite_float(row.get("LPIPS_or_proxy_mean")) for row in rows)
    return _has_metric(fieldnames, rows, ("Struct_LPIPS", "LPIPS"))


def _has_lpips_proxy(fieldnames: list[str], rows: list[dict[str, str]]) -> bool:
    if "LPIPS_metric" in fieldnames:
        return any(
            str(row.get("LPIPS_metric", "")).upper() != "LPIPS"
            and finite_float(row.get("LPIPS_or_proxy_mean")) == finite_float(row.get("LPIPS_or_proxy_mean"))
            for row in rows
        )

    real_keys = [key for key in ("Struct_LPIPS", "LPIPS") if key in fieldnames]
    proxy_keys = [key for key in ("Struct_LPIPS_PROXY", "LPIPS_PROXY") if key in fieldnames]
    for row in rows:
        has_real = any(finite_float(row.get(key)) == finite_float(row.get(key)) for key in real_keys)
        has_proxy = any(finite_float(row.get(key)) == finite_float(row.get(key)) for key in proxy_keys)
        if has_proxy and not has_real:
            return True
    return False


def _official_ranking_missing_metrics(fieldnames: list[str], rows: list[dict[str, str]]) -> list[str]:
    missing: list[str] = []
    for map_name in ("Struct", "OAC", "SC", "RSC"):
        ms_keys = (f"{map_name}_MS-SSIM_median", f"{map_name}_MS-SSIM_mean", f"{map_name}_MS-SSIM")
        lpips_keys = (f"{map_name}_LPIPS_median", f"{map_name}_LPIPS_mean", f"{map_name}_LPIPS")
        if not _has_metric(fieldnames, rows, ms_keys):
            missing.append(f"{map_name}_MS-SSIM")
        if not _has_metric(fieldnames, rows, lpips_keys):
            missing.append(f"{map_name}_LPIPS")
    return missing


def _preliminary_threshold_failures(fieldnames: list[str], rows: list[dict[str, str]]) -> list[str]:
    if "preliminary_threshold_failures" in fieldnames:
        failures = sorted({str(row.get("preliminary_threshold_failures", "")) for row in rows if row.get("preliminary_threshold_failures", "")})
        return [failure for failure in failures if failure]
    if "preliminary_threshold_pass" in fieldnames:
        failing = [str(row.get("method", "")) or "<missing>" for row in rows if int(finite_float(row.get("preliminary_threshold_pass"), 0.0)) != 1]
        return [f"method threshold failure: {method}" for method in failing]
    return []


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{method}={count}" for method, count in sorted(counts.items()))


def _format_float_counts(counts: dict[str, float]) -> str:
    return ", ".join(f"{method}={value:.3f}" for method, value in sorted(counts.items()))


def _format_guardrail_failures(failures: dict[str, float]) -> str:
    return ", ".join(f"{key} delta={value:.6g}" for key, value in sorted(failures.items()))


def _interpretation(status: str) -> str:
    if status == "promotion_ready":
        return "Local grouped true-scanner evidence is strong enough to compare candidates, but final ranking still requires organizer hidden-holdout execution."
    if status == "limited_true_scanner_evidence":
        return "Rows appear to come from a true scanner, but the evidence is too narrow for promotion to final candidate."
    return "Rows are not reliable challenge evidence for promotion."


def _promotion_rank_key(decision: dict[str, object]) -> tuple[float, float, float, float, int, float]:
    return (
        1.0 if decision.get("promote") else 0.0,
        finite_float(decision.get("official_score_delta"), float("-inf")),
        finite_float(decision.get("MS-SSIM_delta"), float("-inf")),
        -finite_float(decision.get("LPIPS_or_proxy_delta"), float("inf")),
        int(decision.get("candidate_MS-SSIM_wins", 0)),
        -finite_float(decision.get("candidate_generation_seconds_max"), float("inf")),
    )


def _promotion_interpretation(promote: bool) -> str:
    if promote:
        return "Candidate beats the named baseline under local grouped true-scanner full-frame evidence; hidden-holdout organizer execution is still required for final ranking."
    return "Candidate is not promoted under the local fair-evidence gate."


def _selection_interpretation(promote: bool) -> str:
    if promote:
        return "Selected candidate is the best local promoted method under the fair-evidence gate; hidden-holdout organizer execution is still required for final ranking."
    return "No candidate beats the baseline under the local fair-evidence gate."


def _load_submission_readiness(path: Path, issues: list[str]) -> dict[str, object] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(f"invalid submission readiness report JSON: {exc}")
    return None


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _readiness_interpretation(status: str) -> str:
    if status == "local_candidate_ready":
        return "Local evidence supports this candidate for submission preparation; official hidden-holdout execution is still required to establish final ranking."
    return "Local evidence is not sufficient to treat this as the current fair challenge submission candidate."
