from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from .summaries import finite_float

TRUE_SCANNER_SOURCES = {"hosted_api_true_scanner", "official_windows_true_scanner"}
GROUPED_VALIDATION_PREFIX = "grouped_validation_"


def audit_challenge_evidence(
    metrics_csv: str | Path,
    *,
    min_samples_per_method: int = 2,
    require_real_lpips: bool = False,
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

    has_real_lpips = _has_real_lpips(fieldnames, rows)
    has_lpips_proxy = _has_metric(fieldnames, rows, ("LPIPS_or_proxy_mean", "Struct_LPIPS_PROXY", "LPIPS_PROXY"))
    if require_real_lpips and not has_real_lpips:
        issues.append("missing finite real LPIPS metric")
    elif not has_real_lpips and has_lpips_proxy:
        warnings.append("LPIPS appears to be proxy/fallback rather than real LPIPS")

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
        "true_scanner_evidence": true_scanner,
        "grouped_validation_evidence": grouped_validation,
        "has_ms_ssim": _has_metric(fieldnames, rows, ("MS-SSIM_mean", "Struct_MS-SSIM", "MS-SSIM")),
        "has_real_lpips": has_real_lpips,
        "has_lpips_proxy": has_lpips_proxy,
        "issues": issues,
        "warnings": warnings,
        "interpretation": _interpretation(status),
    }


def _method_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        method = str(row.get("method", "")) or "<missing>"
        n_value = finite_float(row.get("n"))
        counts[method] += int(n_value) if n_value >= 1 else 1
    return dict(counts)


def _has_metric(fieldnames: list[str], rows: list[dict[str, str]], keys: tuple[str, ...]) -> bool:
    present = [key for key in keys if key in fieldnames]
    return any(any(finite_float(row.get(key)) == finite_float(row.get(key)) for row in rows) for key in present)


def _has_real_lpips(fieldnames: list[str], rows: list[dict[str, str]]) -> bool:
    if "LPIPS_metric" in fieldnames:
        return any(str(row.get("LPIPS_metric", "")).upper() == "LPIPS" and finite_float(row.get("LPIPS_or_proxy_mean")) == finite_float(row.get("LPIPS_or_proxy_mean")) for row in rows)
    return _has_metric(fieldnames, rows, ("Struct_LPIPS", "LPIPS"))


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{method}={count}" for method, count in sorted(counts.items()))


def _interpretation(status: str) -> str:
    if status == "promotion_ready":
        return "Local grouped true-scanner evidence is strong enough to compare candidates, but final ranking still requires organizer hidden-holdout execution."
    if status == "limited_true_scanner_evidence":
        return "Rows appear to come from a true scanner, but the evidence is too narrow for promotion to final candidate."
    return "Rows are not reliable challenge evidence for promotion."
