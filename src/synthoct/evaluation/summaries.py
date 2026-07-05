from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HIGHER_IS_BETTER_GUARDRAILS = (
    "DepthCorr",
    "LateralCorr",
    "OACProfileCorr",
    "OAC_MS-SSIM",
    "SC_MS-SSIM",
    "RSC_MS-SSIM",
)
LOWER_IS_BETTER_GUARDRAILS = ("SCMeanAbsErr",)
OFFICIAL_EVALUATION_MAPS = ("Struct", "OAC", "SC", "RSC")
PRELIMINARY_MS_SSIM_THRESHOLDS = {
    "Struct": 0.3,
    "OAC": 0.4,
    "SC": 0.4,
    "RSC": 0.5,
}
PRELIMINARY_LPIPS_THRESHOLD = 0.4


def challenge_lpips_key(row: dict[str, float | str], suffix: str = "") -> str:
    lpips_key = f"Struct_LPIPS{suffix}"
    proxy_key = f"Struct_LPIPS_PROXY{suffix}"
    lpips_value = finite_float(row.get(lpips_key), np.nan)
    return lpips_key if np.isfinite(lpips_value) else proxy_key


def finite_float(value: float | str | None, default: float = np.nan) -> float:
    try:
        out = float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def summarize_rows(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    numeric_keys = sorted(k for row in rows for k, v in row.items() if k not in {"method", "archive_path"} and isinstance(v, (int, float)))
    by_method: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)
    summary = []
    for method, method_rows in sorted(by_method.items()):
        out: dict[str, float | str] = {"method": method, "n": len(method_rows)}
        for key in numeric_keys:
            values = np.array([float(r[key]) for r in method_rows if key in r and np.isfinite(float(r[key]))], dtype=float)
            if values.size:
                out[f"{key}_mean"] = float(values.mean())
                out[f"{key}_std"] = float(values.std(ddof=0))
        summary.append(out)
    summary.sort(
        key=lambda r: (
            finite_float(r.get("Struct_MS-SSIM_mean"), -1.0),
            -finite_float(r.get(challenge_lpips_key(r, "_mean")), np.inf),
        ),
        reverse=True,
    )
    return summary


def summarize_challenge_metrics(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    by_method: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    by_sample: dict[tuple[str, str, str], list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)
        by_sample[(str(row["fold"]), str(row["sample"]), str(row["archive_path"]))].append(row)

    ms_wins: dict[str, int] = defaultdict(int)
    lpips_wins: dict[str, int] = defaultdict(int)
    for sample_rows in by_sample.values():
        ms_sorted = sorted(sample_rows, key=lambda row: finite_float(row.get("Struct_MS-SSIM"), -1.0), reverse=True)
        if ms_sorted:
            ms_wins[str(ms_sorted[0]["method"])] += 1
        lpips_sorted = sorted(sample_rows, key=lambda row: finite_float(row.get(challenge_lpips_key(row)), np.inf))
        if lpips_sorted:
            lpips_wins[str(lpips_sorted[0]["method"])] += 1

    out: list[dict[str, float | str]] = []
    for method, method_rows in by_method.items():
        lpips_key = challenge_lpips_key(method_rows[0])
        ms_values = np.array([finite_float(row.get("Struct_MS-SSIM")) for row in method_rows], dtype=float)
        lpips_values = np.array([finite_float(row.get(lpips_key)) for row in method_rows], dtype=float)
        generation_values = np.array([finite_float(row.get("generation_seconds")) for row in method_rows], dtype=float)
        ms_values = ms_values[np.isfinite(ms_values)]
        lpips_values = lpips_values[np.isfinite(lpips_values)]
        generation_values = generation_values[np.isfinite(generation_values)]
        guardrails = _guardrail_means(method_rows)
        official = _official_ranking_summary(method_rows)
        out.append(
            {
                "method": method,
                "evidence_source": _common_value(method_rows, "evidence_source"),
                "evidence_scope": _common_value(method_rows, "evidence_scope"),
                "evaluation_region": _common_value(method_rows, "Struct_evaluation_region"),
                "evaluated_shape": _common_value(method_rows, "Struct_evaluated_shape"),
                "prediction_resized_to_reference": _common_value(method_rows, "Struct_prediction_resized_to_reference"),
                "n": len(method_rows),
                "generation_seconds_mean": float(generation_values.mean()) if generation_values.size else np.nan,
                "generation_seconds_max": float(generation_values.max()) if generation_values.size else np.nan,
                "MS-SSIM_mean": float(ms_values.mean()) if ms_values.size else np.nan,
                "MS-SSIM_std": float(ms_values.std(ddof=0)) if ms_values.size else np.nan,
                "LPIPS_metric": lpips_key.replace("Struct_", ""),
                "LPIPS_or_proxy_mean": float(lpips_values.mean()) if lpips_values.size else np.nan,
                "LPIPS_or_proxy_std": float(lpips_values.std(ddof=0)) if lpips_values.size else np.nan,
                "MS-SSIM_wins": int(ms_wins[method]),
                "LPIPS_wins": int(lpips_wins[method]),
                **official,
                **guardrails,
            }
        )
    out.sort(
        key=lambda row: (
            int(finite_float(row.get("official_metric_complete"), 0.0)),
            finite_float(row.get("official_score"), -1.0),
            finite_float(row.get("MS-SSIM_mean"), -1.0),
            -finite_float(row.get("LPIPS_or_proxy_mean"), np.inf),
        ),
        reverse=True,
    )
    return out


def _common_value(rows: list[dict[str, float | str]], key: str) -> str:
    values = {str(row.get(key, "")) for row in rows if row.get(key, "") != ""}
    if not values:
        return ""
    if len(values) == 1:
        return next(iter(values))
    return "mixed"


def _guardrail_means(rows: list[dict[str, float | str]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in (*HIGHER_IS_BETTER_GUARDRAILS, *LOWER_IS_BETTER_GUARDRAILS):
        values = np.array([finite_float(row.get(key)) for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            out[f"{key}_mean"] = float(values.mean())
    return out


def _official_ranking_summary(rows: list[dict[str, float | str]]) -> dict[str, float | str]:
    """Compute the handout-described median score when all official metrics exist."""
    out: dict[str, float | str] = {}
    score_terms: list[float] = []
    missing: list[str] = []
    medians: dict[str, tuple[float, float]] = {}
    for map_name in OFFICIAL_EVALUATION_MAPS:
        ms_values = _finite_values(rows, f"{map_name}_MS-SSIM")
        lpips_values = _finite_values(rows, f"{map_name}_LPIPS")
        if ms_values.size:
            median_ms = float(np.median(ms_values))
            out[f"{map_name}_MS-SSIM_median"] = median_ms
            score_terms.append(median_ms)
        else:
            missing.append(f"{map_name}_MS-SSIM")
        if lpips_values.size:
            median_lpips = float(np.median(lpips_values))
            out[f"{map_name}_LPIPS_median"] = median_lpips
            out[f"{map_name}_LPIPS_inverted_median"] = float(1.0 - median_lpips)
            score_terms.append(float(1.0 - median_lpips))
            if f"{map_name}_MS-SSIM_median" in out:
                medians[map_name] = (float(out[f"{map_name}_MS-SSIM_median"]), median_lpips)
        else:
            missing.append(f"{map_name}_LPIPS")
    complete = not missing and len(score_terms) == len(OFFICIAL_EVALUATION_MAPS) * 2
    out["official_metric_complete"] = int(complete)
    out["official_score"] = float(np.mean(score_terms)) if complete else np.nan
    out["official_missing_metrics"] = ";".join(missing)
    threshold_failures: list[str] = []
    if complete:
        for map_name, ms_threshold in PRELIMINARY_MS_SSIM_THRESHOLDS.items():
            median_ms, median_lpips = medians[map_name]
            if median_ms <= ms_threshold:
                threshold_failures.append(f"{map_name}_MS-SSIM<={ms_threshold:g}")
            if median_lpips >= PRELIMINARY_LPIPS_THRESHOLD:
                threshold_failures.append(f"{map_name}_LPIPS>={PRELIMINARY_LPIPS_THRESHOLD:g}")
    else:
        threshold_failures.extend(missing)
    out["preliminary_threshold_pass"] = int(complete and not threshold_failures)
    out["preliminary_threshold_failures"] = ";".join(threshold_failures)
    return out


def _finite_values(rows: list[dict[str, float | str]], key: str) -> np.ndarray:
    values = np.array([finite_float(row.get(key)) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def summarize_sample_wins(rows: list[dict[str, float | str]], score_key: str = "Struct_MS-SSIM") -> list[dict[str, float | str]]:
    by_sample: dict[tuple[str, str, str], list[dict[str, float | str]]] = defaultdict(list)
    by_method: dict[str, list[float]] = defaultdict(list)
    margins: dict[str, list[float]] = defaultdict(list)
    wins: dict[str, int] = defaultdict(int)

    for row in rows:
        sample_key = (str(row["fold"]), str(row["sample"]), str(row["archive_path"]))
        by_sample[sample_key].append(row)
        by_method[str(row["method"])].append(float(row.get(score_key, row.get("Composite", 0.0))))

    for sample_rows in by_sample.values():
        scored = [(str(row["method"]), float(row.get(score_key, row.get("Composite", 0.0)))) for row in sample_rows]
        if not scored:
            continue
        scored.sort(key=lambda item: item[1], reverse=True)
        best_method, best_score = scored[0]
        wins[best_method] += 1
        for method, score in scored:
            margins[method].append(score - best_score)

    out: list[dict[str, float | str]] = []
    sample_count = max(1, len(by_sample))
    for method in sorted(by_method):
        scores = np.array(by_method[method], dtype=float)
        method_margins = np.array(margins[method], dtype=float)
        out.append(
            {
                "method": method,
                "n": int(scores.size),
                "wins": int(wins[method]),
                "win_rate": float(wins[method] / sample_count),
                f"{score_key}_mean": float(scores.mean()),
                "mean_margin_to_sample_best": float(method_margins.mean()) if method_margins.size else 0.0,
            }
        )
    out.sort(key=lambda row: (int(row["wins"]), float(row[f"{score_key}_mean"])), reverse=True)
    return out


def write_ms_ssim_pair_comparison(
    base_metrics: str | Path,
    candidate_metrics: str | Path,
    out_path: str | Path,
    *,
    base_method: str = "base",
    candidate_method: str = "candidate",
    model_queue: str | Path | None = None,
) -> Path:
    """Write a per-row MS-SSIM comparison for two matched API metrics CSVs."""
    base_rows = _read_csv_rows(base_metrics)
    candidate_rows = _read_csv_rows(candidate_metrics)
    base_by_source = {str(row.get("source_archive_path", "")): row for row in base_rows}
    queue_by_source = (
        {str(row.get("source_archive_path", "")): row for row in _read_csv_rows(model_queue)}
        if model_queue is not None
        else {}
    )
    ranked_base = {
        str(row.get("source_archive_path", "")): rank
        for rank, row in enumerate(
            sorted(base_rows, key=lambda item: finite_float(item.get("MS-SSIM"), np.inf)),
            start=1,
        )
    }
    comparison: list[dict[str, float | str | int]] = []
    missing: list[str] = []
    for idx, candidate in enumerate(candidate_rows, start=1):
        source = str(candidate.get("source_archive_path", ""))
        base = base_by_source.get(source)
        if base is None:
            missing.append(source)
            continue
        base_ms = finite_float(base.get("MS-SSIM"))
        cand_ms = finite_float(candidate.get("MS-SSIM"))
        delta = cand_ms - base_ms if np.isfinite(base_ms) and np.isfinite(cand_ms) else np.nan
        queue_row = queue_by_source.get(source, {})
        comparison.append(
            {
                "row": idx,
                "base_rank_by_ms_ssim": ranked_base.get(source, ""),
                "source_archive_path": source,
                "base_method": base_method,
                "candidate_method": candidate_method,
                "base_ms_ssim": base_ms,
                "candidate_ms_ssim": cand_ms,
                "delta_ms_ssim": delta,
                "winner": "candidate" if delta > 0 else "base" if delta < 0 else "tie",
                "base_lpips": base.get("LPIPS", base.get("LPIPS_PROXY", "")),
                "candidate_lpips": candidate.get("LPIPS", candidate.get("LPIPS_PROXY", "")),
                "model_predicted_delta_lcb": queue_row.get("expected_delta_lcb", ""),
                "model_selected_strength": queue_row.get("selected_strength", ""),
                "model_residual_policy": queue_row.get("residual_control_policy", ""),
                "model_map_safety_status": queue_row.get("map_safety_status", ""),
            }
        )
    if missing:
        raise ValueError(f"Candidate metrics contain sources missing from base metrics: {missing[:5]}")

    out = Path(out_path)
    write_rows(out, comparison)
    finite_deltas = np.array([finite_float(row.get("delta_ms_ssim")) for row in comparison], dtype=float)
    finite_deltas = finite_deltas[np.isfinite(finite_deltas)]
    summary = {
        "comparison_csv": str(out),
        "base_api_metrics": str(Path(base_metrics)),
        "candidate_api_metrics": str(Path(candidate_metrics)),
        "base_method": base_method,
        "candidate_method": candidate_method,
        "row_count": len(comparison),
        "base_ms_ssim_mean": _mean_from_rows(comparison, "base_ms_ssim"),
        "candidate_ms_ssim_mean": _mean_from_rows(comparison, "candidate_ms_ssim"),
        "delta_ms_ssim_mean": float(finite_deltas.mean()) if finite_deltas.size else np.nan,
        "delta_ms_ssim_median": float(np.median(finite_deltas)) if finite_deltas.size else np.nan,
        "delta_ms_ssim_min": float(finite_deltas.min()) if finite_deltas.size else np.nan,
        "delta_ms_ssim_max": float(finite_deltas.max()) if finite_deltas.size else np.nan,
        "candidate_wins": sum(row["winner"] == "candidate" for row in comparison),
        "base_wins": sum(row["winner"] == "base" for row in comparison),
        "ties": sum(row["winner"] == "tie" for row in comparison),
        "model_queue": str(Path(model_queue)) if model_queue is not None else "",
        "model_queue_row_count": len(queue_by_source),
        "evidence_source": "hosted_api_true_scanner",
        "evidence_scope": "full_public_ms_ssim_pair_comparison",
        "hidden_holdout_final_score": False,
    }
    out.with_name(f"{out.stem}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out


def _read_csv_rows(path: str | Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def _mean_from_rows(rows: list[dict[str, float | str | int]], key: str) -> float:
    values = np.array([finite_float(row.get(key)) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else np.nan


def write_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
