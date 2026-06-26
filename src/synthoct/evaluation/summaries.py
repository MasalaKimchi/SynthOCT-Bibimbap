from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


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
        ms_values = ms_values[np.isfinite(ms_values)]
        lpips_values = lpips_values[np.isfinite(lpips_values)]
        out.append(
            {
                "method": method,
                "n": len(method_rows),
                "MS-SSIM_mean": float(ms_values.mean()) if ms_values.size else np.nan,
                "MS-SSIM_std": float(ms_values.std(ddof=0)) if ms_values.size else np.nan,
                "LPIPS_metric": lpips_key.replace("Struct_", ""),
                "LPIPS_or_proxy_mean": float(lpips_values.mean()) if lpips_values.size else np.nan,
                "LPIPS_or_proxy_std": float(lpips_values.std(ddof=0)) if lpips_values.size else np.nan,
                "MS-SSIM_wins": int(ms_wins[method]),
                "LPIPS_wins": int(lpips_wins[method]),
            }
        )
    out.sort(key=lambda row: (finite_float(row.get("MS-SSIM_mean"), -1.0), -finite_float(row.get("LPIPS_or_proxy_mean"), np.inf)), reverse=True)
    return out


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


def write_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
