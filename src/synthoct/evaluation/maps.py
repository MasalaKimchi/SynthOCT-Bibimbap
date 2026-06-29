from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.transform import resize

from synthoct.features import calculate_oac, calculate_speckle_contrast_map, generate_maps, load_and_linearize_image, load_scan

from .metrics import calculate_metrics


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def resize_like(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return resize(arr, shape, anti_aliasing=True, preserve_range=True)


def profile_scores(ref_path: str | Path, pred_path: str | Path) -> dict[str, float]:
    """Compare depth, lateral, OAC, and speckle profiles for rendered OCT PNGs."""
    ref = load_scan(ref_path)
    pred = load_scan(pred_path)
    if ref.shape != pred.shape:
        pred = resize_like(pred, ref.shape)
    ref_depth = ref.mean(axis=1)
    pred_depth = pred.mean(axis=1)
    ref_lat = ref.mean(axis=0)
    pred_lat = pred.mean(axis=0)

    ref_oac = calculate_oac(load_and_linearize_image(ref_path))
    pred_oac = calculate_oac(load_and_linearize_image(pred_path))
    ref_sc = calculate_speckle_contrast_map(load_and_linearize_image(ref_path))
    pred_sc = calculate_speckle_contrast_map(load_and_linearize_image(pred_path))
    if ref_oac.shape != pred_oac.shape:
        pred_oac = resize_like(pred_oac, ref_oac.shape)
    if ref_sc.shape != pred_sc.shape:
        pred_sc = resize_like(pred_sc, ref_sc.shape)
    return {
        "DepthCorr": safe_corr(ref_depth, pred_depth),
        "LateralCorr": safe_corr(ref_lat, pred_lat),
        "OACProfileCorr": safe_corr(ref_oac.mean(axis=1), pred_oac.mean(axis=1)),
        "SCMeanAbsErr": float(abs(np.mean(ref_sc) - np.mean(pred_sc))),
    }


def evaluate_feature_map_metrics(
    ref_path: str | Path,
    pred_path: str | Path,
    ref_output_dir: str | Path,
    pred_output_dir: str | Path,
    *,
    include_lpips: bool = False,
) -> dict[str, float]:
    """Generate and compare OAC, SC, and RSC maps for rendered OCT PNGs."""
    ref_maps = generate_maps(ref_path, output_dir=ref_output_dir)
    pred_maps = generate_maps(pred_path, output_dir=pred_output_dir)
    row: dict[str, float] = {}
    for map_name in ("OAC", "SC", "RSC"):
        map_metrics = calculate_metrics(ref_maps[map_name], pred_maps[map_name], include_lpips=include_lpips)
        for key, value in map_metrics.items():
            row[f"{map_name}_{key}"] = value
    return row


def composite_score(row: dict[str, float | str]) -> float:
    ms = float(row.get("Struct_MS-SSIM", np.nan))
    lpips = float(row.get("Struct_LPIPS", np.nan))
    lpips_proxy = float(row.get("Struct_LPIPS_PROXY", np.nan))
    ssim = float(row.get("Struct_SSIM", np.nan))
    depth = float(row.get("DepthCorr", 0.0))
    oac = float(row.get("OACProfileCorr", 0.0))
    sc_err = float(row.get("SCMeanAbsErr", 1.0))
    oac_ssim = float(row.get("OAC_SSIM", np.nan))
    sc_ssim = float(row.get("SC_SSIM", np.nan))
    rsc_ssim = float(row.get("RSC_SSIM", np.nan))
    if np.isnan(ms):
        ms = ssim
    if np.isnan(lpips):
        lpips = lpips_proxy if not np.isnan(lpips_proxy) else 0.5
    map_terms = [v for v in (oac_ssim, sc_ssim, rsc_ssim) if not np.isnan(v)]
    map_score = float(np.mean(map_terms)) if map_terms else oac
    return float(
        0.25 * ms
        + 0.15 * (1.0 - lpips)
        + 0.10 * ssim
        + 0.15 * depth
        + 0.10 * oac
        + 0.20 * map_score
        + 0.05 * (1.0 / (1.0 + sc_err))
    )


def competition_proxy_score(row: dict[str, float | str]) -> float:
    """Internal triage score: MS-SSIM up, LPIPS down, physical maps as guardrails."""
    struct_ms = float(row.get("Struct_MS-SSIM", np.nan))
    struct_ssim = float(row.get("Struct_SSIM", np.nan))
    lpips = float(row.get("Struct_LPIPS", np.nan))
    lpips_proxy = float(row.get("Struct_LPIPS_PROXY", np.nan))
    if np.isnan(struct_ms):
        struct_ms = struct_ssim
    if np.isnan(lpips):
        lpips = lpips_proxy if not np.isnan(lpips_proxy) else 0.5
    map_ms_values = [
        float(row.get("OAC_MS-SSIM", np.nan)),
        float(row.get("SC_MS-SSIM", np.nan)),
        float(row.get("RSC_MS-SSIM", np.nan)),
    ]
    map_ssim_values = [
        float(row.get("OAC_SSIM", np.nan)),
        float(row.get("SC_SSIM", np.nan)),
        float(row.get("RSC_SSIM", np.nan)),
    ]
    map_terms = [v for v in map_ms_values if not np.isnan(v)] or [v for v in map_ssim_values if not np.isnan(v)]
    map_score = float(np.mean(map_terms)) if map_terms else float(row.get("OACProfileCorr", 0.0))
    return float(0.45 * struct_ms + 0.35 * (1.0 - lpips) + 0.20 * map_score)
