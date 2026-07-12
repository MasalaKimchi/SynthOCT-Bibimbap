from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from skimage.transform import resize

from synthoct.features import (
    ORGANIZER_MAP_MODE,
    SCIENTIFIC_MAP_MODE,
    ScientificMapConfig,
    calculate_oac,
    calculate_scientific_map_arrays,
    calculate_speckle_contrast_map,
    generate_maps,
    load_and_linearize_image,
    load_scan,
)

from .metrics import calculate_metrics


COMPETITION_MAPS = ("Struct", "OAC", "SC", "RSC")


def competition_formula_estimate(
    rows: Mapping[str, float | str] | Sequence[Mapping[str, float | str]],
    *,
    map_mode: str = ORGANIZER_MAP_MODE,
) -> tuple[float, dict[str, float]]:
    """Apply the published eight-median formula to local metric rows.

    This is deliberately named an estimate: only the organizers can issue a
    leaderboard/final score.  Inputs must contain real LPIPS, never the proxy.
    """
    if isinstance(rows, Mapping):
        rows = [rows]
    if not rows:
        raise ValueError("at least one metric row is required")
    row_modes = {str(row.get("Map_Mode", map_mode)) for row in rows}
    if map_mode != ORGANIZER_MAP_MODE or row_modes != {ORGANIZER_MAP_MODE}:
        raise ValueError(
            "the published competition formula is defined only for "
            f"{ORGANIZER_MAP_MODE} maps"
        )

    medians: dict[str, float] = {}
    components: list[float] = []
    for map_name in COMPETITION_MAPS:
        ms_key = f"{map_name}_MS-SSIM"
        lpips_key = f"{map_name}_LPIPS"
        ms_values = np.asarray([float(row[ms_key]) for row in rows], dtype=float)
        lpips_values = np.asarray([float(row[lpips_key]) for row in rows], dtype=float)
        if not np.isfinite(ms_values).all() or not np.isfinite(lpips_values).all():
            raise ValueError(f"non-finite competition metric for {map_name}")
        ms_median = float(np.median(ms_values))
        lpips_median = float(np.median(lpips_values))
        medians[f"{map_name}_MS-SSIM_median"] = ms_median
        medians[f"{map_name}_LPIPS_median"] = lpips_median
        medians[f"{map_name}_LPIPS_inverted_median"] = 1.0 - lpips_median
        components.extend((ms_median, 1.0 - lpips_median))
    return float(np.mean(components)), medians


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def resize_like(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return resize(arr, shape, anti_aliasing=True, preserve_range=True)


def profile_scores(
    ref_path: str | Path,
    pred_path: str | Path,
    *,
    map_mode: str = ORGANIZER_MAP_MODE,
    scientific_config: ScientificMapConfig | None = None,
) -> dict[str, float]:
    """Compare depth, lateral, OAC, and speckle profiles for rendered OCT PNGs."""
    ref = load_scan(ref_path)
    pred = load_scan(pred_path)
    if ref.shape != pred.shape:
        pred = resize_like(pred, ref.shape)
    ref_depth = ref.mean(axis=1)
    pred_depth = pred.mean(axis=1)
    ref_lat = ref.mean(axis=0)
    pred_lat = pred.mean(axis=0)

    if map_mode == ORGANIZER_MAP_MODE:
        ref_intensity = load_and_linearize_image(ref_path)
        pred_intensity = load_and_linearize_image(pred_path)
        ref_oac = calculate_oac(ref_intensity)
        pred_oac = calculate_oac(pred_intensity)
        ref_sc = calculate_speckle_contrast_map(ref_intensity)
        pred_sc = calculate_speckle_contrast_map(pred_intensity)
        oac_valid = np.ones(ref_oac.shape, dtype=bool)
        sc_valid = np.ones(ref_sc.shape, dtype=bool)
    elif map_mode == SCIENTIFIC_MAP_MODE:
        config = scientific_config or ScientificMapConfig()
        ref_arrays = calculate_scientific_map_arrays(ref_path, config)
        pred_arrays = calculate_scientific_map_arrays(pred_path, config)
        ref_oac = ref_arrays["OAC"]
        pred_oac = pred_arrays["OAC"]
        ref_sc = ref_arrays["SC"]
        pred_sc = pred_arrays["SC"]
        if ref_oac.shape != pred_oac.shape:
            raise ValueError("scientific profile arrays must have identical shapes")
        oac_valid = ref_arrays["OAC_valid"] & np.isfinite(pred_oac)
        sc_valid = ref_arrays["SC_valid"] & np.isfinite(pred_sc)
    else:
        raise ValueError(f"unsupported map mode: {map_mode}")
    if ref_oac.shape != pred_oac.shape:
        pred_oac = resize_like(pred_oac, ref_oac.shape)
    if ref_sc.shape != pred_sc.shape:
        pred_sc = resize_like(pred_sc, ref_sc.shape)
    oac_rows = np.any(oac_valid, axis=1)
    ref_oac_profile = np.asarray(
        [np.mean(ref_oac[row, oac_valid[row]]) for row in np.flatnonzero(oac_rows)]
    )
    pred_oac_profile = np.asarray(
        [np.mean(pred_oac[row, oac_valid[row]]) for row in np.flatnonzero(oac_rows)]
    )
    return {
        "DepthCorr": safe_corr(ref_depth, pred_depth),
        "LateralCorr": safe_corr(ref_lat, pred_lat),
        "OACProfileCorr": safe_corr(ref_oac_profile, pred_oac_profile),
        "SCMeanAbsErr": float(
            abs(np.mean(ref_sc[sc_valid]) - np.mean(pred_sc[sc_valid]))
        ),
    }


def evaluate_feature_map_metrics(
    ref_path: str | Path,
    pred_path: str | Path,
    ref_output_dir: str | Path,
    pred_output_dir: str | Path,
    *,
    include_lpips: bool = False,
    map_mode: str = ORGANIZER_MAP_MODE,
) -> dict[str, float | str]:
    """Generate and compare OAC, SC, and RSC maps for rendered OCT PNGs."""
    ref_maps = generate_maps(ref_path, output_dir=ref_output_dir, mode=map_mode)
    pred_maps = generate_maps(pred_path, output_dir=pred_output_dir, mode=map_mode)
    # Preserve the historical organizer CSV schema so resumable retained runs
    # do not mix rows with different field sets. Scientific rows are tagged
    # explicitly because they must never enter the competition formula.
    row: dict[str, float | str] = (
        {"Map_Mode": map_mode} if map_mode == SCIENTIFIC_MAP_MODE else {}
    )
    if map_mode == SCIENTIFIC_MAP_MODE:
        if include_lpips:
            raise ValueError("LPIPS is not defined for masked scientific float maps")
        with np.load(ref_maps["FloatMaps"]) as ref_bundle, np.load(
            pred_maps["FloatMaps"]
        ) as pred_bundle:
            for map_name in ("OAC", "SC", "RSC"):
                ref = np.asarray(ref_bundle[map_name], dtype=np.float64)
                pred = np.asarray(pred_bundle[map_name], dtype=np.float64)
                if ref.shape != pred.shape:
                    raise ValueError(
                        f"scientific {map_name} arrays must have identical shapes: "
                        f"{ref.shape} != {pred.shape}"
                    )
                # The reference defines the support.  A prediction must not be
                # allowed to improve its score by declaring difficult pixels
                # invalid in its own mask.
                valid = np.asarray(ref_bundle[f"{map_name}_valid"], dtype=bool)
                valid &= np.isfinite(ref)
                if not np.any(valid):
                    raise ValueError(f"scientific {map_name} comparison has no valid pixels")
                if not np.isfinite(pred[valid]).all():
                    raise ValueError(
                        f"scientific {map_name} prediction is non-finite on the reference support"
                    )
                ref_valid = ref[valid]
                pred_valid = pred[valid]
                diff = pred_valid - ref_valid
                row[f"{map_name}_Masked_MSE"] = float(np.mean(diff**2))
                row[f"{map_name}_Masked_MAE"] = float(np.mean(np.abs(diff)))
                row[f"{map_name}_Masked_Bias"] = float(np.mean(diff))
                row[f"{map_name}_Masked_Corr"] = safe_corr(ref_valid, pred_valid)
                row[f"{map_name}_Valid_Fraction"] = float(np.mean(valid))
        return row
    if map_mode != ORGANIZER_MAP_MODE:
        raise ValueError(f"unsupported map mode: {map_mode}")
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
