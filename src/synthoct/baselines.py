from __future__ import annotations

from pathlib import Path

import numpy as np

from .cnn import cnn_embedding
from .phantom import ExperimentConfig, generate_two_layers, generate_uniform, save_phantom
from .processor import calculate_oac, calculate_speckle_contrast_map, load_and_linearize_image, load_scan


HYPOTHESIS_CONFIGS = {
    "H1_attenuation_density": {
        "density_power": 1.0,
        "depth_compensation": 1.8,
        "oac_weight": 2.0,
        "texture_weight": 0.35,
        "energy_oac_scale": 35.0,
        "void_fraction": 0.00,
        "band_boost": 0.00,
    },
    "H2_depth_compensated": {
        "density_power": 1.1,
        "depth_compensation": 2.6,
        "oac_weight": 2.2,
        "texture_weight": 0.30,
        "energy_oac_scale": 42.0,
        "void_fraction": 0.00,
        "band_boost": 0.00,
    },
    "H3_speckle_matched": {
        "density_power": 0.95,
        "depth_compensation": 1.7,
        "oac_weight": 1.6,
        "texture_weight": 0.75,
        "energy_oac_scale": 30.0,
        "void_fraction": 0.00,
        "band_boost": 0.00,
    },
    "H4_oac_dominant": {
        "density_power": 1.25,
        "depth_compensation": 2.0,
        "oac_weight": 3.4,
        "texture_weight": 0.20,
        "energy_oac_scale": 55.0,
        "void_fraction": 0.00,
        "band_boost": 0.00,
    },
    "H5_boundary_band": {
        "density_power": 1.0,
        "depth_compensation": 1.9,
        "oac_weight": 2.0,
        "texture_weight": 0.35,
        "energy_oac_scale": 35.0,
        "void_fraction": 0.00,
        "band_boost": 1.10,
    },
    "H6_void_inclusions": {
        "density_power": 1.05,
        "depth_compensation": 2.1,
        "oac_weight": 2.5,
        "texture_weight": 0.45,
        "energy_oac_scale": 38.0,
        "void_fraction": 0.08,
        "band_boost": 0.30,
    },
    "H7_lateral_coherence": {
        "density_power": 1.15,
        "depth_compensation": 2.2,
        "oac_weight": 2.4,
        "texture_weight": 0.30,
        "energy_oac_scale": 40.0,
        "void_fraction": 0.02,
        "band_boost": 0.50,
        "lateral_smooth": 1.8,
    },
    "H8_multilayer": {
        "density_power": 1.10,
        "depth_compensation": 2.4,
        "oac_weight": 2.8,
        "texture_weight": 0.40,
        "energy_oac_scale": 45.0,
        "void_fraction": 0.04,
        "band_boost": 0.80,
        "multi_layer": True,
    },
    "H9_log_energy": {
        "density_power": 0.90,
        "depth_compensation": 2.0,
        "oac_weight": 2.4,
        "texture_weight": 0.55,
        "energy_oac_scale": 28.0,
        "void_fraction": 0.03,
        "band_boost": 0.50,
        "log_energy": True,
    },
    "H10_refined_attenuation_band": {
        "density_power": 0.98,
        "depth_compensation": 1.85,
        "oac_weight": 2.15,
        "texture_weight": 0.38,
        "energy_oac_scale": 36.0,
        "void_fraction": 0.00,
        "band_boost": 0.28,
        "lateral_smooth": 0.95,
    },
    "H11_low_depth_comp": {
        "density_power": 1.0,
        "depth_compensation": 1.45,
        "oac_weight": 2.0,
        "texture_weight": 0.35,
        "energy_oac_scale": 35.0,
        "band_boost": 0.0,
    },
    "H12_mid_depth_comp": {
        "density_power": 1.0,
        "depth_compensation": 1.65,
        "oac_weight": 2.0,
        "texture_weight": 0.35,
        "energy_oac_scale": 35.0,
        "band_boost": 0.0,
    },
    "H13_high_depth_comp": {
        "density_power": 1.0,
        "depth_compensation": 2.05,
        "oac_weight": 2.0,
        "texture_weight": 0.35,
        "energy_oac_scale": 35.0,
        "band_boost": 0.0,
    },
    "H14_low_oac_weight": {
        "density_power": 1.0,
        "depth_compensation": 1.8,
        "oac_weight": 1.65,
        "texture_weight": 0.35,
        "energy_oac_scale": 32.0,
        "band_boost": 0.0,
    },
    "H15_high_oac_weight": {
        "density_power": 1.0,
        "depth_compensation": 1.8,
        "oac_weight": 2.35,
        "texture_weight": 0.35,
        "energy_oac_scale": 40.0,
        "band_boost": 0.0,
    },
    "H16_sublinear_density": {
        "density_power": 0.82,
        "depth_compensation": 1.8,
        "oac_weight": 2.0,
        "texture_weight": 0.35,
        "energy_oac_scale": 35.0,
        "band_boost": 0.0,
    },
    "H17_superlinear_density": {
        "density_power": 1.18,
        "depth_compensation": 1.8,
        "oac_weight": 2.0,
        "texture_weight": 0.35,
        "energy_oac_scale": 35.0,
        "band_boost": 0.0,
    },
    "H18_texture_light": {
        "density_power": 1.0,
        "depth_compensation": 1.8,
        "oac_weight": 2.0,
        "texture_weight": 0.22,
        "energy_oac_scale": 35.0,
        "band_boost": 0.0,
    },
    "H19_texture_heavy": {
        "density_power": 1.0,
        "depth_compensation": 1.8,
        "oac_weight": 2.0,
        "texture_weight": 0.52,
        "energy_oac_scale": 35.0,
        "band_boost": 0.0,
    },
    "H20_weak_boundary": {
        "density_power": 1.0,
        "depth_compensation": 1.8,
        "oac_weight": 2.0,
        "texture_weight": 0.35,
        "energy_oac_scale": 35.0,
        "band_boost": 0.18,
    },
    "H21_medium_boundary": {
        "density_power": 1.0,
        "depth_compensation": 1.8,
        "oac_weight": 2.0,
        "texture_weight": 0.35,
        "energy_oac_scale": 35.0,
        "band_boost": 0.45,
    },
    "H22_low_energy_scale": {
        "density_power": 1.0,
        "depth_compensation": 1.8,
        "oac_weight": 2.0,
        "texture_weight": 0.35,
        "energy_oac_scale": 25.0,
        "band_boost": 0.0,
    },
    "H23_high_energy_scale": {
        "density_power": 1.0,
        "depth_compensation": 1.8,
        "oac_weight": 2.0,
        "texture_weight": 0.35,
        "energy_oac_scale": 48.0,
        "band_boost": 0.0,
    },
    "H24_lateral_sharp": {
        "density_power": 1.0,
        "depth_compensation": 1.8,
        "oac_weight": 2.0,
        "texture_weight": 0.35,
        "energy_oac_scale": 35.0,
        "lateral_smooth": 0.35,
    },
    "H25_lateral_smooth": {
        "density_power": 1.0,
        "depth_compensation": 1.8,
        "oac_weight": 2.0,
        "texture_weight": 0.35,
        "energy_oac_scale": 35.0,
        "lateral_smooth": 1.45,
    },
    "H26_gentle_log_energy": {
        "density_power": 1.0,
        "depth_compensation": 1.8,
        "oac_weight": 2.0,
        "texture_weight": 0.35,
        "energy_oac_scale": 35.0,
        "log_energy": True,
    },
    "H27_sparse_voids": {
        "density_power": 1.0,
        "depth_compensation": 1.8,
        "oac_weight": 2.0,
        "texture_weight": 0.35,
        "energy_oac_scale": 35.0,
        "void_fraction": 0.025,
    },
    "H28_boundary_void_combo": {
        "density_power": 1.0,
        "depth_compensation": 1.8,
        "oac_weight": 2.0,
        "texture_weight": 0.35,
        "energy_oac_scale": 35.0,
        "band_boost": 0.25,
        "void_fraction": 0.025,
    },
    "H29_low_depth_high_oac": {
        "density_power": 1.0,
        "depth_compensation": 1.55,
        "oac_weight": 2.35,
        "texture_weight": 0.35,
        "energy_oac_scale": 40.0,
    },
    "H30_high_depth_low_oac": {
        "density_power": 1.0,
        "depth_compensation": 2.05,
        "oac_weight": 1.65,
        "texture_weight": 0.35,
        "energy_oac_scale": 32.0,
    },
    "H31_sublinear_boundary": {
        "density_power": 0.88,
        "depth_compensation": 1.8,
        "oac_weight": 2.0,
        "texture_weight": 0.35,
        "energy_oac_scale": 35.0,
        "band_boost": 0.25,
    },
    "H32_superlinear_boundary": {
        "density_power": 1.12,
        "depth_compensation": 1.8,
        "oac_weight": 2.0,
        "texture_weight": 0.35,
        "energy_oac_scale": 35.0,
        "band_boost": 0.25,
    },
    "H33_speckle_oac_balance": {
        "density_power": 0.96,
        "depth_compensation": 1.75,
        "oac_weight": 2.15,
        "texture_weight": 0.48,
        "energy_oac_scale": 38.0,
        "lateral_smooth": 0.75,
    },
    "H34_epidermal_emphasis": {
        "density_power": 0.95,
        "depth_compensation": 1.35,
        "oac_weight": 2.2,
        "texture_weight": 0.35,
        "energy_oac_scale": 42.0,
        "band_boost": 0.35,
    },
    "H35_deep_dermis_emphasis": {
        "density_power": 1.05,
        "depth_compensation": 2.15,
        "oac_weight": 1.9,
        "texture_weight": 0.35,
        "energy_oac_scale": 34.0,
        "band_boost": 0.15,
    },
    "H36_multi_layer_light": {
        "density_power": 0.98,
        "depth_compensation": 1.75,
        "oac_weight": 2.0,
        "texture_weight": 0.35,
        "energy_oac_scale": 35.0,
        "multi_layer": True,
    },
    "H37_multi_layer_smooth": {
        "density_power": 0.95,
        "depth_compensation": 1.7,
        "oac_weight": 2.0,
        "texture_weight": 0.30,
        "energy_oac_scale": 35.0,
        "multi_layer": True,
        "lateral_smooth": 1.4,
    },
    "H38_h1_h5_midpoint": {
        "density_power": 1.0,
        "depth_compensation": 1.85,
        "oac_weight": 2.0,
        "texture_weight": 0.35,
        "energy_oac_scale": 35.0,
        "band_boost": 0.55,
    },
    "H39_best_pair_tuned": {
        "density_power": 0.96,
        "depth_compensation": 1.72,
        "oac_weight": 2.08,
        "texture_weight": 0.32,
        "energy_oac_scale": 36.0,
        "band_boost": 0.12,
        "lateral_smooth": 0.85,
    },
    "H40_conservative_winner": {
        "density_power": 0.98,
        "depth_compensation": 1.68,
        "oac_weight": 1.95,
        "texture_weight": 0.30,
        "energy_oac_scale": 34.0,
        "band_boost": 0.08,
        "lateral_smooth": 0.9,
    },
    "H41_final_optimized": {
        "density_power": 1.1950643094718914,
        "depth_compensation": 1.232449572578015,
        "oac_weight": 2.1664221698464448,
        "texture_weight": 0.3718513055230929,
        "energy_oac_scale": 28.504499297377823,
        "band_boost": 0.32309648245405514,
        "lateral_smooth": 0.5263087292628759,
        "oac_percentile": 81.52821510765463,
        "base_energy_mix": 0.7099850978498786,
        "texture_sigma_scale": 1.2453729328071115,
    },
    "H61_api_low_depth_prelim": {
        "density_power": 0.90,
        "depth_compensation": -0.80,
        "oac_weight": 1.20,
        "texture_weight": 0.22,
        "energy_oac_scale": 10.0,
        "band_boost": 0.00,
        "lateral_smooth": 0.90,
        "oac_percentile": 70.0,
        "base_energy_mix": 0.40,
        "texture_sigma_scale": 0.70,
    },
}

FINAL_CONFIG_NAME = "H56_h41_anti_anatomy"
PORTFOLIO_CONFIGS = ("H41_final_optimized", "H11_low_depth_comp", "H34_epidermal_emphasis", "H29_low_depth_high_oac")
AGENT_METHOD_CONFIGS = {
    "agent_inverse_psf": {
        "density_power": 1.08,
        "depth_compensation": 1.35,
        "oac_weight": 2.45,
        "texture_weight": 0.22,
        "energy_oac_scale": 31.0,
        "band_boost": 0.18,
        "lateral_smooth": 0.45,
        "oac_percentile": 82.0,
        "base_energy_mix": 0.74,
        "texture_sigma_scale": 0.75,
    },
    "agent_speckle_moment": {
        "density_power": 0.94,
        "depth_compensation": 1.42,
        "oac_weight": 1.90,
        "texture_weight": 0.68,
        "energy_oac_scale": 30.0,
        "band_boost": 0.05,
        "lateral_smooth": 0.72,
        "oac_percentile": 68.0,
        "base_energy_mix": 0.50,
        "texture_sigma_scale": 1.45,
    },
    "agent_anatomical_boundary": {
        "density_power": 1.04,
        "depth_compensation": 1.32,
        "oac_weight": 2.10,
        "texture_weight": 0.30,
        "energy_oac_scale": 37.0,
        "band_boost": 0.70,
        "lateral_smooth": 1.15,
        "oac_percentile": 72.0,
        "base_energy_mix": 0.58,
        "texture_sigma_scale": 0.95,
    },
    "agent_perceptual_retrieval": {
        "density_power": 1.16,
        "depth_compensation": 1.24,
        "oac_weight": 2.20,
        "texture_weight": 0.42,
        "energy_oac_scale": 29.0,
        "band_boost": 0.30,
        "lateral_smooth": 0.52,
        "oac_percentile": 80.0,
        "base_energy_mix": 0.70,
        "texture_sigma_scale": 1.20,
    },
}


def _blend_configs(*weighted_configs: tuple[float, dict[str, float | bool]]) -> dict[str, float | bool]:
    keys = sorted({key for _, config in weighted_configs for key in config})
    total = sum(weight for weight, _ in weighted_configs)
    blended: dict[str, float | bool] = {}
    for key in keys:
        values = [(weight, config[key]) for weight, config in weighted_configs if key in config]
        if any(isinstance(value, bool) for _, value in values):
            blended[key] = sum(weight for weight, value in values if bool(value)) / total >= 0.5
        else:
            blended[key] = float(sum(weight * float(value) for weight, value in values) / sum(weight for weight, _ in values))
    return blended


def _extrapolate_config(
    base: dict[str, float | bool],
    other: dict[str, float | bool],
    amount: float,
) -> dict[str, float | bool]:
    out = dict(base)
    bounds = {
        "density_power": (0.75, 1.45),
        "depth_compensation": (0.85, 2.4),
        "oac_weight": (1.45, 3.2),
        "texture_weight": (0.05, 0.9),
        "energy_oac_scale": (20.0, 55.0),
        "band_boost": (0.0, 1.1),
        "lateral_smooth": (0.25, 1.8),
        "oac_percentile": (55.0, 88.0),
        "base_energy_mix": (0.35, 0.9),
        "texture_sigma_scale": (0.55, 1.7),
    }
    for key, base_value in base.items():
        if isinstance(base_value, bool) or key not in other or isinstance(other[key], bool):
            continue
        value = float(base_value) + amount * (float(base_value) - float(other[key]))
        lo, hi = bounds.get(key, (-np.inf, np.inf))
        out[key] = float(np.clip(value, lo, hi))
    return out


COUNCIL_COMBO_CONFIGS = {
    "H42_h41_anatomy_blend": _blend_configs(
        (0.70, HYPOTHESIS_CONFIGS["H41_final_optimized"]),
        (0.30, AGENT_METHOD_CONFIGS["agent_anatomical_boundary"]),
    ),
    "H43_h41_oac_guarded": _blend_configs(
        (0.62, HYPOTHESIS_CONFIGS["H41_final_optimized"]),
        (0.25, HYPOTHESIS_CONFIGS["H11_low_depth_comp"]),
        (0.13, AGENT_METHOD_CONFIGS["agent_anatomical_boundary"]),
    ),
    "H44_h41_perceptual_blend": _blend_configs(
        (0.75, HYPOTHESIS_CONFIGS["H41_final_optimized"]),
        (0.25, AGENT_METHOD_CONFIGS["agent_perceptual_retrieval"]),
    ),
    "H45_h41_speckle_calibrated": _blend_configs(
        (0.72, HYPOTHESIS_CONFIGS["H41_final_optimized"]),
        (0.28, AGENT_METHOD_CONFIGS["agent_speckle_moment"]),
    ),
    "H46_h41_inverse_structure": _blend_configs(
        (0.74, HYPOTHESIS_CONFIGS["H41_final_optimized"]),
        (0.26, AGENT_METHOD_CONFIGS["agent_inverse_psf"]),
    ),
    "H47_h41_anatomy_perceptual": _blend_configs(
        (0.60, HYPOTHESIS_CONFIGS["H41_final_optimized"]),
        (0.25, AGENT_METHOD_CONFIGS["agent_anatomical_boundary"]),
        (0.15, AGENT_METHOD_CONFIGS["agent_perceptual_retrieval"]),
    ),
    "H48_h41_council_balanced": _blend_configs(
        (0.55, HYPOTHESIS_CONFIGS["H41_final_optimized"]),
        (0.20, HYPOTHESIS_CONFIGS["H11_low_depth_comp"]),
        (0.15, AGENT_METHOD_CONFIGS["agent_anatomical_boundary"]),
        (0.10, AGENT_METHOD_CONFIGS["agent_perceptual_retrieval"]),
    ),
    "H49_h41_structural_guarded": _blend_configs(
        (0.50, HYPOTHESIS_CONFIGS["H41_final_optimized"]),
        (0.25, AGENT_METHOD_CONFIGS["agent_anatomical_boundary"]),
        (0.15, AGENT_METHOD_CONFIGS["agent_inverse_psf"]),
        (0.10, HYPOTHESIS_CONFIGS["H34_epidermal_emphasis"]),
    ),
    "H50_h41_texture_guarded": _blend_configs(
        (0.60, HYPOTHESIS_CONFIGS["H41_final_optimized"]),
        (0.20, AGENT_METHOD_CONFIGS["agent_speckle_moment"]),
        (0.10, AGENT_METHOD_CONFIGS["agent_perceptual_retrieval"]),
        (0.10, HYPOTHESIS_CONFIGS["H33_speckle_oac_balance"]),
    ),
    "H51_h41_micro_perceptual": _blend_configs(
        (0.90, HYPOTHESIS_CONFIGS["H41_final_optimized"]),
        (0.10, AGENT_METHOD_CONFIGS["agent_perceptual_retrieval"]),
    ),
    "H52_h41_micro_anatomy": _blend_configs(
        (0.90, HYPOTHESIS_CONFIGS["H41_final_optimized"]),
        (0.10, AGENT_METHOD_CONFIGS["agent_anatomical_boundary"]),
    ),
    "H53_h41_micro_inverse": _blend_configs(
        (0.90, HYPOTHESIS_CONFIGS["H41_final_optimized"]),
        (0.10, AGENT_METHOD_CONFIGS["agent_inverse_psf"]),
    ),
    "H54_h41_micro_speckle": _blend_configs(
        (0.90, HYPOTHESIS_CONFIGS["H41_final_optimized"]),
        (0.10, AGENT_METHOD_CONFIGS["agent_speckle_moment"]),
    ),
    "H55_h41_micro_anatomy_perceptual": _blend_configs(
        (0.90, HYPOTHESIS_CONFIGS["H41_final_optimized"]),
        (0.05, AGENT_METHOD_CONFIGS["agent_anatomical_boundary"]),
        (0.05, AGENT_METHOD_CONFIGS["agent_perceptual_retrieval"]),
    ),
    "H56_h41_anti_anatomy": _extrapolate_config(
        HYPOTHESIS_CONFIGS["H41_final_optimized"],
        AGENT_METHOD_CONFIGS["agent_anatomical_boundary"],
        0.08,
    ),
    "H57_h41_anti_speckle": _extrapolate_config(
        HYPOTHESIS_CONFIGS["H41_final_optimized"],
        AGENT_METHOD_CONFIGS["agent_speckle_moment"],
        0.06,
    ),
    "H58_h41_anti_inverse": _extrapolate_config(
        HYPOTHESIS_CONFIGS["H41_final_optimized"],
        AGENT_METHOD_CONFIGS["agent_inverse_psf"],
        0.06,
    ),
    "H59_h41_anti_oac_guard": _extrapolate_config(
        HYPOTHESIS_CONFIGS["H41_final_optimized"],
        HYPOTHESIS_CONFIGS["H11_low_depth_comp"],
        0.05,
    ),
    "H60_h41_micro_portfolio_centroid": _blend_configs(
        (0.88, HYPOTHESIS_CONFIGS["H41_final_optimized"]),
        (0.04, HYPOTHESIS_CONFIGS["H11_low_depth_comp"]),
        (0.04, HYPOTHESIS_CONFIGS["H34_epidermal_emphasis"]),
        (0.04, HYPOTHESIS_CONFIGS["H29_low_depth_high_oac"]),
    ),
}
HYPOTHESIS_CONFIGS.update(COUNCIL_COMBO_CONFIGS)

INVERSE_WAVE_CONFIGS = {
    "H62_born_linearized": {
        "density_power": 1.05,
        "depth_compensation": 1.08,
        "oac_weight": 2.55,
        "texture_weight": 0.18,
        "energy_oac_scale": 26.0,
        "band_boost": 0.14,
        "lateral_smooth": 0.42,
        "oac_percentile": 84.0,
        "base_energy_mix": 0.78,
        "texture_sigma_scale": 0.62,
    },
    "H63_bayesian_oac_smooth": {
        "density_power": 1.02,
        "depth_compensation": 1.18,
        "oac_weight": 2.35,
        "texture_weight": 0.26,
        "energy_oac_scale": 30.0,
        "band_boost": 0.16,
        "lateral_smooth": 0.95,
        "oac_percentile": 76.0,
        "base_energy_mix": 0.66,
        "texture_sigma_scale": 0.82,
    },
    "H64_speckle_posterior": {
        "density_power": 1.14,
        "depth_compensation": 1.20,
        "oac_weight": 2.05,
        "texture_weight": 0.58,
        "energy_oac_scale": 28.0,
        "band_boost": 0.08,
        "lateral_smooth": 0.50,
        "oac_percentile": 72.0,
        "base_energy_mix": 0.60,
        "texture_sigma_scale": 1.58,
    },
    "H65_ot_depth_hist": {
        "density_power": 0.92,
        "depth_compensation": 1.00,
        "oac_weight": 2.65,
        "texture_weight": 0.34,
        "energy_oac_scale": 24.0,
        "band_boost": 0.02,
        "lateral_smooth": 0.70,
        "oac_percentile": 86.0,
        "base_energy_mix": 0.72,
        "texture_sigma_scale": 1.05,
    },
    "H66_patch_retrieval_proxy": {
        "density_power": 1.24,
        "depth_compensation": 1.26,
        "oac_weight": 2.18,
        "texture_weight": 0.46,
        "energy_oac_scale": 29.5,
        "band_boost": 0.22,
        "lateral_smooth": 0.34,
        "oac_percentile": 80.0,
        "base_energy_mix": 0.70,
        "texture_sigma_scale": 1.28,
    },
    "H67_coarse_to_fine_crisp": {
        "density_power": 1.32,
        "depth_compensation": 1.16,
        "oac_weight": 2.28,
        "texture_weight": 0.32,
        "energy_oac_scale": 27.0,
        "band_boost": 0.26,
        "lateral_smooth": 0.28,
        "oac_percentile": 82.0,
        "base_energy_mix": 0.76,
        "texture_sigma_scale": 1.10,
    },
    "H68_layer_map_prior": {
        "density_power": 1.08,
        "depth_compensation": 1.34,
        "oac_weight": 2.05,
        "texture_weight": 0.30,
        "energy_oac_scale": 34.0,
        "band_boost": 0.48,
        "lateral_smooth": 0.82,
        "oac_percentile": 74.0,
        "base_energy_mix": 0.58,
        "texture_sigma_scale": 0.90,
        "multi_layer": True,
    },
    "H69_h56_anti_smooth": _extrapolate_config(
        HYPOTHESIS_CONFIGS["H56_h41_anti_anatomy"],
        HYPOTHESIS_CONFIGS["H37_multi_layer_smooth"],
        0.05,
    ),
    "H70_h56_born_microblend": _blend_configs(
        (0.86, HYPOTHESIS_CONFIGS["H56_h41_anti_anatomy"]),
        (0.14, {
            "density_power": 1.05,
            "depth_compensation": 1.08,
            "oac_weight": 2.55,
            "texture_weight": 0.18,
            "energy_oac_scale": 26.0,
            "band_boost": 0.14,
            "lateral_smooth": 0.42,
            "oac_percentile": 84.0,
            "base_energy_mix": 0.78,
            "texture_sigma_scale": 0.62,
        }),
    ),
}
HYPOTHESIS_CONFIGS.update(INVERSE_WAVE_CONFIGS)

HYPOTHESIS_WAVES = {
    "inverse-wave-1": (
        "H41_final_optimized",
        "H56_h41_anti_anatomy",
        "H59_h41_anti_oac_guard",
        "portfolio",
        *INVERSE_WAVE_CONFIGS.keys(),
    ),
}


def official_baseline(
    output_path: str | Path,
    method: str = "two-layer",
    seed: int = 7,
    scatterers_count: int = 300_000,
) -> Path:
    config = ExperimentConfig(scatterers_count=scatterers_count)
    if method == "uniform":
        data = generate_uniform(config, seed=seed, amp=1.0)
    elif method == "two-layer":
        data = generate_two_layers(config, seed=seed, boundary_z_mcm=400.0, amp_top=0.05, amp_bottom=2.5)
    else:
        raise ValueError("method must be 'uniform' or 'two-layer'.")
    return save_phantom(data, output_path, config=config)


def estimate_layer_params(scan_path: str | Path) -> dict[str, float]:
    img = load_scan(scan_path)
    intensity = load_and_linearize_image(scan_path)
    oac = calculate_oac(intensity)
    sc = calculate_speckle_contrast_map(intensity)

    profile = img.mean(axis=1)
    window = min(15, max(3, len(profile) // 4))
    smooth = np.convolve(profile, np.ones(window) / window, mode="same")
    grad = np.abs(np.gradient(smooth))
    margin = min(10, max(1, len(profile) // 8))
    start = max(1, int(0.08 * len(profile)))
    stop = min(len(profile) - margin, max(start + 1, int(0.75 * len(profile))))
    boundary_idx = int(np.argmax(grad[start:stop]) + start)

    shallow = slice(0, max(boundary_idx, 1))
    deep = slice(min(boundary_idx, len(profile) - 1), None)
    shallow_energy = float(np.clip(np.percentile(oac[shallow], 75) * 30, 0.01, 8.0))
    deep_energy = float(np.clip(np.percentile(oac[deep], 75) * 60, 0.01, 12.0))
    heterogeneity = float(np.clip(np.nanmean(sc), 0.2, 5.0))
    return {
        "boundary_z_mcm": boundary_idx * ExperimentConfig().pixel_size_z,
        "amp_top": shallow_energy,
        "amp_bottom": deep_energy,
        "heterogeneity": heterogeneity,
    }


def heuristic_baseline(
    input_path: str | Path,
    output_path: str | Path,
    seed: int = 7,
    scatterers_count: int = 300_000,
) -> Path:
    config = ExperimentConfig(scatterers_count=scatterers_count)
    params = estimate_layer_params(input_path)
    data = generate_two_layers(
        config,
        seed=seed,
        boundary_z_mcm=params["boundary_z_mcm"],
        amp_top=params["amp_top"],
        amp_bottom=params["amp_bottom"],
    )
    rng = np.random.default_rng(seed + 1009)
    jitter = rng.lognormal(mean=0.0, sigma=0.12 * params["heterogeneity"], size=len(data))
    data[:, 3] = np.clip(data[:, 3] * jitter, 0, 100)
    return save_phantom(data, output_path, config=config)


def pretrained_cnn_baseline(
    input_path: str | Path,
    output_path: str | Path,
    backbone: str = "resnet50",
    seed: int = 7,
    scatterers_count: int = 300_000,
    pretrained: bool = True,
) -> Path:
    emb = cnn_embedding(input_path, backbone=backbone, pretrained=pretrained)
    config = ExperimentConfig(scatterers_count=scatterers_count)
    rng = np.random.default_rng(seed)
    digest = np.tanh(emb[:16] if emb.size >= 16 else np.pad(emb, (0, 16 - emb.size)))
    boundary = float(np.clip(500 + 220 * digest[0], 180, 1100))
    amp_top = float(np.clip(0.8 + 1.5 * abs(digest[1]), 0.02, 5.0))
    amp_bottom = float(np.clip(1.5 + 4.0 * abs(digest[2]), 0.02, 12.0))
    data = generate_two_layers(config, seed=seed, boundary_z_mcm=boundary, amp_top=amp_top, amp_bottom=amp_bottom)

    # Use frozen features as a deterministic parameter source, while keeping output a valid phantom.
    texture_sigma = float(np.clip(0.1 + 0.35 * abs(digest[3]), 0.05, 0.6))
    data[:, 3] = np.clip(data[:, 3] * rng.lognormal(0, texture_sigma, len(data)), 0, 100)
    return save_phantom(data, output_path, config=config)


def physics_guided_baseline(
    input_path: str | Path,
    output_path: str | Path,
    seed: int = 7,
    scatterers_count: int = 300_000,
    lateral_bins: int = 64,
    depth_bins: int = 64,
    **variant: float | bool,
) -> Path:
    """Generate scatterers from an attenuation-aware image-derived density field."""
    config = ExperimentConfig(scatterers_count=scatterers_count)
    img = load_scan(input_path)
    intensity = load_and_linearize_image(input_path)
    oac = calculate_oac(intensity)
    sc = calculate_speckle_contrast_map(intensity)
    params = estimate_layer_params(input_path)

    z_edges = np.linspace(0, img.shape[0], depth_bins + 1).astype(int)
    x_edges = np.linspace(0, img.shape[1], lateral_bins + 1).astype(int)
    density = np.zeros((depth_bins, lateral_bins), dtype=np.float64)
    energy_map = np.zeros_like(density)

    density_power = float(variant.get("density_power", 1.0))
    depth_compensation = float(variant.get("depth_compensation", 1.8))
    oac_weight = float(variant.get("oac_weight", 2.0))
    texture_weight = float(variant.get("texture_weight", 0.35))
    energy_oac_scale = float(variant.get("energy_oac_scale", 35.0))
    band_boost = float(variant.get("band_boost", 0.0))
    void_fraction = float(variant.get("void_fraction", 0.0))
    lateral_smooth = float(variant.get("lateral_smooth", 0.8))
    oac_percentile = float(variant.get("oac_percentile", 70.0))
    base_energy_mix = float(variant.get("base_energy_mix", 0.55))
    texture_sigma_scale = float(variant.get("texture_sigma_scale", 1.0))
    depth_prior = np.exp(np.linspace(0.0, depth_compensation, depth_bins))[:, None]
    for zi in range(depth_bins):
        z0, z1 = z_edges[zi], max(z_edges[zi + 1], z_edges[zi] + 1)
        for xi in range(lateral_bins):
            x0, x1 = x_edges[xi], max(x_edges[xi + 1], x_edges[xi] + 1)
            patch_img = img[z0:z1, x0:x1]
            patch_oac = oac[z0:z1, x0:x1]
            patch_sc = sc[z0:z1, x0:x1]
            contrast = float(np.std(patch_img) + 0.25 * np.mean(patch_sc))
            atten = float(np.percentile(patch_oac, oac_percentile))
            density[zi, xi] = max(1e-6, np.mean(patch_img) + texture_weight * contrast + oac_weight * atten)
            energy_map[zi, xi] = max(0.01, energy_oac_scale * atten + 1.5 * contrast)

    density *= depth_prior
    density = gaussian_smooth_density(density, sigma=(1.0, lateral_smooth))
    if density_power != 1.0:
        density = np.maximum(density, 1e-8) ** density_power
    if band_boost:
        boundary_bin = int(np.clip(params["boundary_z_mcm"] / config.z_max * depth_bins, 0, depth_bins - 1))
        zz = np.arange(depth_bins)[:, None]
        density *= 1.0 + band_boost * np.exp(-0.5 * ((zz - boundary_bin) / 2.0) ** 2)
    if variant.get("multi_layer"):
        profile = density.mean(axis=1)
        peaks = np.argsort(profile)[-3:]
        zz = np.arange(depth_bins)[:, None]
        for peak in peaks:
            density *= 1.0 + 0.18 * np.exp(-0.5 * ((zz - peak) / 1.5) ** 2)
    if void_fraction:
        rng_void = np.random.default_rng(seed + 12345)
        voids = rng_void.random(density.shape) < void_fraction
        density = np.where(voids, density * 0.18, density)
    probs = density.ravel() / density.sum()

    rng = np.random.default_rng(seed)
    cells = rng.choice(probs.size, size=scatterers_count, replace=True, p=probs)
    z_cell, x_cell = np.divmod(cells, lateral_bins)

    xs = ((x_cell + rng.random(scatterers_count)) / lateral_bins - 0.5) * config.x_max
    zs = ((z_cell + rng.random(scatterers_count)) / depth_bins) * config.z_max
    ys = (rng.random(scatterers_count) - 0.5) * (2 * config.beam_radius)
    base_energy = energy_map[z_cell, x_cell]
    if variant.get("log_energy"):
        base_energy = np.log1p(base_energy) * 4.0

    layer_boost = np.where(zs > params["boundary_z_mcm"], params["amp_bottom"], params["amp_top"])
    texture_sigma = texture_sigma_scale * (0.18 + 0.03 * params["heterogeneity"])
    texture = rng.lognormal(mean=0.0, sigma=texture_sigma, size=scatterers_count)
    amps = np.clip(base_energy_mix * base_energy + (1.0 - base_energy_mix) * layer_boost, 0.01, 100.0) * texture
    data = np.column_stack((xs, ys, zs, np.clip(amps, 0.01, 100.0)))
    return save_phantom(data, output_path, config=config)


def gaussian_smooth_density(density: np.ndarray, sigma: tuple[float, float] = (1.0, 0.8)) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    smoothed = gaussian_filter(density, sigma=sigma, mode="reflect")
    return np.maximum(smoothed, 1e-8)


def hypothesis_baseline(
    input_path: str | Path,
    output_path: str | Path,
    hypothesis: str,
    seed: int = 7,
    scatterers_count: int = 300_000,
) -> Path:
    if hypothesis == "H0_official":
        return official_baseline(output_path, seed=seed, scatterers_count=scatterers_count)
    config = HYPOTHESIS_CONFIGS[hypothesis]
    return physics_guided_baseline(
        input_path,
        output_path,
        seed=seed,
        scatterers_count=scatterers_count,
        **config,
    )


def final_baseline(
    input_path: str | Path,
    output_path: str | Path,
    seed: int = 7,
    scatterers_count: int = 300_000,
) -> Path:
    return hypothesis_baseline(input_path, output_path, FINAL_CONFIG_NAME, seed=seed, scatterers_count=scatterers_count)


def portfolio_baseline(
    input_path: str | Path,
    output_path: str | Path,
    seed: int = 7,
    scatterers_count: int = 300_000,
) -> Path:
    """Per-scan selector among validated physics hypotheses using reference-only features."""
    img = load_scan(input_path)
    intensity = load_and_linearize_image(input_path)
    oac = calculate_oac(intensity)
    depth_profile = img.mean(axis=1)
    shallow = float(np.mean(depth_profile[: max(4, len(depth_profile) // 5)]))
    deep = float(np.mean(depth_profile[len(depth_profile) // 2 :]))
    decay_ratio = shallow / (deep + 1e-6)
    oac_peak_depth = int(np.argmax(oac.mean(axis=1))) / max(1, oac.shape[0] - 1)
    texture = float(np.std(img) / (np.mean(img) + 1e-6))

    if shallow > 0.18 and oac_peak_depth < 0.35:
        choice = "H34_epidermal_emphasis"
    elif decay_ratio > 2.6 and texture < 1.4:
        choice = "H11_low_depth_comp"
    elif texture > 1.8:
        choice = "H29_low_depth_high_oac"
    else:
        choice = "H41_final_optimized"
    return hypothesis_baseline(input_path, output_path, choice, seed=seed, scatterers_count=scatterers_count)


def scan_descriptor_features(input_path: str | Path) -> dict[str, float]:
    img = load_scan(input_path)
    intensity = load_and_linearize_image(input_path)
    oac = calculate_oac(intensity)
    sc = calculate_speckle_contrast_map(intensity)
    profile = img.mean(axis=1)
    shallow = float(np.mean(profile[: max(4, len(profile) // 5)]))
    mid = float(np.mean(profile[len(profile) // 5 : max(len(profile) // 5 + 1, len(profile) // 2)]))
    deep = float(np.mean(profile[len(profile) // 2 :]))
    grad = np.abs(np.gradient(np.convolve(profile, np.ones(min(15, max(3, len(profile) // 4))) / min(15, max(3, len(profile) // 4)), mode="same")))
    return {
        "decay_ratio": shallow / (deep + 1e-6),
        "deep_ratio": deep / (shallow + 1e-6),
        "texture": float(np.std(img) / (np.mean(img) + 1e-6)),
        "speckle": float(np.nanmean(sc)),
        "boundary_sharpness": float(np.max(grad) / (np.mean(profile) + 1e-6)),
        "oac_peak_depth": float(int(np.argmax(oac.mean(axis=1))) / max(1, oac.shape[0] - 1)),
        "mid_shallow_ratio": mid / (shallow + 1e-6),
    }


def council_combo_baseline(
    input_path: str | Path,
    output_path: str | Path,
    seed: int = 7,
    scatterers_count: int = 300_000,
) -> Path:
    """Adaptive H41-centered blend of the strongest council hypotheses."""
    features = scan_descriptor_features(input_path)
    weighted: list[tuple[float, dict[str, float | bool]]] = [(1.00, HYPOTHESIS_CONFIGS["H41_final_optimized"])]

    if features["boundary_sharpness"] > 1.2 or features["oac_peak_depth"] < 0.38:
        weighted.append((0.32, AGENT_METHOD_CONFIGS["agent_anatomical_boundary"]))
    if features["texture"] > 1.55 or features["speckle"] > 0.95:
        weighted.append((0.22, AGENT_METHOD_CONFIGS["agent_speckle_moment"]))
        weighted.append((0.12, AGENT_METHOD_CONFIGS["agent_perceptual_retrieval"]))
    if features["decay_ratio"] > 2.5:
        weighted.append((0.28, HYPOTHESIS_CONFIGS["H11_low_depth_comp"]))
    if features["deep_ratio"] > 0.55 and features["boundary_sharpness"] < 0.9:
        weighted.append((0.20, AGENT_METHOD_CONFIGS["agent_inverse_psf"]))
    if features["mid_shallow_ratio"] > 0.78:
        weighted.append((0.16, HYPOTHESIS_CONFIGS["H34_epidermal_emphasis"]))

    config = _blend_configs(*weighted)
    return physics_guided_baseline(input_path, output_path, seed=seed, scatterers_count=scatterers_count, **config)


def council_combo_conservative_baseline(
    input_path: str | Path,
    output_path: str | Path,
    seed: int = 7,
    scatterers_count: int = 300_000,
) -> Path:
    """Lower-variance council blend for hidden-test robustness."""
    features = scan_descriptor_features(input_path)
    weighted: list[tuple[float, dict[str, float | bool]]] = [
        (1.00, HYPOTHESIS_CONFIGS["H41_final_optimized"]),
        (0.18, HYPOTHESIS_CONFIGS["H11_low_depth_comp"]),
        (0.12, AGENT_METHOD_CONFIGS["agent_anatomical_boundary"]),
    ]
    if features["texture"] > 1.7:
        weighted.append((0.10, AGENT_METHOD_CONFIGS["agent_perceptual_retrieval"]))
    if features["boundary_sharpness"] > 1.4:
        weighted.append((0.10, HYPOTHESIS_CONFIGS["H34_epidermal_emphasis"]))
    config = _blend_configs(*weighted)
    return physics_guided_baseline(input_path, output_path, seed=seed, scatterers_count=scatterers_count, **config)


def agent_method_baseline(
    input_path: str | Path,
    output_path: str | Path,
    method: str,
    seed: int = 7,
    scatterers_count: int = 300_000,
) -> Path:
    return physics_guided_baseline(
        input_path,
        output_path,
        seed=seed,
        scatterers_count=scatterers_count,
        **AGENT_METHOD_CONFIGS[method],
    )


def parameter_search_baseline(
    input_path: str | Path,
    output_path: str | Path,
    seed: int = 7,
    scatterers_count: int = 300_000,
) -> Path:
    # Scanner-in-the-loop scoring is unavailable on macOS by default, so this initializes
    # search from image-derived parameters and leaves real scoring to the Windows adapter.
    params = estimate_layer_params(input_path)
    grid_boundaries = [params["boundary_z_mcm"] - 60, params["boundary_z_mcm"], params["boundary_z_mcm"] + 60]
    grid_top = [params["amp_top"] * 0.75, params["amp_top"], params["amp_top"] * 1.25]
    grid_bottom = [params["amp_bottom"] * 0.75, params["amp_bottom"], params["amp_bottom"] * 1.25]
    config = ExperimentConfig(scatterers_count=scatterers_count)
    best = generate_two_layers(
        config,
        seed=seed,
        boundary_z_mcm=float(np.clip(grid_boundaries[1], 0, config.z_max)),
        amp_top=float(np.clip(grid_top[1], 0.01, 100)),
        amp_bottom=float(np.clip(grid_bottom[1], 0.01, 100)),
    )
    return save_phantom(best, output_path, config=config)
