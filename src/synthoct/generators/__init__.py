"""Digital phantom generators that map real OCT scans to scatterer tables."""

from .phantom_generators import (
    FINAL_CONFIG_NAME,
    HYPOTHESIS_CONFIGS,
    HYPOTHESIS_WAVES,
    PROMISING_PIPELINE_CONFIGS,
    PROMISING_PIPELINE_WAVES,
    PhantomConfig,
    VISUAL_PIPELINE_CONFIGS,
    estimate_layer_params,
    final_phantom,
    heuristic_layer_phantom,
    hypothesis_phantom,
    official_baseline_phantom,
    pipeline_phantom,
    physics_guided_phantom,
    visual_inversion_phantom,
)

__all__ = [
    "FINAL_CONFIG_NAME",
    "HYPOTHESIS_CONFIGS",
    "HYPOTHESIS_WAVES",
    "PROMISING_PIPELINE_CONFIGS",
    "PROMISING_PIPELINE_WAVES",
    "PhantomConfig",
    "VISUAL_PIPELINE_CONFIGS",
    "estimate_layer_params",
    "final_phantom",
    "heuristic_layer_phantom",
    "hypothesis_phantom",
    "official_baseline_phantom",
    "pipeline_phantom",
    "physics_guided_phantom",
    "visual_inversion_phantom",
]
