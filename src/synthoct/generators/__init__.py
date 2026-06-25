"""Digital phantom generators that map real OCT scans to scatterer tables."""

from .phantom_generators import (
    FINAL_CONFIG_NAME,
    HYPOTHESIS_CONFIGS,
    HYPOTHESIS_WAVES,
    PhantomConfig,
    estimate_layer_params,
    final_phantom,
    heuristic_layer_phantom,
    hypothesis_phantom,
    official_baseline_phantom,
    physics_guided_phantom,
)

__all__ = [
    "FINAL_CONFIG_NAME",
    "HYPOTHESIS_CONFIGS",
    "HYPOTHESIS_WAVES",
    "PhantomConfig",
    "estimate_layer_params",
    "final_phantom",
    "heuristic_layer_phantom",
    "hypothesis_phantom",
    "official_baseline_phantom",
    "physics_guided_phantom",
]
