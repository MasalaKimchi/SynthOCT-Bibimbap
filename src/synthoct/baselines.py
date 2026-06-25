"""Backward-compatible phantom generator facade.

New code should import from :mod:`synthoct.generators`. The functions below keep
the historical CLI/API names while the implementation is now explicit that these
methods generate digital phantoms, not rendered final OCT images.
"""

from .generators import (
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

official_baseline = official_baseline_phantom
heuristic_baseline = heuristic_layer_phantom
physics_guided_baseline = physics_guided_phantom
hypothesis_baseline = hypothesis_phantom
final_baseline = final_phantom

__all__ = [
    "FINAL_CONFIG_NAME",
    "HYPOTHESIS_CONFIGS",
    "HYPOTHESIS_WAVES",
    "PhantomConfig",
    "estimate_layer_params",
    "final_baseline",
    "final_phantom",
    "heuristic_baseline",
    "heuristic_layer_phantom",
    "hypothesis_baseline",
    "hypothesis_phantom",
    "official_baseline",
    "official_baseline_phantom",
    "physics_guided_baseline",
    "physics_guided_phantom",
]
