"""Feature extraction from real OCT reference scans."""

from .extraction import (
    HPX,
    WINDOW_SIZE,
    calculate_oac,
    calculate_speckle_contrast_map,
    estimate_layer_boundary,
    generate_maps,
    load_and_linearize_image,
    load_scan,
    normalize_map,
)

__all__ = [
    "HPX",
    "WINDOW_SIZE",
    "calculate_oac",
    "calculate_speckle_contrast_map",
    "estimate_layer_boundary",
    "generate_maps",
    "load_and_linearize_image",
    "load_scan",
    "normalize_map",
]
