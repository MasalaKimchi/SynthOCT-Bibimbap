"""Part3 (Processor) — FIXED parametric mapping, in the baseline's flat shape.

Baseline role (``Part3_Processor.py``): convert a raw OCT scan into OAC (Optical
Attenuation Coefficient), SC (Speckle Contrast) and RSC (Refined Speckle
Contrast) maps, and expose ``generate_maps(scan_path) -> {Struct, OAC, SC, RSC}``.

You are not meant to modify Part3.  This file re-exports the organizer-compatible
implementation from ``synthoct.features.extraction`` (the
``organizer-compatible-v1`` mode, which freezes the published Part3 behaviour
byte-for-byte, including the 40 dB linearization ``I = 10**(4*P)`` and the
save-time autoscaling).  ``generate_maps`` returns the same dict, with the same
``_OAC/_SC/_RSC.png`` filenames written next to the input, as the baseline.

The ``scientific-v1`` calibrated/masked map mode also lives in ``synthoct`` but
is audit-only and never enters a competition score; it is deliberately not
surfaced here so this file stays a faithful stand-in for the fixed baseline Part3.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (must precede any `synthoct` import; sets sys.path)

import os
from pathlib import Path

import matplotlib.pyplot as plt

from synthoct.features.extraction import (
    HPX,
    ORGANIZER_MAP_MODE,
    WINDOW_SIZE,
    calculate_oac,
    calculate_speckle_contrast_map,
    load_and_linearize_image,
    normalize_map,
)
from synthoct.features.extraction import generate_maps as _synth_generate_maps

__all__ = [
    "HPX",
    "WINDOW_SIZE",
    "load_and_linearize_image",
    "calculate_oac",
    "calculate_speckle_contrast_map",
    "save_map",
    "generate_maps",
]


def save_map(data, filename, vmin=None, vmax=None) -> Path:
    """Save a normalized grayscale map (baseline ``save_map`` signature).

    Matches the organizer path's ``plt.imsave(..., cmap="gray")`` encoding,
    which is metric-significant at the 1e-4 scale (do not substitute skimage).
    """
    plt.imsave(filename, normalize_map(data, vmin=vmin, vmax=vmax), cmap="gray")
    return Path(filename)


def generate_maps(input_image_path, output_dir=None) -> dict:
    """Generate OAC/SC/RSC maps for a structural OCT image.

    Returns ``{'Struct', 'OAC', 'SC', 'RSC'}`` file paths.  When ``output_dir``
    is omitted the maps are written next to the input as ``<stem>_OAC.png`` etc.,
    exactly like the official ``Part3_Processor.generate_maps``.
    """
    paths = _synth_generate_maps(input_image_path, output_dir, mode=ORGANIZER_MAP_MODE)
    print(f"[Processor] Generated maps for {input_image_path}")
    return paths


if __name__ == "__main__":
    # Baseline-style test run.
    if os.path.exists("Test_Scan.png"):
        generate_maps("Test_Scan.png")
    else:
        print("Run Part2 first to generate a scan PNG, then pass it to generate_maps().")
