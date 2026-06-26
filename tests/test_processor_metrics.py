from __future__ import annotations

import numpy as np
from skimage import io

from synthoct.evaluation import calculate_metrics
from synthoct.features import calculate_oac, calculate_speckle_contrast_map, generate_maps


def test_processor_maps_and_metrics(tmp_path):
    image = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    ref = tmp_path / "ref.png"
    pred = tmp_path / "pred.png"
    io.imsave(ref, image)
    io.imsave(pred, image)

    maps = generate_maps(ref, output_dir=tmp_path / "maps")
    assert {"Struct", "OAC", "SC", "RSC"} == set(maps)
    assert maps["OAC"].exists()

    metrics = calculate_metrics(ref, pred, include_lpips=False)
    assert metrics["MSE"] == 0.0
    assert metrics["SSIM"] > 0.99


def test_oac_and_speckle_shapes():
    arr = np.ones((32, 48), dtype=np.float32)
    assert calculate_oac(arr).shape == arr.shape
    assert calculate_speckle_contrast_map(arr, window_size=6).shape == arr.shape
