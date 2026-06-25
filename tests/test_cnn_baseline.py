from __future__ import annotations

import numpy as np
from skimage import io

from synthoct.baselines import physics_guided_baseline, pretrained_cnn_baseline
from synthoct.cnn import cnn_embedding


def test_cnn_embedding_is_deterministic_without_weights(tmp_path):
    img = (np.random.default_rng(0).random((64, 64)) * 255).astype(np.uint8)
    path = tmp_path / "scan.png"
    io.imsave(path, img)
    a = cnn_embedding(path, backbone="resnet50", pretrained=False)
    b = cnn_embedding(path, backbone="resnet50", pretrained=False)
    assert a.shape == b.shape
    assert np.allclose(a, b)


def test_pretrained_cnn_baseline_outputs_valid_phantom(tmp_path):
    img = (np.random.default_rng(1).random((64, 64)) * 255).astype(np.uint8)
    scan = tmp_path / "scan.png"
    phantom = tmp_path / "phantom.txt"
    io.imsave(scan, img)
    pretrained_cnn_baseline(scan, phantom, pretrained=False, scatterers_count=128)
    data = np.loadtxt(phantom)
    assert data.shape == (128, 4)
    assert data[:, 3].min() >= 0
    assert data[:, 3].max() <= 100


def test_physics_guided_baseline_outputs_valid_phantom(tmp_path):
    img = np.tile(np.linspace(20, 220, 64, dtype=np.uint8), (64, 1)).T
    scan = tmp_path / "scan.png"
    phantom = tmp_path / "physics.txt"
    io.imsave(scan, img)
    physics_guided_baseline(scan, phantom, scatterers_count=128, lateral_bins=8, depth_bins=8)
    data = np.loadtxt(phantom)
    assert data.shape == (128, 4)
    assert data[:, 0].min() >= -1536
    assert data[:, 2].max() <= 1536
