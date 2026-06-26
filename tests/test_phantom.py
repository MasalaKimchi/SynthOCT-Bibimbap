from __future__ import annotations

import numpy as np

from synthoct.generators import FINAL_CONFIG_NAME, HYPOTHESIS_CONFIGS, PROMISING_PIPELINE_CONFIGS, VISUAL_PIPELINE_CONFIGS, final_phantom, pipeline_phantom
from synthoct.phantom import ExperimentConfig, generate_two_layers, load_phantom, save_phantom, validate_phantom


def test_generate_and_save_valid_phantom(tmp_path):
    config = ExperimentConfig(scatterers_count=128)
    data = generate_two_layers(config, seed=1, boundary_z_mcm=400, amp_top=0.2, amp_bottom=2.0)
    validate_phantom(data, config)
    out = save_phantom(data, tmp_path / "phantom.txt", config)
    loaded = np.loadtxt(out)
    assert loaded.shape == (128, 4)
    assert loaded[:, 0].min() >= -1536
    assert loaded[:, 0].max() <= 1536
    assert loaded[:, 2].min() >= 0
    assert loaded[:, 2].max() <= 1536


def test_validate_rejects_bad_energy():
    config = ExperimentConfig(scatterers_count=1)
    data = np.array([[0.0, 0.0, 1.0, 101.0]])
    try:
        validate_phantom(data, config)
    except ValueError as exc:
        assert "Energy" in str(exc)
    else:
        raise AssertionError("Expected invalid energy to fail")


def test_final_generator_writes_four_column_phantom_from_reference_scan(tmp_path):
    scan = tmp_path / "reference.npy"
    phantom = tmp_path / "h61_phantom.txt"
    np.save(scan, np.tile(np.linspace(0.0, 1.0, 24, dtype=np.float32), (24, 1)))

    assert FINAL_CONFIG_NAME == "H61_api_low_depth_prelim"
    assert HYPOTHESIS_CONFIGS[FINAL_CONFIG_NAME]["depth_compensation"] == -0.80

    final_phantom(scan, phantom, seed=3, scatterers_count=96)
    data = load_phantom(phantom)
    assert data.shape == (96, 4)
    assert data[:, 3].min() >= 0


def test_all_promising_pipelines_write_valid_phantoms(tmp_path):
    scan = tmp_path / "reference.npy"
    base = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    layered = np.tile(base[:, None], (1, 48))
    layered[14:18, :] += 0.25
    np.save(scan, np.clip(layered, 0.0, 1.0))

    assert len(PROMISING_PIPELINE_CONFIGS) == 5
    for idx, name in enumerate(PROMISING_PIPELINE_CONFIGS):
        phantom = tmp_path / f"{name}.txt"
        pipeline_phantom(scan, phantom, name, seed=idx, scatterers_count=80)
        data = load_phantom(phantom)
        assert data.shape == (80, 4)
        assert np.isfinite(data).all()
        assert data[:, 3].min() >= 0


def test_visual_pipelines_write_valid_phantoms(tmp_path):
    scan = tmp_path / "reference.npy"
    img = np.zeros((40, 60), dtype=np.float32)
    img[6:18, :] = np.linspace(0.7, 0.35, 12, dtype=np.float32)[:, None]
    img[18:, :] = 0.04
    img[10:16, 35:50] += 0.2
    np.save(scan, np.clip(img, 0.0, 1.0))

    assert set(VISUAL_PIPELINE_CONFIGS) == {
        "P06_visual_surface_dark_body",
        "P07_surface_cutoff_broad_mix",
        "P08_sparse_top_texture_ssim",
        "P09_gamma_sparse_lowfloor_ssim",
    }
    for idx, name in enumerate(VISUAL_PIPELINE_CONFIGS):
        phantom = tmp_path / f"{name}.txt"
        pipeline_phantom(scan, phantom, name, seed=20 + idx, scatterers_count=96)
        data = load_phantom(phantom)
        assert data.shape == (96, 4)
        assert np.isfinite(data).all()
        assert data[:, 3].min() >= 0
