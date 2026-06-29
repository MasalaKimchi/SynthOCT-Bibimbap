from __future__ import annotations

import csv

import numpy as np
import pytest

from synthoct.generators import (
    FINAL_CONFIG_NAME,
    HYPOTHESIS_CONFIGS,
    PROMISING_PIPELINE_CONFIGS,
    VISUAL_PIPELINE_CONFIGS,
    final_phantom,
    hybrid_neural_prior_phantom,
    learned_prior_phantom,
    pipeline_phantom,
)
from synthoct.learned_prior import LEARNED_PRIOR_CONFIGS
from synthoct.neural_prior import discover_neural_training_pairs, neural_prior_phantom, train_neural_phantom_prior
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


def test_learned_prior_phantom_uses_artifact_and_writes_valid_scatterers(tmp_path):
    scan = tmp_path / "reference.npy"
    img = np.zeros((40, 60), dtype=np.float32)
    img[8:22, :] = np.linspace(0.75, 0.20, 14, dtype=np.float32)[:, None]
    img[12:18, 24:40] += 0.15
    np.save(scan, np.clip(img, 0.0, 1.0))
    artifact = tmp_path / "prior.npz"
    density = np.ones((16, 24), dtype=np.float32)
    density[4:10, :] *= 4.0
    energy = np.linspace(0.2, 0.8, 16, dtype=np.float32)[:, None] * np.ones((16, 24), dtype=np.float32)
    np.savez_compressed(artifact, density_prior=density, energy_prior=energy)

    phantom = tmp_path / "learned_prior.txt"
    learned_prior_phantom(scan, phantom, artifact, seed=9, scatterers_count=128)
    data = load_phantom(phantom)
    assert data.shape == (128, 4)
    assert np.isfinite(data).all()
    assert data[:, 3].min() >= 0


def test_learned_prior_energy_sigma_controls_energy_variance(tmp_path):
    scan = tmp_path / "reference.npy"
    img = np.zeros((40, 60), dtype=np.float32)
    img[8:22, :] = np.linspace(0.75, 0.20, 14, dtype=np.float32)[:, None]
    np.save(scan, np.clip(img, 0.0, 1.0))
    artifact = tmp_path / "prior.npz"
    np.savez_compressed(
        artifact,
        density_prior=np.ones((16, 24), dtype=np.float32),
        energy_prior=np.ones((16, 24), dtype=np.float32),
    )

    low_sigma = tmp_path / "low_sigma.txt"
    high_sigma = tmp_path / "high_sigma.txt"
    learned_prior_phantom(scan, low_sigma, artifact, seed=9, scatterers_count=128, energy_sigma=0.02)
    learned_prior_phantom(scan, high_sigma, artifact, seed=9, scatterers_count=128, energy_sigma=0.22)

    low = load_phantom(low_sigma)
    high = load_phantom(high_sigma)
    assert np.allclose(low[:, :3], high[:, :3])
    assert high[:, 3].std() > low[:, 3].std()


def test_learned_prior_configs_include_lpips_tuning_variants():
    assert LEARNED_PRIOR_CONFIGS["learned-prior-sparse-p140"]["density_power"] == 14.0
    assert LEARNED_PRIOR_CONFIGS["learned-prior-sparse-p140-sigma16"]["energy_sigma"] == 0.16
    assert LEARNED_PRIOR_CONFIGS["learned-prior-sparse-p140-t32"]["texture_weight"] == 0.32


def test_neural_prior_trains_artifact_and_writes_valid_scatterers(tmp_path):
    pytest.importorskip("torch")

    reference = tmp_path / "reference.npy"
    rendered = tmp_path / "rendered.npy"
    phantom_source = tmp_path / "source_phantom.txt"
    img = np.zeros((40, 60), dtype=np.float32)
    img[8:22, :] = np.linspace(0.75, 0.18, 14, dtype=np.float32)[:, None]
    img[12:18, 20:38] += 0.12
    np.save(reference, np.clip(img, 0.0, 1.0))
    np.save(rendered, np.clip(img * 0.9, 0.0, 1.0))
    save_phantom(
        generate_two_layers(ExperimentConfig(scatterers_count=96), seed=11),
        phantom_source,
        ExperimentConfig(scatterers_count=96),
    )

    metrics = tmp_path / "internal_validation_detail.csv"
    with metrics.open("w", newline="") as fobj:
        writer = csv.DictWriter(
            fobj,
            fieldnames=[
                "method",
                "evidence_source",
                "reference_png",
                "phantom_path",
                "synthetic_gray_png",
                "Struct_MS-SSIM",
                "Struct_LPIPS",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": "scanner_pair",
                "evidence_source": "hosted_api_true_scanner",
                "reference_png": str(reference),
                "phantom_path": str(phantom_source),
                "synthetic_gray_png": str(rendered),
                "Struct_MS-SSIM": "0.72",
                "Struct_LPIPS": "0.31",
            }
        )

    pairs = discover_neural_training_pairs(tmp_path)
    assert len(pairs) == 1
    assert pairs[0].label == "scanner_pair"

    artifact = tmp_path / "neural_prior.pt"
    train_neural_phantom_prior(tmp_path, artifact, shape=(16, 24), epochs=2, pair_limit=4)
    assert artifact.exists()
    assert artifact.with_suffix(".json").exists()

    out = tmp_path / "neural_prior_phantom.txt"
    neural_prior_phantom(reference, out, artifact, seed=17, scatterers_count=72)
    data = load_phantom(out)
    assert data.shape == (72, 4)
    assert np.isfinite(data).all()
    assert data[:, 3].min() >= 0


def test_hybrid_neural_prior_blends_artifacts_and_writes_valid_scatterers(tmp_path):
    pytest.importorskip("torch")

    reference = tmp_path / "reference.npy"
    rendered = tmp_path / "rendered.npy"
    source_phantom = tmp_path / "source_phantom.txt"
    img = np.zeros((40, 60), dtype=np.float32)
    img[8:22, :] = np.linspace(0.72, 0.20, 14, dtype=np.float32)[:, None]
    np.save(reference, np.clip(img, 0.0, 1.0))
    np.save(rendered, np.clip(img * 0.85, 0.0, 1.0))
    save_phantom(
        generate_two_layers(ExperimentConfig(scatterers_count=96), seed=12),
        source_phantom,
        ExperimentConfig(scatterers_count=96),
    )
    metrics = tmp_path / "internal_validation_detail.csv"
    with metrics.open("w", newline="") as fobj:
        writer = csv.DictWriter(
            fobj,
            fieldnames=["method", "evidence_source", "reference_png", "phantom_path", "synthetic_gray_png", "Struct_SSIM"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": "scanner_pair",
                "evidence_source": "hosted_api_true_scanner",
                "reference_png": str(reference),
                "phantom_path": str(source_phantom),
                "synthetic_gray_png": str(rendered),
                "Struct_SSIM": "0.64",
            }
        )
    learned_artifact = tmp_path / "prior.npz"
    np.savez_compressed(
        learned_artifact,
        density_prior=np.ones((16, 24), dtype=np.float32),
        energy_prior=np.ones((16, 24), dtype=np.float32) * 0.5,
    )
    neural_artifact = tmp_path / "neural.pt"
    train_neural_phantom_prior(tmp_path, neural_artifact, shape=(16, 24), epochs=1)

    out = tmp_path / "hybrid.txt"
    hybrid_neural_prior_phantom(
        reference,
        out,
        learned_artifact,
        neural_artifact,
        seed=18,
        scatterers_count=80,
        neural_density_blend=0.10,
    )
    data = load_phantom(out)
    assert data.shape == (80, 4)
    assert np.isfinite(data).all()
