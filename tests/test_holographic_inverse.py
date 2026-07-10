from __future__ import annotations

import json

import numpy as np
from skimage import io

from synthoct.cli import main
from synthoct.holographic_inverse import HolographicInverseConfig, holographic_inverse_phantom
from synthoct.phantom import ExperimentConfig, load_phantom, save_phantom


def _reference(path, shape=(16, 8)):
    rows, cols = shape
    image = np.zeros(shape, dtype=np.uint8)
    image[2 : rows // 2] = np.linspace(220, 30, rows // 2 - 2, dtype=np.uint8)[:, None]
    image[:, 1::2] = np.clip(image[:, 1::2] + 12, 0, 255)
    io.imsave(path, image)


def test_holographic_inverse_writes_contract_rows_and_zero_fillers(tmp_path):
    reference = tmp_path / "reference.png"
    _reference(reference)
    scanner = ExperimentConfig(n_depth=16, n_lateral=8, scatterers_count=256)
    phantom_path = tmp_path / "phantom.txt"
    diagnostics = tmp_path / "diagnostics.json"
    holographic_inverse_phantom(
        reference,
        phantom_path,
        scatterers_count=256,
        scanner=scanner,
        inverse=HolographicInverseConfig(axial_regularization=0.03, lateral_regularization=0.2),
        diagnostics_path=diagnostics,
    )
    phantom = load_phantom(phantom_path)
    assert phantom.shape == (256, 4)
    assert np.count_nonzero(phantom[:, 3] == 0.0) >= 128
    report = json.loads(diagnostics.read_text())
    assert report["generator"] == "holographic-inverse-v1"
    assert report["active_scatterers"] == 128


def test_holographic_inverse_cli(tmp_path, monkeypatch):
    reference = tmp_path / "reference.png"
    _reference(reference, shape=(256, 512))
    phantom = tmp_path / "phantom.txt"
    def fake_generator(_input, output, *, scatterers_count, **_kwargs):
        config = ExperimentConfig(scatterers_count=scatterers_count)
        data = np.zeros((scatterers_count, 4), dtype=np.float64)
        return save_phantom(data, output, config=config)

    monkeypatch.setattr("synthoct.cli.holographic_inverse_phantom", fake_generator)
    assert (
        main(
            [
                "baseline",
                "holographic-inverse",
                "--input",
                str(reference),
                "--out",
                str(phantom),
                "--scatterers-count",
                "131072",
            ]
        )
        == 0
    )
    assert load_phantom(phantom).shape == (131072, 4)
