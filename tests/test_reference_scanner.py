from __future__ import annotations

import numpy as np

from synthoct.phantom import ExperimentConfig
from synthoct.scanners.reference import render_reference_array, scanner_wavenumbers


def test_scanner_wavenumbers_have_expected_ifft_order():
    config = ExperimentConfig(n_depth=8)
    k = scanner_wavenumbers(config)
    k0 = 2.0 * np.pi / config.wavelength
    assert k[0] == k0
    assert np.all(np.diff(k[:5]) > 0)
    assert np.all(k[5:] < k0)
    assert np.all(np.diff(k[5:]) > 0)


def test_reference_scanner_is_finite_and_normalized():
    config = ExperimentConfig(n_depth=16, n_lateral=8, scatterers_count=64)
    rng = np.random.default_rng(7)
    phantom = np.column_stack(
        (
            rng.uniform(-config.x_max / 2, config.x_max / 2, config.scatterers_count),
            rng.uniform(-config.beam_radius, config.beam_radius, config.scatterers_count),
            rng.uniform(0.0, config.z_max, config.scatterers_count),
            rng.uniform(0.1, 2.0, config.scatterers_count),
        )
    )
    image = render_reference_array(phantom, config=config)
    assert image.shape == (config.n_depth, config.n_lateral)
    assert np.isfinite(image).all()
    assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0
    assert float(image.max()) == 1.0
