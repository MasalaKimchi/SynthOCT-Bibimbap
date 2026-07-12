from __future__ import annotations

import json
import hashlib

import numpy as np
import pytest
from skimage import io

from synthoct.evaluation import competition_formula_estimate
from synthoct.features import (
    ORGANIZER_MAP_MODE,
    SCIENTIFIC_MAP_MODE,
    ScientificMapConfig,
    generate_maps,
    load_and_linearize_image,
)
from synthoct.evaluation import evaluate_feature_map_metrics, profile_scores
from synthoct.features.extraction import _save_fixed_scale_map


def _random_scan(path, *, seed=17, shape=(64, 64)):
    rng = np.random.default_rng(seed)
    scan = rng.integers(0, 256, size=shape, dtype=np.uint8)
    io.imsave(path, scan, check_contrast=False)
    return scan


def test_organizer_mode_is_default_and_pixel_stable(tmp_path):
    scan = tmp_path / "scan.png"
    _random_scan(scan)
    default = generate_maps(scan, tmp_path / "default")
    explicit = generate_maps(
        scan,
        tmp_path / "explicit",
        mode=ORGANIZER_MAP_MODE,
    )
    for name in ("OAC", "SC", "RSC"):
        np.testing.assert_array_equal(io.imread(default[name]), io.imread(explicit[name]))


def test_organizer_mode_matches_upstream_golden_pixels(tmp_path):
    scan = tmp_path / "scan.png"
    _random_scan(scan)
    maps = generate_maps(scan, tmp_path / "maps")
    expected = {
        "OAC": "7bdc20280c9c1a3cd937d9b3b33626ad1ef396e897ba6145d7db3ff3644a1f8e",
        "SC": "eb73a02d6ae1966cff70896d3b3e3ade2db88ec0e9c1146f585c40c25093c22b",
        "RSC": "b79328996c3a4d5aff7a53547751c51b0b8fc9b06701bce940e0887add341f90",
    }
    for name, expected_hash in expected.items():
        pixels = io.imread(maps[name]).tobytes()
        assert hashlib.sha256(pixels).hexdigest() == expected_hash


def test_linearization_separates_organizer_40db_from_scientific_51db(tmp_path):
    scan = tmp_path / "ramp.png"
    io.imsave(scan, np.asarray([[0, 255]], dtype=np.uint8), check_contrast=False)
    organizer = load_and_linearize_image(scan)
    scientific = load_and_linearize_image(scan, dynamic_range_db=51.0)
    assert organizer[0, 0] == pytest.approx(1.0)
    assert organizer[0, 1] == pytest.approx(10_000.0, rel=1e-6)
    assert scientific[0, 1] == pytest.approx(10 ** 5.1, rel=1e-6)


def test_fixed_scale_png_does_not_stretch_observed_maximum(tmp_path):
    path = tmp_path / "fixed.png"
    data = np.asarray([[0.0, 0.25], [0.5, 0.75]], dtype=np.float32)
    _save_fixed_scale_map(
        path,
        data,
        np.ones(data.shape, dtype=bool),
        vmin=0.0,
        vmax=1.0,
    )
    saved = io.imread(path, as_gray=True)
    assert float(saved.max()) == pytest.approx(0.75, abs=1.0 / 255.0)


def test_scientific_mode_writes_float_maps_masks_and_metadata(tmp_path):
    scan = tmp_path / "scan.png"
    _random_scan(scan)
    paths = generate_maps(
        scan,
        tmp_path / "scientific",
        mode=SCIENTIFIC_MAP_MODE,
        scientific_config=ScientificMapConfig(dynamic_range_db=51.0),
    )
    with np.load(paths["FloatMaps"]) as bundle:
        for name in ("OAC", "SC", "RSC"):
            assert bundle[name].shape == (64, 64)
            assert bundle[name].dtype == np.float32
            assert bundle[f"{name}_valid"].dtype == bool
        assert not bundle["OAC_valid"][-1].any()
        assert "TissueSupport" in bundle
        assert not bundle["SC_valid"][:10].any()
        assert not bundle["SC_valid"][:, :10].any()
        assert bundle["SC_valid"][10:-10, 10:-10].all()
        assert not np.array_equal(bundle["SC"][0], bundle["SC"][10])
    metadata = json.loads(paths["Metadata"].read_text(encoding="utf-8"))
    assert metadata["mode"] == SCIENTIFIC_MAP_MODE
    assert metadata["config"]["dynamic_range_db"] == 51.0
    assert SCIENTIFIC_MAP_MODE in str(paths["OAC"])


def test_map_modes_cannot_overwrite_each_other(tmp_path):
    scan = tmp_path / "scan.png"
    _random_scan(scan)
    organizer = generate_maps(scan, tmp_path / "maps", mode=ORGANIZER_MAP_MODE)
    before = {name: io.imread(organizer[name]).copy() for name in ("OAC", "SC", "RSC")}
    scientific = generate_maps(scan, tmp_path / "maps", mode=SCIENTIFIC_MAP_MODE)
    for name in ("OAC", "SC", "RSC"):
        assert organizer[name] != scientific[name]
        np.testing.assert_array_equal(io.imread(organizer[name]), before[name])


def test_scientific_metrics_use_reference_mask_and_51db_profiles(tmp_path):
    ref = tmp_path / "ref.png"
    pred = tmp_path / "pred.png"
    _random_scan(ref, seed=3)
    _random_scan(pred, seed=4)
    row = evaluate_feature_map_metrics(
        ref,
        pred,
        tmp_path / "ref_maps",
        tmp_path / "pred_maps",
        map_mode=SCIENTIFIC_MAP_MODE,
    )
    assert row["Map_Mode"] == SCIENTIFIC_MAP_MODE
    assert 0.0 < row["OAC_Valid_Fraction"] < 1.0
    scientific = profile_scores(ref, pred, map_mode=SCIENTIFIC_MAP_MODE)
    organizer = profile_scores(ref, pred, map_mode=ORGANIZER_MAP_MODE)
    assert scientific["SCMeanAbsErr"] != pytest.approx(organizer["SCMeanAbsErr"])


def test_oac_invalid_tail_propagates_into_rsc_mask(tmp_path):
    scan = tmp_path / "scan.png"
    _random_scan(scan, shape=(64, 64))
    paths = generate_maps(
        scan,
        tmp_path / "scientific",
        mode=SCIENTIFIC_MAP_MODE,
        scientific_config=ScientificMapConfig(
            tissue_display_threshold=0.0,
            oac_invalid_tail_rows=12,
        ),
    )
    with np.load(paths["FloatMaps"]) as bundle:
        assert not bundle["RSC_valid"][-21:].any()


def test_competition_formula_rejects_scientific_map_rows():
    row = {"Map_Mode": SCIENTIFIC_MAP_MODE}
    for name in ("Struct", "OAC", "SC", "RSC"):
        row[f"{name}_MS-SSIM"] = 0.99
        row[f"{name}_LPIPS"] = 0.01
    with pytest.raises(ValueError, match="organizer-compatible"):
        competition_formula_estimate(row, map_mode=SCIENTIFIC_MAP_MODE)
