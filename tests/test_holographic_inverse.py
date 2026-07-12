from __future__ import annotations

import json

import numpy as np
from skimage import io
from skimage.util import img_as_float
from sewar.full_ref import msssim

from synthoct.cli import main
from synthoct.benchmark import run_local_benchmark
from synthoct.evidence import build_evidence_manifest, verify_evidence_manifest
from synthoct.evaluation import calculate_metrics, competition_formula_estimate
from synthoct.holographic_inverse import (
    HolographicInverseConfig,
    _encode_coefficients,
    holographic_inverse_phantom,
)
from synthoct.features import ORGANIZER_MAP_MODE
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
    scanner = ExperimentConfig(n_depth=16, n_lateral=8, scatterers_count=384)
    phantom_path = tmp_path / "phantom.txt"
    diagnostics = tmp_path / "diagnostics.json"
    holographic_inverse_phantom(
        reference,
        phantom_path,
        scatterers_count=384,
        scanner=scanner,
        inverse=HolographicInverseConfig(
            axial_regularization=0.02,
            lateral_regularization=0.05,
            phase_iterations=2,
        ),
        diagnostics_path=diagnostics,
    )
    phantom = load_phantom(phantom_path)
    assert phantom.shape == (384, 4)
    assert np.count_nonzero(phantom[:, 3] == 0.0) >= 128
    report = json.loads(diagnostics.read_text())
    assert report["generator"] == "holographic-inverse-v3-phase-pair"
    assert report["active_scatterers"] == 256
    assert report["inverse"]["phase_iterations"] == 2
    assert report["inverse"]["phase_encoding"] == "dispersion-canceling-pair"
    assert report["projected_magnitude_mae"] >= 0.0
    assert report["reference"]["sha256"]
    assert report["phantom"]["sha256"]


def test_phase_pair_preserves_carrier_and_cancels_first_order_dispersion(tmp_path):
    scanner = ExperimentConfig(n_depth=2, n_lateral=3, scatterers_count=12)
    coefficients = np.asarray(
        [
            [1.0 + 2.0j, -0.5 + 0.25j, 0.2 - 1.3j],
            [-2.0 - 0.3j, 0.8 - 0.6j, -0.1 + 0.4j],
        ]
    )

    active, energies = _encode_coefficients(
        coefficients,
        scanner,
        HolographicInverseConfig(),
    )
    serialized = tmp_path / "phase_pair.txt"
    save_phantom(active, serialized, config=scanner)
    reloaded = load_phantom(serialized)
    amplitudes = np.sqrt(reloaded[:, 3].reshape(-1, 2) / 100.0)
    z_centers = np.repeat(
        (np.arange(scanner.n_depth) + 0.5) * scanner.pixel_size_z,
        scanner.n_lateral,
    )
    offsets = reloaded[:, 2].reshape(-1, 2) - z_centers[:, None]

    np.testing.assert_allclose(np.sum(amplitudes * offsets, axis=1), 0.0, atol=5e-10)
    k0 = 2.0 * np.pi / scanner.wavelength
    encoded = np.sum(amplitudes * np.exp(-2j * k0 * offsets), axis=1)
    scale = encoded / coefficients.ravel()
    np.testing.assert_allclose(scale.imag, 0.0, atol=2e-9)
    np.testing.assert_allclose(scale.real, scale.real[0], rtol=1e-7, atol=1e-12)
    assert scale.real[0] > 0.0


def test_holographic_inverse_config_rejects_invalid_phase_controls():
    for config in (
        HolographicInverseConfig(phase_iterations=-1),
        HolographicInverseConfig(phase_iterations=1.5),
        HolographicInverseConfig(phase_momentum=float("nan")),
        HolographicInverseConfig(phase_momentum=-0.1),
        HolographicInverseConfig(phase_momentum=1.1),
        HolographicInverseConfig(phase_encoding="invalid"),
    ):
        try:
            config.validate()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid phase controls were accepted")


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


def test_competition_formula_estimate_uses_eight_published_components():
    first = {}
    second = {}
    for index, map_name in enumerate(("Struct", "OAC", "SC", "RSC")):
        first[f"{map_name}_MS-SSIM"] = 0.8 + index * 0.01
        first[f"{map_name}_LPIPS"] = 0.2 - index * 0.01
        second[f"{map_name}_MS-SSIM"] = 0.9 + index * 0.01
        second[f"{map_name}_LPIPS"] = 0.1 - index * 0.01
    score, medians = competition_formula_estimate([first, second])
    expected = np.mean(
        [
            value
            for map_name in ("Struct", "OAC", "SC", "RSC")
            for value in (
                np.median([first[f"{map_name}_MS-SSIM"], second[f"{map_name}_MS-SSIM"]]),
                1.0 - np.median([first[f"{map_name}_LPIPS"], second[f"{map_name}_LPIPS"]]),
            )
        ]
    )
    assert score == expected
    assert medians["Struct_LPIPS_inverted_median"] == 0.85


def test_ms_ssim_matches_official_orchestrator_uint8_path(tmp_path):
    rng = np.random.default_rng(11)
    reference = rng.integers(0, 256, size=(256, 512), dtype=np.uint8)
    prediction = np.clip(reference.astype(np.int16) + 3, 0, 255).astype(np.uint8)
    reference_path = tmp_path / "reference.png"
    prediction_path = tmp_path / "prediction.png"
    io.imsave(reference_path, reference, check_contrast=False)
    io.imsave(prediction_path, prediction, check_contrast=False)

    loaded_reference = img_as_float(io.imread(reference_path, as_gray=True))
    loaded_prediction = img_as_float(io.imread(prediction_path, as_gray=True))
    expected = float(
        np.real(
            msssim(
                (loaded_reference * 255).astype(np.uint8),
                (loaded_prediction * 255).astype(np.uint8),
            )
        )
    )
    actual = calculate_metrics(reference_path, prediction_path, include_lpips=False)
    assert actual["MS-SSIM"] == expected


def test_scan_cli_records_request_provenance(tmp_path, monkeypatch):
    phantom = tmp_path / "phantom.txt"
    save_phantom(np.zeros((8, 4)), phantom, config=ExperimentConfig(scatterers_count=8))
    raw = tmp_path / "raw.png"
    manifest = tmp_path / "manifest.json"
    reference = tmp_path / "reference.png"
    diagnostics = tmp_path / "diagnostics.json"
    _reference(reference)
    diagnostics.write_text("{}\n")

    def fake_render(_phantom, _config, output, **_kwargs):
        io.imsave(output, np.zeros((16, 8), dtype=np.uint8), check_contrast=False)
        return "request-123", output, 2.5, 3

    monkeypatch.setattr("synthoct.cli.render_with_api", fake_render)
    assert (
        main(
            [
                "scan",
                "--phantom",
                str(phantom),
                "--out",
                str(raw),
                "--scatterers-count",
                "8",
                "--manifest",
                str(manifest),
                "--reference",
                str(reference),
                "--generator-diagnostics",
                str(diagnostics),
            ]
        )
        == 0
    )
    payload = json.loads(manifest.read_text())
    assert payload["request"]["request_id"] == "request-123"
    assert payload["inputs"]["phantom"]["sha256"]
    assert payload["inputs"]["evaluation_reference"]["sha256"]
    assert payload["inputs"]["generator_diagnostics"]["sha256"]
    assert payload["outputs"]["raw_scanner_png"]["sha256"]


def test_evidence_manifest_builds_and_fails_closed_on_extra_files(tmp_path):
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "phantom.txt").write_text("0 0 0 0\n", encoding="utf-8")
    (root / "metrics.csv").write_text("score\n0.9\n", encoding="utf-8")

    manifest = build_evidence_manifest(
        root,
        hosted_request_id="request-123",
        claims={"score": 0.9},
    )
    payload = verify_evidence_manifest(root)
    assert manifest == root / "evidence_manifest.json"
    assert payload["hosted_request_id"] == "request-123"
    assert set(payload["files"]) == {"metrics.csv", "phantom.txt"}

    (root / "untracked.json").write_text("{}\n", encoding="utf-8")
    try:
        verify_evidence_manifest(root)
    except ValueError as error:
        assert "file-set mismatch" in str(error)
    else:
        raise AssertionError("extra retained file was not detected")

    try:
        build_evidence_manifest(
            root,
            hosted_request_id="request-escape",
            claims={},
            phantom="../outside.txt",
        )
    except ValueError as error:
        assert "stay below" in str(error)
    else:
        raise AssertionError("out-of-root phantom path was accepted")

    linked = tmp_path / "linked-evidence"
    linked.mkdir()
    (tmp_path / "external.txt").write_text("external\n", encoding="utf-8")
    (linked / "phantom.txt").symlink_to(tmp_path / "external.txt")
    try:
        build_evidence_manifest(linked, hosted_request_id="request-link", claims={})
    except ValueError as error:
        assert "regular file" in str(error)
    else:
        raise AssertionError("symlinked phantom was accepted")


def test_local_benchmark_records_effective_controls_and_environment(tmp_path, monkeypatch):
    input_root = tmp_path / "references"
    input_root.mkdir()
    reference = input_root / "reference.png"
    _reference(reference, shape=(256, 512))

    def fake_generator(_reference, output, **_kwargs):
        output.write_text("0 0 0 0\n", encoding="utf-8")
        return output

    def fake_render(_phantom, output):
        io.imsave(output, io.imread(reference), check_contrast=False)
        return output

    monkeypatch.setattr("synthoct.benchmark.holographic_inverse_phantom", fake_generator)
    monkeypatch.setattr("synthoct.benchmark.render_reference_scanner", fake_render)
    monkeypatch.setattr(
        "synthoct.benchmark.calculate_metrics",
        lambda *_args, **_kwargs: {"MS-SSIM": 0.9},
    )
    monkeypatch.setattr(
        "synthoct.benchmark.profile_scores",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr("synthoct.benchmark.dependency_versions", lambda: {"numpy": "test"})
    monkeypatch.setattr(
        "synthoct.benchmark.git_provenance",
        lambda: {"commit": "test", "source_tree_sha256": "abc"},
    )

    _, summary_path = run_local_benchmark(
        input_root,
        tmp_path / "benchmark",
        scatterers_count=8,
        include_maps=False,
        include_lpips=False,
        limit=1,
    )
    summary = json.loads(summary_path.read_text())
    assert summary["benchmark_controls"] == {
        "scatterers_count": 8,
        "include_maps": False,
        "include_lpips": False,
        "limit": 1,
        "map_mode": ORGANIZER_MAP_MODE,
    }
    assert summary["scanner"]["scatterers_count"] == 8
    assert summary["environment"]["dependencies"] == {"numpy": "test"}
    assert summary["environment"]["git"]["source_tree_sha256"] == "abc"
