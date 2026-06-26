from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
from skimage import io as skio

import synthoct.correction_refinement as correction_refinement
import synthoct.energy_ratio_refinement as energy_ratio_refinement
import synthoct.candidate_rendering as candidate_rendering
import synthoct.scanners.api as scanner_api
import synthoct.seed_search as seed_search
import synthoct.transfer_refinement as transfer_refinement
import synthoct.validation as validation
from synthoct.cli import main
from synthoct.dataset import iter_records, prepare_dataset
from synthoct.scanners import prepare_api_render_request, render_phantom, write_api_config, write_scanner_config
from synthoct.validation import run_internal_validation


def _make_nested_dataset(path):
    inner_buf = io.BytesIO()
    with zipfile.ZipFile(inner_buf, "w") as inner:
        npy_buf = io.BytesIO()
        np.save(npy_buf, np.ones((8, 8), dtype=np.float32))
        inner.writestr("DATASET_NPY/Female/1990-2000/Cheek/sample_frame50.npy", npy_buf.getvalue())
        png_buf = io.BytesIO()
        skio.imsave(png_buf, np.ones((8, 8), dtype=np.uint8) * 127, extension=".png")
        inner.writestr("DATASET_PNG/Female/1990-2000/Cheek/sample_frame50.png", png_buf.getvalue())
    with zipfile.ZipFile(path, "w") as outer:
        outer.writestr("DATASET.zip", inner_buf.getvalue())


def test_dataset_manifest_from_nested_zip(tmp_path):
    archive = tmp_path / "dataset.zip"
    _make_nested_dataset(archive)
    records = iter_records(archive)
    assert len(records) == 2
    manifest = prepare_dataset(archive, tmp_path / "data", extract=False)
    assert manifest.read_text().count("\n") == 3


def test_cli_official_and_precomputed_scan(tmp_path):
    phantom = tmp_path / "phantom.txt"
    scan = tmp_path / "scan.png"
    precomputed = tmp_path / "precomputed.png"
    skio.imsave(precomputed, np.ones((8, 8), dtype=np.uint8) * 64)
    assert main(["baseline", "official", "--out", str(phantom), "--scatterers-count", "256"]) == 0
    assert phantom.exists()
    assert main(["scan", "--phantom", str(phantom), "--out", str(scan), "--mode", "precomputed", "--precomputed", str(precomputed)]) == 0
    assert scan.exists()


def test_scanner_config_generation_and_backend_interface(tmp_path):
    phantom = tmp_path / "Scatterers.txt"
    phantom.write_text("0 0 0 1\n", encoding="utf-8")
    api_config = write_api_config(tmp_path / "Configuration_api.ini", scatterers_count=1)
    windows_config = write_scanner_config(tmp_path / "Configuration_windows.ini", phantom, tmp_path / "scan.png")

    assert "scatterers coordinates file = Scatterers.txt" in api_config.read_text(encoding="utf-8")
    assert "number of scatterers in b-scan = 300000" in windows_config.read_text(encoding="utf-8").lower()

    class FakeBackend:
        def render(self, phantom_path, config_path, output_png):
            output_png = tmp_path / "fake_scan.png"
            output_png.write_bytes(b"png")
            return output_png

    rendered = render_phantom(phantom, api_config, tmp_path / "fake_scan.png", backend=FakeBackend())
    assert rendered.read_bytes() == b"png"


def test_api_request_preparation_redacts_secret(tmp_path):
    phantom = tmp_path / "Scatterers.txt"
    config = tmp_path / "Configuration.ini"
    phantom.write_text("0 0 0 1\n", encoding="utf-8")
    config.write_text("[Parameters]\n", encoding="utf-8")

    request = prepare_api_render_request(phantom, config, api_key="super-secret")
    assert request.headers["X-API-Key"] == "super-secret"
    assert request.redacted_headers["X-API-Key"] == "<redacted>"
    assert "super-secret" not in repr(request)


def test_api_request_retries_retryable_status(monkeypatch):
    calls = []

    class FakeResponse:
        def __init__(self, status_code):
            self.status_code = status_code

    def fake_request(method, url, **kwargs):
        del method, url, kwargs
        calls.append(1)
        return FakeResponse(500 if len(calls) == 1 else 200)

    monkeypatch.setattr(scanner_api.requests, "request", fake_request)
    monkeypatch.setattr(scanner_api.time, "sleep", lambda seconds: None)

    response = scanner_api._request_with_retries("POST", "https://example.test", attempts=3)

    assert response.status_code == 200
    assert len(calls) == 2


def test_api_request_returns_final_retryable_status(monkeypatch):
    calls = []

    class FakeResponse:
        status_code = 500

    def fake_request(method, url, **kwargs):
        del method, url, kwargs
        calls.append(1)
        return FakeResponse()

    monkeypatch.setattr(scanner_api.requests, "request", fake_request)
    monkeypatch.setattr(scanner_api.time, "sleep", lambda seconds: None)

    response = scanner_api._request_with_retries("POST", "https://example.test", attempts=2)

    assert response.status_code == 500
    assert len(calls) == 2


def test_internal_validation_writes_summary(tmp_path, monkeypatch):
    def fake_render_with_api(phantom_path, config_path, out_png, **kwargs):
        image = np.ones((256, 512), dtype=np.uint8) * 96
        skio.imsave(out_png, image)
        return "fake-request", out_png, 0.0, 0

    monkeypatch.setattr(validation, "render_with_api", fake_render_with_api)
    archive = tmp_path / "dataset.zip"
    _make_nested_dataset(archive)
    detail, summary = run_internal_validation(
        archive,
        tmp_path / "validation",
        methods=["official", "physics-guided"],
        folds=2,
        max_per_fold=1,
        scatterers_count=128,
        include_maps=False,
        include_lpips=False,
    )
    assert detail.exists()
    assert summary.exists()
    assert (tmp_path / "validation" / "challenge_metrics_summary.csv").exists()
    assert (tmp_path / "validation" / "hypothesis_wins.csv").exists()
    text = summary.read_text()
    assert "physics-guided" in text
    assert "official" in text


def test_internal_validation_accepts_promising_pipeline_wave(tmp_path, monkeypatch):
    def fake_render_with_api(phantom_path, config_path, out_png, **kwargs):
        image = np.ones((256, 512), dtype=np.uint8) * 112
        skio.imsave(out_png, image)
        return "fake-request", out_png, 0.0, 0

    monkeypatch.setattr(validation, "render_with_api", fake_render_with_api)
    archive = tmp_path / "dataset.zip"
    _make_nested_dataset(archive)
    methods = validation.resolve_method_wave("promising-pipelines")
    detail, summary = run_internal_validation(
        archive,
        tmp_path / "pipeline_validation",
        methods=methods,
        folds=1,
        max_per_fold=1,
        scatterers_count=96,
        include_maps=False,
        include_lpips=False,
    )
    assert detail.exists()
    text = summary.read_text()
    assert "P01_simulator_constrained_prior" in text
    assert "P05_attenuation_layer_map" in text


def test_cli_final_baseline(tmp_path):
    img = np.tile(np.linspace(0, 255, 32, dtype=np.uint8), (32, 1))
    scan = tmp_path / "scan.png"
    phantom = tmp_path / "final.txt"
    skio.imsave(scan, img)
    assert main(["baseline", "final", "--input", str(scan), "--out", str(phantom), "--scatterers-count", "128"]) == 0
    assert phantom.exists()


def test_cli_pipeline_baseline(tmp_path):
    img = np.tile(np.linspace(0, 255, 32, dtype=np.uint8), (32, 1))
    scan = tmp_path / "scan.png"
    phantom = tmp_path / "pipeline.txt"
    skio.imsave(scan, img)
    assert (
        main(
            [
                "baseline",
                "pipeline",
                "--input",
                str(scan),
                "--name",
                "P03_speckle_preserving_texture",
                "--out",
                str(phantom),
                "--scatterers-count",
                "96",
            ]
        )
        == 0
    )
    assert phantom.exists()


def test_cli_optimize_seeds_writes_ranked_metrics(tmp_path, monkeypatch):
    def fake_render_with_api(phantom_path, config_path, out_png, **kwargs):
        image = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (256, 1))
        skio.imsave(out_png, image)
        return "fake-request", out_png, 0.0, 0

    monkeypatch.setattr(seed_search, "render_with_api", fake_render_with_api)
    scan = tmp_path / "reference.png"
    image = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (256, 1))
    skio.imsave(scan, image)

    assert (
        main(
            [
                "optimize-seeds",
                "--input",
                str(scan),
                "--out",
                str(tmp_path / "seed_search"),
                "--method",
                "P09_gamma_sparse_lowfloor_ssim",
                "--seeds",
                "1",
                "2",
                "--scatterers-count",
                "64",
            ]
        )
        == 0
    )
    metrics = tmp_path / "seed_search" / "seed_search_metrics.csv"
    assert metrics.exists()
    assert "P09_gamma_sparse_lowfloor_ssim" in metrics.read_text()


def test_cli_optimize_correction_writes_iterative_metrics(tmp_path, monkeypatch):
    def fake_render_with_api(phantom_path, config_path, out_png, **kwargs):
        del phantom_path, config_path, kwargs
        image = np.tile(np.linspace(12, 224, 512, dtype=np.uint8), (256, 1))
        skio.imsave(out_png, image)
        return "fake-request", out_png, 0.0, 0

    monkeypatch.setattr(correction_refinement, "render_with_api", fake_render_with_api)
    scan = tmp_path / "reference.png"
    image = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (256, 1))
    skio.imsave(scan, image)

    assert (
        main(
            [
                "optimize-correction",
                "--input",
                str(scan),
                "--out",
                str(tmp_path / "correction"),
                "--scatterers-count",
                "64",
                "--exponents",
                "0.7",
                "0.5",
                "--ratio-highs",
                "2.0",
                "1.8",
            ]
        )
        == 0
    )
    metrics = tmp_path / "correction" / "correction_refinement_metrics.csv"
    assert metrics.exists()
    text = metrics.read_text()
    assert "base" in text
    assert "iter1" in text
    assert "iter2" in text


def test_cli_optimize_transfer_writes_ranked_metrics(tmp_path, monkeypatch):
    def fake_render_with_api(phantom_path, config_path, out_png, **kwargs):
        del phantom_path, config_path, kwargs
        image = np.tile(np.linspace(0, 220, 512, dtype=np.uint8), (256, 1))
        skio.imsave(out_png, image)
        return "fake-request", out_png, 0.0, 0

    monkeypatch.setattr(transfer_refinement, "render_with_api", fake_render_with_api)
    scan = tmp_path / "reference.png"
    rendered = tmp_path / "rendered_gray.png"
    image = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (256, 1))
    skio.imsave(scan, image)
    skio.imsave(rendered, image)
    phantom = tmp_path / "phantom.txt"
    assert main(["baseline", "official", "--out", str(phantom), "--scatterers-count", "64"]) == 0

    assert (
        main(
            [
                "optimize-transfer",
                "--ref",
                str(scan),
                "--phantom",
                str(phantom),
                "--rendered-gray",
                str(rendered),
                "--out",
                str(tmp_path / "transfer"),
                "--exponents",
                "0.1",
                "0.2",
            ]
        )
        == 0
    )
    metrics = tmp_path / "transfer" / "selective_transfer_metrics.csv"
    assert metrics.exists()
    text = metrics.read_text()
    assert "transfer_e0p100" in text
    assert "transfer_e0p200" in text


def test_cli_optimize_energy_ratio_writes_ranked_metrics(tmp_path, monkeypatch):
    def fake_render_with_api(phantom_path, config_path, out_png, **kwargs):
        del phantom_path, config_path, kwargs
        image = np.tile(np.linspace(4, 230, 512, dtype=np.uint8), (256, 1))
        skio.imsave(out_png, image)
        return "fake-request", out_png, 0.0, 0

    monkeypatch.setattr(energy_ratio_refinement, "render_with_api", fake_render_with_api)
    scan = tmp_path / "reference.png"
    rendered = tmp_path / "rendered_gray.png"
    image = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (256, 1))
    skio.imsave(scan, image)
    skio.imsave(rendered, image)
    phantom = tmp_path / "phantom.txt"
    assert main(["baseline", "official", "--out", str(phantom), "--scatterers-count", "64"]) == 0

    assert (
        main(
            [
                "optimize-energy-ratio",
                "--ref",
                str(scan),
                "--phantom",
                str(phantom),
                "--rendered-gray",
                str(rendered),
                "--out",
                str(tmp_path / "energy_ratio"),
                "--exponents",
                "0.008",
                "0.012",
            ]
        )
        == 0
    )
    metrics = tmp_path / "energy_ratio" / "energy_ratio_metrics.csv"
    assert metrics.exists()
    text = metrics.read_text()
    assert "energy_ratio_e0p008" in text
    assert "energy_ratio_e0p012" in text


def test_cli_optimize_energy_ratio_accepts_shape_controls(tmp_path, monkeypatch):
    captured = {}

    def fake_run_energy_ratio_refinement(ref, phantom, rendered_gray, out_dir, **kwargs):
        del ref, phantom, rendered_gray
        captured.update(kwargs)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics = out_dir / "energy_ratio_metrics.csv"
        metrics.write_text("method,SSIM\nenergy_ratio_e0p100,0.1\n", encoding="utf-8")
        return metrics

    monkeypatch.setattr("synthoct.cli.run_energy_ratio_refinement", fake_run_energy_ratio_refinement)

    scan = tmp_path / "reference.png"
    rendered = tmp_path / "rendered_gray.png"
    phantom = tmp_path / "phantom.txt"
    image = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (256, 1))
    skio.imsave(scan, image)
    skio.imsave(rendered, image)
    assert main(["baseline", "official", "--out", str(phantom), "--scatterers-count", "64"]) == 0

    out_dir = tmp_path / "energy_ratio_controls"
    assert (
        main(
            [
                "optimize-energy-ratio",
                "--ref",
                str(scan),
                "--phantom",
                str(phantom),
                "--rendered-gray",
                str(rendered),
                "--out",
                str(out_dir),
                "--exponents",
                "0.1",
                "--stabilizer",
                "0.05",
                "--sigma",
                "4.0",
                "--ratio-low",
                "0.5",
                "--ratio-high",
                "1.1",
                "--clip-low",
                "0.7",
                "--clip-high",
                "1.05",
                "--air-boundary",
                "34",
                "--no-match-total-energy",
            ]
        )
        == 0
    )

    assert captured["stabilizer"] == 0.05
    assert captured["sigma"] == 4.0
    assert captured["ratio_low"] == 0.5
    assert captured["ratio_high"] == 1.1
    assert captured["clip_low"] == 0.7
    assert captured["clip_high"] == 1.05
    assert captured["air_boundary"] == 34
    assert captured["match_total_energy"] is False


def test_cli_optimize_learned_surrogate_passes_controls(tmp_path, monkeypatch):
    captured = {}

    def fake_run_learned_surrogate_refinement(ref, out_dir, **kwargs):
        captured["ref"] = ref
        captured["out_dir"] = out_dir
        captured.update(kwargs)
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        metrics = out_path / "learned_surrogate_metrics.csv"
        metrics.write_text("method,SSIM\nlearned_surrogate_r0p80,0.2\n", encoding="utf-8")
        return metrics

    monkeypatch.setattr("synthoct.cli.run_learned_surrogate_refinement", fake_run_learned_surrogate_refinement)
    scan = tmp_path / "reference.png"
    image = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (256, 1))
    skio.imsave(scan, image)

    assert (
        main(
            [
                "optimize-learned-surrogate",
                "--ref",
                str(scan),
                "--out",
                str(tmp_path / "learned"),
                "--outputs-dir",
                str(tmp_path / "outputs"),
                "--shape",
                "64",
                "128",
                "--train-limit",
                "5",
                "--epochs",
                "3",
                "--optimize-steps",
                "4",
                "--scatterers-count",
                "256",
                "--seed",
                "11",
                "--energy-ratios",
                "0.8",
                "0.9",
                "--texture-strengths",
                "0.2",
                "0.6",
            ]
        )
        == 0
    )

    assert captured["ref"] == str(scan)
    assert captured["shape"] == (64, 128)
    assert captured["train_limit"] == 5
    assert captured["epochs"] == 3
    assert captured["optimize_steps"] == 4
    assert captured["scatterers_count"] == 256
    assert captured["seed"] == 11
    assert captured["energy_ratios"] == [0.8, 0.9]
    assert captured["texture_strengths"] == [0.2, 0.6]


def test_cli_render_candidate_queue_writes_metrics(tmp_path, monkeypatch):
    def fake_render_with_api(phantom_path, config_path, out_png, **kwargs):
        del phantom_path, config_path, kwargs
        image = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (256, 1))
        skio.imsave(out_png, image)
        return "fake-request", out_png, 0.0, 1

    monkeypatch.setattr(candidate_rendering, "render_with_api", fake_render_with_api)
    scan = tmp_path / "reference.png"
    image = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (256, 1))
    skio.imsave(scan, image)
    phantom = tmp_path / "phantom.txt"
    assert main(["baseline", "official", "--out", str(phantom), "--scatterers-count", "64"]) == 0
    queue = tmp_path / "queue.csv"
    queue.write_text(f"priority,method,phantom_path\n1,test_candidate,{phantom}\n", encoding="utf-8")

    assert (
        main(
            [
                "render-candidate-queue",
                "--queue",
                str(queue),
                "--ref",
                str(scan),
                "--out",
                str(tmp_path / "rendered_queue"),
                "--max-candidates",
                "1",
                "--poll-interval-seconds",
                "0.1",
            ]
        )
        == 0
    )
    metrics = tmp_path / "rendered_queue" / "candidate_queue_metrics.csv"
    assert metrics.exists()
    text = metrics.read_text()
    assert "test_candidate" in text
    assert "fake-request" in text


def test_cli_optimize_empirical_basis_passes_controls(tmp_path, monkeypatch):
    captured = {}

    def fake_run_empirical_basis_refinement(ref, out_dir, **kwargs):
        captured["ref"] = ref
        captured["out_dir"] = out_dir
        captured.update(kwargs)
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        metrics = out_path / "empirical_basis_metrics.csv"
        metrics.write_text("method,SSIM\nempirical_basis,0.2\n", encoding="utf-8")
        return metrics

    monkeypatch.setattr("synthoct.cli.run_empirical_basis_refinement", fake_run_empirical_basis_refinement)
    scan = tmp_path / "reference.png"
    image = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (256, 1))
    skio.imsave(scan, image)

    assert (
        main(
            [
                "optimize-empirical-basis",
                "--ref",
                str(scan),
                "--out",
                str(tmp_path / "basis"),
                "--outputs-dir",
                str(tmp_path / "outputs"),
                "--shape",
                "64",
                "128",
                "--pair-limit",
                "9",
                "--basis-count",
                "5",
                "--scatterers-count",
                "512",
                "--seed",
                "17",
                "--residual-exponents",
                "0.1",
                "0.3",
                "--energy-ratios",
                "0.8",
                "1.0",
                "--texture-strengths",
                "0.0",
                "0.4",
            ]
        )
        == 0
    )

    assert captured["ref"] == str(scan)
    assert captured["shape"] == (64, 128)
    assert captured["pair_limit"] == 9
    assert captured["basis_count"] == 5
    assert captured["scatterers_count"] == 512
    assert captured["seed"] == 17
    assert captured["residual_exponents"] == [0.1, 0.3]
    assert captured["energy_ratios"] == [0.8, 1.0]
    assert captured["texture_strengths"] == [0.0, 0.4]


def test_cli_optimize_patch_basis_passes_controls(tmp_path, monkeypatch):
    captured = {}

    def fake_run_patch_basis_refinement(ref, out_dir, **kwargs):
        captured["ref"] = ref
        captured["out_dir"] = out_dir
        captured.update(kwargs)
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        metrics = out_path / "patch_basis_metrics.csv"
        metrics.write_text("method,SSIM\npatch_basis,0.2\n", encoding="utf-8")
        return metrics

    monkeypatch.setattr("synthoct.cli.run_patch_basis_refinement", fake_run_patch_basis_refinement)
    scan = tmp_path / "reference.png"
    image = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (256, 1))
    skio.imsave(scan, image)

    assert (
        main(
            [
                "optimize-patch-basis",
                "--ref",
                str(scan),
                "--out",
                str(tmp_path / "patch_basis"),
                "--outputs-dir",
                str(tmp_path / "outputs"),
                "--shape",
                "64",
                "128",
                "--basis-count",
                "7",
                "--tile-shape",
                "16",
                "24",
                "--scatterers-count",
                "512",
                "--seed",
                "19",
                "--residual-exponents",
                "0.1",
                "--texture-strengths",
                "0.2",
                "--energy-ratios",
                "0.8",
            ]
        )
        == 0
    )

    assert captured["ref"] == str(scan)
    assert captured["shape"] == (64, 128)
    assert captured["basis_count"] == 7
    assert captured["tile_shape"] == (16, 24)
    assert captured["scatterers_count"] == 512
    assert captured["seed"] == 19
    assert captured["residual_exponents"] == [0.1]
    assert captured["texture_strengths"] == [0.2]
    assert captured["energy_ratios"] == [0.8]


def test_cli_optimize_direct_lattice_passes_controls(tmp_path, monkeypatch):
    captured = {}

    def fake_run_direct_lattice_refinement(ref, out_dir, **kwargs):
        captured["ref"] = ref
        captured["out_dir"] = out_dir
        captured.update(kwargs)
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        metrics = out_path / "direct_lattice_metrics.csv"
        metrics.write_text("method,SSIM\ndirect_lattice,0.2\n", encoding="utf-8")
        return metrics

    monkeypatch.setattr("synthoct.cli.run_direct_lattice_refinement", fake_run_direct_lattice_refinement)
    scan = tmp_path / "reference.png"
    image = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (256, 1))
    skio.imsave(scan, image)

    assert (
        main(
            [
                "optimize-direct-lattice",
                "--ref",
                str(scan),
                "--out",
                str(tmp_path / "lattice"),
                "--scatterers-count",
                "512",
                "--seed",
                "23",
                "--recipes",
                "sqrt_attn",
                "surface_locked",
                "--energy-scales",
                "0.02",
                "0.03",
            ]
        )
        == 0
    )

    assert captured["ref"] == str(scan)
    assert captured["scatterers_count"] == 512
    assert captured["seed"] == 23
    assert captured["recipes"] == ["sqrt_attn", "surface_locked"]
    assert captured["energy_scales"] == [0.02, 0.03]
