from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import numpy as np
from skimage import io as skio

import synthoct.correction_refinement as correction_refinement
import synthoct.energy_ratio_refinement as energy_ratio_refinement
import synthoct.candidate_rendering as candidate_rendering
import synthoct.api_recovery as api_recovery
import synthoct.learned_surrogate as learned_surrogate
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
    detail_rows = list(csv.DictReader(detail.open()))
    assert detail_rows[0]["evidence_source"] == "hosted_api_true_scanner"
    assert detail_rows[0]["evidence_scope"] == "grouped_validation_2fold_1perfold"
    assert detail_rows[0]["Struct_evaluation_region"] == "full_frame"
    assert detail_rows[0]["Struct_evaluated_shape"] == "8x8"
    challenge_rows = list(csv.DictReader((tmp_path / "validation" / "challenge_metrics_summary.csv").open()))
    assert challenge_rows[0]["evidence_source"] == "hosted_api_true_scanner"
    assert challenge_rows[0]["evaluation_region"] == "full_frame"
    assert float(challenge_rows[0]["generation_seconds_max"]) < 600.0
    assert "DepthCorr_mean" in challenge_rows[0]
    assert "OACProfileCorr_mean" in challenge_rows[0]
    assert "SCMeanAbsErr_mean" in challenge_rows[0]


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


def test_audit_evidence_accepts_grouped_true_scanner_summary(tmp_path):
    metrics = tmp_path / "challenge_metrics_summary.csv"
    with metrics.open("w", newline="") as fobj:
        writer = csv.DictWriter(
            fobj,
            fieldnames=[
                "method",
                "evidence_source",
                "evidence_scope",
                "n",
                "MS-SSIM_mean",
                "LPIPS_metric",
                "LPIPS_or_proxy_mean",
                "evaluation_region",
                "generation_seconds_max",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": "candidate",
                "evidence_source": "hosted_api_true_scanner",
                "evidence_scope": "grouped_validation_2fold_1perfold",
                "n": "2",
                "MS-SSIM_mean": "0.72",
                "LPIPS_metric": "LPIPS",
                "LPIPS_or_proxy_mean": "0.18",
                "evaluation_region": "full_frame",
                "generation_seconds_max": "10.0",
            }
        )

    assert main(["audit-evidence", "--metrics", str(metrics), "--strict", "--require-real-lpips"]) == 0


def test_audit_evidence_rejects_surrogate_preview_metrics(tmp_path):
    metrics = tmp_path / "learned_surrogate_metrics.csv"
    with metrics.open("w", newline="") as fobj:
        writer = csv.DictWriter(
            fobj,
            fieldnames=["method", "evidence_source", "evidence_scope", "surrogate_MS-SSIM", "phantom_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": "surrogate_candidate",
                "evidence_source": "learned_surrogate_preview",
                "evidence_scope": "not_challenge_evidence",
                "surrogate_MS-SSIM": "0.90",
                "phantom_path": "candidate.txt",
            }
        )

    assert main(["audit-evidence", "--metrics", str(metrics), "--strict"]) == 2


def test_audit_evidence_treats_single_reference_true_scanner_as_limited(tmp_path):
    metrics = tmp_path / "candidate_queue_metrics.csv"
    with metrics.open("w", newline="") as fobj:
        writer = csv.DictWriter(
            fobj,
            fieldnames=["method", "evidence_source", "evidence_scope", "evaluation_region", "MS-SSIM", "LPIPS_PROXY"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": "candidate",
                "evidence_source": "hosted_api_true_scanner",
                "evidence_scope": "single_reference_candidate_queue",
                "evaluation_region": "full_frame",
                "MS-SSIM": "0.773537950274",
                "LPIPS_PROXY": "0.21",
            }
        )

    assert main(["audit-evidence", "--metrics", str(metrics)]) == 0
    assert main(["audit-evidence", "--metrics", str(metrics), "--strict"]) == 2


def test_decide_promotion_accepts_candidate_that_beats_baseline(tmp_path):
    metrics = tmp_path / "challenge_metrics_summary.csv"
    with metrics.open("w", newline="") as fobj:
        writer = csv.DictWriter(
            fobj,
            fieldnames=[
                "method",
                "evidence_source",
                "evidence_scope",
                "evaluation_region",
                "n",
                "MS-SSIM_mean",
                "LPIPS_metric",
                "LPIPS_or_proxy_mean",
                "MS-SSIM_wins",
                "LPIPS_wins",
                "generation_seconds_max",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": "baseline",
                "evidence_source": "hosted_api_true_scanner",
                "evidence_scope": "grouped_validation_2fold_1perfold",
                "evaluation_region": "full_frame",
                "n": "2",
                "MS-SSIM_mean": "0.70",
                "LPIPS_metric": "LPIPS",
                "LPIPS_or_proxy_mean": "0.25",
                "MS-SSIM_wins": "0",
                "LPIPS_wins": "0",
                "generation_seconds_max": "12.0",
            }
        )
        writer.writerow(
            {
                "method": "candidate",
                "evidence_source": "hosted_api_true_scanner",
                "evidence_scope": "grouped_validation_2fold_1perfold",
                "evaluation_region": "full_frame",
                "n": "2",
                "MS-SSIM_mean": "0.74",
                "LPIPS_metric": "LPIPS",
                "LPIPS_or_proxy_mean": "0.24",
                "MS-SSIM_wins": "2",
                "LPIPS_wins": "2",
                "generation_seconds_max": "11.0",
            }
        )

    assert main(["decide-promotion", "--metrics", str(metrics), "--candidate", "candidate", "--baseline", "baseline", "--strict"]) == 0


def test_decide_promotion_rejects_candidate_that_loses_lpips(tmp_path):
    metrics = tmp_path / "challenge_metrics_summary.csv"
    with metrics.open("w", newline="") as fobj:
        writer = csv.DictWriter(
            fobj,
            fieldnames=[
                "method",
                "evidence_source",
                "evidence_scope",
                "evaluation_region",
                "n",
                "MS-SSIM_mean",
                "LPIPS_metric",
                "LPIPS_or_proxy_mean",
                "MS-SSIM_wins",
                "LPIPS_wins",
                "generation_seconds_max",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": "baseline",
                "evidence_source": "hosted_api_true_scanner",
                "evidence_scope": "grouped_validation_2fold_1perfold",
                "evaluation_region": "full_frame",
                "n": "2",
                "MS-SSIM_mean": "0.70",
                "LPIPS_metric": "LPIPS",
                "LPIPS_or_proxy_mean": "0.20",
                "MS-SSIM_wins": "1",
                "LPIPS_wins": "2",
                "generation_seconds_max": "12.0",
            }
        )
        writer.writerow(
            {
                "method": "candidate",
                "evidence_source": "hosted_api_true_scanner",
                "evidence_scope": "grouped_validation_2fold_1perfold",
                "evaluation_region": "full_frame",
                "n": "2",
                "MS-SSIM_mean": "0.74",
                "LPIPS_metric": "LPIPS",
                "LPIPS_or_proxy_mean": "0.30",
                "MS-SSIM_wins": "2",
                "LPIPS_wins": "0",
                "generation_seconds_max": "11.0",
            }
        )

    assert main(["decide-promotion", "--metrics", str(metrics), "--candidate", "candidate", "--baseline", "baseline", "--strict"]) == 2


def test_decide_promotion_rejects_candidate_over_runtime_budget(tmp_path):
    metrics = tmp_path / "challenge_metrics_summary.csv"
    with metrics.open("w", newline="") as fobj:
        writer = csv.DictWriter(
            fobj,
            fieldnames=[
                "method",
                "evidence_source",
                "evidence_scope",
                "evaluation_region",
                "n",
                "MS-SSIM_mean",
                "LPIPS_metric",
                "LPIPS_or_proxy_mean",
                "MS-SSIM_wins",
                "LPIPS_wins",
                "generation_seconds_max",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": "baseline",
                "evidence_source": "hosted_api_true_scanner",
                "evidence_scope": "grouped_validation_2fold_1perfold",
                "evaluation_region": "full_frame",
                "n": "2",
                "MS-SSIM_mean": "0.70",
                "LPIPS_metric": "LPIPS",
                "LPIPS_or_proxy_mean": "0.25",
                "MS-SSIM_wins": "0",
                "LPIPS_wins": "0",
                "generation_seconds_max": "20.0",
            }
        )
        writer.writerow(
            {
                "method": "candidate",
                "evidence_source": "hosted_api_true_scanner",
                "evidence_scope": "grouped_validation_2fold_1perfold",
                "evaluation_region": "full_frame",
                "n": "2",
                "MS-SSIM_mean": "0.74",
                "LPIPS_metric": "LPIPS",
                "LPIPS_or_proxy_mean": "0.24",
                "MS-SSIM_wins": "2",
                "LPIPS_wins": "2",
                "generation_seconds_max": "601.0",
            }
        )

    assert main(["decide-promotion", "--metrics", str(metrics), "--candidate", "candidate", "--baseline", "baseline", "--strict"]) == 2


def test_decide_promotion_rejects_candidate_with_guardrail_regression(tmp_path):
    metrics = tmp_path / "challenge_metrics_summary.csv"
    with metrics.open("w", newline="") as fobj:
        writer = csv.DictWriter(
            fobj,
            fieldnames=[
                "method",
                "evidence_source",
                "evidence_scope",
                "evaluation_region",
                "n",
                "MS-SSIM_mean",
                "LPIPS_metric",
                "LPIPS_or_proxy_mean",
                "MS-SSIM_wins",
                "LPIPS_wins",
                "generation_seconds_max",
                "DepthCorr_mean",
                "OACProfileCorr_mean",
                "SCMeanAbsErr_mean",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": "baseline",
                "evidence_source": "hosted_api_true_scanner",
                "evidence_scope": "grouped_validation_2fold_1perfold",
                "evaluation_region": "full_frame",
                "n": "2",
                "MS-SSIM_mean": "0.70",
                "LPIPS_metric": "LPIPS",
                "LPIPS_or_proxy_mean": "0.25",
                "MS-SSIM_wins": "0",
                "LPIPS_wins": "0",
                "generation_seconds_max": "20.0",
                "DepthCorr_mean": "0.90",
                "OACProfileCorr_mean": "0.80",
                "SCMeanAbsErr_mean": "0.10",
            }
        )
        writer.writerow(
            {
                "method": "candidate",
                "evidence_source": "hosted_api_true_scanner",
                "evidence_scope": "grouped_validation_2fold_1perfold",
                "evaluation_region": "full_frame",
                "n": "2",
                "MS-SSIM_mean": "0.74",
                "LPIPS_metric": "LPIPS",
                "LPIPS_or_proxy_mean": "0.24",
                "MS-SSIM_wins": "2",
                "LPIPS_wins": "2",
                "generation_seconds_max": "11.0",
                "DepthCorr_mean": "0.70",
                "OACProfileCorr_mean": "0.80",
                "SCMeanAbsErr_mean": "0.10",
            }
        )

    assert main(["decide-promotion", "--metrics", str(metrics), "--candidate", "candidate", "--baseline", "baseline", "--strict"]) == 2


def test_select_best_picks_top_promoted_candidate(tmp_path):
    metrics = tmp_path / "challenge_metrics_summary.csv"
    with metrics.open("w", newline="") as fobj:
        writer = csv.DictWriter(
            fobj,
            fieldnames=[
                "method",
                "evidence_source",
                "evidence_scope",
                "evaluation_region",
                "n",
                "MS-SSIM_mean",
                "LPIPS_metric",
                "LPIPS_or_proxy_mean",
                "MS-SSIM_wins",
                "LPIPS_wins",
                "generation_seconds_max",
            ],
        )
        writer.writeheader()
        for row in [
            ("baseline", "0.70", "0.25", "0", "0", "12.0"),
            ("candidate_a", "0.74", "0.24", "2", "2", "11.0"),
            ("candidate_b", "0.76", "0.24", "2", "2", "10.0"),
        ]:
            method, ms, lpips, ms_wins, lpips_wins, runtime = row
            writer.writerow(
                {
                    "method": method,
                    "evidence_source": "hosted_api_true_scanner",
                    "evidence_scope": "grouped_validation_2fold_1perfold",
                    "evaluation_region": "full_frame",
                    "n": "2",
                    "MS-SSIM_mean": ms,
                    "LPIPS_metric": "LPIPS",
                    "LPIPS_or_proxy_mean": lpips,
                    "MS-SSIM_wins": ms_wins,
                    "LPIPS_wins": lpips_wins,
                    "generation_seconds_max": runtime,
                }
            )

    assert main(["select-best", "--metrics", str(metrics), "--baseline", "baseline", "--strict"]) == 0


def test_select_best_rejects_when_no_candidate_beats_baseline(tmp_path):
    metrics = tmp_path / "challenge_metrics_summary.csv"
    with metrics.open("w", newline="") as fobj:
        writer = csv.DictWriter(
            fobj,
            fieldnames=[
                "method",
                "evidence_source",
                "evidence_scope",
                "evaluation_region",
                "n",
                "MS-SSIM_mean",
                "LPIPS_metric",
                "LPIPS_or_proxy_mean",
                "MS-SSIM_wins",
                "LPIPS_wins",
                "generation_seconds_max",
            ],
        )
        writer.writeheader()
        for row in [
            ("baseline", "0.75", "0.20", "2", "2", "12.0"),
            ("candidate_a", "0.74", "0.19", "0", "2", "11.0"),
            ("candidate_b", "0.76", "0.30", "2", "0", "10.0"),
        ]:
            method, ms, lpips, ms_wins, lpips_wins, runtime = row
            writer.writerow(
                {
                    "method": method,
                    "evidence_source": "hosted_api_true_scanner",
                    "evidence_scope": "grouped_validation_2fold_1perfold",
                    "evaluation_region": "full_frame",
                    "n": "2",
                    "MS-SSIM_mean": ms,
                    "LPIPS_metric": "LPIPS",
                    "LPIPS_or_proxy_mean": lpips,
                    "MS-SSIM_wins": ms_wins,
                    "LPIPS_wins": lpips_wins,
                    "generation_seconds_max": runtime,
                }
            )

    assert main(["select-best", "--metrics", str(metrics), "--baseline", "baseline", "--strict"]) == 2


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
                "--holdout-fraction",
                "0.4",
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
    assert captured["holdout_fraction"] == 0.4


def test_learned_surrogate_split_keeps_train_and_holdout_disjoint(tmp_path):
    pairs = [
        learned_surrogate.ScannerPair(
            label=f"pair_{idx}",
            phantom_path=tmp_path / f"phantom_{idx}.txt",
            rendered_gray_path=tmp_path / f"render_{idx}.png",
            ssim=0.1 * idx,
        )
        for idx in range(5)
    ]

    train, holdout = learned_surrogate._split_train_holdout(pairs, holdout_fraction=0.4, seed=7)

    assert len(train) == 3
    assert len(holdout) == 2
    assert {pair.label for pair in train}.isdisjoint({pair.label for pair in holdout})


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
    rows = list(csv.DictReader(metrics.open()))
    assert rows[0]["evidence_source"] == "hosted_api_true_scanner"
    assert rows[0]["evidence_scope"] == "single_reference_candidate_queue"
    assert rows[0]["evaluation_region"] == "full_frame"
    assert rows[0]["reference_shape"] == "256x512"
    assert rows[0]["evaluated_shape"] == "256x512"


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


def test_cli_recover_api_results_updates_failed_row(tmp_path, monkeypatch):
    scan = tmp_path / "reference.png"
    image = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (256, 1))
    skio.imsave(scan, image)
    png = tmp_path / "synthetic.png"
    gray = tmp_path / "synthetic_gray.png"
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "status,method,request_id,synthetic_png,synthetic_gray_png,error,MSE,PSNR,SSIM,MS-SSIM,VIF,LPIPS,LPIPS_PROXY\n"
        f"failed,candidate,failed,{png},{gray},API result was not ready: https://synthoct.com/results/result_abc123.png,nan,nan,nan,nan,nan,nan,nan\n",
        encoding="utf-8",
    )

    class FakeResponse:
        status_code = 200

        def __init__(self, content):
            self.content = content

    rendered = tmp_path / "rendered_source.png"
    skio.imsave(rendered, image)
    monkeypatch.setattr(api_recovery.requests, "get", lambda url, timeout: FakeResponse(rendered.read_bytes()))

    assert main(["recover-api-results", "--metrics", str(metrics), "--ref", str(scan)]) == 0
    text = metrics.read_text()
    assert "ok,candidate,abc123" in text
    assert gray.exists()
