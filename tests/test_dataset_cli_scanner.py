from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from skimage import io as skio

import synthoct.correction_refinement as correction_refinement
import synthoct.energy_ratio_refinement as energy_ratio_refinement
import synthoct.adaptive_flow_batch as adaptive_flow_batch
import synthoct.candidate_rendering as candidate_rendering
import synthoct.api_recovery as api_recovery
import synthoct.cli as cli_module
import synthoct.learned_surrogate as learned_surrogate
import synthoct.scanners.api as scanner_api
import synthoct.seed_search as seed_search
import synthoct.transfer_refinement as transfer_refinement
import synthoct.validation as validation
from synthoct.cli import main
from synthoct.dataset import iter_records, prepare_dataset, write_scan_png_from_zip
from synthoct.evaluation import audit_challenge_evidence
from synthoct.phantom import ExperimentConfig, generate_two_layers, load_phantom, save_phantom
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


def test_write_scan_png_from_zip_preserves_float_png_scale(tmp_path):
    archive = tmp_path / "dataset.zip"
    _make_nested_dataset(archive)
    out_png = write_scan_png_from_zip(
        archive,
        "DATASET_PNG/Female/1990-2000/Cheek/sample_frame50.png",
        tmp_path / "reference.png",
    )

    image = skio.imread(out_png, as_gray=True)
    assert image.max() > 1.0
    assert image.max() == 127


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


def test_render_with_api_attaches_request_id_to_poll_failure(monkeypatch, tmp_path):
    def fake_submit_api_render(*args, **kwargs):
        del args, kwargs
        return "abc123", 0.1

    def fake_poll_api_result(*args, **kwargs):
        del args, kwargs
        raise TimeoutError("not ready")

    monkeypatch.setattr(scanner_api, "submit_api_render", fake_submit_api_render)
    monkeypatch.setattr(scanner_api, "poll_api_result", fake_poll_api_result)

    try:
        scanner_api.render_with_api(tmp_path / "phantom.txt", tmp_path / "config.ini", tmp_path / "out.png")
    except TimeoutError as exc:
        assert getattr(exc, "request_id") == "abc123"
    else:
        raise AssertionError("render_with_api should propagate the poll failure")


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
    reference_path = Path(detail_rows[0]["reference_png"])
    assert reference_path.exists()
    assert tmp_path / "validation" in reference_path.parents
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


def test_internal_validation_accepts_learned_prior_with_artifact(tmp_path, monkeypatch):
    def fake_render_with_api(phantom_path, config_path, out_png, **kwargs):
        image = np.ones((256, 512), dtype=np.uint8) * 104
        skio.imsave(out_png, image)
        return "fake-request", out_png, 0.0, 0

    monkeypatch.setattr(validation, "render_with_api", fake_render_with_api)
    artifact = tmp_path / "prior.npz"
    np.savez_compressed(
        artifact,
        density_prior=np.ones((16, 24), dtype=np.float32),
        energy_prior=np.ones((16, 24), dtype=np.float32) * 0.5,
    )
    archive = tmp_path / "dataset.zip"
    _make_nested_dataset(archive)
    detail, summary = run_internal_validation(
        archive,
        tmp_path / "learned_prior_validation",
        methods=["learned-prior-balanced"],
        folds=1,
        max_per_fold=1,
        scatterers_count=80,
        include_maps=False,
        include_lpips=False,
        learned_prior_artifact=artifact,
    )
    detail_rows = list(csv.DictReader(detail.open()))
    assert detail_rows[0]["method"] == "learned-prior-balanced"
    assert detail_rows[0]["evidence_source"] == "hosted_api_true_scanner"
    assert "learned-prior" in summary.read_text()


def test_discover_scanner_pairs_reads_internal_validation_detail(tmp_path):
    phantom = tmp_path / "phantom.txt"
    rendered = tmp_path / "api_scan_gray.png"
    save_phantom(generate_two_layers(ExperimentConfig(scatterers_count=32), seed=8), phantom, ExperimentConfig(scatterers_count=32))
    skio.imsave(rendered, np.ones((8, 8), dtype=np.uint8) * 100)
    detail = tmp_path / "internal_validation_detail.csv"
    with detail.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=["method", "phantom_path", "synthetic_gray_png", "Struct_SSIM"])
        writer.writeheader()
        writer.writerow(
            {
                "method": "candidate",
                "phantom_path": str(phantom),
                "synthetic_gray_png": str(rendered),
                "Struct_SSIM": "0.42",
            }
        )

    pairs = learned_surrogate.discover_scanner_pairs(tmp_path)
    assert len(pairs) == 1
    assert pairs[0].label == "candidate"
    assert pairs[0].ssim == 0.42


def test_internal_validation_sample_offset_uses_later_records(tmp_path, monkeypatch):
    def fake_render_with_api(phantom_path, config_path, out_png, **kwargs):
        image = np.ones((256, 512), dtype=np.uint8) * 88
        skio.imsave(out_png, image)
        return "fake-request", out_png, 0.0, 0

    monkeypatch.setattr(validation, "render_with_api", fake_render_with_api)
    inner_buf = io.BytesIO()
    with zipfile.ZipFile(inner_buf, "w") as inner:
        for idx in range(2):
            png_buf = io.BytesIO()
            skio.imsave(png_buf, np.ones((8, 8), dtype=np.uint8) * (90 + idx), extension=".png")
            inner.writestr(f"DATASET_PNG/Female/1990-2000/Cheek/sample_frame{idx}.png", png_buf.getvalue())
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as outer:
        outer.writestr("DATASET.zip", inner_buf.getvalue())

    detail, _summary = run_internal_validation(
        archive,
        tmp_path / "offset_validation",
        methods=["official"],
        folds=1,
        max_per_fold=1,
        sample_offset=1,
        scatterers_count=48,
        include_maps=False,
        include_lpips=False,
    )
    rows = list(csv.DictReader(detail.open()))
    assert rows[0]["archive_path"].endswith("sample_frame1.png")
    assert rows[0]["evidence_scope"] == "grouped_validation_1fold_1perfold_offset1"


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


def test_audit_evidence_real_lpips_does_not_report_proxy_usage(tmp_path):
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

    audit = audit_challenge_evidence(metrics, require_real_lpips=True)
    assert audit["has_real_lpips"] is True
    assert audit["has_lpips_proxy"] is False
    assert audit["official_ranking_metric_complete"] is False
    assert "OAC_LPIPS" in audit["official_ranking_metric_missing"]
    assert audit["surrogate_scanner_is_true_scanner"] is False
    assert "hosted_api_true_scanner" in audit["true_scanner_sources"]
    assert "learned_surrogate_preview" in audit["non_challenge_scanner_sources"]


def test_audit_evidence_reports_complete_official_metric_set(tmp_path):
    metrics = tmp_path / "challenge_metrics_summary.csv"
    fieldnames = [
        "method",
        "evidence_source",
        "evidence_scope",
        "n",
        "MS-SSIM_mean",
        "LPIPS_metric",
        "LPIPS_or_proxy_mean",
        "evaluation_region",
        "generation_seconds_max",
        "Struct_MS-SSIM_median",
        "Struct_LPIPS_median",
        "OAC_MS-SSIM_median",
        "OAC_LPIPS_median",
        "SC_MS-SSIM_median",
        "SC_LPIPS_median",
        "RSC_MS-SSIM_median",
        "RSC_LPIPS_median",
    ]
    with metrics.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
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
                "Struct_MS-SSIM_median": "0.72",
                "Struct_LPIPS_median": "0.18",
                "OAC_MS-SSIM_median": "0.45",
                "OAC_LPIPS_median": "0.21",
                "SC_MS-SSIM_median": "0.50",
                "SC_LPIPS_median": "0.30",
                "RSC_MS-SSIM_median": "0.60",
                "RSC_LPIPS_median": "0.25",
            }
        )

    audit = audit_challenge_evidence(metrics, require_real_lpips=True)
    assert audit["official_ranking_metric_complete"] is True
    assert audit["official_ranking_metric_missing"] == []


def test_decide_promotion_rejects_lower_complete_official_score(tmp_path):
    metrics = tmp_path / "challenge_metrics_summary.csv"
    fieldnames = [
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
        "official_score",
        "Struct_MS-SSIM_median",
        "Struct_LPIPS_median",
        "OAC_MS-SSIM_median",
        "OAC_LPIPS_median",
        "SC_MS-SSIM_median",
        "SC_LPIPS_median",
        "RSC_MS-SSIM_median",
        "RSC_LPIPS_median",
    ]
    with metrics.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        for method, ms, lpips, official_score in [
            ("baseline", "0.70", "0.25", "0.72"),
            ("candidate", "0.74", "0.24", "0.68"),
        ]:
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
                    "MS-SSIM_wins": "2" if method == "candidate" else "0",
                    "LPIPS_wins": "2" if method == "candidate" else "0",
                    "generation_seconds_max": "10.0",
                    "official_score": official_score,
                    "Struct_MS-SSIM_median": ms,
                    "Struct_LPIPS_median": lpips,
                    "OAC_MS-SSIM_median": "0.45",
                    "OAC_LPIPS_median": "0.21",
                    "SC_MS-SSIM_median": "0.50",
                    "SC_LPIPS_median": "0.30",
                    "RSC_MS-SSIM_median": "0.60",
                    "RSC_LPIPS_median": "0.25",
                }
            )

    assert main(["decide-promotion", "--metrics", str(metrics), "--candidate", "candidate", "--baseline", "baseline", "--strict"]) == 2


def test_decide_promotion_prefers_higher_complete_official_score(tmp_path):
    metrics = tmp_path / "challenge_metrics_summary.csv"
    fieldnames = [
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
        "official_score",
        "Struct_MS-SSIM_median",
        "Struct_LPIPS_median",
        "OAC_MS-SSIM_median",
        "OAC_LPIPS_median",
        "SC_MS-SSIM_median",
        "SC_LPIPS_median",
        "RSC_MS-SSIM_median",
        "RSC_LPIPS_median",
    ]
    with metrics.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        for method, ms, lpips, official_score in [
            ("baseline", "0.6763", "0.5847", "0.6719"),
            ("candidate", "0.6771", "0.5860", "0.6770"),
        ]:
            writer.writerow(
                {
                    "method": method,
                    "evidence_source": "hosted_api_true_scanner",
                    "evidence_scope": "grouped_validation_5fold_1perfold",
                    "evaluation_region": "full_frame",
                    "n": "5",
                    "MS-SSIM_mean": ms,
                    "LPIPS_metric": "LPIPS",
                    "LPIPS_or_proxy_mean": lpips,
                    "MS-SSIM_wins": "5",
                    "LPIPS_wins": "5",
                    "generation_seconds_max": "12.0",
                    "official_score": official_score,
                    "Struct_MS-SSIM_median": ms,
                    "Struct_LPIPS_median": lpips,
                    "OAC_MS-SSIM_median": "0.81",
                    "OAC_LPIPS_median": "0.32",
                    "SC_MS-SSIM_median": "0.70",
                    "SC_LPIPS_median": "0.34",
                    "RSC_MS-SSIM_median": "0.75",
                    "RSC_LPIPS_median": "0.30",
                }
            )

    assert main(["decide-promotion", "--metrics", str(metrics), "--candidate", "candidate", "--baseline", "baseline", "--strict"]) == 0


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


def test_challenge_readiness_accepts_selected_candidate(tmp_path):
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
            ("candidate", "0.74", "0.24", "2", "2", "11.0"),
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

    assert main(["challenge-readiness", "--metrics", str(metrics), "--baseline", "baseline", "--method", "candidate", "--strict"]) == 0


def test_challenge_readiness_rejects_unselected_method(tmp_path):
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

    assert main(["challenge-readiness", "--metrics", str(metrics), "--baseline", "baseline", "--method", "candidate_a", "--strict"]) == 2


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


def test_cli_train_phantom_prior_and_generate_learned_prior(tmp_path):
    outputs = tmp_path / "outputs" / "run"
    outputs.mkdir(parents=True)
    phantom_source = outputs / "source_phantom.txt"
    rendered = outputs / "rendered_gray.png"
    save_phantom(generate_two_layers(ExperimentConfig(scatterers_count=96), seed=4), phantom_source, ExperimentConfig(scatterers_count=96))
    skio.imsave(rendered, np.tile(np.linspace(0, 255, 32, dtype=np.uint8), (32, 1)))
    metrics = outputs / "api_metrics.csv"
    with metrics.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=["method", "phantom_path", "synthetic_gray_png", "SSIM"])
        writer.writeheader()
        writer.writerow(
            {
                "method": "true_scanner_pair",
                "phantom_path": str(phantom_source),
                "synthetic_gray_png": str(rendered),
                "SSIM": "0.82",
            }
        )
    prior = tmp_path / "prior.npz"
    assert main(["train-phantom-prior", "--outputs-dir", str(tmp_path / "outputs"), "--out", str(prior), "--shape", "16", "24"]) == 0
    assert prior.exists()

    scan = tmp_path / "reference.npy"
    np.save(scan, np.tile(np.linspace(0.0, 1.0, 40, dtype=np.float32)[:, None], (1, 60)))
    out = tmp_path / "learned_prior.txt"
    assert (
        main(
            [
                "baseline",
                "learned-prior",
                "--input",
                str(scan),
                "--artifact",
                str(prior),
                "--out",
                str(out),
                "--scatterers-count",
                "72",
            ]
        )
        == 0
    )
    data = load_phantom(out)
    assert data.shape == (72, 4)
    assert np.isfinite(data).all()


def test_cli_train_and_apply_stage1_residual_prior(tmp_path):
    pytest.importorskip("torch")
    config = ExperimentConfig(scatterers_count=128)
    base = generate_two_layers(config, seed=11)
    flow_good = base.copy()
    flow_good[:, 2] = np.clip(flow_good[:, 2] + 4.0, 0.0, config.z_max)
    flow_bad = base.copy()
    flow_bad[:, 0] = np.clip(flow_bad[:, 0] + 14.0, -config.x_max / 2, config.x_max / 2)

    base_path = tmp_path / "base.txt"
    flow_good_path = tmp_path / "flow_good.txt"
    flow_bad_path = tmp_path / "flow_bad.txt"
    save_phantom(base, base_path, config)
    save_phantom(flow_good, flow_good_path, config)
    save_phantom(flow_bad, flow_bad_path, config)

    ref = tmp_path / "reference.png"
    image = np.tile(np.linspace(0, 255, 96, dtype=np.uint8), (48, 1))
    skio.imsave(ref, image)
    teacher = tmp_path / "teacher.csv"
    with teacher.open("w", newline="") as fobj:
        writer = csv.DictWriter(
            fobj,
            fieldnames=[
                "evidence_source",
                "source_archive_path",
                "reference_png",
                "current_phantom_path",
                "flow_phantom_path",
                "current_ms_ssim",
                "flow_ms_ssim",
                "flow_delta_vs_current",
                "flow_map_delta_vs_current",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "evidence_source": "hosted_api_true_scanner",
                "source_archive_path": "DATASET_PNG/Female/1990-2000/Cheek/good_frame50.png",
                "reference_png": str(ref),
                "current_phantom_path": str(base_path),
                "flow_phantom_path": str(flow_good_path),
                "current_ms_ssim": "0.62",
                "flow_ms_ssim": "0.63",
                "flow_delta_vs_current": "0.01",
                "flow_map_delta_vs_current": "0.004",
            }
        )
        writer.writerow(
            {
                "evidence_source": "hosted_api_true_scanner",
                "source_archive_path": "DATASET_PNG/Female/1990-2000/Cheek/bad_frame50.png",
                "reference_png": str(ref),
                "current_phantom_path": str(base_path),
                "flow_phantom_path": str(flow_bad_path),
                "current_ms_ssim": "0.62",
                "flow_ms_ssim": "0.59",
                "flow_delta_vs_current": "-0.03",
                "flow_map_delta_vs_current": "-0.010",
            }
        )

    artifact = tmp_path / "stage1_residual.pt"
    assert (
        main(
            [
                "train-stage1-residual-prior",
                "--teacher-metrics",
                str(teacher),
                "--out",
                str(artifact),
                "--shape",
                "16",
                "24",
                "--epochs",
                "1",
            ]
        )
        == 0
    )
    metadata = json.loads(artifact.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["example_count"] == 2
    assert metadata["positive_example_count"] == 1
    assert metadata["regression_anchor_count"] == 1
    assert metadata["promotion_allowed_without_true_scanner"] is False

    learned_artifact = tmp_path / "learned_prior.npz"
    prior = np.tile(np.linspace(0.0, 1.0, 24, dtype=np.float32), (16, 1))
    np.savez_compressed(learned_artifact, density_prior=prior + 0.01, energy_prior=np.flipud(prior) + 0.01)
    out = tmp_path / "stage1_phantom.txt"
    assert (
        main(
            [
                "stage1-residual-prior-phantom",
                "--input",
                str(ref),
                "--learned-artifact",
                str(learned_artifact),
                "--residual-artifact",
                str(artifact),
                "--out",
                str(out),
                "--scatterers-count",
                "96",
            ]
        )
        == 0
    )
    data = load_phantom(out)
    assert data.shape == (96, 4)
    assert np.isfinite(data).all()

    base_metrics = tmp_path / "base_api_metrics.csv"
    with base_metrics.open("w", newline="") as fobj:
        writer = csv.DictWriter(
            fobj,
            fieldnames=["source_archive_path", "reference_png", "phantom_path", "MS-SSIM"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_archive_path": "DATASET_PNG/Female/1990-2000/Cheek/good_frame50.png",
                "reference_png": str(ref),
                "phantom_path": str(base_path),
                "MS-SSIM": "0.90",
            }
        )
        writer.writerow(
            {
                "source_archive_path": "DATASET_PNG/Male/1950-1960/Eye_corner/heldout_frame450.png",
                "reference_png": str(ref),
                "phantom_path": str(base_path),
                "MS-SSIM": "0.82",
            }
        )
    heldout_queue = tmp_path / "stage1_heldout_queue.csv"
    assert (
        main(
            [
                "plan-stage1-residual-validation-queue",
                "--base-api-metrics",
                str(base_metrics),
                "--residual-artifact",
                str(artifact),
                "--out-queue",
                str(heldout_queue),
                "--exclude-metrics",
                str(teacher),
                "--max-sources",
                "1",
                "--residual-blends",
                "0.25",
                "1.0",
            ]
        )
        == 0
    )
    heldout_rows = list(csv.DictReader(heldout_queue.open()))
    assert len(heldout_rows) == 4
    assert {row["source_archive_path"] for row in heldout_rows} == {
        "DATASET_PNG/Male/1950-1960/Eye_corner/heldout_frame450.png"
    }
    assert {row["preserve_depth_profile"] for row in heldout_rows} == {"False", "True"}
    assert {row["preserve_local_profile"] for row in heldout_rows} == {"False"}
    assert all(Path(row["phantom_path"]).exists() for row in heldout_rows)
    heldout_summary = json.loads(heldout_queue.with_suffix(".json").read_text(encoding="utf-8"))
    assert heldout_summary["source_count"] == 1
    assert heldout_summary["queue_count"] == 4
    assert heldout_summary["hidden_holdout_final_score"] is False

    overlay_out = tmp_path / "stage2_overlay.txt"
    assert (
        main(
            [
                "stage2-texture-overlay-phantom",
                "--input",
                str(ref),
                "--base-phantom",
                str(base_path),
                "--out",
                str(overlay_out),
                "--replace-fraction",
                "0.05",
                "--texture-weight",
                "0.6",
            ]
        )
        == 0
    )
    overlay = load_phantom(overlay_out)
    assert overlay.shape == base.shape
    assert np.isclose(overlay[:, 3].sum(), base[:, 3].sum(), rtol=0.02)

    overlay_queue = tmp_path / "stage2_overlay_queue.csv"
    assert (
        main(
            [
                "plan-stage2-texture-overlay-validation-queue",
                "--base-api-metrics",
                str(base_metrics),
                "--out-queue",
                str(overlay_queue),
                "--max-sources",
                "1",
                "--replace-fractions",
                "0.01",
                "0.02",
                "--texture-weights",
                "0.3",
            ]
        )
        == 0
    )
    overlay_rows = list(csv.DictReader(overlay_queue.open()))
    assert len(overlay_rows) == 2
    assert all(Path(row["phantom_path"]).exists() for row in overlay_rows)
    overlay_summary = json.loads(overlay_queue.with_suffix(".json").read_text(encoding="utf-8"))
    assert overlay_summary["queue_count"] == 2
    assert overlay_summary["promotion_allowed_without_true_scanner"] is False


def test_cli_enrich_stage1_residual_smoke_maps(tmp_path):
    ref = tmp_path / "reference.png"
    current = tmp_path / "current.png"
    stage1 = tmp_path / "stage1.png"
    image = np.tile(np.linspace(20, 220, 64, dtype=np.uint8), (48, 1))
    skio.imsave(ref, image)
    skio.imsave(current, np.clip(image * 0.82, 0, 255).astype(np.uint8))
    skio.imsave(stage1, image)

    queue = tmp_path / "queue.csv"
    with queue.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=["method", "source_archive_path", "reference_png"])
        writer.writeheader()
        writer.writerow(
            {
                "method": "stage1_residual_preserve_b1p00_01",
                "source_archive_path": "DATASET_PNG/Female/1990-2000/Cheek/sample_frame50.png",
                "reference_png": str(ref),
            }
        )

    candidate_metrics = tmp_path / "candidate_queue_metrics.csv"
    with candidate_metrics.open("w", newline="") as fobj:
        writer = csv.DictWriter(
            fobj,
            fieldnames=["status", "method", "reference_png", "synthetic_gray_png", "MS-SSIM"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "status": "ok",
                "method": "stage1_residual_preserve_b1p00_01",
                "reference_png": str(ref),
                "synthetic_gray_png": str(stage1),
                "MS-SSIM": "1.0",
            }
        )

    base_metrics = tmp_path / "base_api_metrics.csv"
    with base_metrics.open("w", newline="") as fobj:
        writer = csv.DictWriter(
            fobj,
            fieldnames=["source_archive_path", "reference_png", "synthetic_gray_png", "MS-SSIM"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_archive_path": "DATASET_PNG/Female/1990-2000/Cheek/sample_frame50.png",
                "reference_png": str(ref),
                "synthetic_gray_png": str(current),
                "MS-SSIM": "0.72",
            }
        )

    out = tmp_path / "stage1_enriched.csv"
    assert (
        main(
            [
                "enrich-stage1-residual-smoke-maps",
                "--candidate-metrics",
                str(candidate_metrics),
                "--candidate-queue",
                str(queue),
                "--base-api-metrics",
                str(base_metrics),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 1
    assert rows[0]["guardrail_pass"] == "True"
    assert float(rows[0]["delta_Struct_MS-SSIM"]) > 0.0
    summary = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert summary["guardrail_pass_count"] == 1
    assert summary["hidden_holdout_final_score"] is False


def test_cli_select_stage1_residual_variants_uses_multi_objective_gate(tmp_path):
    headers = [
        "status",
        "method",
        "source_archive_path",
        "reference_png",
        "phantom_path",
        "synthetic_gray_png",
        "MS-SSIM",
        "current_ms_ssim",
        "stage1_ms_ssim",
        "current_map_enrichment_status",
        "stage1_map_enrichment_status",
        "delta_Struct_MS-SSIM",
        "delta_OAC_MS-SSIM",
        "delta_SC_MS-SSIM",
        "delta_RSC_MS-SSIM",
        "delta_Struct_LPIPS_PROXY",
    ]
    source_a = "DATASET_PNG/Female/1990-2000/Cheek/a_frame50.png"
    source_b = "DATASET_PNG/Male/1950-1960/Eye_corner/b_frame450.png"
    raw = tmp_path / "raw_enriched.csv"
    depth = tmp_path / "depth_enriched.csv"
    with raw.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=headers)
        writer.writeheader()
        writer.writerow(
            {
                "status": "ok",
                "method": "raw_a",
                "source_archive_path": source_a,
                "reference_png": "ref_a.png",
                "phantom_path": "raw_a.txt",
                "synthetic_gray_png": "raw_a.png",
                "MS-SSIM": "0.710",
                "current_ms_ssim": "0.704",
                "stage1_ms_ssim": "0.710",
                "current_map_enrichment_status": "ok",
                "stage1_map_enrichment_status": "ok",
                "delta_Struct_MS-SSIM": "0.0060",
                "delta_OAC_MS-SSIM": "-0.0018",
                "delta_SC_MS-SSIM": "0.0120",
                "delta_RSC_MS-SSIM": "0.0130",
                "delta_Struct_LPIPS_PROXY": "0.0020",
            }
        )
        writer.writerow(
            {
                "status": "ok",
                "method": "raw_b",
                "source_archive_path": source_b,
                "reference_png": "ref_b.png",
                "phantom_path": "raw_b.txt",
                "synthetic_gray_png": "raw_b.png",
                "MS-SSIM": "0.705",
                "current_ms_ssim": "0.701",
                "stage1_ms_ssim": "0.705",
                "current_map_enrichment_status": "ok",
                "stage1_map_enrichment_status": "ok",
                "delta_Struct_MS-SSIM": "0.0040",
                "delta_OAC_MS-SSIM": "-0.0002",
                "delta_SC_MS-SSIM": "0.0030",
                "delta_RSC_MS-SSIM": "0.0040",
                "delta_Struct_LPIPS_PROXY": "0.0005",
            }
        )
    with depth.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=headers)
        writer.writeheader()
        writer.writerow(
            {
                "status": "ok",
                "method": "depth_a",
                "source_archive_path": source_a,
                "reference_png": "ref_a.png",
                "phantom_path": "depth_a.txt",
                "synthetic_gray_png": "depth_a.png",
                "MS-SSIM": "0.708",
                "current_ms_ssim": "0.704",
                "stage1_ms_ssim": "0.708",
                "current_map_enrichment_status": "ok",
                "stage1_map_enrichment_status": "ok",
                "delta_Struct_MS-SSIM": "0.0040",
                "delta_OAC_MS-SSIM": "-0.0001",
                "delta_SC_MS-SSIM": "0.0060",
                "delta_RSC_MS-SSIM": "0.0050",
                "delta_Struct_LPIPS_PROXY": "0.0010",
            }
        )

    out_queue = tmp_path / "selected_stage1.csv"
    assert (
        main(
            [
                "select-stage1-residual-variants",
                "--enriched-metrics",
                str(raw),
                str(depth),
                "--out-queue",
                str(out_queue),
                "--min-oac-delta",
                "-0.001",
            ]
        )
        == 0
    )
    rows = list(csv.DictReader(out_queue.open()))
    assert [row["method"] for row in rows] == ["depth_a", "raw_b"]
    assert rows[0]["stage1_selection_status"] == "selected"
    assert rows[0]["promotion_allowed_without_true_scanner"] == "False"
    assert float(rows[0]["stage1_multi_objective_score"]) > float(rows[1]["stage1_multi_objective_score"])
    summary = json.loads(out_queue.with_suffix(".json").read_text(encoding="utf-8"))
    assert summary["candidate_count"] == 3
    assert summary["selected_count"] == 2
    assert summary["status_counts"]["blocked_oac_delta"] == 1
    assert summary["hidden_holdout_final_score"] is False


def test_cli_optimize_seeds_writes_ranked_metrics(tmp_path, monkeypatch):
    def fake_render_with_api(phantom_path, config_path, out_png, **kwargs):
        image = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (256, 1))
        skio.imsave(out_png, image)
        return "fake-request", out_png, 0.0, 0

    monkeypatch.setattr(seed_search, "render_with_api", fake_render_with_api)
    scan = tmp_path / "reference.png"
    scan_2 = tmp_path / "reference_2.png"
    image = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (256, 1))
    skio.imsave(scan, image)
    skio.imsave(scan_2, np.flipud(image))

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
                "--anchored-residual",
                "--density-residual-scale",
                "0.12",
                "--energy-residual-scale",
                "0.18",
                "--anchor-weight",
                "0.7",
                "--residual-kernel",
                "9",
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
    assert captured["anchored_residual"] is True
    assert captured["density_residual_scale"] == 0.12
    assert captured["energy_residual_scale"] == 0.18
    assert captured["anchor_weight"] == 0.7
    assert captured["residual_kernel"] == 9


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
    scan_2 = tmp_path / "reference_2.png"
    image = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (256, 1))
    skio.imsave(scan, image)
    skio.imsave(scan_2, np.flipud(image))
    phantom = tmp_path / "phantom.txt"
    phantom_2 = tmp_path / "phantom_2.txt"
    assert main(["baseline", "official", "--out", str(phantom), "--scatterers-count", "64"]) == 0
    assert main(["baseline", "official", "--out", str(phantom_2), "--scatterers-count", "64", "--seed", "9"]) == 0
    queue = tmp_path / "queue.csv"
    queue.write_text(
        f"priority,method,phantom_path,reference_png\n"
        f"1,test_candidate,{phantom},\n"
        f"2,test_candidate_2,{phantom_2},{scan_2}\n",
        encoding="utf-8",
    )

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
                "2",
                "--api-concurrency",
                "2",
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
    assert "test_candidate_2" in text
    assert "fake-request" in text
    rows = list(csv.DictReader(metrics.open()))
    assert len(rows) == 2
    assert rows[0]["evidence_source"] == "hosted_api_true_scanner"
    assert rows[0]["evidence_scope"] == "single_reference_candidate_queue"
    assert rows[0]["evaluation_region"] == "full_frame"
    assert rows[0]["reference_shape"] == "256x512"
    assert rows[0]["evaluated_shape"] == "256x512"
    assert {row["reference_png"] for row in rows} == {str(scan.resolve()), str(scan_2.resolve())}


def test_cli_flow_energy_rank_batch_passes_controls(tmp_path, monkeypatch):
    captured = {}

    def fake_run_flow_energy_rank_batch(base_api_metrics, out_dir, **kwargs):
        captured["base_api_metrics"] = base_api_metrics
        captured["out_dir"] = out_dir
        captured.update(kwargs)
        metrics_path = Path(out_dir) / "flow_energy_correct_metrics.csv"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("rank,flow_energy_ms_ssim\n1,0.7\n", encoding="utf-8")
        return metrics_path

    monkeypatch.setattr(cli_module, "run_flow_energy_rank_batch", fake_run_flow_energy_rank_batch)
    base_metrics = tmp_path / "api_metrics.csv"
    key_file = tmp_path / "api_key"
    base_metrics.write_text("MS-SSIM\n0.5\n", encoding="utf-8")
    key_file.write_text("secret", encoding="utf-8")

    assert (
        main(
            [
                "flow-energy-rank-batch",
                "--base-api-metrics",
                str(base_metrics),
                "--out",
                str(tmp_path / "batch"),
                "--rank-start",
                "13",
                "--rank-end",
                "36",
                "--api-key-file",
                str(key_file),
                "--api-concurrency",
                "2",
                "--poll-interval-seconds",
                "3.5",
                "--max-polls",
                "77",
                "--flow-strength",
                "0.38",
                "--flow-smooth-sigma",
                "1.4",
                "--flow-attachment",
                "5.5",
                "--energy-exponent",
                "0.9",
            ]
        )
        == 0
    )
    assert captured["base_api_metrics"] == str(base_metrics)
    assert captured["out_dir"] == str(tmp_path / "batch")
    assert captured["rank_start"] == 13
    assert captured["rank_end"] == 36
    assert captured["api_key_file"] == str(key_file)
    assert captured["api_concurrency"] == 2
    assert captured["poll_interval_seconds"] == 3.5
    assert captured["max_polls"] == 77
    assert captured["flow_strength"] == 0.38
    assert captured["flow_smooth_sigma"] == 1.4
    assert captured["flow_attachment"] == 5.5
    assert captured["energy_exponent"] == 0.9


def test_cli_adaptive_flow_strength_batch_passes_controls(tmp_path, monkeypatch):
    captured = {}

    def fake_run_adaptive_flow_strength_batch(current_api_metrics, out_dir, **kwargs):
        captured["current_api_metrics"] = current_api_metrics
        captured["out_dir"] = out_dir
        captured.update(kwargs)
        metrics_path = Path(out_dir) / "adaptive_strength_metrics.csv"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("row_index,best_ms_ssim\n1,0.7\n", encoding="utf-8")
        return metrics_path

    monkeypatch.setattr(cli_module, "run_adaptive_flow_strength_batch", fake_run_adaptive_flow_strength_batch)
    current_metrics = tmp_path / "api_metrics.csv"
    key_file = tmp_path / "api_key"
    current_metrics.write_text("MS-SSIM\n0.5\n", encoding="utf-8")
    key_file.write_text("secret", encoding="utf-8")

    assert (
        main(
            [
                "adaptive-flow-strength-batch",
                "--current-api-metrics",
                str(current_metrics),
                "--out",
                str(tmp_path / "adaptive"),
                "--rank-start",
                "13",
                "--rank-end",
                "36",
                "--strengths",
                "0.08",
                "0.24",
                "0.44",
                "--api-key-file",
                str(key_file),
                "--api-concurrency",
                "2",
                "--poll-interval-seconds",
                "3.5",
                "--max-polls",
                "77",
                "--flow-smooth-sigma",
                "1.4",
                "--flow-attachment",
                "5.5",
                "--energy-exponent",
                "0.9",
            ]
        )
        == 0
    )
    assert captured["current_api_metrics"] == str(current_metrics)
    assert captured["out_dir"] == str(tmp_path / "adaptive")
    assert captured["rank_start"] == 13
    assert captured["rank_end"] == 36
    assert captured["strengths"] == (0.08, 0.24, 0.44)
    assert captured["api_key_file"] == str(key_file)
    assert captured["api_concurrency"] == 2
    assert captured["poll_interval_seconds"] == 3.5
    assert captured["max_polls"] == 77
    assert captured["flow_smooth_sigma"] == 1.4
    assert captured["flow_attachment"] == 5.5
    assert captured["energy_exponent"] == 0.9


def test_cli_residual_selector_flow_batch_passes_controls(tmp_path, monkeypatch):
    captured = {}

    def fake_run_residual_selector_flow_batch(selector_queue, out_dir, **kwargs):
        captured["selector_queue"] = selector_queue
        captured["out_dir"] = out_dir
        captured.update(kwargs)
        metrics_path = Path(out_dir) / "residual_selector_probe_metrics.csv"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("priority,best_ms_ssim\n1,0.7\n", encoding="utf-8")
        return metrics_path

    monkeypatch.setattr(cli_module, "run_residual_selector_flow_batch", fake_run_residual_selector_flow_batch)
    selector_queue = tmp_path / "selector_queue.csv"
    key_file = tmp_path / "api_key"
    selector_queue.write_text("priority,selected_strength\n1,0.25\n", encoding="utf-8")
    key_file.write_text("secret", encoding="utf-8")

    assert (
        main(
            [
                "residual-selector-flow-batch",
                "--selector-queue",
                str(selector_queue),
                "--out",
                str(tmp_path / "selector_batch"),
                "--max-candidates",
                "12",
                "--max-energy-followups",
                "4",
                "--min-energy-flow-delta",
                "-0.006",
                "--api-key-file",
                str(key_file),
                "--api-concurrency",
                "3",
                "--poll-interval-seconds",
                "3.5",
                "--max-polls",
                "77",
                "--flow-smooth-sigma",
                "1.4",
                "--flow-attachment",
                "5.5",
                "--energy-exponent",
                "0.9",
            ]
        )
        == 0
    )
    assert captured["selector_queue"] == str(selector_queue)
    assert captured["out_dir"] == str(tmp_path / "selector_batch")
    assert captured["max_candidates"] == 12
    assert captured["max_energy_followups"] == 4
    assert captured["min_energy_flow_delta"] == -0.006
    assert captured["api_key_file"] == str(key_file)
    assert captured["api_concurrency"] == 3
    assert captured["poll_interval_seconds"] == 3.5
    assert captured["max_polls"] == 77
    assert captured["flow_smooth_sigma"] == 1.4
    assert captured["flow_attachment"] == 5.5
    assert captured["energy_exponent"] == 0.9


def test_cli_plan_residual_api_budget_expands_selector_queue(tmp_path):
    selector_queue = tmp_path / "selector_queue.csv"
    with selector_queue.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "priority",
                "row_index",
                "current_rank",
                "source_archive_path",
                "current_ms_ssim",
                "current_lpips",
                "selected_strength",
                "residual_control_policy",
                "flow_smooth_sigma",
                "flow_attachment",
                "energy_exponent",
                "energy_sigma",
                "texture_exponent",
                "expected_delta_mean",
                "expected_delta_lcb",
                "expected_delta_uncertainty",
                "acquisition_score",
                "nearest_teacher_count",
                "nearest_teacher_sources",
                "selector_train_examples",
                "reference_png",
                "current_phantom_path",
                "current_synthetic_gray_png",
                "surrogate_gate_status",
                "evidence_scope",
            ],
        )
        writer.writeheader()
        for priority, acquisition in [(1, "0.020"), (2, "0.015")]:
            writer.writerow(
                {
                    "priority": priority,
                    "row_index": priority - 1,
                    "current_rank": priority + 10,
                    "source_archive_path": f"DATASET_PNG/Female/1990-2000/Cheek/sample{priority}_frame50.png",
                    "current_ms_ssim": "0.62",
                    "current_lpips": "0.58",
                    "selected_strength": "0.25",
                    "residual_control_policy": "row_wise_physics_dl_v1",
                    "flow_smooth_sigma": "1.5",
                    "flow_attachment": "5.8",
                    "energy_exponent": "0.84",
                    "energy_sigma": "2.4",
                    "texture_exponent": "0.31",
                    "expected_delta_mean": "0.025",
                    "expected_delta_lcb": "0.010",
                    "expected_delta_uncertainty": "0.006",
                    "acquisition_score": acquisition,
                    "nearest_teacher_count": "12",
                    "nearest_teacher_sources": "teacher_a;teacher_b",
                    "selector_train_examples": "140",
                    "reference_png": str(tmp_path / f"ref{priority}.png"),
                    "current_phantom_path": str(tmp_path / f"base{priority}.txt"),
                    "current_synthetic_gray_png": str(tmp_path / f"gray{priority}.png"),
                    "surrogate_gate_status": "pass",
                    "evidence_scope": "selector_planning_not_challenge_evidence",
                }
            )

    out = tmp_path / "budget_queue.csv"
    assert (
        main(
            [
                "plan-residual-api-budget",
                "--selector-queue",
                str(selector_queue),
                "--out",
                str(out),
                "--total-api-calls",
                "6",
                "--energy-followup-fraction",
                "0.333",
                "--strength-multipliers",
                "1.0",
                "0.5",
                "1.5",
                "--energy-exponents",
                "0.8",
                "0.7",
                "0.9",
            ]
        )
        == 0
    )
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 4
    assert [row["priority"] for row in rows] == ["1", "2", "3", "4"]
    assert rows[0]["candidate_kind"] == "exploit"
    assert rows[1]["candidate_kind"] == "exploit"
    assert rows[2]["candidate_kind"] == "explore_low_strength"
    assert rows[3]["candidate_kind"] == "explore_high_strength"
    assert rows[0]["selected_strength"] == "0.25"
    assert rows[2]["selected_strength"] == "0.125"
    assert rows[0]["residual_control_policy"] == "row_wise_physics_dl_v1"
    assert rows[0]["flow_smooth_sigma"] == "1.5"
    assert rows[0]["energy_exponent"] == "0.84"
    assert rows[2]["energy_exponent"] == "0.735"
    assert rows[0]["texture_exponent"] == "0.31"
    assert rows[0]["reserved_energy_followups"] == "2"
    assert rows[0]["flow_budget"] == "4"
    assert rows[0]["evidence_scope"] == "selector_budget_planning_not_challenge_evidence"
    summary = json.loads(out.with_name("budget_queue_budget_summary.json").read_text(encoding="utf-8"))
    assert summary["total_api_budget"] == 6
    assert summary["planned_flow_calls"] == 4
    assert summary["reserved_energy_followups"] == 2
    assert summary["max_possible_api_calls"] == 6
    assert summary["planned_candidate_kinds"] == {
        "exploit": 2,
        "explore_high_strength": 1,
        "explore_low_strength": 1,
    }
    assert summary["promotion_allowed_without_true_scanner"] is False


def test_cli_plan_residual_api_budget_can_cap_diversity_strata(tmp_path):
    selector_queue = tmp_path / "selector_queue.csv"
    with selector_queue.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "priority",
                "row_index",
                "current_rank",
                "source_archive_path",
                "current_ms_ssim",
                "current_lpips",
                "selected_strength",
                "expected_delta_mean",
                "expected_delta_lcb",
                "expected_delta_uncertainty",
                "acquisition_score",
                "nearest_teacher_count",
                "nearest_teacher_sources",
                "selector_train_examples",
                "reference_png",
                "current_phantom_path",
                "current_synthetic_gray_png",
                "surrogate_gate_status",
                "evidence_scope",
            ],
        )
        writer.writeheader()
        rows = [
            (1, "Cheek", "0.040"),
            (2, "Cheek", "0.038"),
            (3, "Cheek", "0.036"),
            (4, "Eye_corner", "0.020"),
        ]
        for priority, site, acquisition in rows:
            writer.writerow(
                {
                    "priority": priority,
                    "row_index": priority - 1,
                    "current_rank": priority,
                    "source_archive_path": f"DATASET_PNG/Female/1990-2000/{site}/sample{priority}_frame50.png",
                    "current_ms_ssim": "0.62",
                    "current_lpips": "0.58",
                    "selected_strength": "0.25",
                    "expected_delta_mean": "0.025",
                    "expected_delta_lcb": "0.010",
                    "expected_delta_uncertainty": "0.006",
                    "acquisition_score": acquisition,
                    "nearest_teacher_count": "12",
                    "nearest_teacher_sources": "teacher_a;teacher_b",
                    "selector_train_examples": "140",
                    "reference_png": str(tmp_path / f"ref{priority}.png"),
                    "current_phantom_path": str(tmp_path / f"base{priority}.txt"),
                    "current_synthetic_gray_png": str(tmp_path / f"gray{priority}.png"),
                    "surrogate_gate_status": "pass",
                    "evidence_scope": "selector_planning_not_challenge_evidence",
                }
            )

    out = tmp_path / "diverse_budget_queue.csv"
    assert (
        main(
            [
                "plan-residual-api-budget",
                "--selector-queue",
                str(selector_queue),
                "--out",
                str(out),
                "--total-api-calls",
                "9",
                "--energy-followup-fraction",
                "0.333",
                "--strength-multipliers",
                "1.0",
                "0.5",
                "1.5",
                "--diversity-fields",
                "body_site",
                "--max-per-stratum",
                "3",
            ]
        )
        == 0
    )
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 6
    stratum_counts = {stratum: sum(row["diversity_stratum"] == stratum for row in rows) for stratum in {"Cheek", "Eye_corner"}}
    assert stratum_counts == {"Cheek": 3, "Eye_corner": 3}
    assert {row["candidate_kind"] for row in rows if row["diversity_stratum"] == "Eye_corner"} == {
        "exploit",
        "explore_low_strength",
        "explore_high_strength",
    }
    summary = json.loads(out.with_name("diverse_budget_queue_budget_summary.json").read_text(encoding="utf-8"))
    assert summary["diversity_fields"] == ["body_site"]
    assert summary["max_per_stratum"] == 3
    assert summary["planned_diversity_strata"] == {"Cheek": 3, "Eye_corner": 3}


def test_cli_plan_residual_api_budget_penalizes_risky_holdout_group(tmp_path):
    selector_queue = tmp_path / "selector_queue.csv"
    with selector_queue.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "priority",
                "row_index",
                "current_rank",
                "source_archive_path",
                "current_ms_ssim",
                "current_lpips",
                "selected_strength",
                "selector_holdout_group_key",
                "selector_holdout_group_status",
                "selector_holdout_score_multiplier",
                "expected_delta_mean",
                "expected_delta_lcb",
                "expected_delta_uncertainty",
                "acquisition_score",
                "nearest_teacher_count",
                "nearest_teacher_sources",
                "selector_train_examples",
                "reference_png",
                "current_phantom_path",
                "current_synthetic_gray_png",
                "surrogate_gate_status",
                "evidence_scope",
            ],
        )
        writer.writeheader()
        for priority, source, status, multiplier in [
            (1, "DATASET_PNG/Male/1990-2000/Cheek/risky_frame50.png", "holdout_group_no_wins", "0.5"),
            (2, "DATASET_PNG/Female/1950-1960/Eye_corner/supported_frame50.png", "holdout_group_supported", "1.0"),
        ]:
            writer.writerow(
                {
                    "priority": priority,
                    "row_index": priority - 1,
                    "current_rank": priority,
                    "source_archive_path": source,
                    "current_ms_ssim": "0.62",
                    "current_lpips": "0.58",
                    "selected_strength": "0.25",
                    "selector_holdout_group_key": "|".join(source.split("/")[1:4]),
                    "selector_holdout_group_status": status,
                    "selector_holdout_score_multiplier": multiplier,
                    "expected_delta_mean": "0.025",
                    "expected_delta_lcb": "0.010",
                    "expected_delta_uncertainty": "0.006",
                    "acquisition_score": "0.020",
                    "nearest_teacher_count": "12",
                    "nearest_teacher_sources": "teacher_a;teacher_b",
                    "selector_train_examples": "140",
                    "reference_png": str(tmp_path / f"ref{priority}.png"),
                    "current_phantom_path": str(tmp_path / f"base{priority}.txt"),
                    "current_synthetic_gray_png": str(tmp_path / f"gray{priority}.png"),
                    "surrogate_gate_status": "pass",
                    "evidence_scope": "selector_planning_not_challenge_evidence",
                }
            )

    out = tmp_path / "risk_budget_queue.csv"
    assert (
        main(
            [
                "plan-residual-api-budget",
                "--selector-queue",
                str(selector_queue),
                "--out",
                str(out),
                "--total-api-calls",
                "2",
                "--energy-followup-fraction",
                "0",
                "--strength-multipliers",
                "1.0",
            ]
        )
        == 0
    )
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["selector_holdout_group_status"] == "holdout_group_supported"
    assert rows[1]["selector_holdout_group_status"] == "holdout_group_no_wins"
    assert float(rows[0]["budget_score"]) > float(rows[1]["budget_score"])


def test_cli_plan_residual_api_budget_penalizes_surrogate_out_of_domain_row(tmp_path):
    selector_queue = tmp_path / "selector_queue.csv"
    with selector_queue.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "priority",
                "row_index",
                "current_rank",
                "source_archive_path",
                "current_ms_ssim",
                "current_lpips",
                "selected_strength",
                "surrogate_calibration_status",
                "surrogate_calibration_support_n",
                "surrogate_calibration_distance",
                "surrogate_calibration_score_multiplier",
                "expected_delta_mean",
                "expected_delta_lcb",
                "expected_delta_uncertainty",
                "acquisition_score",
                "nearest_teacher_count",
                "nearest_teacher_sources",
                "selector_train_examples",
                "reference_png",
                "current_phantom_path",
                "current_synthetic_gray_png",
                "surrogate_gate_status",
                "evidence_scope",
            ],
        )
        writer.writeheader()
        for priority, source, status, multiplier in [
            (1, "DATASET_PNG/Female/1990-2000/Cheek/out_domain_frame50.png", "calibration_out_of_domain", "0.65"),
            (2, "DATASET_PNG/Female/1990-2000/Cheek/supported_frame50.png", "calibration_supported", "1.0"),
        ]:
            writer.writerow(
                {
                    "priority": priority,
                    "row_index": priority - 1,
                    "current_rank": priority,
                    "source_archive_path": source,
                    "current_ms_ssim": "0.62",
                    "current_lpips": "0.58",
                    "selected_strength": "0.25",
                    "surrogate_calibration_status": status,
                    "surrogate_calibration_support_n": "4",
                    "surrogate_calibration_distance": "2.0" if priority == 1 else "0.0",
                    "surrogate_calibration_score_multiplier": multiplier,
                    "expected_delta_mean": "0.025",
                    "expected_delta_lcb": "0.010",
                    "expected_delta_uncertainty": "0.006",
                    "acquisition_score": "0.020",
                    "nearest_teacher_count": "12",
                    "nearest_teacher_sources": "teacher_a;teacher_b",
                    "selector_train_examples": "140",
                    "reference_png": str(tmp_path / f"ref{priority}.png"),
                    "current_phantom_path": str(tmp_path / f"base{priority}.txt"),
                    "current_synthetic_gray_png": str(tmp_path / f"gray{priority}.png"),
                    "surrogate_gate_status": "pass",
                    "evidence_scope": "selector_planning_not_challenge_evidence",
                }
            )

    out = tmp_path / "surrogate_risk_budget_queue.csv"
    assert (
        main(
            [
                "plan-residual-api-budget",
                "--selector-queue",
                str(selector_queue),
                "--out",
                str(out),
                "--total-api-calls",
                "2",
                "--energy-followup-fraction",
                "0",
                "--strength-multipliers",
                "1.0",
            ]
        )
        == 0
    )
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["surrogate_calibration_status"] == "calibration_supported"
    assert rows[1]["surrogate_calibration_status"] == "calibration_out_of_domain"
    assert float(rows[0]["budget_score"]) > float(rows[1]["budget_score"])
    summary = json.loads(out.with_name("surrogate_risk_budget_queue_budget_summary.json").read_text(encoding="utf-8"))
    assert summary["planned_surrogate_calibration_status"] == {
        "calibration_out_of_domain": 1,
        "calibration_supported": 1,
    }


def test_cli_plan_residual_api_budget_uses_true_probe_feedback(tmp_path):
    selector_queue = tmp_path / "selector_queue.csv"
    with selector_queue.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "priority",
                "row_index",
                "current_rank",
                "source_archive_path",
                "current_ms_ssim",
                "current_lpips",
                "selected_strength",
                "expected_delta_mean",
                "expected_delta_lcb",
                "expected_delta_uncertainty",
                "acquisition_score",
                "nearest_teacher_count",
                "nearest_teacher_sources",
                "selector_train_examples",
                "reference_png",
                "current_phantom_path",
                "current_synthetic_gray_png",
                "surrogate_gate_status",
                "evidence_scope",
            ],
        )
        writer.writeheader()
        for priority, source in [
            (1, "DATASET_PNG/Male/1950-1960/Cheek/flow_negative_frame50.png"),
            (2, "DATASET_PNG/Male/1950-1960/Eye_corner/energy_supported_frame50.png"),
        ]:
            writer.writerow(
                {
                    "priority": priority,
                    "row_index": priority - 1,
                    "current_rank": priority,
                    "source_archive_path": source,
                    "current_ms_ssim": "0.64",
                    "current_lpips": "0.58",
                    "selected_strength": "0.38",
                    "expected_delta_mean": "0.020",
                    "expected_delta_lcb": "0.010",
                    "expected_delta_uncertainty": "0.004",
                    "acquisition_score": "0.020",
                    "nearest_teacher_count": "12",
                    "nearest_teacher_sources": "teacher_a;teacher_b",
                    "selector_train_examples": "140",
                    "reference_png": str(tmp_path / f"ref{priority}.png"),
                    "current_phantom_path": str(tmp_path / f"base{priority}.txt"),
                    "current_synthetic_gray_png": str(tmp_path / f"gray{priority}.png"),
                    "surrogate_gate_status": "pass",
                    "evidence_scope": "selector_planning_not_challenge_evidence",
                }
            )

    feedback = tmp_path / "probe_feedback.csv"
    with feedback.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_archive_path",
                "current_ms_ssim",
                "flow_delta_vs_current",
                "flow_energy_delta_vs_current",
                "best_ms_ssim",
                "energy_status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_archive_path": "DATASET_PNG/Male/1950-1960/Cheek/probed_frame50.png",
                "current_ms_ssim": "0.64",
                "flow_delta_vs_current": "-0.006",
                "flow_energy_delta_vs_current": "",
                "best_ms_ssim": "0.64",
                "energy_status": "deferred_budget",
            }
        )
        writer.writerow(
            {
                "source_archive_path": "DATASET_PNG/Male/1950-1960/Eye_corner/probed_frame50.png",
                "current_ms_ssim": "0.64",
                "flow_delta_vs_current": "-0.001",
                "flow_energy_delta_vs_current": "0.005",
                "best_ms_ssim": "0.645",
                "energy_status": "ok",
            }
        )

    out = tmp_path / "feedback_budget_queue.csv"
    assert (
        main(
            [
                "plan-residual-api-budget",
                "--selector-queue",
                str(selector_queue),
                "--out",
                str(out),
                "--total-api-calls",
                "2",
                "--energy-followup-fraction",
                "0",
                "--strength-multipliers",
                "1.0",
                "--probe-feedback-metrics",
                str(feedback),
            ]
        )
        == 0
    )
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["true_probe_feedback_status"] == "energy_single_probe_positive"
    assert rows[1]["true_probe_feedback_status"] == "flow_regression_risky"
    assert rows[1]["true_probe_feedback_flow_regression_rate"] == "1.0"
    assert rows[0]["true_probe_feedback_group_key"] == "Male|1950-1960|Eye_corner"
    assert float(rows[0]["budget_score"]) > float(rows[1]["budget_score"])
    summary = json.loads(out.with_name("feedback_budget_queue_budget_summary.json").read_text(encoding="utf-8"))
    assert summary["planned_true_probe_feedback_status"] == {
        "energy_single_probe_positive": 1,
        "flow_regression_risky": 1,
    }
    assert summary["probe_feedback_metrics"] == [str(feedback)]


def test_cli_plan_residual_api_budget_penalizes_exact_source_regression(tmp_path):
    selector_queue = tmp_path / "selector_queue.csv"
    with selector_queue.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "priority",
                "row_index",
                "current_rank",
                "source_archive_path",
                "current_ms_ssim",
                "current_lpips",
                "selected_strength",
                "expected_delta_mean",
                "expected_delta_lcb",
                "expected_delta_uncertainty",
                "acquisition_score",
                "nearest_teacher_count",
                "nearest_teacher_sources",
                "selector_train_examples",
                "reference_png",
                "current_phantom_path",
                "current_synthetic_gray_png",
                "surrogate_gate_status",
                "evidence_scope",
            ],
        )
        writer.writeheader()
        for priority, source in [
            (1, "DATASET_PNG/Male/1950-1960/Cheek/retry_bad_frame50.png"),
            (2, "DATASET_PNG/Male/1950-1960/Cheek/fresh_same_group_frame50.png"),
        ]:
            writer.writerow(
                {
                    "priority": priority,
                    "row_index": priority - 1,
                    "current_rank": priority,
                    "source_archive_path": source,
                    "current_ms_ssim": "0.64",
                    "current_lpips": "0.58",
                    "selected_strength": "0.38",
                    "expected_delta_mean": "0.020",
                    "expected_delta_lcb": "0.010",
                    "expected_delta_uncertainty": "0.004",
                    "acquisition_score": "0.020",
                    "nearest_teacher_count": "12",
                    "nearest_teacher_sources": "teacher_a;teacher_b",
                    "selector_train_examples": "140",
                    "reference_png": str(tmp_path / f"ref{priority}.png"),
                    "current_phantom_path": str(tmp_path / f"base{priority}.txt"),
                    "current_synthetic_gray_png": str(tmp_path / f"gray{priority}.png"),
                    "surrogate_gate_status": "pass",
                    "evidence_scope": "selector_planning_not_challenge_evidence",
                }
            )

    feedback = tmp_path / "probe_feedback.csv"
    with feedback.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_archive_path",
                "current_ms_ssim",
                "flow_delta_vs_current",
                "flow_energy_delta_vs_current",
                "best_ms_ssim",
                "energy_status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_archive_path": "DATASET_PNG/Male/1950-1960/Cheek/retry_bad_frame50.png",
                "current_ms_ssim": "0.64",
                "flow_delta_vs_current": "-0.019",
                "flow_energy_delta_vs_current": "",
                "best_ms_ssim": "0.64",
                "energy_status": "deferred_flow_regression",
            }
        )
        writer.writerow(
            {
                "source_archive_path": "DATASET_PNG/Male/1950-1960/Cheek/nearby_supported_frame50.png",
                "current_ms_ssim": "0.64",
                "flow_delta_vs_current": "0.002",
                "flow_energy_delta_vs_current": "0.006",
                "best_ms_ssim": "0.646",
                "energy_status": "ok",
            }
        )

    out = tmp_path / "exact_feedback_budget_queue.csv"
    assert (
        main(
            [
                "plan-residual-api-budget",
                "--selector-queue",
                str(selector_queue),
                "--out",
                str(out),
                "--total-api-calls",
                "2",
                "--energy-followup-fraction",
                "0",
                "--strength-multipliers",
                "1.0",
                "--probe-feedback-metrics",
                str(feedback),
            ]
        )
        == 0
    )
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["source_archive_path"].endswith("fresh_same_group_frame50.png")
    assert rows[0]["true_probe_feedback_status"] == "flow_regression_risky"
    assert rows[0]["true_probe_feedback_flow_regression_rate"] == "0.5"
    assert rows[0]["exact_probe_feedback_status"] == "not_probed"
    assert rows[1]["source_archive_path"].endswith("retry_bad_frame50.png")
    assert rows[1]["true_probe_feedback_status"] == "flow_regression_risky"
    assert rows[1]["exact_probe_feedback_status"] == "exact_flow_regression"
    assert float(rows[1]["exact_probe_feedback_score_multiplier"]) == 0.45
    assert float(rows[0]["budget_score"]) > float(rows[1]["budget_score"])
    summary = json.loads(out.with_name("exact_feedback_budget_queue_budget_summary.json").read_text(encoding="utf-8"))
    assert summary["planned_exact_probe_feedback_status"] == {
        "exact_flow_regression": 1,
        "not_probed": 1,
    }


def test_cli_plan_residual_api_budget_can_exclude_exact_probed_sources(tmp_path):
    selector_queue = tmp_path / "selector_queue.csv"
    with selector_queue.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "priority",
                "row_index",
                "current_rank",
                "source_archive_path",
                "current_ms_ssim",
                "current_lpips",
                "selected_strength",
                "expected_delta_mean",
                "expected_delta_lcb",
                "expected_delta_uncertainty",
                "acquisition_score",
                "nearest_teacher_count",
                "nearest_teacher_sources",
                "selector_train_examples",
                "reference_png",
                "current_phantom_path",
                "current_synthetic_gray_png",
                "surrogate_gate_status",
                "evidence_scope",
            ],
        )
        writer.writeheader()
        for priority, source in [
            (1, "DATASET_PNG/Male/1950-1960/Cheek/probed_positive_frame50.png"),
            (2, "DATASET_PNG/Male/1950-1960/Cheek/probed_negative_frame50.png"),
            (3, "DATASET_PNG/Male/1950-1960/Cheek/fresh_frame50.png"),
        ]:
            writer.writerow(
                {
                    "priority": priority,
                    "row_index": priority - 1,
                    "current_rank": priority,
                    "source_archive_path": source,
                    "current_ms_ssim": "0.64",
                    "current_lpips": "0.58",
                    "selected_strength": "0.38",
                    "expected_delta_mean": "0.020",
                    "expected_delta_lcb": "0.010",
                    "expected_delta_uncertainty": "0.004",
                    "acquisition_score": "0.020",
                    "nearest_teacher_count": "12",
                    "nearest_teacher_sources": "teacher_a;teacher_b",
                    "selector_train_examples": "140",
                    "reference_png": str(tmp_path / f"ref{priority}.png"),
                    "current_phantom_path": str(tmp_path / f"base{priority}.txt"),
                    "current_synthetic_gray_png": str(tmp_path / f"gray{priority}.png"),
                    "surrogate_gate_status": "pass",
                    "evidence_scope": "selector_planning_not_challenge_evidence",
                }
            )

    feedback = tmp_path / "probe_feedback.csv"
    with feedback.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_archive_path",
                "current_ms_ssim",
                "flow_delta_vs_current",
                "flow_energy_delta_vs_current",
                "best_ms_ssim",
                "energy_status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_archive_path": "DATASET_PNG/Male/1950-1960/Cheek/probed_positive_frame50.png",
                "current_ms_ssim": "0.64",
                "flow_delta_vs_current": "0.003",
                "flow_energy_delta_vs_current": "0.008",
                "best_ms_ssim": "0.648",
                "energy_status": "ok",
            }
        )
        writer.writerow(
            {
                "source_archive_path": "DATASET_PNG/Male/1950-1960/Cheek/probed_negative_frame50.png",
                "current_ms_ssim": "0.64",
                "flow_delta_vs_current": "-0.012",
                "flow_energy_delta_vs_current": "",
                "best_ms_ssim": "0.64",
                "energy_status": "deferred_flow_regression",
            }
        )

    out = tmp_path / "fresh_budget_queue.csv"
    assert (
        main(
            [
                "plan-residual-api-budget",
                "--selector-queue",
                str(selector_queue),
                "--out",
                str(out),
                "--total-api-calls",
                "3",
                "--energy-followup-fraction",
                "0",
                "--strength-multipliers",
                "1.0",
                "--probe-feedback-metrics",
                str(feedback),
                "--exclude-exact-probed",
            ]
        )
        == 0
    )
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 1
    assert rows[0]["source_archive_path"].endswith("fresh_frame50.png")
    assert rows[0]["exact_probe_feedback_status"] == "not_probed"
    summary = json.loads(out.with_name("fresh_budget_queue_budget_summary.json").read_text(encoding="utf-8"))
    assert summary["exclude_exact_probed"] is True
    assert summary["candidate_pool_size"] == 1


def test_cli_plan_residual_api_budget_applies_energy_gate_summary(tmp_path):
    selector_queue = tmp_path / "selector_queue.csv"
    with selector_queue.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "priority",
                "row_index",
                "current_rank",
                "source_archive_path",
                "current_ms_ssim",
                "current_lpips",
                "selected_strength",
                "expected_delta_mean",
                "expected_delta_lcb",
                "expected_delta_uncertainty",
                "acquisition_score",
                "expected_flow_delta_mean",
                "nearest_teacher_count",
                "nearest_teacher_sources",
                "selector_train_examples",
                "reference_png",
                "current_phantom_path",
                "current_synthetic_gray_png",
                "surrogate_gate_status",
                "evidence_scope",
            ],
        )
        writer.writeheader()
        for priority, acquisition, flow_delta in [
            (1, "0.020", "-0.010"),
            (2, "0.010", "0.002"),
        ]:
            writer.writerow(
                {
                    "priority": priority,
                    "row_index": priority - 1,
                    "current_rank": priority,
                    "source_archive_path": f"DATASET_PNG/Female/1990-2000/Cheek/sample{priority}_frame50.png",
                    "current_ms_ssim": "0.64",
                    "current_lpips": "0.58",
                    "selected_strength": "0.38",
                    "expected_delta_mean": "0.020",
                    "expected_delta_lcb": "0.010",
                    "expected_delta_uncertainty": "0.004",
                    "acquisition_score": acquisition,
                    "expected_flow_delta_mean": flow_delta,
                    "nearest_teacher_count": "12",
                    "nearest_teacher_sources": "teacher_a;teacher_b",
                    "selector_train_examples": "140",
                    "reference_png": str(tmp_path / f"ref{priority}.png"),
                    "current_phantom_path": str(tmp_path / f"base{priority}.txt"),
                    "current_synthetic_gray_png": str(tmp_path / f"gray{priority}.png"),
                    "surrogate_gate_status": "pass",
                    "evidence_scope": "selector_planning_not_challenge_evidence",
                }
            )

    gate = tmp_path / "energy_gate.json"
    gate.write_text(
        json.dumps(
            {
                "recommended_min_energy_flow_delta": -0.004,
                "selected": {
                    "retained_best_delta_fraction": 0.996,
                    "retained_map_delta_fraction": 0.975,
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "gated_budget_queue.csv"
    assert (
        main(
            [
                "plan-residual-api-budget",
                "--selector-queue",
                str(selector_queue),
                "--out",
                str(out),
                "--total-api-calls",
                "2",
                "--energy-followup-fraction",
                "0",
                "--strength-multipliers",
                "1.0",
                "--energy-gate-summary",
                str(gate),
            ]
        )
        == 0
    )
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["source_archive_path"].endswith("sample2_frame50.png")
    assert rows[0]["energy_gate_expected_status"] == "energy_gate_pass"
    assert rows[0]["energy_gate_min_flow_delta"] == "-0.004"
    assert rows[0]["energy_gate_retained_map_delta_fraction"] == "0.975"
    assert rows[1]["energy_gate_expected_status"] == "energy_gate_skip"
    assert rows[1]["energy_gate_score_multiplier"] == "0.2"
    assert float(rows[0]["budget_score"]) > float(rows[1]["budget_score"])
    summary = json.loads(out.with_name("gated_budget_queue_budget_summary.json").read_text(encoding="utf-8"))
    assert summary["energy_gate_summary"] == str(gate)
    assert summary["energy_gate_min_flow_delta"] == -0.004
    assert summary["energy_gate_retained_map_delta_fraction"] == 0.975
    assert summary["candidate_energy_gate_expected_status"] == {
        "energy_gate_pass": 1,
        "energy_gate_skip": 1,
    }
    assert summary["planned_energy_gate_expected_status"] == {
        "energy_gate_pass": 1,
        "energy_gate_skip": 1,
    }
    assert summary["planned_energy_gate_eligible_followups"] == 0


def test_residual_selector_flow_batch_skips_energy_after_failed_flow(tmp_path, monkeypatch):
    selector_queue = tmp_path / "selector_queue.csv"
    with selector_queue.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "priority",
                "row_index",
                "current_rank",
                "source_archive_path",
                "selected_strength",
                "current_ms_ssim",
                "current_lpips",
                "expected_delta_mean",
                "expected_delta_lcb",
                "expected_delta_uncertainty",
                "acquisition_score",
                "nearest_teacher_count",
                "nearest_teacher_sources",
                "reference_png",
                "current_phantom_path",
                "current_synthetic_gray_png",
                "surrogate_gate_status",
                "evidence_scope",
            ],
        )
        writer.writeheader()
        for priority, ms in [(1, "0.62"), (2, "0.61")]:
            writer.writerow(
                {
                    "priority": priority,
                    "row_index": priority - 1,
                    "current_rank": priority,
                    "source_archive_path": f"DATASET_PNG/Female/1990-2000/Cheek/sample{priority}_frame50.png",
                    "selected_strength": "0.25",
                    "current_ms_ssim": ms,
                    "current_lpips": "0.58",
                    "expected_delta_mean": "0.02",
                    "expected_delta_lcb": "0.01",
                    "expected_delta_uncertainty": "0.004",
                    "acquisition_score": "0.011",
                    "nearest_teacher_count": "2",
                    "nearest_teacher_sources": "teacher_a;teacher_b",
                    "reference_png": str(tmp_path / f"ref{priority}.png"),
                    "current_phantom_path": str(tmp_path / f"base{priority}.txt"),
                    "current_synthetic_gray_png": str(tmp_path / f"gray{priority}.png"),
                    "surrogate_gate_status": "pass",
                    "evidence_scope": "selector_planning_not_challenge_evidence",
                }
            )

    flow_outputs = []
    energy_outputs = []

    def fake_write_flow_transport_phantom(_ref, _phantom, _gray, output_path, **_kwargs):
        flow_outputs.append(Path(output_path))
        Path(output_path).write_text("flow", encoding="utf-8")
        return Path(output_path)

    def fake_write_energy_ratio_phantom(_ref, _phantom, _gray, output_path, **_kwargs):
        energy_outputs.append(Path(output_path))
        Path(output_path).write_text("energy", encoding="utf-8")
        return Path(output_path)

    def fake_render_candidate_queue(queue_csv, _reference_path, out_dir, **_kwargs):
        queue_rows = list(csv.DictReader(Path(queue_csv).open()))
        metrics_path = Path(out_dir) / "candidate_queue_metrics.csv"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "status",
            "priority",
            "method",
            "request_id",
            "reference_png",
            "phantom_path",
            "synthetic_gray_png",
            "MS-SSIM",
            "LPIPS_PROXY",
            "error",
        ]
        with metrics_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            if "selector_flow_queue" in str(queue_csv):
                assert len(queue_rows) == 2
                for row in queue_rows:
                    if row["priority"] == "1":
                        writer.writerow(
                            {
                                "status": "ok",
                                "priority": row["priority"],
                                "method": row["method"],
                                "request_id": "flow_1",
                                "reference_png": row["reference_png"],
                                "phantom_path": row["phantom_path"],
                                "synthetic_gray_png": str(tmp_path / "flow_1_gray.png"),
                                "MS-SSIM": "0.64",
                                "LPIPS_PROXY": "0.2",
                                "error": "",
                            }
                        )
                    else:
                        writer.writerow(
                            {
                                "status": "failed",
                                "priority": row["priority"],
                                "method": row["method"],
                                "request_id": "failed",
                                "reference_png": row["reference_png"],
                                "phantom_path": row["phantom_path"],
                                "synthetic_gray_png": str(tmp_path / "flow_2_gray.png"),
                                "MS-SSIM": "nan",
                                "LPIPS_PROXY": "nan",
                                "error": "connect timeout",
                            }
                        )
            else:
                assert len(queue_rows) == 1
                row = queue_rows[0]
                assert row["priority"] == "1"
                writer.writerow(
                    {
                        "status": "ok",
                        "priority": row["priority"],
                        "method": row["method"],
                        "request_id": "energy_1",
                        "reference_png": row["reference_png"],
                        "phantom_path": row["phantom_path"],
                        "synthetic_gray_png": str(tmp_path / "energy_1_gray.png"),
                        "MS-SSIM": "0.66",
                        "LPIPS_PROXY": "0.18",
                        "error": "",
                    }
                )
        return metrics_path

    monkeypatch.setattr(adaptive_flow_batch, "write_flow_transport_phantom", fake_write_flow_transport_phantom)
    monkeypatch.setattr(adaptive_flow_batch, "write_energy_ratio_phantom", fake_write_energy_ratio_phantom)
    monkeypatch.setattr(adaptive_flow_batch, "render_candidate_queue", fake_render_candidate_queue)

    metrics_path = adaptive_flow_batch.run_residual_selector_flow_batch(
        selector_queue,
        tmp_path / "selector_batch",
        api_concurrency=3,
    )
    rows = list(csv.DictReader(metrics_path.open()))
    assert len(rows) == 2
    assert rows[0]["priority"] == "1"
    assert rows[0]["flow_status"] == "ok"
    assert rows[0]["energy_status"] == "ok"
    assert rows[0]["best_stage"] == "flow_energy"
    assert rows[0]["flow_energy_ms_ssim"] == "0.66"
    assert rows[0]["evidence_scope"] == "selector_residual_probe_not_challenge_evidence"
    assert rows[1]["priority"] == "2"
    assert rows[1]["flow_status"] == "failed"
    assert rows[1]["energy_status"] == "not_run"
    assert rows[1]["best_stage"] == "current"
    assert rows[1]["flow_error"] == "connect timeout"
    assert len(flow_outputs) == 2
    assert len(energy_outputs) == 1
    energy_queue_rows = list(csv.DictReader((tmp_path / "selector_batch" / "selector_energy_queue_with_refs.csv").open()))
    assert len(energy_queue_rows) == 1
    assert energy_queue_rows[0]["priority"] == "1"


def test_residual_selector_flow_batch_limits_energy_followups(tmp_path, monkeypatch):
    selector_queue = tmp_path / "budget_queue.csv"
    with selector_queue.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "priority",
                "row_index",
                "current_rank",
                "source_archive_path",
                "selected_strength",
                "candidate_kind",
                "energy_exponent",
                "current_ms_ssim",
                "current_lpips",
                "acquisition_score",
                "reference_png",
                "current_phantom_path",
                "current_synthetic_gray_png",
                "surrogate_gate_status",
                "evidence_scope",
            ],
        )
        writer.writeheader()
        for priority in [1, 2, 3]:
            writer.writerow(
                {
                    "priority": priority,
                    "row_index": priority - 1,
                    "current_rank": priority,
                    "source_archive_path": f"DATASET_PNG/Female/1990-2000/Cheek/sample{priority}_frame50.png",
                    "selected_strength": "0.25",
                    "candidate_kind": "exploit",
                    "energy_exponent": "0.9",
                    "current_ms_ssim": "0.62",
                    "current_lpips": "0.58",
                    "acquisition_score": "0.01",
                    "reference_png": str(tmp_path / f"ref{priority}.png"),
                    "current_phantom_path": str(tmp_path / f"base{priority}.txt"),
                    "current_synthetic_gray_png": str(tmp_path / f"gray{priority}.png"),
                    "surrogate_gate_status": "pass",
                    "evidence_scope": "selector_budget_planning_not_challenge_evidence",
                }
            )

    def fake_write_flow_transport_phantom(_ref, _phantom, _gray, output_path, **_kwargs):
        Path(output_path).write_text("flow", encoding="utf-8")
        return Path(output_path)

    def fake_write_energy_ratio_phantom(_ref, _phantom, _gray, output_path, **_kwargs):
        Path(output_path).write_text("energy", encoding="utf-8")
        return Path(output_path)

    def fake_render_candidate_queue(queue_csv, _reference_path, out_dir, **_kwargs):
        queue_rows = list(csv.DictReader(Path(queue_csv).open()))
        metrics_path = Path(out_dir) / "candidate_queue_metrics.csv"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "status",
            "priority",
            "method",
            "request_id",
            "reference_png",
            "phantom_path",
            "synthetic_gray_png",
            "MS-SSIM",
            "LPIPS_PROXY",
            "error",
        ]
        flow_scores = {"1": "0.63", "2": "0.70", "3": "0.65"}
        with metrics_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            if "selector_flow_queue" in str(queue_csv):
                for row in queue_rows:
                    writer.writerow(
                        {
                            "status": "ok",
                            "priority": row["priority"],
                            "method": row["method"],
                            "request_id": f"flow_{row['priority']}",
                            "reference_png": row["reference_png"],
                            "phantom_path": row["phantom_path"],
                            "synthetic_gray_png": str(tmp_path / f"flow_{row['priority']}_gray.png"),
                            "MS-SSIM": flow_scores[row["priority"]],
                            "LPIPS_PROXY": "0.2",
                            "error": "",
                        }
                    )
            else:
                assert [row["priority"] for row in queue_rows] == ["2"]
                row = queue_rows[0]
                writer.writerow(
                    {
                        "status": "ok",
                        "priority": row["priority"],
                        "method": row["method"],
                        "request_id": "energy_2",
                        "reference_png": row["reference_png"],
                        "phantom_path": row["phantom_path"],
                        "synthetic_gray_png": str(tmp_path / "energy_2_gray.png"),
                        "MS-SSIM": "0.72",
                        "LPIPS_PROXY": "0.18",
                        "error": "",
                    }
                )
        return metrics_path

    monkeypatch.setattr(adaptive_flow_batch, "write_flow_transport_phantom", fake_write_flow_transport_phantom)
    monkeypatch.setattr(adaptive_flow_batch, "write_energy_ratio_phantom", fake_write_energy_ratio_phantom)
    monkeypatch.setattr(adaptive_flow_batch, "render_candidate_queue", fake_render_candidate_queue)

    metrics_path = adaptive_flow_batch.run_residual_selector_flow_batch(
        selector_queue,
        tmp_path / "selector_batch",
        max_energy_followups=1,
    )
    rows = {row["priority"]: row for row in csv.DictReader(metrics_path.open())}
    assert rows["2"]["energy_status"] == "ok"
    assert rows["2"]["best_stage"] == "flow_energy"
    assert rows["1"]["energy_status"] == "deferred_budget"
    assert rows["3"]["energy_status"] == "deferred_budget"
    energy_queue_rows = list(csv.DictReader((tmp_path / "selector_batch" / "selector_energy_queue_with_refs.csv").open()))
    assert [row["priority"] for row in energy_queue_rows] == ["2"]


def test_residual_selector_flow_batch_gates_energy_after_large_flow_regression(tmp_path, monkeypatch):
    selector_queue = tmp_path / "budget_queue.csv"
    with selector_queue.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "priority",
                "row_index",
                "current_rank",
                "source_archive_path",
                "selected_strength",
                "current_ms_ssim",
                "current_lpips",
                "acquisition_score",
                "reference_png",
                "current_phantom_path",
                "current_synthetic_gray_png",
                "surrogate_gate_status",
                "evidence_scope",
            ],
        )
        writer.writeheader()
        for priority in [1, 2]:
            writer.writerow(
                {
                    "priority": priority,
                    "row_index": priority - 1,
                    "current_rank": priority,
                    "source_archive_path": f"DATASET_PNG/Female/1990-2000/Cheek/sample{priority}_frame50.png",
                    "selected_strength": "0.25",
                    "current_ms_ssim": "0.62",
                    "current_lpips": "0.58",
                    "acquisition_score": "0.01",
                    "reference_png": str(tmp_path / f"ref{priority}.png"),
                    "current_phantom_path": str(tmp_path / f"base{priority}.txt"),
                    "current_synthetic_gray_png": str(tmp_path / f"gray{priority}.png"),
                    "surrogate_gate_status": "pass",
                    "evidence_scope": "selector_budget_planning_not_challenge_evidence",
                }
            )

    def fake_write_flow_transport_phantom(_ref, _phantom, _gray, output_path, **_kwargs):
        Path(output_path).write_text("flow", encoding="utf-8")
        return Path(output_path)

    def fake_write_energy_ratio_phantom(_ref, _phantom, _gray, output_path, **_kwargs):
        Path(output_path).write_text("energy", encoding="utf-8")
        return Path(output_path)

    def fake_render_candidate_queue(queue_csv, _reference_path, out_dir, **_kwargs):
        queue_rows = list(csv.DictReader(Path(queue_csv).open()))
        metrics_path = Path(out_dir) / "candidate_queue_metrics.csv"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "status",
            "priority",
            "method",
            "request_id",
            "reference_png",
            "phantom_path",
            "synthetic_gray_png",
            "MS-SSIM",
            "LPIPS_PROXY",
            "error",
        ]
        flow_scores = {"1": "0.60", "2": "0.619"}
        with metrics_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            if "selector_flow_queue" in str(queue_csv):
                for row in queue_rows:
                    writer.writerow(
                        {
                            "status": "ok",
                            "priority": row["priority"],
                            "method": row["method"],
                            "request_id": f"flow_{row['priority']}",
                            "reference_png": row["reference_png"],
                            "phantom_path": row["phantom_path"],
                            "synthetic_gray_png": str(tmp_path / f"flow_{row['priority']}_gray.png"),
                            "MS-SSIM": flow_scores[row["priority"]],
                            "LPIPS_PROXY": "0.2",
                            "error": "",
                        }
                    )
            else:
                assert [row["priority"] for row in queue_rows] == ["2"]
                row = queue_rows[0]
                writer.writerow(
                    {
                        "status": "ok",
                        "priority": row["priority"],
                        "method": row["method"],
                        "request_id": "energy_2",
                        "reference_png": row["reference_png"],
                        "phantom_path": row["phantom_path"],
                        "synthetic_gray_png": str(tmp_path / "energy_2_gray.png"),
                        "MS-SSIM": "0.625",
                        "LPIPS_PROXY": "0.18",
                        "error": "",
                    }
                )
        return metrics_path

    monkeypatch.setattr(adaptive_flow_batch, "write_flow_transport_phantom", fake_write_flow_transport_phantom)
    monkeypatch.setattr(adaptive_flow_batch, "write_energy_ratio_phantom", fake_write_energy_ratio_phantom)
    monkeypatch.setattr(adaptive_flow_batch, "render_candidate_queue", fake_render_candidate_queue)

    metrics_path = adaptive_flow_batch.run_residual_selector_flow_batch(
        selector_queue,
        tmp_path / "selector_batch",
        max_energy_followups=2,
        min_energy_flow_delta=-0.005,
    )
    rows = {row["priority"]: row for row in csv.DictReader(metrics_path.open())}
    assert rows["1"]["energy_status"] == "deferred_flow_regression"
    assert rows["1"]["best_stage"] == "current"
    assert rows["2"]["energy_status"] == "ok"
    assert rows["2"]["best_stage"] == "flow_energy"
    energy_queue_rows = list(csv.DictReader((tmp_path / "selector_batch" / "selector_energy_queue_with_refs.csv").open()))
    assert [row["priority"] for row in energy_queue_rows] == ["2"]


def test_calibrate_energy_followup_gate_selects_true_scanner_threshold(tmp_path):
    metrics = tmp_path / "probe_metrics.csv"
    with metrics.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "priority",
                "current_ms_ssim",
                "flow_ms_ssim",
                "flow_energy_ms_ssim",
                "flow_delta_vs_current",
                "flow_energy_delta_vs_current",
            ],
        )
        writer.writeheader()
        for row in [
            {
                "priority": 1,
                "current_ms_ssim": "0.62",
                "flow_ms_ssim": "0.60",
                "flow_energy_ms_ssim": "0.61",
                "flow_delta_vs_current": "-0.02",
                "flow_energy_delta_vs_current": "-0.01",
            },
            {
                "priority": 2,
                "current_ms_ssim": "0.62",
                "flow_ms_ssim": "0.617",
                "flow_energy_ms_ssim": "0.624",
                "flow_delta_vs_current": "-0.003",
                "flow_energy_delta_vs_current": "0.004",
            },
            {
                "priority": 3,
                "current_ms_ssim": "0.62",
                "flow_ms_ssim": "0.63",
                "flow_energy_ms_ssim": "0.635",
                "flow_delta_vs_current": "0.01",
                "flow_energy_delta_vs_current": "0.015",
            },
            {
                "priority": 4,
                "current_ms_ssim": "0.62",
                "flow_ms_ssim": "0.625",
                "flow_energy_ms_ssim": "0.63",
                "flow_delta_vs_current": "0.005",
                "flow_energy_delta_vs_current": "0.01",
            },
        ]:
            writer.writerow(row)

    out = adaptive_flow_batch.calibrate_energy_followup_gate(
        (metrics,),
        tmp_path / "energy_gate.json",
        min_retained_delta_fraction=0.99,
        max_negative_energy_calls=0,
    )
    summary = json.loads(out.read_text())
    selected = summary["selected"]

    assert summary["recommended_min_energy_flow_delta"] == selected["min_energy_flow_delta"]
    assert selected["energy_calls"] == 3
    assert selected["skipped_energy_calls"] == 1
    assert selected["negative_energy_calls"] == 0
    assert selected["skipped_positive_energy_calls"] == 0
    assert selected["retained_best_delta_fraction"] == 1.0
    assert Path(summary["sweep_csv"]).exists()


def test_calibrate_energy_followup_gate_can_require_map_objective_retention(tmp_path):
    metrics = tmp_path / "probe_metrics.csv"
    with metrics.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "priority",
                "current_ms_ssim",
                "flow_ms_ssim",
                "flow_energy_ms_ssim",
                "flow_map_delta_vs_current",
                "flow_energy_map_delta_vs_current",
            ],
        )
        writer.writeheader()
        for row in [
            {
                "priority": 1,
                "current_ms_ssim": "0.62",
                "flow_ms_ssim": "0.60",
                "flow_energy_ms_ssim": "0.61",
                "flow_map_delta_vs_current": "-0.02",
                "flow_energy_map_delta_vs_current": "-0.01",
            },
            {
                "priority": 2,
                "current_ms_ssim": "0.62",
                "flow_ms_ssim": "0.617",
                "flow_energy_ms_ssim": "0.624",
                "flow_map_delta_vs_current": "-0.003",
                "flow_energy_map_delta_vs_current": "0.004",
            },
            {
                "priority": 3,
                "current_ms_ssim": "0.62",
                "flow_ms_ssim": "0.63",
                "flow_energy_ms_ssim": "0.635",
                "flow_map_delta_vs_current": "0.006",
                "flow_energy_map_delta_vs_current": "0.012",
            },
        ]:
            writer.writerow(row)

    out = adaptive_flow_batch.calibrate_energy_followup_gate(
        (metrics,),
        tmp_path / "energy_map_gate.json",
        min_retained_delta_fraction=0.99,
        min_retained_map_delta_fraction=0.99,
        max_negative_energy_calls=0,
        max_negative_map_calls=0,
    )
    summary = json.loads(out.read_text())
    selected = summary["selected"]

    assert summary["observed_map_objective_examples"] == 3
    assert selected["energy_calls"] == 2
    assert selected["negative_map_calls"] == 0
    assert selected["retained_map_delta_fraction"] == 1.0
    sweep_rows = list(csv.DictReader(Path(summary["sweep_csv"]).open()))
    assert "negative_map_calls" in sweep_rows[0]


def test_cli_calibrate_energy_followup_gate_reports_summary(tmp_path):
    metrics = tmp_path / "probe_metrics.csv"
    metrics.write_text(
        "\n".join(
            [
                "priority,current_ms_ssim,flow_ms_ssim,flow_energy_ms_ssim",
                "1,0.62,0.60,0.61",
                "2,0.62,0.63,0.635",
                "",
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "energy_gate.json"

    assert (
        main(
            [
                "calibrate-energy-followup-gate",
                "--probe-metrics",
                str(metrics),
                "--out",
                str(out),
                "--min-retained-delta-fraction",
                "0.9",
            ]
        )
        == 0
    )
    summary = json.loads(out.read_text())
    assert summary["observed_energy_examples"] == 2
    assert summary["evidence_scope"] == "true_scanner_probe_calibration_not_hidden_holdout"


def test_cli_enrich_residual_probe_energy_maps_backfills_objective(tmp_path):
    ref = tmp_path / "reference.png"
    energy = tmp_path / "energy.png"
    base = np.tile(np.linspace(0, 255, 32, dtype=np.uint8), (32, 1))
    skio.imsave(ref, base)
    skio.imsave(energy, np.clip(base + 2, 0, 255).astype(np.uint8))

    metrics = tmp_path / "probe_metrics.csv"
    with metrics.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "priority",
                "reference_png",
                "flow_energy_synthetic_gray_png",
                "current_map_objective_score",
                "flow_map_objective_score",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "priority": "1",
                "reference_png": str(ref),
                "flow_energy_synthetic_gray_png": str(energy),
                "current_map_objective_score": "0.50",
                "flow_map_objective_score": "0.55",
            }
        )

    out = tmp_path / "enriched_probe_metrics.csv"
    assert (
        main(
            [
                "enrich-residual-probe-energy-maps",
                "--probe-metrics",
                str(metrics),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["flow_energy_map_objective_status"] == "ok"
    assert rows[0]["flow_energy_map_objective_score"] != ""
    assert float(rows[0]["flow_energy_map_delta_vs_current"]) != 0.0
    summary = json.loads(out.with_name("enriched_probe_metrics_summary.json").read_text(encoding="utf-8"))
    assert summary["rows_with_flow_energy_map_objective"] == 1
    assert summary["promotion_allowed_without_true_scanner"] is False


def test_residual_selector_flow_batch_prioritizes_map_delta_for_energy_followups(tmp_path, monkeypatch):
    selector_queue = tmp_path / "budget_queue.csv"
    with selector_queue.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "priority",
                "row_index",
                "current_rank",
                "source_archive_path",
                "selected_strength",
                "current_ms_ssim",
                "current_lpips",
                "acquisition_score",
                "reference_png",
                "current_phantom_path",
                "current_synthetic_gray_png",
                "surrogate_gate_status",
                "evidence_scope",
            ],
        )
        writer.writeheader()
        for priority in [1, 2]:
            writer.writerow(
                {
                    "priority": priority,
                    "row_index": priority - 1,
                    "current_rank": priority,
                    "source_archive_path": f"DATASET_PNG/Female/1990-2000/Cheek/sample{priority}_frame50.png",
                    "selected_strength": "0.25",
                    "current_ms_ssim": "0.62",
                    "current_lpips": "0.58",
                    "acquisition_score": "0.01",
                    "reference_png": str(tmp_path / f"ref{priority}.png"),
                    "current_phantom_path": str(tmp_path / f"base{priority}.txt"),
                    "current_synthetic_gray_png": str(tmp_path / f"gray{priority}.png"),
                    "surrogate_gate_status": "pass",
                    "evidence_scope": "selector_budget_planning_not_challenge_evidence",
                }
            )

    def fake_write_flow_transport_phantom(_ref, _phantom, _gray, output_path, **_kwargs):
        Path(output_path).write_text("flow", encoding="utf-8")
        return Path(output_path)

    def fake_write_energy_ratio_phantom(_ref, _phantom, _gray, output_path, **_kwargs):
        Path(output_path).write_text("energy", encoding="utf-8")
        return Path(output_path)

    def fake_stage_map_objective(_reference_path, _prediction_path, maps_dir):
        label = Path(maps_dir).name
        scores = {
            "p1_current": 0.50,
            "p1_flow": 0.86,
            "p1_flow_energy": 0.90,
            "p2_current": 0.50,
            "p2_flow": 0.55,
        }
        return scores[label], "ok", ""

    def fake_render_candidate_queue(queue_csv, _reference_path, out_dir, **_kwargs):
        queue_rows = list(csv.DictReader(Path(queue_csv).open()))
        metrics_path = Path(out_dir) / "candidate_queue_metrics.csv"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "status",
            "priority",
            "method",
            "request_id",
            "reference_png",
            "phantom_path",
            "synthetic_gray_png",
            "MS-SSIM",
            "LPIPS_PROXY",
            "error",
        ]
        flow_scores = {"1": "0.63", "2": "0.70"}
        with metrics_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            if "selector_flow_queue" in str(queue_csv):
                for row in queue_rows:
                    writer.writerow(
                        {
                            "status": "ok",
                            "priority": row["priority"],
                            "method": row["method"],
                            "request_id": f"flow_{row['priority']}",
                            "reference_png": row["reference_png"],
                            "phantom_path": row["phantom_path"],
                            "synthetic_gray_png": str(tmp_path / f"flow_{row['priority']}_gray.png"),
                            "MS-SSIM": flow_scores[row["priority"]],
                            "LPIPS_PROXY": "0.2",
                            "error": "",
                        }
                    )
            else:
                assert [row["priority"] for row in queue_rows] == ["1"]
                row = queue_rows[0]
                writer.writerow(
                    {
                        "status": "ok",
                        "priority": row["priority"],
                        "method": row["method"],
                        "request_id": "energy_1",
                        "reference_png": row["reference_png"],
                        "phantom_path": row["phantom_path"],
                        "synthetic_gray_png": str(tmp_path / "energy_1_gray.png"),
                        "MS-SSIM": "0.66",
                        "LPIPS_PROXY": "0.18",
                        "error": "",
                    }
                )
        return metrics_path

    monkeypatch.setattr(adaptive_flow_batch, "write_flow_transport_phantom", fake_write_flow_transport_phantom)
    monkeypatch.setattr(adaptive_flow_batch, "write_energy_ratio_phantom", fake_write_energy_ratio_phantom)
    monkeypatch.setattr(adaptive_flow_batch, "_stage_map_objective", fake_stage_map_objective)
    monkeypatch.setattr(adaptive_flow_batch, "render_candidate_queue", fake_render_candidate_queue)

    metrics_path = adaptive_flow_batch.run_residual_selector_flow_batch(
        selector_queue,
        tmp_path / "selector_batch",
        max_energy_followups=1,
    )
    rows = {row["priority"]: row for row in csv.DictReader(metrics_path.open())}
    assert rows["1"]["energy_status"] == "ok"
    assert rows["2"]["energy_status"] == "deferred_budget"
    assert rows["1"]["flow_map_objective_status"] == "ok"
    assert float(rows["1"]["flow_map_delta_vs_current"]) > float(rows["2"]["flow_map_delta_vs_current"])
    energy_queue_rows = list(csv.DictReader((tmp_path / "selector_batch" / "selector_energy_queue_with_refs.csv").open()))
    assert [row["priority"] for row in energy_queue_rows] == ["1"]


def test_residual_selector_flow_batch_uses_row_wise_stage2_controls(tmp_path, monkeypatch):
    selector_queue = tmp_path / "controlled_queue.csv"
    with selector_queue.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "priority",
                "row_index",
                "current_rank",
                "source_archive_path",
                "selected_strength",
                "selector_holdout_group_key",
                "selector_holdout_group_status",
                "selector_holdout_score_multiplier",
                "surrogate_calibration_status",
                "surrogate_calibration_support_n",
                "surrogate_calibration_distance",
                "surrogate_calibration_score_multiplier",
                "diversity_stratum",
                "residual_control_policy",
                "flow_smooth_sigma",
                "flow_attachment",
                "energy_exponent",
                "energy_sigma",
                "energy_ratio_low",
                "energy_ratio_high",
                "energy_clip_low",
                "energy_clip_high",
                "texture_mean_exponent",
                "texture_exponent",
                "texture_deep_exponent",
                "current_ms_ssim",
                "current_ssim",
                "current_lpips",
                "acquisition_score",
                "reference_png",
                "current_phantom_path",
                "current_synthetic_gray_png",
                "surrogate_gate_status",
                "evidence_scope",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "priority": "1",
                "row_index": "0",
                "current_rank": "1",
                "source_archive_path": "DATASET_PNG/Female/1990-2000/Cheek/sample_frame50.png",
                "selected_strength": "0.38",
                "selector_holdout_group_key": "Female|1990-2000|Cheek",
                "selector_holdout_group_status": "holdout_group_no_wins",
                "selector_holdout_score_multiplier": "0.5",
                "surrogate_calibration_status": "calibration_supported",
                "surrogate_calibration_support_n": "4",
                "surrogate_calibration_distance": "0.0",
                "surrogate_calibration_score_multiplier": "1.0",
                "diversity_stratum": "Female|1990-2000|Cheek",
                "residual_control_policy": "row_wise_physics_dl_v1",
                "flow_smooth_sigma": "1.62",
                "flow_attachment": "5.35",
                "energy_exponent": "0.91",
                "energy_sigma": "2.55",
                "energy_ratio_low": "0.61",
                "energy_ratio_high": "1.34",
                "energy_clip_low": "0.76",
                "energy_clip_high": "1.24",
                "texture_mean_exponent": "0.03",
                "texture_exponent": "0.42",
                "texture_deep_exponent": "0.95",
                "current_ms_ssim": "0.62",
                "current_ssim": "0.08",
                "current_lpips": "0.58",
                "acquisition_score": "0.02",
                "reference_png": str(tmp_path / "ref.png"),
                "current_phantom_path": str(tmp_path / "base.txt"),
                "current_synthetic_gray_png": str(tmp_path / "base_gray.png"),
                "surrogate_gate_status": "pass",
                "evidence_scope": "selector_budget_planning_not_challenge_evidence",
            }
        )

    captured: dict[str, dict[str, float]] = {}

    def fake_write_flow_transport_phantom(_ref, _phantom, _gray, output_path, **kwargs):
        captured["flow"] = kwargs
        Path(output_path).write_text("flow", encoding="utf-8")
        return Path(output_path)

    def fake_write_energy_ratio_phantom(_ref, _phantom, _gray, output_path, **kwargs):
        captured["energy"] = kwargs
        Path(output_path).write_text("ratio", encoding="utf-8")
        return Path(output_path)

    def fake_write_texture_matched_phantom(_ref, _phantom, _gray, output_path, **kwargs):
        captured["texture"] = kwargs
        Path(output_path).write_text("texture", encoding="utf-8")
        return Path(output_path)

    def fake_render_candidate_queue(queue_csv, _reference_path, out_dir, **_kwargs):
        queue_rows = list(csv.DictReader(Path(queue_csv).open()))
        metrics_path = Path(out_dir) / "candidate_queue_metrics.csv"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "status",
                    "priority",
                    "method",
                    "request_id",
                    "reference_png",
                    "phantom_path",
                    "synthetic_gray_png",
                    "MS-SSIM",
                    "LPIPS_PROXY",
                    "error",
                ],
            )
            writer.writeheader()
            row = queue_rows[0]
            stage = "flow" if "selector_flow_queue" in str(queue_csv) else "energy"
            writer.writerow(
                {
                    "status": "ok",
                    "priority": row["priority"],
                    "method": row["method"],
                    "request_id": f"{stage}_1",
                    "reference_png": row["reference_png"],
                    "phantom_path": row["phantom_path"],
                    "synthetic_gray_png": str(tmp_path / f"{stage}_gray.png"),
                    "MS-SSIM": "0.64" if stage == "flow" else "0.67",
                    "LPIPS_PROXY": "0.2",
                    "error": "",
                }
            )
        return metrics_path

    monkeypatch.setattr(adaptive_flow_batch, "write_flow_transport_phantom", fake_write_flow_transport_phantom)
    monkeypatch.setattr(adaptive_flow_batch, "write_energy_ratio_phantom", fake_write_energy_ratio_phantom)
    monkeypatch.setattr(adaptive_flow_batch, "write_texture_matched_phantom", fake_write_texture_matched_phantom)
    monkeypatch.setattr(adaptive_flow_batch, "render_candidate_queue", fake_render_candidate_queue)

    metrics_path = adaptive_flow_batch.run_residual_selector_flow_batch(
        selector_queue,
        tmp_path / "selector_batch",
        flow_smooth_sigma=1.0,
        flow_attachment=7.0,
        energy_exponent=0.7,
    )

    assert captured["flow"]["strength"] == 0.38
    assert captured["flow"]["smooth_sigma"] == 1.62
    assert captured["flow"]["attachment"] == 5.35
    assert captured["energy"]["exponent"] == 0.91
    assert captured["energy"]["sigma"] == 2.55
    assert captured["energy"]["ratio_low"] == 0.61
    assert captured["energy"]["ratio_high"] == 1.34
    assert captured["energy"]["clip_low"] == 0.76
    assert captured["energy"]["clip_high"] == 1.24
    assert captured["texture"]["mean_exponent"] == 0.03
    assert captured["texture"]["texture_exponent"] == 0.42
    assert captured["texture"]["deep_exponent"] == 0.95
    rows = list(csv.DictReader(metrics_path.open()))
    assert rows[0]["residual_control_policy"] == "row_wise_physics_dl_v1"
    assert rows[0]["selector_holdout_group_status"] == "holdout_group_no_wins"
    assert rows[0]["selector_holdout_score_multiplier"] == "0.5"
    assert rows[0]["surrogate_calibration_status"] == "calibration_supported"
    assert rows[0]["surrogate_calibration_score_multiplier"] == "1.0"
    assert rows[0]["current_ssim"] == "0.08"
    assert rows[0]["diversity_stratum"] == "Female|1990-2000|Cheek"
    assert rows[0]["flow_smooth_sigma"] == "1.62"
    assert rows[0]["energy_sigma"] == "2.55"
    assert rows[0]["texture_exponent"] == "0.42"


def test_residual_selector_flow_batch_can_branch_energy_from_current(tmp_path, monkeypatch):
    selector_queue = tmp_path / "branch_queue.csv"
    with selector_queue.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "priority",
                "row_index",
                "current_rank",
                "source_archive_path",
                "selected_strength",
                "residual_control_policy",
                "energy_base_policy",
                "energy_base_flow_delta_threshold",
                "current_ms_ssim",
                "current_lpips",
                "reference_png",
                "current_phantom_path",
                "current_synthetic_gray_png",
                "surrogate_gate_status",
                "evidence_scope",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "priority": "1",
                "row_index": "0",
                "current_rank": "1",
                "source_archive_path": "DATASET_PNG/Female/1950-1960/Cheek/mild_flow_regression_frame50.png",
                "selected_strength": "0.38",
                "residual_control_policy": "teacher_neighbor_feedback_distilled_v3",
                "energy_base_policy": "current_on_flow_regression",
                "energy_base_flow_delta_threshold": "0.0",
                "current_ms_ssim": "0.62",
                "current_lpips": "0.58",
                "reference_png": str(tmp_path / "ref.png"),
                "current_phantom_path": str(tmp_path / "base.txt"),
                "current_synthetic_gray_png": str(tmp_path / "base_gray.png"),
                "surrogate_gate_status": "pass",
                "evidence_scope": "selector_budget_planning_not_challenge_evidence",
            }
        )

    captured: dict[str, str] = {}

    def fake_write_flow_transport_phantom(_ref, _phantom, _gray, output_path, **_kwargs):
        Path(output_path).write_text("flow", encoding="utf-8")
        return Path(output_path)

    def fake_write_energy_ratio_phantom(_ref, phantom, gray, output_path, **_kwargs):
        captured["energy_phantom"] = str(phantom)
        captured["energy_gray"] = str(gray)
        Path(output_path).write_text("energy", encoding="utf-8")
        return Path(output_path)

    def fake_render_candidate_queue(queue_csv, _reference_path, out_dir, **_kwargs):
        queue_rows = list(csv.DictReader(Path(queue_csv).open()))
        metrics_path = Path(out_dir) / "candidate_queue_metrics.csv"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "status",
                    "priority",
                    "method",
                    "request_id",
                    "reference_png",
                    "phantom_path",
                    "synthetic_gray_png",
                    "MS-SSIM",
                    "LPIPS_PROXY",
                    "error",
                ],
            )
            writer.writeheader()
            row = queue_rows[0]
            stage = "flow" if "selector_flow_queue" in str(queue_csv) else "energy"
            writer.writerow(
                {
                    "status": "ok",
                    "priority": row["priority"],
                    "method": row["method"],
                    "request_id": f"{stage}_1",
                    "reference_png": row["reference_png"],
                    "phantom_path": row["phantom_path"],
                    "synthetic_gray_png": str(tmp_path / f"{stage}_gray.png"),
                    "MS-SSIM": "0.618" if stage == "flow" else "0.64",
                    "LPIPS_PROXY": "0.2",
                    "error": "",
                }
            )
        return metrics_path

    monkeypatch.setattr(adaptive_flow_batch, "write_flow_transport_phantom", fake_write_flow_transport_phantom)
    monkeypatch.setattr(adaptive_flow_batch, "write_energy_ratio_phantom", fake_write_energy_ratio_phantom)
    monkeypatch.setattr(adaptive_flow_batch, "render_candidate_queue", fake_render_candidate_queue)

    metrics_path = adaptive_flow_batch.run_residual_selector_flow_batch(
        selector_queue,
        tmp_path / "selector_branch_batch",
        min_energy_flow_delta=-0.01,
    )

    assert captured["energy_phantom"] == str(tmp_path / "base.txt")
    assert captured["energy_gray"] == str(tmp_path / "base_gray.png")
    row = next(csv.DictReader(metrics_path.open()))
    assert row["energy_base_policy"] == "current_on_flow_regression"
    assert row["energy_base_stage"] == "current"
    assert row["energy_base_phantom_path"] == str(tmp_path / "base.txt")
    assert row["energy_status"] == "ok"
    assert row["best_stage"] == "flow_energy"


def test_cli_residual_selector_trains_and_plans_queue(tmp_path):
    base_metrics = tmp_path / "base_api_metrics.csv"
    with base_metrics.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_archive_path", "SSIM", "MS-SSIM", "LPIPS", "reference_png", "phantom_path", "synthetic_gray_png"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_archive_path": "DATASET_PNG/Female/1990-2000/Cheek/a_frame50.png",
                "SSIM": "0.61",
                "MS-SSIM": "0.62",
                "LPIPS": "0.58",
                "reference_png": str(tmp_path / "a_ref.png"),
                "phantom_path": str(tmp_path / "a_phantom.txt"),
                "synthetic_gray_png": str(tmp_path / "a_gray.png"),
            }
        )
        writer.writerow(
            {
                "source_archive_path": "DATASET_PNG/Female/1990-2000/Cheek/b_frame50.png",
                "SSIM": "0.62",
                "MS-SSIM": "0.61",
                "LPIPS": "0.59",
                "reference_png": str(tmp_path / "b_ref.png"),
                "phantom_path": str(tmp_path / "b_phantom.txt"),
                "synthetic_gray_png": str(tmp_path / "b_gray.png"),
            }
        )

    teacher_metrics = tmp_path / "adaptive_strength_metrics.csv"
    with teacher_metrics.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_archive_path",
                "row_index",
                "current_rank",
                "strength",
                "current_ms_ssim",
                "current_lpips",
                "flow_ms_ssim",
                "flow_energy_ms_ssim",
                "best_stage",
                "best_ms_ssim",
                "reference_png",
                "current_phantom_path",
                "current_synthetic_gray_png",
                "flow_phantom_path",
                "flow_energy_phantom_path",
                "current_Struct_MS-SSIM",
                "current_OAC_MS-SSIM",
                "current_SC_MS-SSIM",
                "current_RSC_MS-SSIM",
                "current_Struct_LPIPS",
                "current_OAC_LPIPS",
                "current_SC_LPIPS",
                "current_RSC_LPIPS",
                "flow_energy_Struct_MS-SSIM",
                "flow_energy_OAC_MS-SSIM",
                "flow_energy_SC_MS-SSIM",
                "flow_energy_RSC_MS-SSIM",
                "flow_energy_Struct_LPIPS",
                "flow_energy_OAC_LPIPS",
                "flow_energy_SC_LPIPS",
                "flow_energy_RSC_LPIPS",
            ],
        )
        writer.writeheader()
        for source, row_index, base, flow, energy in [
            ("DATASET_PNG/Female/1990-2000/Cheek/a_frame50.png", 0, 0.62, 0.625, 0.64),
            ("DATASET_PNG/Female/1990-2000/Cheek/b_frame50.png", 1, 0.61, 0.620, 0.625),
        ]:
            writer.writerow(
                {
                    "source_archive_path": source,
                    "row_index": row_index,
                    "current_rank": row_index + 1,
                    "strength": "0.12",
                    "current_ms_ssim": base,
                    "current_lpips": "0.58",
                    "flow_ms_ssim": flow,
                    "flow_energy_ms_ssim": energy,
                    "best_stage": "flow_energy",
                    "best_ms_ssim": energy,
                    "reference_png": str(tmp_path / f"{row_index}_ref.png"),
                    "current_phantom_path": str(tmp_path / f"{row_index}_base.txt"),
                    "current_synthetic_gray_png": str(tmp_path / f"{row_index}_base_gray.png"),
                    "flow_phantom_path": str(tmp_path / f"{row_index}_flow.txt"),
                    "flow_energy_phantom_path": str(tmp_path / f"{row_index}_energy.txt"),
                    "current_Struct_MS-SSIM": "0.70",
                    "current_OAC_MS-SSIM": "0.70",
                    "current_SC_MS-SSIM": "0.70",
                    "current_RSC_MS-SSIM": "0.70",
                    "current_Struct_LPIPS": "0.30",
                    "current_OAC_LPIPS": "0.30",
                    "current_SC_LPIPS": "0.30",
                    "current_RSC_LPIPS": "0.30",
                    "flow_energy_Struct_MS-SSIM": "0.75",
                    "flow_energy_OAC_MS-SSIM": "0.75",
                    "flow_energy_SC_MS-SSIM": "0.75",
                    "flow_energy_RSC_MS-SSIM": "0.75",
                    "flow_energy_Struct_LPIPS": "0.25",
                    "flow_energy_OAC_LPIPS": "0.25",
                    "flow_energy_SC_LPIPS": "0.25",
                    "flow_energy_RSC_LPIPS": "0.25",
                }
            )
            writer.writerow(
                {
                    "source_archive_path": source,
                    "row_index": row_index,
                    "current_rank": row_index + 1,
                    "strength": "0.38",
                    "current_ms_ssim": base,
                    "current_lpips": "0.58",
                    "flow_ms_ssim": base - 0.002,
                    "flow_energy_ms_ssim": base + 0.003,
                    "best_stage": "flow_energy",
                    "best_ms_ssim": base + 0.003,
                    "reference_png": str(tmp_path / f"{row_index}_ref.png"),
                    "current_phantom_path": str(tmp_path / f"{row_index}_base.txt"),
                    "current_synthetic_gray_png": str(tmp_path / f"{row_index}_base_gray.png"),
                    "flow_phantom_path": str(tmp_path / f"{row_index}_flow_s38.txt"),
                    "flow_energy_phantom_path": str(tmp_path / f"{row_index}_energy_s38.txt"),
                    "current_Struct_MS-SSIM": "0.70",
                    "current_OAC_MS-SSIM": "0.70",
                    "current_SC_MS-SSIM": "0.70",
                    "current_RSC_MS-SSIM": "0.70",
                    "current_Struct_LPIPS": "0.30",
                    "current_OAC_LPIPS": "0.30",
                    "current_SC_LPIPS": "0.30",
                    "current_RSC_LPIPS": "0.30",
                    "flow_energy_Struct_MS-SSIM": "0.62",
                    "flow_energy_OAC_MS-SSIM": "0.62",
                    "flow_energy_SC_MS-SSIM": "0.62",
                    "flow_energy_RSC_MS-SSIM": "0.62",
                    "flow_energy_Struct_LPIPS": "0.45",
                    "flow_energy_OAC_LPIPS": "0.45",
                    "flow_energy_SC_LPIPS": "0.45",
                    "flow_energy_RSC_LPIPS": "0.45",
                }
            )

    artifact = tmp_path / "selector.json"
    surrogate_calibration = tmp_path / "surrogate_calibration_metrics.csv"
    with surrogate_calibration.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["surrogate_MS-SSIM", "surrogate_LPIPS_PROXY", "training_row_ssim"])
        writer.writeheader()
        writer.writerow({"surrogate_MS-SSIM": "0.95", "surrogate_LPIPS_PROXY": "0.015", "training_row_ssim": "0.60"})
        writer.writerow({"surrogate_MS-SSIM": "0.94", "surrogate_LPIPS_PROXY": "0.018", "training_row_ssim": "0.63"})

    assert (
        main(
            [
                "train-residual-selector",
                "--base-api-metrics",
                str(base_metrics),
                "--teacher-metrics",
                str(teacher_metrics),
                "--out",
                str(artifact),
                "--holdout-fraction",
                "0",
                "--surrogate-calibration-metrics",
                str(surrogate_calibration),
            ]
        )
        == 0
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["evidence_scope"] == "not_challenge_evidence"
    assert payload["surrogate_scanner_is_true_scanner"] is False
    assert payload["official_final_ranking_proven"] is False
    assert payload["selected_strength"] == 0.12
    assert payload["objective"]["primary"].startswith("Struct/OAC/SC/RSC")
    assert payload["surrogate_gate"]["status"] == "pass"
    assert payload["surrogate_gate"]["training_row_ssim_n"] == 2
    assert Path(payload["teacher_examples_csv"]).exists()

    queue = tmp_path / "selector_queue.csv"
    assert (
        main(
            [
                "plan-residual-selector-queue",
                "--base-api-metrics",
                str(base_metrics),
                "--artifact",
                str(artifact),
                "--out",
                str(queue),
                "--limit",
                "2",
            ]
        )
        == 0
    )
    rows = list(csv.DictReader(queue.open()))
    assert len(rows) == 2
    assert rows[0]["source_archive_path"].endswith("b_frame50.png")
    assert rows[0]["selected_strength"] == "0.12"
    assert rows[0]["current_ssim"] == "0.62"
    assert rows[0]["residual_control_policy"] == "teacher_neighbor_physics_dl_v2"
    assert rows[0]["flow_smooth_sigma"] != ""
    assert rows[0]["energy_exponent"] != ""
    assert rows[0]["texture_exponent"] != ""
    assert rows[0]["expected_delta_uncertainty"] != ""
    assert rows[0]["acquisition_score"] != ""
    assert rows[0]["expected_flow_delta_mean"] != ""
    assert rows[0]["expected_energy_delta_mean"] != ""
    assert rows[0]["expected_energy_extra_mean"] != ""
    assert rows[0]["nearest_teacher_count"] != ""
    assert rows[0]["surrogate_gate_status"] == "pass"
    assert rows[0]["surrogate_calibration_status"] == "calibration_supported"
    assert rows[0]["surrogate_calibration_score_multiplier"] == "1.0"
    assert rows[0]["evidence_scope"] == "selector_planning_not_challenge_evidence"


def test_cli_residual_selector_can_choose_row_wise_strengths(tmp_path):
    base_metrics = tmp_path / "base_api_metrics.csv"
    with base_metrics.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_archive_path", "MS-SSIM", "LPIPS", "reference_png", "phantom_path", "synthetic_gray_png"],
        )
        writer.writeheader()
        for source, label in [
            ("DATASET_PNG/Female/1990-2000/Cheek/a_frame50.png", "cheek"),
            ("DATASET_PNG/Female/1990-2000/Eye_corner/b_frame50.png", "eye"),
        ]:
            writer.writerow(
                {
                    "source_archive_path": source,
                    "MS-SSIM": "0.60",
                    "LPIPS": "0.50",
                    "reference_png": str(tmp_path / f"{label}_ref.png"),
                    "phantom_path": str(tmp_path / f"{label}_phantom.txt"),
                    "synthetic_gray_png": str(tmp_path / f"{label}_gray.png"),
                }
            )

    teacher_metrics = tmp_path / "teacher.csv"
    with teacher_metrics.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_archive_path",
                "row_index",
                "current_rank",
                "strength",
                "current_ms_ssim",
                "current_lpips",
                "flow_ms_ssim",
                "flow_energy_ms_ssim",
                "best_stage",
                "best_ms_ssim",
            ],
        )
        writer.writeheader()
        rows = [
            ("DATASET_PNG/Female/1990-2000/Cheek/a_frame50.png", 0, 0.12, 0.65),
            ("DATASET_PNG/Female/1990-2000/Cheek/a_frame50.png", 0, 0.38, 0.60),
            ("DATASET_PNG/Female/1990-2000/Eye_corner/b_frame50.png", 1, 0.12, 0.60),
            ("DATASET_PNG/Female/1990-2000/Eye_corner/b_frame50.png", 1, 0.38, 0.66),
        ]
        for source, row_index, strength, best in rows:
            writer.writerow(
                {
                    "source_archive_path": source,
                    "row_index": row_index,
                    "current_rank": row_index + 1,
                    "strength": strength,
                    "current_ms_ssim": "0.60",
                    "current_lpips": "0.50",
                    "flow_ms_ssim": best,
                    "flow_energy_ms_ssim": best,
                    "best_stage": "flow_energy" if best > 0.60 else "base",
                    "best_ms_ssim": best,
                }
            )

    artifact = tmp_path / "selector.json"
    assert (
        main(
            [
                "train-residual-selector",
                "--base-api-metrics",
                str(base_metrics),
                "--teacher-metrics",
                str(teacher_metrics),
                "--out",
                str(artifact),
                "--holdout-fraction",
                "0",
                "--neighbor-count",
                "1",
            ]
        )
        == 0
    )

    queue = tmp_path / "row_wise_queue.csv"
    assert (
        main(
            [
                "plan-residual-selector-queue",
                "--base-api-metrics",
                str(base_metrics),
                "--artifact",
                str(artifact),
                "--out",
                str(queue),
                "--limit",
                "2",
                "--row-wise-strengths",
            ]
        )
        == 0
    )
    by_source = {row["source_archive_path"]: row for row in csv.DictReader(queue.open())}
    cheek = by_source["DATASET_PNG/Female/1990-2000/Cheek/a_frame50.png"]
    eye = by_source["DATASET_PNG/Female/1990-2000/Eye_corner/b_frame50.png"]
    assert cheek["selected_strength"] == "0.12"
    assert eye["selected_strength"] == "0.38"
    assert cheek["strength_policy"] == "row_wise_neighbor_strength"
    assert eye["evaluated_strengths"] == "0.12;0.38"
    assert eye["strength_candidate_count"] == "2"


def test_cli_residual_selector_learns_selected_strength_probe_rows(tmp_path):
    base_metrics = tmp_path / "base_api_metrics.csv"
    base_metrics.write_text(
        "source_archive_path,MS-SSIM,LPIPS,reference_png,phantom_path,synthetic_gray_png\n"
        "DATASET_PNG/Male/1950-1960/Eye_corner/probe_frame50.png,0.64,0.50,ref.png,base.txt,base_gray.png\n",
        encoding="utf-8",
    )
    probe_metrics = tmp_path / "probe_metrics.csv"
    probe_metrics.write_text(
        "source_archive_path,row_index,current_rank,selected_strength,current_ms_ssim,current_lpips,"
        "flow_ms_ssim,flow_energy_ms_ssim,best_stage,best_ms_ssim\n"
        "DATASET_PNG/Male/1950-1960/Eye_corner/probe_frame50.png,0,1,0.38,0.64,0.50,0.65,0.66,flow_energy,0.66\n",
        encoding="utf-8",
    )
    artifact = tmp_path / "selector.json"

    assert (
        main(
            [
                "train-residual-selector",
                "--base-api-metrics",
                str(base_metrics),
                "--teacher-metrics",
                str(probe_metrics),
                "--out",
                str(artifact),
                "--holdout-fraction",
                "0",
            ]
        )
        == 0
    )

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["selected_strength"] == 0.38
    examples = list(csv.DictReader(Path(payload["teacher_examples_csv"]).open()))
    assert examples[0]["strength"] == "0.38"


def test_cli_residual_selector_controls_follow_teacher_stage_signal(tmp_path):
    base_metrics = tmp_path / "base_api_metrics.csv"
    with base_metrics.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_archive_path", "MS-SSIM", "LPIPS", "reference_png", "phantom_path", "synthetic_gray_png"],
        )
        writer.writeheader()
        for source, label in [
            ("DATASET_PNG/Female/1990-2000/Cheek/flow_case_frame50.png", "flow_case"),
            ("DATASET_PNG/Female/1990-2000/Eye_corner/energy_case_frame50.png", "energy_case"),
        ]:
            writer.writerow(
                {
                    "source_archive_path": source,
                    "MS-SSIM": "0.60",
                    "LPIPS": "0.52",
                    "reference_png": str(tmp_path / f"{label}_ref.png"),
                    "phantom_path": str(tmp_path / f"{label}_phantom.txt"),
                    "synthetic_gray_png": str(tmp_path / f"{label}_gray.png"),
                }
            )

    teacher_metrics = tmp_path / "stage_teacher.csv"
    with teacher_metrics.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_archive_path",
                "row_index",
                "current_rank",
                "strength",
                "current_ms_ssim",
                "current_lpips",
                "flow_ms_ssim",
                "flow_energy_ms_ssim",
                "best_stage",
                "best_ms_ssim",
            ],
        )
        writer.writeheader()
        for row in [
            ("DATASET_PNG/Female/1990-2000/Cheek/flow_case_frame50.png", 0, 0.25, 0.680, 0.681, "flow"),
            ("DATASET_PNG/Female/1990-2000/Eye_corner/energy_case_frame50.png", 1, 0.25, 0.602, 0.680, "flow_energy"),
        ]:
            source, row_index, strength, flow_ms, energy_ms, best_stage = row
            writer.writerow(
                {
                    "source_archive_path": source,
                    "row_index": row_index,
                    "current_rank": row_index + 1,
                    "strength": strength,
                    "current_ms_ssim": "0.60",
                    "current_lpips": "0.52",
                    "flow_ms_ssim": flow_ms,
                    "flow_energy_ms_ssim": energy_ms,
                    "best_stage": best_stage,
                    "best_ms_ssim": max(flow_ms, energy_ms),
                }
            )

    artifact = tmp_path / "selector.json"
    assert (
        main(
            [
                "train-residual-selector",
                "--base-api-metrics",
                str(base_metrics),
                "--teacher-metrics",
                str(teacher_metrics),
                "--out",
                str(artifact),
                "--holdout-fraction",
                "0",
                "--neighbor-count",
                "1",
            ]
        )
        == 0
    )

    queue = tmp_path / "stage_queue.csv"
    assert (
        main(
            [
                "plan-residual-selector-queue",
                "--base-api-metrics",
                str(base_metrics),
                "--artifact",
                str(artifact),
                "--out",
                str(queue),
                "--limit",
                "2",
            ]
        )
        == 0
    )
    by_source = {row["source_archive_path"]: row for row in csv.DictReader(queue.open())}
    flow_case = by_source["DATASET_PNG/Female/1990-2000/Cheek/flow_case_frame50.png"]
    energy_case = by_source["DATASET_PNG/Female/1990-2000/Eye_corner/energy_case_frame50.png"]
    assert flow_case["residual_control_policy"] == "teacher_neighbor_physics_dl_v2"
    assert float(flow_case["expected_flow_delta_mean"]) > float(energy_case["expected_flow_delta_mean"])
    assert float(energy_case["expected_energy_extra_mean"]) > float(flow_case["expected_energy_extra_mean"])
    assert float(flow_case["flow_smooth_sigma"]) > float(energy_case["flow_smooth_sigma"])
    assert float(energy_case["texture_exponent"]) > float(flow_case["texture_exponent"])


def test_cli_residual_selector_reports_grouped_holdout(tmp_path):
    base_metrics = tmp_path / "base_api_metrics.csv"
    base_metrics.write_text(
        "source_archive_path,MS-SSIM,LPIPS,reference_png,phantom_path,synthetic_gray_png\n"
        "DATASET_PNG/Female/1990-2000/Cheek/a_frame50.png,0.60,0.50,a_ref.png,a.txt,a_gray.png\n"
        "DATASET_PNG/Female/1990-2000/Eye_corner/b_frame50.png,0.60,0.50,b_ref.png,b.txt,b_gray.png\n",
        encoding="utf-8",
    )
    teacher_metrics = tmp_path / "teacher.csv"
    with teacher_metrics.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_archive_path",
                "row_index",
                "current_rank",
                "strength",
                "current_ms_ssim",
                "current_lpips",
                "flow_ms_ssim",
                "flow_energy_ms_ssim",
                "best_stage",
                "best_ms_ssim",
            ],
        )
        writer.writeheader()
        for source, row_index, strength, best in [
            ("DATASET_PNG/Female/1990-2000/Cheek/a_frame50.png", 0, 0.25, 0.66),
            ("DATASET_PNG/Female/1990-2000/Cheek/a_frame50.png", 0, 0.38, 0.61),
            ("DATASET_PNG/Female/1990-2000/Eye_corner/b_frame50.png", 1, 0.25, 0.65),
            ("DATASET_PNG/Female/1990-2000/Eye_corner/b_frame50.png", 1, 0.38, 0.60),
        ]:
            writer.writerow(
                {
                    "source_archive_path": source,
                    "row_index": row_index,
                    "current_rank": row_index + 1,
                    "strength": strength,
                    "current_ms_ssim": "0.60",
                    "current_lpips": "0.50",
                    "flow_ms_ssim": best,
                    "flow_energy_ms_ssim": best,
                    "best_stage": "flow_energy",
                    "best_ms_ssim": best,
                }
            )

    artifact = tmp_path / "selector.json"
    assert (
        main(
            [
                "train-residual-selector",
                "--base-api-metrics",
                str(base_metrics),
                "--teacher-metrics",
                str(teacher_metrics),
                "--out",
                str(artifact),
                "--holdout-fraction",
                "0.25",
                "--holdout-group-fields",
                "body_site",
            ]
        )
        == 0
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["holdout_group_fields"] == ["body_site"]
    assert payload["train_example_count"] == 2
    assert payload["holdout_example_count"] == 2
    assert payload["holdout_groups"] == ["Eye_corner"]
    assert payload["selected_strength"] == 0.25
    assert payload["holdout"]["n"] == 1
    assert payload["holdout_by_group"] == [
        {
            "group": "Eye_corner",
            "strength": 0.25,
            "n": 1,
            "delta_mean": 0.050000000000000044,
            "delta_median": 0.050000000000000044,
            "delta_min": 0.050000000000000044,
            "delta_max": 0.050000000000000044,
            "win_rate": 1.0,
        }
    ]

    queue = tmp_path / "selector_queue.csv"
    assert (
        main(
            [
                "plan-residual-selector-queue",
                "--base-api-metrics",
                str(base_metrics),
                "--artifact",
                str(artifact),
                "--out",
                str(queue),
                "--limit",
                "2",
            ]
        )
        == 0
    )
    by_source = {row["source_archive_path"]: row for row in csv.DictReader(queue.open())}
    cheek = by_source["DATASET_PNG/Female/1990-2000/Cheek/a_frame50.png"]
    eye = by_source["DATASET_PNG/Female/1990-2000/Eye_corner/b_frame50.png"]
    assert cheek["selector_holdout_group_key"] == "Cheek"
    assert cheek["selector_holdout_group_status"] == "not_held_out"
    assert cheek["selector_holdout_score_multiplier"] == "1.0"
    assert eye["selector_holdout_group_key"] == "Eye_corner"
    assert eye["selector_holdout_group_n"] == "1"
    assert eye["selector_holdout_group_status"] == "holdout_group_supported"
    assert eye["selector_holdout_score_multiplier"] == "1.0"


def test_cli_residual_selector_surrogate_gate_blocks_queue(tmp_path):
    base_metrics = tmp_path / "base_api_metrics.csv"
    base_metrics.write_text(
        "source_archive_path,MS-SSIM,LPIPS,reference_png,phantom_path,synthetic_gray_png\n"
        "DATASET_PNG/Female/1990-2000/Cheek/a_frame50.png,0.62,0.58,a_ref.png,a.txt,a_gray.png\n",
        encoding="utf-8",
    )
    teacher_metrics = tmp_path / "teacher.csv"
    teacher_metrics.write_text(
        "source_archive_path,row_index,current_rank,strength,current_ms_ssim,current_lpips,flow_ms_ssim,flow_energy_ms_ssim,best_stage,best_ms_ssim\n"
        "DATASET_PNG/Female/1990-2000/Cheek/a_frame50.png,0,1,0.25,0.62,0.58,0.63,0.64,flow_energy,0.64\n",
        encoding="utf-8",
    )
    surrogate_calibration = tmp_path / "bad_surrogate_calibration.csv"
    surrogate_calibration.write_text(
        "surrogate_MS-SSIM,surrogate_LPIPS_PROXY\n0.81,0.09\n",
        encoding="utf-8",
    )
    artifact = tmp_path / "selector_bad_gate.json"
    assert (
        main(
            [
                "train-residual-selector",
                "--base-api-metrics",
                str(base_metrics),
                "--teacher-metrics",
                str(teacher_metrics),
                "--out",
                str(artifact),
                "--surrogate-calibration-metrics",
                str(surrogate_calibration),
            ]
        )
        == 0
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["surrogate_gate"]["status"] == "fail"

    queue = tmp_path / "blocked_queue.csv"
    assert (
        main(
            [
                "plan-residual-selector-queue",
                "--base-api-metrics",
                str(base_metrics),
                "--artifact",
                str(artifact),
                "--out",
                str(queue),
            ]
        )
        == 0
    )
    rows = list(csv.DictReader(queue.open()))
    assert rows == []
    assert "surrogate_gate_status" in queue.read_text(encoding="utf-8").splitlines()[0]


def test_cli_enrich_residual_teacher_maps_feeds_multi_objective_selector(tmp_path):
    x = np.linspace(0, 1, 64, dtype=np.float32)
    ref_arr = np.tile(x, (64, 1))
    base_arr = np.clip(ref_arr * 0.72 + 0.08, 0, 1)
    flow_arr = np.clip(ref_arr * 0.86 + 0.04, 0, 1)
    energy_arr = np.clip(ref_arr * 0.94 + 0.02, 0, 1)
    ref_png = tmp_path / "ref.png"
    base_png = tmp_path / "base_gray.png"
    flow_png = tmp_path / "flow_gray.png"
    energy_png = tmp_path / "energy_gray.png"
    for path, arr in [(ref_png, ref_arr), (base_png, base_arr), (flow_png, flow_arr), (energy_png, energy_arr)]:
        skio.imsave(path, (arr * 255).astype(np.uint8))

    source = "DATASET_PNG/Female/1990-2000/Cheek/sample_frame50.png"
    base_metrics = tmp_path / "base_api_metrics.csv"
    with base_metrics.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_archive_path", "MS-SSIM", "LPIPS", "reference_png", "phantom_path", "synthetic_gray_png"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_archive_path": source,
                "MS-SSIM": "0.62",
                "LPIPS": "0.58",
                "reference_png": str(ref_png),
                "phantom_path": str(tmp_path / "base.txt"),
                "synthetic_gray_png": str(base_png),
            }
        )

    teacher_metrics = tmp_path / "teacher.csv"
    with teacher_metrics.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_archive_path",
                "row_index",
                "current_rank",
                "strength",
                "current_ms_ssim",
                "current_lpips",
                "flow_ms_ssim",
                "flow_energy_ms_ssim",
                "best_stage",
                "best_ms_ssim",
                "flow_synthetic_gray_png",
                "flow_energy_synthetic_gray_png",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_archive_path": source,
                "row_index": "0",
                "current_rank": "1",
                "strength": "0.25",
                "current_ms_ssim": "0.62",
                "current_lpips": "0.58",
                "flow_ms_ssim": "0.66",
                "flow_energy_ms_ssim": "0.70",
                "best_stage": "flow_energy",
                "best_ms_ssim": "0.70",
                "flow_synthetic_gray_png": str(flow_png),
                "flow_energy_synthetic_gray_png": str(energy_png),
            }
        )

    enriched = tmp_path / "teacher_enriched.csv"
    assert (
        main(
            [
                "enrich-residual-teacher-maps",
                "--base-api-metrics",
                str(base_metrics),
                "--teacher-metrics",
                str(teacher_metrics),
                "--out",
                str(enriched),
            ]
        )
        == 0
    )
    enriched_rows = list(csv.DictReader(enriched.open()))
    assert enriched_rows[0]["current_map_enrichment_status"] == "ok"
    assert enriched_rows[0]["flow_map_enrichment_status"] == "ok"
    assert enriched_rows[0]["flow_energy_map_enrichment_status"] == "ok"
    assert enriched_rows[0]["current_Struct_MS-SSIM"] != ""
    assert enriched_rows[0]["flow_energy_OAC_MS-SSIM"] != ""
    assert enriched_rows[0]["flow_energy_RSC_LPIPS_PROXY"] != ""
    summary = json.loads(enriched.with_name("teacher_enriched_summary.json").read_text(encoding="utf-8"))
    assert summary["rows_with_flow_energy_struct"] == 1
    assert summary["objective_source"] == "multi_objective_maps_proxy_lpips"

    artifact = tmp_path / "selector.json"
    assert (
        main(
            [
                "train-residual-selector",
                "--base-api-metrics",
                str(base_metrics),
                "--teacher-metrics",
                str(enriched),
                "--out",
                str(artifact),
                "--holdout-fraction",
                "0",
            ]
        )
        == 0
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    examples = list(csv.DictReader(Path(payload["teacher_examples_csv"]).open()))
    assert examples[0]["objective_source"] == "multi_objective_maps"
    assert float(examples[0]["best_objective_delta"]) != float(examples[0]["best_delta"])


def test_residual_selector_uses_aggregate_map_objective_fields(tmp_path):
    teacher_metrics = tmp_path / "aggregate_probe_metrics.csv"
    with teacher_metrics.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_archive_path",
                "row_index",
                "current_rank",
                "selected_strength",
                "current_ms_ssim",
                "current_lpips",
                "flow_ms_ssim",
                "flow_energy_ms_ssim",
                "best_stage",
                "best_ms_ssim",
                "current_map_objective_score",
                "flow_map_objective_score",
                "flow_energy_map_objective_score",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_archive_path": "DATASET_PNG/Female/1990-2000/Cheek/sample_frame50.png",
                "row_index": "0",
                "current_rank": "1",
                "selected_strength": "0.38",
                "current_ms_ssim": "0.62",
                "current_lpips": "0.58",
                "flow_ms_ssim": "0.63",
                "flow_energy_ms_ssim": "0.70",
                "best_stage": "flow_energy",
                "best_ms_ssim": "0.70",
                "current_map_objective_score": "0.40",
                "flow_map_objective_score": "0.42",
                "flow_energy_map_objective_score": "0.46",
            }
        )

    from synthoct.residual_selector import load_residual_teacher_examples

    parsed = load_residual_teacher_examples([teacher_metrics], default_strength=0.25)
    assert parsed[0].objective_source == "multi_objective_map_objective"
    assert parsed[0].base_objective_score == 0.40
    assert parsed[0].flow_energy_objective_score == 0.46
    assert abs(parsed[0].best_objective_delta - 0.06) < 1e-12
    assert abs(parsed[0].best_delta - 0.08) < 1e-12


def test_cli_residual_selector_can_require_uncertainty_gated_map_safety(tmp_path):
    base_metrics = tmp_path / "base_api_metrics.csv"
    base_metrics.write_text(
        "source_archive_path,MS-SSIM,LPIPS,reference_png,phantom_path,synthetic_gray_png\n"
        "DATASET_PNG/Female/1990-2000/Cheek/safe_frame50.png,0.62,0.50,safe_ref.png,safe.txt,safe_gray.png\n"
        "DATASET_PNG/Female/1990-2000/Eye_corner/unsafe_frame50.png,0.62,0.50,unsafe_ref.png,unsafe.txt,unsafe_gray.png\n",
        encoding="utf-8",
    )
    teacher_metrics = tmp_path / "map_teacher.csv"
    fieldnames = [
        "source_archive_path",
        "row_index",
        "current_rank",
        "strength",
        "current_ms_ssim",
        "current_lpips",
        "flow_ms_ssim",
        "flow_energy_ms_ssim",
        "best_stage",
        "best_ms_ssim",
        "current_Struct_MS-SSIM",
        "current_OAC_MS-SSIM",
        "current_SC_MS-SSIM",
        "current_RSC_MS-SSIM",
        "flow_energy_Struct_MS-SSIM",
        "flow_energy_OAC_MS-SSIM",
        "flow_energy_SC_MS-SSIM",
        "flow_energy_RSC_MS-SSIM",
    ]
    with teacher_metrics.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "source_archive_path": "DATASET_PNG/Female/1990-2000/Cheek/safe_frame50.png",
                "row_index": "0",
                "current_rank": "1",
                "strength": "0.25",
                "current_ms_ssim": "0.62",
                "current_lpips": "0.50",
                "flow_ms_ssim": "0.64",
                "flow_energy_ms_ssim": "0.66",
                "best_stage": "flow_energy",
                "best_ms_ssim": "0.66",
                "current_Struct_MS-SSIM": "0.70",
                "current_OAC_MS-SSIM": "0.70",
                "current_SC_MS-SSIM": "0.70",
                "current_RSC_MS-SSIM": "0.70",
                "flow_energy_Struct_MS-SSIM": "0.72",
                "flow_energy_OAC_MS-SSIM": "0.71",
                "flow_energy_SC_MS-SSIM": "0.715",
                "flow_energy_RSC_MS-SSIM": "0.705",
            }
        )
        writer.writerow(
            {
                "source_archive_path": "DATASET_PNG/Female/1990-2000/Eye_corner/unsafe_frame50.png",
                "row_index": "1",
                "current_rank": "2",
                "strength": "0.25",
                "current_ms_ssim": "0.62",
                "current_lpips": "0.50",
                "flow_ms_ssim": "0.65",
                "flow_energy_ms_ssim": "0.68",
                "best_stage": "flow_energy",
                "best_ms_ssim": "0.68",
                "current_Struct_MS-SSIM": "0.70",
                "current_OAC_MS-SSIM": "0.70",
                "current_SC_MS-SSIM": "0.70",
                "current_RSC_MS-SSIM": "0.70",
                "flow_energy_Struct_MS-SSIM": "0.735",
                "flow_energy_OAC_MS-SSIM": "0.675",
                "flow_energy_SC_MS-SSIM": "0.71",
                "flow_energy_RSC_MS-SSIM": "0.685",
            }
        )

    artifact = tmp_path / "selector.json"
    assert (
        main(
            [
                "train-residual-selector",
                "--base-api-metrics",
                str(base_metrics),
                "--teacher-metrics",
                str(teacher_metrics),
                "--out",
                str(artifact),
                "--holdout-fraction",
                "0",
                "--neighbor-count",
                "1",
            ]
        )
        == 0
    )
    examples = list(csv.DictReader(Path(json.loads(artifact.read_text(encoding="utf-8"))["teacher_examples_csv"]).open()))
    by_source = {row["source_archive_path"]: row for row in examples}
    assert float(by_source["DATASET_PNG/Female/1990-2000/Cheek/safe_frame50.png"]["flow_energy_oac_delta"]) > 0
    assert float(by_source["DATASET_PNG/Female/1990-2000/Eye_corner/unsafe_frame50.png"]["flow_energy_oac_delta"]) < 0

    queue = tmp_path / "selector_queue.csv"
    assert (
        main(
            [
                "plan-residual-selector-queue",
                "--base-api-metrics",
                str(base_metrics),
                "--artifact",
                str(artifact),
                "--out",
                str(queue),
                "--limit",
                "2",
                "--require-map-safe",
                "--min-map-delta-lcb",
                "0",
                "--min-map-safe-win-rate",
                "1.0",
            ]
        )
        == 0
    )
    rows = list(csv.DictReader(queue.open()))
    assert [row["source_archive_path"] for row in rows] == ["DATASET_PNG/Female/1990-2000/Cheek/safe_frame50.png"]
    assert rows[0]["map_safety_status"] == "map_safe_pass"
    assert float(rows[0]["expected_oac_delta_lcb"]) > 0
    assert rows[0]["map_safe_support_n"] == "1"


def test_residual_selector_distills_observed_stage2_controls(tmp_path):
    base_metrics = tmp_path / "base_api_metrics.csv"
    base_metrics.write_text(
        "source_archive_path,MS-SSIM,SSIM,LPIPS,reference_png,phantom_path,synthetic_gray_png\n"
        "DATASET_PNG/Female/1990-2000/Cheek/sample_frame50.png,0.62,0.08,0.58,ref.png,base.txt,base_gray.png\n",
        encoding="utf-8",
    )
    teacher_metrics = tmp_path / "controlled_probe_metrics.csv"
    with teacher_metrics.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_archive_path",
                "row_index",
                "current_rank",
                "selected_strength",
                "current_ms_ssim",
                "current_lpips",
                "flow_ms_ssim",
                "flow_energy_ms_ssim",
                "best_stage",
                "best_ms_ssim",
                "current_map_objective_score",
                "flow_map_objective_score",
                "flow_energy_map_objective_score",
                "residual_control_policy",
                "flow_smooth_sigma",
                "flow_attachment",
                "energy_exponent",
                "energy_sigma",
                "energy_ratio_low",
                "energy_ratio_high",
                "energy_clip_low",
                "energy_clip_high",
                "texture_mean_exponent",
                "texture_exponent",
                "texture_deep_exponent",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_archive_path": "DATASET_PNG/Female/1990-2000/Cheek/sample_frame50.png",
                "row_index": "0",
                "current_rank": "1",
                "selected_strength": "0.38",
                "current_ms_ssim": "0.62",
                "current_lpips": "0.58",
                "flow_ms_ssim": "0.64",
                "flow_energy_ms_ssim": "0.70",
                "best_stage": "flow_energy",
                "best_ms_ssim": "0.70",
                "current_map_objective_score": "0.40",
                "flow_map_objective_score": "0.43",
                "flow_energy_map_objective_score": "0.49",
                "residual_control_policy": "teacher_neighbor_physics_dl_v2",
                "flow_smooth_sigma": "1.80",
                "flow_attachment": "4.90",
                "energy_exponent": "1.10",
                "energy_sigma": "2.90",
                "energy_ratio_low": "0.57",
                "energy_ratio_high": "1.36",
                "energy_clip_low": "0.70",
                "energy_clip_high": "1.24",
                "texture_mean_exponent": "0.05",
                "texture_exponent": "0.60",
                "texture_deep_exponent": "1.20",
            }
        )

    artifact = tmp_path / "selector.json"
    assert (
        main(
            [
                "train-residual-selector",
                "--base-api-metrics",
                str(base_metrics),
                "--teacher-metrics",
                str(teacher_metrics),
                "--out",
                str(artifact),
                "--holdout-fraction",
                "0",
                "--neighbor-count",
                "1",
            ]
        )
        == 0
    )
    examples = list(csv.DictReader(Path(json.loads(artifact.read_text(encoding="utf-8"))["teacher_examples_csv"]).open()))
    assert examples[0]["flow_smooth_sigma"] == "1.8"
    assert examples[0]["texture_deep_exponent"] == "1.2"

    queue = tmp_path / "selector_queue.csv"
    assert (
        main(
            [
                "plan-residual-selector-queue",
                "--base-api-metrics",
                str(base_metrics),
                "--artifact",
                str(artifact),
                "--out",
                str(queue),
                "--limit",
                "1",
            ]
        )
        == 0
    )
    row = next(csv.DictReader(queue.open()))
    assert row["residual_control_policy"] == "teacher_neighbor_feedback_distilled_v3"
    assert float(row["flow_smooth_sigma"]) > 1.55
    assert float(row["flow_attachment"]) < 5.75
    assert float(row["energy_exponent"]) > 0.85
    assert float(row["texture_deep_exponent"]) > 0.85


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
