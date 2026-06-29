from __future__ import annotations

import csv
import io
import json
import zipfile

import numpy as np
import pytest
from skimage import io as skio

from synthoct.cli import main
from synthoct.neural_prior import train_neural_phantom_prior
from synthoct.phantom import ExperimentConfig, generate_two_layers, save_phantom
from synthoct.scanners import resolve_api_key
from synthoct.submission import prepare_preliminary_png_pairs, prepare_submission_bundle, validate_phantom_submission, write_preliminary_upload_plan


def _make_nested_png_dataset(path):
    inner_buf = io.BytesIO()
    with zipfile.ZipFile(inner_buf, "w") as inner:
        for idx in range(2):
            png_buf = io.BytesIO()
            image = np.full((16, 16), 80 + idx * 40, dtype=np.uint8)
            skio.imsave(png_buf, image, extension=".png")
            inner.writestr(f"DATASET_PNG/Female/1990-2000/Cheek/sample_frame{idx}.png", png_buf.getvalue())
    with zipfile.ZipFile(path, "w") as outer:
        outer.writestr("DATASET.zip", inner_buf.getvalue())


def test_prepare_submission_bundle_writes_official_artifacts(tmp_path):
    archive = tmp_path / "dataset.zip"
    _make_nested_png_dataset(archive)

    bundle = prepare_submission_bundle(
        archive,
        tmp_path,
        tmp_path / "submission",
        scatterers_count=64,
        limit=1,
    )

    assert bundle.manifest.exists()
    assert bundle.phantom_zip.exists()
    assert bundle.code_zip.exists()
    assert bundle.readme.exists()
    assert bundle.validation_csv.exists()
    assert bundle.artifact_manifest is not None
    assert bundle.artifact_manifest.exists()
    assert "SHA-256" in bundle.artifact_manifest.read_text(encoding="utf-8")
    validation = list(csv.DictReader(bundle.validation_csv.open()))
    assert validation[0]["columns"] == "4"
    assert validation[0]["row_count_ok"] == "1"


def test_prepare_submission_scales_float_png_references(tmp_path, monkeypatch):
    import synthoct.submission.packaging as packaging

    archive = tmp_path / "dataset.zip"
    _make_nested_png_dataset(archive)
    observed = []

    def fake_final_phantom(ref_path, phantom_path, *, seed=7, scatterers_count=300_000):
        del seed
        image = skio.imread(ref_path, as_gray=True)
        observed.append(float(image.max()))
        save_phantom(
            generate_two_layers(ExperimentConfig(scatterers_count=scatterers_count), seed=3),
            phantom_path,
            ExperimentConfig(scatterers_count=scatterers_count),
        )
        return phantom_path

    monkeypatch.setattr(packaging, "final_phantom", fake_final_phantom)

    prepare_submission_bundle(
        archive,
        tmp_path,
        tmp_path / "submission_scaled",
        scatterers_count=32,
        limit=1,
    )

    assert observed
    assert observed[0] > 1.0


def test_cli_prepare_submission_reports_artifacts(tmp_path):
    archive = tmp_path / "dataset.zip"
    _make_nested_png_dataset(archive)
    assert (
        main(
            [
                "prepare-submission",
                "--zip",
                str(archive),
                "--out",
                str(tmp_path / "submission"),
                "--scatterers-count",
                "32",
                "--limit",
                "1",
            ]
        )
        == 0
    )
    assert (tmp_path / "submission" / "submission_validation.csv").exists()


def test_prepare_submission_can_package_candidate_method(tmp_path):
    archive = tmp_path / "dataset.zip"
    _make_nested_png_dataset(archive)
    bundle = prepare_submission_bundle(
        archive,
        tmp_path,
        tmp_path / "submission_h68",
        scatterers_count=32,
        limit=1,
        method="H68_layer_map_prior",
    )

    assert bundle.phantom_zip.name == "synthoct_h68_phantoms.zip"
    assert "H68_layer_map_prior" in bundle.readme.read_text(encoding="utf-8")


def test_prepare_submission_can_package_learned_prior_artifact(tmp_path):
    archive = tmp_path / "dataset.zip"
    _make_nested_png_dataset(archive)
    artifact = tmp_path / "prior.npz"
    np.savez_compressed(
        artifact,
        density_prior=np.ones((16, 24), dtype=np.float32),
        energy_prior=np.ones((16, 24), dtype=np.float32) * 0.5,
    )

    bundle = prepare_submission_bundle(
        archive,
        tmp_path,
        tmp_path / "submission_learned_prior",
        scatterers_count=32,
        limit=1,
        method="learned-prior-balanced",
        learned_prior_artifact=artifact,
    )

    assert bundle.phantom_zip.name == "synthoct_learned-prior-balanced_phantoms.zip"
    assert "learned-prior-balanced" in bundle.readme.read_text(encoding="utf-8")
    with zipfile.ZipFile(bundle.code_zip) as zf:
        names = zf.namelist()
        assert "artifacts/prior.npz" in names
        assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)


def test_prepare_submission_can_package_neural_prior_artifact(tmp_path):
    pytest.importorskip("torch")
    archive = tmp_path / "dataset.zip"
    _make_nested_png_dataset(archive)
    outputs = tmp_path / "outputs" / "neural_training"
    outputs.mkdir(parents=True)
    reference = outputs / "reference.npy"
    rendered = outputs / "rendered.npy"
    source_phantom = outputs / "source_phantom.txt"
    np.save(reference, np.tile(np.linspace(0.1, 0.8, 32, dtype=np.float32)[:, None], (1, 48)))
    np.save(rendered, np.tile(np.linspace(0.1, 0.7, 32, dtype=np.float32)[:, None], (1, 48)))
    save_phantom(
        generate_two_layers(ExperimentConfig(scatterers_count=64), seed=19),
        source_phantom,
        ExperimentConfig(scatterers_count=64),
    )
    metrics = outputs / "internal_validation_detail.csv"
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
                "phantom_path": str(source_phantom),
                "synthetic_gray_png": str(rendered),
                "Struct_MS-SSIM": "0.70",
                "Struct_LPIPS": "0.35",
            }
        )
    artifact = tmp_path / "neural_prior.pt"
    train_neural_phantom_prior(tmp_path / "outputs", artifact, shape=(16, 24), epochs=1)

    bundle = prepare_submission_bundle(
        archive,
        tmp_path,
        tmp_path / "submission_neural_prior",
        scatterers_count=32,
        limit=1,
        method="neural-prior",
        neural_prior_artifact=artifact,
    )

    assert bundle.phantom_zip.name == "synthoct_neural-prior_phantoms.zip"
    assert "neural-prior" in bundle.readme.read_text(encoding="utf-8")
    with zipfile.ZipFile(bundle.code_zip) as zf:
        assert "artifacts/neural_prior.pt" in zf.namelist()


def _write_selection_metrics(path, promoted_method="H68_layer_map_prior"):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
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
        rows = [
            ("H61_api_low_depth_prelim", "0.70", "0.25", "0", "0", "12.0"),
            (promoted_method, "0.74", "0.24", "2", "2", "11.0"),
        ]
        for method, ms, lpips, ms_wins, lpips_wins, runtime in rows:
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


def test_prepare_submission_writes_readiness_report_for_selected_method(tmp_path):
    archive = tmp_path / "dataset.zip"
    _make_nested_png_dataset(archive)
    metrics = tmp_path / "challenge_metrics_summary.csv"
    _write_selection_metrics(metrics)

    bundle = prepare_submission_bundle(
        archive,
        tmp_path,
        tmp_path / "submission_h68_ready",
        scatterers_count=32,
        limit=1,
        method="H68_layer_map_prior",
        evidence_metrics=metrics,
        baseline="H61_api_low_depth_prelim",
        strict_evidence=True,
    )

    assert bundle.readiness_report is not None
    report = json.loads(bundle.readiness_report.read_text(encoding="utf-8"))
    assert report["status"] == "ready"
    assert report["packaged_method"] == "H68_layer_map_prior"
    artifact_text = bundle.artifact_manifest.read_text(encoding="utf-8") if bundle.artifact_manifest else ""
    assert "Readiness status: `ready`" in artifact_text
    assert "submission_readiness_report.json" in artifact_text
    with zipfile.ZipFile(bundle.phantom_zip) as zf:
        assert "submission_readiness_report.json" in zf.namelist()
    assert (
        main(
            [
                "challenge-readiness",
                "--metrics",
                str(metrics),
                "--baseline",
                "H61_api_low_depth_prelim",
                "--method",
                "H68_layer_map_prior",
                "--submission-dir",
                str(tmp_path / "submission_h68_ready"),
                "--strict",
            ]
        )
        == 0
    )


def test_prepare_submission_strict_evidence_rejects_unselected_method(tmp_path):
    archive = tmp_path / "dataset.zip"
    _make_nested_png_dataset(archive)
    metrics = tmp_path / "challenge_metrics_summary.csv"
    _write_selection_metrics(metrics, promoted_method="H67_coarse_to_fine_crisp")

    try:
        prepare_submission_bundle(
            archive,
            tmp_path,
            tmp_path / "submission_h68_rejected",
            scatterers_count=32,
            limit=1,
            method="H68_layer_map_prior",
            evidence_metrics=metrics,
            baseline="H61_api_low_depth_prelim",
            strict_evidence=True,
        )
    except RuntimeError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("Expected strict evidence packaging to reject unselected method")


def test_validate_phantom_submission_rejects_missing_manifest(tmp_path):
    try:
        validate_phantom_submission(tmp_path)
    except FileNotFoundError as exc:
        assert "submission_manifest.csv" in str(exc)
    else:
        raise AssertionError("Expected missing manifest to fail")


def test_resolve_api_key_from_env_and_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNTHOCT_API_KEY", " env-key ")
    assert resolve_api_key() == "env-key"

    monkeypatch.delenv("SYNTHOCT_API_KEY")
    key_file = tmp_path / "key.env"
    key_file.write_text("SYNTHOCT_API_KEY=file-key\n", encoding="utf-8")
    assert resolve_api_key(api_key_file=key_file) == "file-key"


def test_write_preliminary_upload_plan_sorts_by_ms_ssim(tmp_path):
    metrics = tmp_path / "api_metrics.csv"
    with metrics.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "evidence_source",
                "evidence_scope",
                "source_archive_path",
                "reference_png",
                "synthetic_png",
                "synthetic_gray_png",
                "phantom_path",
                "request_id",
                "MS-SSIM",
                "SSIM",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "evidence_source": "hosted_api_true_scanner",
                "evidence_scope": "submission_manifest_render",
                "source_archive_path": "low.png",
                "MS-SSIM": "0.1",
                "SSIM": "0.2",
            }
        )
        writer.writerow(
            {
                "evidence_source": "hosted_api_true_scanner",
                "evidence_scope": "submission_manifest_render",
                "source_archive_path": "high.png",
                "MS-SSIM": "0.9",
                "SSIM": "0.8",
            }
        )

    plan = write_preliminary_upload_plan(metrics, tmp_path / "plan.csv")
    rows = list(csv.DictReader(plan.open()))
    assert rows[0]["source_archive_path"] == "high.png"
    assert rows[0]["upload_order"] == "1"
    assert rows[0]["evidence_source"] == "hosted_api_true_scanner"
    assert rows[0]["evidence_scope"] == "submission_manifest_render"


def test_prepare_preliminary_png_pairs_copies_ranked_files(tmp_path):
    synthetic = tmp_path / "source_synthetic.png"
    synthetic_gray = tmp_path / "source_synthetic_gray.png"
    reference = tmp_path / "source_reference.png"
    synthetic.write_bytes(b"synthetic")
    synthetic_gray.write_bytes(b"synthetic-gray")
    reference.write_bytes(b"reference")
    plan = tmp_path / "plan.csv"
    with plan.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "upload_order",
                "evidence_source",
                "evidence_scope",
                "source_archive_path",
                "synthetic_png",
                "synthetic_gray_png",
                "reference_png",
                "MS-SSIM",
                "SSIM",
                "LPIPS_PROXY",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "upload_order": "1",
                "evidence_source": "hosted_api_true_scanner",
                "evidence_scope": "submission_manifest_render",
                "source_archive_path": "DATASET_PNG/Female/scan_frame1.png",
                "synthetic_png": str(synthetic),
                "synthetic_gray_png": str(synthetic_gray),
                "reference_png": str(reference),
                "MS-SSIM": "0.5",
            }
        )

    manifest = prepare_preliminary_png_pairs(plan, tmp_path / "pairs")
    rows = list(csv.DictReader(manifest.open()))
    assert len(rows) == 1
    assert rows[0]["evidence_source"] == "hosted_api_true_scanner"
    assert rows[0]["evidence_scope"] == "submission_manifest_render"
    assert (tmp_path / "pairs" / "synthetic_scans" / "001_scan_frame1_synthetic.png").read_bytes() == b"synthetic-gray"
    assert (tmp_path / "pairs" / "real_reference_scans" / "001_scan_frame1_reference.png").read_bytes() == b"reference"
