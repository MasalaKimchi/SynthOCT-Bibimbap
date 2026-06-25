from __future__ import annotations

import csv
import io
import zipfile

import numpy as np
from skimage import io as skio

from synthoct.api import prepare_preliminary_png_pairs, resolve_api_key, write_preliminary_upload_plan
from synthoct.cli import main
from synthoct.submission import prepare_submission_bundle, validate_phantom_submission


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
    validation = list(csv.DictReader(bundle.validation_csv.open()))
    assert validation[0]["columns"] == "4"
    assert validation[0]["row_count_ok"] == "1"


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
            fieldnames=["source_archive_path", "reference_png", "synthetic_png", "synthetic_gray_png", "phantom_path", "request_id", "MS-SSIM", "SSIM"],
        )
        writer.writeheader()
        writer.writerow({"source_archive_path": "low.png", "MS-SSIM": "0.1", "SSIM": "0.2"})
        writer.writerow({"source_archive_path": "high.png", "MS-SSIM": "0.9", "SSIM": "0.8"})

    plan = write_preliminary_upload_plan(metrics, tmp_path / "plan.csv")
    rows = list(csv.DictReader(plan.open()))
    assert rows[0]["source_archive_path"] == "high.png"
    assert rows[0]["upload_order"] == "1"


def test_prepare_preliminary_png_pairs_copies_ranked_files(tmp_path):
    synthetic = tmp_path / "source_synthetic.png"
    reference = tmp_path / "source_reference.png"
    synthetic.write_bytes(b"synthetic")
    reference.write_bytes(b"reference")
    plan = tmp_path / "plan.csv"
    with plan.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["upload_order", "source_archive_path", "synthetic_png", "reference_png", "MS-SSIM", "SSIM", "LPIPS_PROXY"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "upload_order": "1",
                "source_archive_path": "DATASET_PNG/Female/scan_frame1.png",
                "synthetic_png": str(synthetic),
                "reference_png": str(reference),
                "MS-SSIM": "0.5",
            }
        )

    manifest = prepare_preliminary_png_pairs(plan, tmp_path / "pairs")
    rows = list(csv.DictReader(manifest.open()))
    assert len(rows) == 1
    assert (tmp_path / "pairs" / "synthetic_scans" / "001_scan_frame1_synthetic.png").read_bytes() == b"synthetic"
    assert (tmp_path / "pairs" / "real_reference_scans" / "001_scan_frame1_reference.png").read_bytes() == b"reference"
