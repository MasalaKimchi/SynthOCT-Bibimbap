from __future__ import annotations

import io
import zipfile

import numpy as np
from skimage import io as skio

from synthoct.cli import main
from synthoct.dataset import iter_records, prepare_dataset
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


def test_cli_official_and_stub_scan(tmp_path):
    phantom = tmp_path / "phantom.txt"
    scan = tmp_path / "scan.png"
    assert main(["baseline", "official", "--out", str(phantom), "--scatterers-count", "256"]) == 0
    assert phantom.exists()
    assert main(["scan", "--phantom", str(phantom), "--out", str(scan), "--mode", "stub"]) == 0
    assert scan.exists()


def test_internal_validation_writes_summary(tmp_path):
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
    text = summary.read_text()
    assert "physics-guided" in text
    assert "official" in text


def test_cli_final_baseline(tmp_path):
    img = np.tile(np.linspace(0, 255, 32, dtype=np.uint8), (32, 1))
    scan = tmp_path / "scan.png"
    phantom = tmp_path / "final.txt"
    skio.imsave(scan, img)
    assert main(["baseline", "final", "--input", str(scan), "--out", str(phantom), "--scatterers-count", "128"]) == 0
    assert phantom.exists()
