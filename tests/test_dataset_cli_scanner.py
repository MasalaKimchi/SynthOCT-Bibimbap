from __future__ import annotations

import io
import zipfile

import numpy as np
from skimage import io as skio

from synthoct.cli import main
from synthoct.dataset import iter_records, prepare_dataset


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
