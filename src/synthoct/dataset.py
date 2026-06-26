from __future__ import annotations

import csv
import io
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage import io as skio


@dataclass(frozen=True)
class ScanRecord:
    split_hint: str
    sex: str
    age_band: str
    body_site: str
    filename: str
    modality: str
    archive_path: str

    @property
    def subject_key(self) -> str:
        return self.filename.rsplit("_frame", 1)[0]

    @property
    def frame(self) -> str:
        if "_frame" not in self.filename:
            return ""
        return self.filename.rsplit("_frame", 1)[-1].split(".")[0]


def _open_inner_dataset(zip_path: str | Path) -> tuple[zipfile.ZipFile, io.BytesIO | None]:
    outer = zipfile.ZipFile(zip_path)
    names = set(outer.namelist())
    if "DATASET.zip" in names:
        data = io.BytesIO(outer.read("DATASET.zip"))
        outer.close()
        return zipfile.ZipFile(data), data
    return outer, None


def iter_records(zip_path: str | Path) -> list[ScanRecord]:
    zf, backing = _open_inner_dataset(zip_path)
    try:
        records: list[ScanRecord] = []
        for name in zf.namelist():
            lower = name.lower()
            if not (lower.endswith(".npy") or lower.endswith(".png")):
                continue
            parts = Path(name).parts
            if len(parts) < 5:
                continue
            root, sex, age_band, body_site, filename = parts[-5:]
            modality = "npy" if root == "DATASET_NPY" else "png" if root == "DATASET_PNG" else Path(filename).suffix[1:]
            records.append(
                ScanRecord(
                    split_hint="all",
                    sex=sex,
                    age_band=age_band,
                    body_site=body_site,
                    filename=filename,
                    modality=modality,
                    archive_path=name,
                )
            )
        return records
    finally:
        zf.close()
        if backing is not None:
            backing.close()


def load_scan_from_zip(zip_path: str | Path, archive_path: str) -> np.ndarray:
    zf, backing = _open_inner_dataset(zip_path)
    try:
        data = zf.read(archive_path)
        if archive_path.lower().endswith(".npy"):
            return np.load(io.BytesIO(data))
        return skio.imread(io.BytesIO(data), as_gray=True)
    finally:
        zf.close()
        if backing is not None:
            backing.close()


def make_grouped_folds(records: list[ScanRecord], folds: int = 3) -> dict[int, list[ScanRecord]]:
    """Group OCT records by subject metadata, then distribute groups across folds."""
    groups: dict[str, list[ScanRecord]] = defaultdict(list)
    for record in records:
        groups[f"{record.sex}/{record.age_band}/{record.body_site}/{record.subject_key}"].append(record)
    folded: dict[int, list[ScanRecord]] = {i: [] for i in range(folds)}
    for idx, key in enumerate(sorted(groups)):
        folded[idx % folds].extend(groups[key])
    return folded


def prepare_dataset(zip_path: str | Path, out_dir: str | Path, extract: bool = True) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = iter_records(zip_path)
    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        fieldnames = [
            "split",
            "sex",
            "age_band",
            "body_site",
            "subject_key",
            "frame",
            "filename",
            "modality",
            "archive_path",
            "local_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, record in enumerate(records):
            split = "val" if idx % 5 == 0 else "train"
            local_path = ""
            if extract:
                local = out_dir / record.archive_path
                local.parent.mkdir(parents=True, exist_ok=True)
                if not local.exists():
                    arr = load_scan_from_zip(zip_path, record.archive_path)
                    if record.archive_path.lower().endswith(".npy"):
                        np.save(local, arr)
                    else:
                        skio.imsave(local, arr.astype(np.uint8) if arr.max(initial=0) > 1 else (arr * 255).astype(np.uint8))
                local_path = str(local)
            writer.writerow(
                {
                    "split": split,
                    "sex": record.sex,
                    "age_band": record.age_band,
                    "body_site": record.body_site,
                    "subject_key": record.subject_key,
                    "frame": record.frame,
                    "filename": record.filename,
                    "modality": record.modality,
                    "archive_path": record.archive_path,
                    "local_path": local_path,
                }
            )
    return manifest_path
