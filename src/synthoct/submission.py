from __future__ import annotations

import csv
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from skimage import io

from .baselines import FINAL_CONFIG_NAME, final_baseline
from .dataset import iter_records, load_scan_from_zip


def safe_stem(archive_path: str) -> str:
    return Path(archive_path).with_suffix("").as_posix().replace("/", "__")


def prepare_phantom_submission(
    zip_path: str | Path,
    out_dir: str | Path,
    scatterers_count: int = 300_000,
    limit: int | None = None,
    seed: int = 7,
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    phantom_dir = out_dir / "phantoms"
    phantom_dir.mkdir(parents=True, exist_ok=True)
    records = [record for record in iter_records(zip_path) if record.modality == "png"]
    if limit is not None:
        records = records[:limit]

    manifest_path = out_dir / "submission_manifest.csv"
    with tempfile.TemporaryDirectory(prefix="synthoct-submit-") as tmp_name, manifest_path.open("w", newline="") as f:
        tmp = Path(tmp_name)
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_archive_path",
                "phantom_path",
                "sex",
                "age_band",
                "body_site",
                "subject_key",
                "frame",
                "scatterers_count",
            ],
        )
        writer.writeheader()
        for idx, record in enumerate(records):
            arr = load_scan_from_zip(zip_path, record.archive_path)
            ref_path = tmp / f"{idx:04d}.png"
            io.imsave(ref_path, np.clip(arr, 0, 255).astype(np.uint8))
            phantom_path = phantom_dir / f"{safe_stem(record.archive_path)}__{FINAL_CONFIG_NAME.split('_', 1)[0]}.txt"
            final_baseline(ref_path, phantom_path, seed=seed + idx, scatterers_count=scatterers_count)
            writer.writerow(
                {
                    "source_archive_path": record.archive_path,
                    "phantom_path": phantom_path.relative_to(out_dir).as_posix(),
                    "sex": record.sex,
                    "age_band": record.age_band,
                    "body_site": record.body_site,
                    "subject_key": record.subject_key,
                    "frame": record.frame,
                    "scatterers_count": scatterers_count,
                }
            )

    zip_out = out_dir / f"synthoct_{FINAL_CONFIG_NAME.split('_', 1)[0].lower()}_phantoms.zip"
    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(manifest_path, manifest_path.name)
        for phantom_path in sorted(phantom_dir.glob("*.txt")):
            zf.write(phantom_path, phantom_path.relative_to(out_dir).as_posix())
    return manifest_path, zip_out


def prepare_code_submission(repo_root: str | Path, out_dir: str | Path) -> Path:
    repo_root = Path(repo_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_out = out_dir / "synthoct_bibimbap_code_submission.zip"
    include_roots = ["src", "tests", "docs", "configs"]
    include_files = ["README.md", "pyproject.toml", "requirements.txt", "environment.yml", ".gitignore"]
    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in include_files:
            path = repo_root / rel
            if path.exists():
                zf.write(path, rel)
        for root in include_roots:
            base = repo_root / root
            if not base.exists():
                continue
            for path in base.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    zf.write(path, path.relative_to(repo_root).as_posix())
    return zip_out
