from __future__ import annotations

import csv
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage import io

from synthoct.dataset import iter_records, load_scan_from_zip
from synthoct.generators import FINAL_CONFIG_NAME, PROMISING_PIPELINE_CONFIGS, VISUAL_PIPELINE_CONFIGS, final_phantom, hypothesis_phantom, pipeline_phantom
from synthoct.phantom import ExperimentConfig, load_phantom


SUBMISSION_METHOD_TAG = FINAL_CONFIG_NAME.split("_", 1)[0]


@dataclass(frozen=True)
class SubmissionArtifact:
    manifest: Path
    phantom_zip: Path
    code_zip: Path
    readme: Path
    validation_csv: Path


def safe_stem(archive_path: str) -> str:
    return Path(archive_path).with_suffix("").as_posix().replace("/", "__")


def method_tag(method: str) -> str:
    return method.split("_", 1)[0].lower()


def _validate_phantom_file(path: Path, scatterers_count: int) -> dict[str, str | int | float]:
    data = load_phantom(path)
    config = ExperimentConfig(scatterers_count=scatterers_count)
    return {
        "phantom_path": str(path),
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "expected_rows": scatterers_count,
        "row_count_ok": int(data.shape[0] == scatterers_count),
        "x_min": float(data[:, 0].min()),
        "x_max": float(data[:, 0].max()),
        "z_min": float(data[:, 2].min()),
        "z_max": float(data[:, 2].max()),
        "energy_min": float(data[:, 3].min()),
        "energy_max": float(data[:, 3].max()),
        "scanner_x_min": -config.x_max / 2,
        "scanner_x_max": config.x_max / 2,
        "scanner_z_min": 0.0,
        "scanner_z_max": config.z_max,
    }


def validate_phantom_submission(out_dir: str | Path, scatterers_count: int = 300_000) -> Path:
    out_dir = Path(out_dir)
    manifest_path = out_dir / "submission_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    rows = []
    with manifest_path.open(newline="") as f:
        for row in csv.DictReader(f):
            phantom_path = out_dir / row["phantom_path"]
            rows.append(_validate_phantom_file(phantom_path, scatterers_count=scatterers_count))

    validation_path = out_dir / "submission_validation.csv"
    if rows:
        with validation_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        validation_path.write_text("", encoding="utf-8")
    return validation_path


def write_submission_readme(out_dir: str | Path, scatterers_count: int = 300_000, method: str = FINAL_CONFIG_NAME) -> Path:
    out_dir = Path(out_dir)
    tag = method_tag(method)
    readme = out_dir / "SUBMISSION_README.md"
    readme.write_text(
        "\n".join(
            [
                "# SynthOCT Bibimbap Submission Package",
                "",
                f"Primary phantom generator: `{method}`.",
                "",
                "## Official Phantom Contract",
                "",
                "- Each primary output is a plain text scatterer table with exactly four columns: `X`, `Y`, `Z`, `Energy`.",
                "- Coordinates are in micrometers.",
                "- `X` is bounded to `[-1536, 1536]`, `Z` is bounded to `[0, 1536]`, and `Energy` is bounded to `[0, 100]`.",
                f"- Each generated package row uses `{scatterers_count}` scatterers unless a smoke-test count is explicitly supplied.",
                "- The SynthOCT scanner, not this repository, renders final OCT B-scans for scoring.",
                "",
                "## Included Files",
                "",
                "- `submission_manifest.csv`: maps each Zenodo reference B-scan to its generated phantom.",
                "- `submission_validation.csv`: local schema/bounds check for every phantom.",
                f"- `synthoct_{tag}_phantoms.zip`: challenge-format phantom archive.",
                "- `synthoct_bibimbap_code_submission.zip`: source package for final code/model review.",
                "",
                "## Preliminary Portal PNG Pairs",
                "",
                "Use the hosted SynthOCT API key from the environment or a local, untracked key file to render phantoms:",
                "",
                "```bash",
                "synthoct api-evaluate-submission \\",
                "  --zip 18095266.zip \\",
                f"  --submission-dir {out_dir.as_posix()} \\",
                f"  --out outputs/api_preliminary_{tag}",
                "```",
                "",
                "Then run `synthoct prepare-png-pairs` to copy rendered synthetic/reference PNG pairs for the preliminary portal.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return readme


def prepare_phantom_submission(
    zip_path: str | Path,
    out_dir: str | Path,
    scatterers_count: int = 300_000,
    limit: int | None = None,
    seed: int = 7,
    method: str = FINAL_CONFIG_NAME,
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    tag = method_tag(method)
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
            phantom_path = phantom_dir / f"{safe_stem(record.archive_path)}__{tag.upper()}.txt"
            if method == FINAL_CONFIG_NAME:
                final_phantom(ref_path, phantom_path, seed=seed + idx, scatterers_count=scatterers_count)
            elif method in PROMISING_PIPELINE_CONFIGS or method in VISUAL_PIPELINE_CONFIGS:
                pipeline_phantom(ref_path, phantom_path, method, seed=seed + idx, scatterers_count=scatterers_count)
            else:
                hypothesis_phantom(ref_path, phantom_path, method, seed=seed + idx, scatterers_count=scatterers_count)
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

    validation_path = validate_phantom_submission(out_dir, scatterers_count=scatterers_count)
    readme_path = write_submission_readme(out_dir, scatterers_count=scatterers_count, method=method)
    zip_out = out_dir / f"synthoct_{tag}_phantoms.zip"
    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(manifest_path, manifest_path.name)
        zf.write(validation_path, validation_path.name)
        zf.write(readme_path, readme_path.name)
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


def prepare_submission_bundle(
    zip_path: str | Path,
    repo_root: str | Path,
    out_dir: str | Path,
    scatterers_count: int = 300_000,
    limit: int | None = None,
    seed: int = 7,
    method: str = FINAL_CONFIG_NAME,
) -> SubmissionArtifact:
    manifest, phantom_zip = prepare_phantom_submission(
        zip_path,
        out_dir,
        scatterers_count=scatterers_count,
        limit=limit,
        seed=seed,
        method=method,
    )
    code_zip = prepare_code_submission(repo_root, out_dir)
    out_dir = Path(out_dir)
    return SubmissionArtifact(
        manifest=manifest,
        phantom_zip=phantom_zip,
        code_zip=code_zip,
        readme=out_dir / "SUBMISSION_README.md",
        validation_csv=out_dir / "submission_validation.csv",
    )
