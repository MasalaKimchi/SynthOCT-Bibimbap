from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from synthoct.dataset import iter_records, write_scan_png_from_zip
from synthoct.evaluation import select_best_candidate
from synthoct.generators import (
    FINAL_CONFIG_NAME,
    PROMISING_PIPELINE_CONFIGS,
    VISUAL_PIPELINE_CONFIGS,
    final_phantom,
    hybrid_neural_prior_phantom,
    hypothesis_phantom,
    learned_prior_phantom,
    pipeline_phantom,
)
from synthoct.learned_prior import LEARNED_PRIOR_CONFIGS
from synthoct.neural_prior import HYBRID_PRIOR_CONFIGS, HYBRID_PRIOR_METHODS, NEURAL_PRIOR_METHODS, neural_prior_phantom
from synthoct.phantom import ExperimentConfig, load_phantom


SUBMISSION_METHOD_TAG = FINAL_CONFIG_NAME.split("_", 1)[0]


@dataclass(frozen=True)
class SubmissionArtifact:
    manifest: Path
    phantom_zip: Path
    code_zip: Path
    readme: Path
    validation_csv: Path
    readiness_report: Path | None = None
    artifact_manifest: Path | None = None


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
    artifact_line = (
        "- `artifacts/`: learned-prior/model artifacts included in the code submission zip."
        if method.startswith("learned-prior") or method in NEURAL_PRIOR_METHODS or method in HYBRID_PRIOR_METHODS
        else ""
    )
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
                artifact_line,
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
    learned_prior_artifact: str | Path | None = None,
    neural_prior_artifact: str | Path | None = None,
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
            ref_path = tmp / f"{idx:04d}.png"
            write_scan_png_from_zip(zip_path, record.archive_path, ref_path)
            phantom_path = phantom_dir / f"{safe_stem(record.archive_path)}__{tag.upper()}.txt"
            if method == FINAL_CONFIG_NAME:
                final_phantom(ref_path, phantom_path, seed=seed + idx, scatterers_count=scatterers_count)
            elif method == "learned-prior" or method in LEARNED_PRIOR_CONFIGS:
                if learned_prior_artifact is None:
                    raise ValueError(f"--learned-prior-artifact is required when --method {method}.")
                learned_prior_phantom(
                    ref_path,
                    phantom_path,
                    learned_prior_artifact,
                    seed=seed + idx,
                    scatterers_count=scatterers_count,
                    **LEARNED_PRIOR_CONFIGS.get(method, {}),
                )
            elif method in NEURAL_PRIOR_METHODS:
                if neural_prior_artifact is None:
                    raise ValueError(f"--neural-prior-artifact is required when --method {method}.")
                neural_prior_phantom(
                    ref_path,
                    phantom_path,
                    neural_prior_artifact,
                    seed=seed + idx,
                    scatterers_count=scatterers_count,
                )
            elif method in HYBRID_PRIOR_METHODS:
                if learned_prior_artifact is None:
                    raise ValueError(f"--learned-prior-artifact is required when --method {method}.")
                if neural_prior_artifact is None:
                    raise ValueError(f"--neural-prior-artifact is required when --method {method}.")
                hybrid_neural_prior_phantom(
                    ref_path,
                    phantom_path,
                    learned_prior_artifact,
                    neural_prior_artifact,
                    seed=seed + idx,
                    scatterers_count=scatterers_count,
                    **HYBRID_PRIOR_CONFIGS[method],
                )
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


def prepare_code_submission(repo_root: str | Path, out_dir: str | Path, extra_files: list[str | Path] | None = None) -> Path:
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
        for extra_file in extra_files or []:
            path = Path(extra_file)
            if path.exists():
                zf.write(path, f"artifacts/{path.name}")
    return zip_out


def write_submission_readiness_report(
    out_dir: str | Path,
    *,
    method: str,
    metrics_csv: str | Path,
    baseline: str,
    min_samples_per_method: int = 2,
    require_real_lpips: bool = False,
    max_generation_seconds: float = 600.0,
    max_guardrail_regression: float = 0.0,
    strict: bool = False,
) -> Path:
    out_dir = Path(out_dir)
    selection = select_best_candidate(
        metrics_csv,
        baseline=baseline,
        min_samples_per_method=min_samples_per_method,
        require_real_lpips=require_real_lpips,
        max_generation_seconds=max_generation_seconds,
        max_guardrail_regression=max_guardrail_regression,
    )
    selected = selection.get("selected")
    ready = bool(selection.get("promote")) and selected == method
    report = {
        "status": "ready" if ready else "not_ready",
        "packaged_method": method,
        "selected_method": selected,
        "baseline": baseline,
        "metrics_csv": str(Path(metrics_csv)),
        "strict": strict,
        "selection": selection,
        "interpretation": (
            "Packaged method matches the best local promoted candidate under fair challenge evidence."
            if ready
            else "Packaged method does not match a fair-evidence promoted candidate."
        ),
    }
    report_path = out_dir / "submission_readiness_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if strict and not ready:
        raise RuntimeError(report["interpretation"])
    return report_path


def write_submission_artifact_manifest(
    out_dir: str | Path,
    *,
    method: str,
    manifest: str | Path,
    validation_csv: str | Path,
    phantom_zip: str | Path,
    code_zip: str | Path,
    readiness_report: str | Path | None = None,
) -> Path:
    """Write an output-side checksum manifest after zip files are finalized."""
    out_dir = Path(out_dir)
    manifest = Path(manifest)
    validation_csv = Path(validation_csv)
    phantom_zip = Path(phantom_zip)
    code_zip = Path(code_zip)
    readiness_path = Path(readiness_report) if readiness_report is not None else None

    manifest_rows = _csv_row_count(manifest)
    validation_rows = _csv_row_count(validation_csv)
    row_count_failures = _validation_failure_count(validation_csv)
    readiness_status = ""
    packaged_method = method
    selected_method = ""
    hidden_holdout = False
    if readiness_path is not None and readiness_path.exists():
        report = json.loads(readiness_path.read_text(encoding="utf-8"))
        readiness_status = str(report.get("status", ""))
        packaged_method = str(report.get("packaged_method", packaged_method))
        selected_method = str(report.get("selected_method", ""))
        selection = report.get("selection", {})
        if isinstance(selection, dict):
            best = selection.get("best_decision", {})
            if isinstance(best, dict):
                hidden_holdout = bool(best.get("hidden_holdout_final_score", False))

    entries = [code_zip, phantom_zip, manifest, validation_csv]
    if readiness_path is not None:
        entries.append(readiness_path)

    lines = [
        "# Submission Artifact Manifest",
        "",
        f"- Directory: `{out_dir.as_posix()}`",
        f"- Method: `{packaged_method}`",
        f"- Selected method: `{selected_method}`" if selected_method else "",
        f"- Manifest rows: `{manifest_rows}`",
        f"- Validation rows: `{validation_rows}`",
        f"- Row-count failures: `{row_count_failures}`",
        f"- Readiness status: `{readiness_status}`" if readiness_status else "",
        f"- Hidden hold-out final score: `{str(hidden_holdout).lower()}`",
        "",
        "This is a local package-integrity manifest. It is written after the zip files are finalized so the hashes below are stable.",
        "",
        "## SHA-256",
        "",
        "```text",
    ]
    lines.extend(f"{_sha256_file(path)}  {path.as_posix()}" for path in entries)
    lines.extend(
        [
            "```",
            "",
            "## Zip Integrity",
            "",
            "Both submission zip files should pass:",
            "",
            "```bash",
            f"unzip -t {phantom_zip.as_posix()}",
            f"unzip -t {code_zip.as_posix()}",
            "```",
            "",
            "Official final ranking still requires organizer execution on hidden hold-out data.",
            "",
        ]
    )
    out = out_dir / "artifact_manifest.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_row_count(path: Path) -> int:
    with path.open(newline="") as f:
        return sum(1 for _row in csv.DictReader(f))


def _validation_failure_count(path: Path) -> int:
    with path.open(newline="") as f:
        return sum(1 for row in csv.DictReader(f) if str(row.get("row_count_ok", "")) != "1")


def prepare_submission_bundle(
    zip_path: str | Path,
    repo_root: str | Path,
    out_dir: str | Path,
    scatterers_count: int = 300_000,
    limit: int | None = None,
    seed: int = 7,
    method: str = FINAL_CONFIG_NAME,
    evidence_metrics: str | Path | None = None,
    baseline: str | None = None,
    min_samples_per_method: int = 2,
    require_real_lpips: bool = False,
    max_generation_seconds: float = 600.0,
    max_guardrail_regression: float = 0.0,
    strict_evidence: bool = False,
    learned_prior_artifact: str | Path | None = None,
    neural_prior_artifact: str | Path | None = None,
) -> SubmissionArtifact:
    manifest, phantom_zip = prepare_phantom_submission(
        zip_path,
        out_dir,
        scatterers_count=scatterers_count,
        limit=limit,
        seed=seed,
        method=method,
        learned_prior_artifact=learned_prior_artifact,
        neural_prior_artifact=neural_prior_artifact,
    )
    extra_code_files = [path for path in (learned_prior_artifact, neural_prior_artifact) if path is not None] or None
    code_zip = prepare_code_submission(repo_root, out_dir, extra_files=extra_code_files)
    out_dir = Path(out_dir)
    readiness_report = None
    if evidence_metrics is not None:
        if baseline is None:
            raise ValueError("--baseline is required when --evidence-metrics is provided.")
        readiness_report = write_submission_readiness_report(
            out_dir,
            method=method,
            metrics_csv=evidence_metrics,
            baseline=baseline,
            min_samples_per_method=min_samples_per_method,
            require_real_lpips=require_real_lpips,
            max_generation_seconds=max_generation_seconds,
            max_guardrail_regression=max_guardrail_regression,
            strict=strict_evidence,
        )
        with zipfile.ZipFile(phantom_zip, "a", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(readiness_report, readiness_report.name)
    artifact_manifest = write_submission_artifact_manifest(
        out_dir,
        method=method,
        manifest=manifest,
        validation_csv=out_dir / "submission_validation.csv",
        phantom_zip=phantom_zip,
        code_zip=code_zip,
        readiness_report=readiness_report,
    )
    return SubmissionArtifact(
        manifest=manifest,
        phantom_zip=phantom_zip,
        code_zip=code_zip,
        readme=out_dir / "SUBMISSION_README.md",
        validation_csv=out_dir / "submission_validation.csv",
        readiness_report=readiness_report,
        artifact_manifest=artifact_manifest,
    )
