from __future__ import annotations

import hashlib
import io
import itertools
import json
import shutil
import time
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .phantom import ExperimentConfig, validate_phantom

if TYPE_CHECKING:
    from .holographic_inverse import HolographicInverseConfig


SUBMISSION_SCHEMA_VERSION = 1
DEFAULT_PRELIMINARY_COUNT = 60


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_reference(reference: str | Path, reference_root: str | Path) -> str:
    reference = Path(reference)
    reference_root = Path(reference_root)
    relative = reference.relative_to(reference_root)
    archive_name = "_".join(relative.with_suffix(".txt").parts)
    _validate_archive_name(archive_name)
    return archive_name


def _validate_archive_name(name: str) -> None:
    if (
        not name
        or "/" in name
        or "\\" in name
        or name.startswith((".", "__MACOSX"))
        or not name.endswith(".txt")
    ):
        raise ValueError(f"unsafe submission member name: {name!r}")


def _validate_archive_names(names: list[str]) -> None:
    for name in names:
        _validate_archive_name(name)
    if len(names) != len(set(names)):
        raise ValueError("ZIP contains duplicate member names")
    folded = [name.casefold() for name in names]
    if len(folded) != len(set(folded)):
        raise ValueError("ZIP contains case-insensitive member-name collisions")


def _reference_inventory(reference_root: Path) -> list[Path]:
    references = [
        path
        for path in sorted(reference_root.rglob("*.png"))
        if not path.stem.endswith(("_OAC", "_SC", "_RSC"))
    ]
    if not references:
        raise ValueError(f"no reference PNGs found below {reference_root}")
    flattened = [flatten_reference(path, reference_root) for path in references]
    folded = [name.casefold() for name in flattened]
    duplicates = [name for name, count in Counter(folded).items() if count > 1]
    if duplicates:
        raise ValueError(f"flattened reference-name collisions: {duplicates}")
    return references


def _stratum(reference: Path, reference_root: Path) -> tuple[str, str, str]:
    parts = reference.relative_to(reference_root).parts
    if len(parts) < 4:
        raise ValueError(
            "expected references below sex/age/site directories, got "
            f"{reference.relative_to(reference_root)}"
        )
    return parts[0], parts[1], parts[2]


def _balanced_quotas(
    groups: list[tuple[str, str, str]],
    count: int,
    capacities: dict[tuple[str, str, str], int],
) -> dict[tuple[str, str, str], int]:
    if count <= 0:
        raise ValueError("submission count must be positive")
    if count > sum(capacities.values()):
        raise ValueError("submission count exceeds the reference inventory")
    base, remainder = divmod(count, len(groups))
    if any(capacities[group] < base for group in groups):
        raise ValueError("a stratum cannot satisfy the balanced base quota")
    if remainder == 0:
        return {group: base for group in groups}

    candidates = [group for group in groups if capacities[group] > base]
    if len(candidates) < remainder:
        raise ValueError("insufficient stratum capacity for the requested count")

    levels = [sorted({group[axis] for group in groups}) for axis in range(3)]
    targets = [count / len(axis_levels) for axis_levels in levels]

    def score(extra_groups: tuple[tuple[str, str, str], ...]) -> tuple[float, tuple]:
        extra = set(extra_groups)
        quotas = {group: base + int(group in extra) for group in groups}
        imbalance = 0.0
        for axis, axis_levels in enumerate(levels):
            for value in axis_levels:
                marginal = sum(
                    quota for group, quota in quotas.items() if group[axis] == value
                )
                imbalance += (marginal - targets[axis]) ** 2
        return imbalance, extra_groups

    # The public corpus has eight strata, so an exhaustive tie-safe choice is
    # inexpensive and yields exact sex/age/site marginals for 60 of 120 cases.
    if len(candidates) <= 24:
        best = min(itertools.combinations(candidates, remainder), key=score)
    else:
        best = tuple(candidates[:remainder])
    extras = set(best)
    return {group: base + int(group in extras) for group in groups}


def select_balanced_references(
    reference_root: str | Path,
    *,
    count: int = DEFAULT_PRELIMINARY_COUNT,
    seed: int = 2026,
) -> list[Path]:
    reference_root = Path(reference_root).resolve()
    references = _reference_inventory(reference_root)
    grouped: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    for reference in references:
        grouped[_stratum(reference, reference_root)].append(reference)
    groups = sorted(grouped)
    quotas = _balanced_quotas(
        groups,
        count,
        {group: len(grouped[group]) for group in groups},
    )

    selected: list[Path] = []
    for group in groups:
        ranked = sorted(
            grouped[group],
            key=lambda path: hashlib.sha256(
                f"{seed}\0{path.relative_to(reference_root).as_posix()}".encode()
            ).hexdigest(),
        )
        selected.extend(ranked[: quotas[group]])
    return sorted(selected, key=lambda path: flatten_reference(path, reference_root))


def audit_phantom_file(
    path: str | Path,
    *,
    config: ExperimentConfig = ExperimentConfig(),
) -> dict[str, object]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"phantom must be a regular non-symlink file: {path}")
    try:
        data = np.loadtxt(path, dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"could not parse four numeric columns in {path}") from exc
    if data.ndim == 1:
        data = data.reshape(1, -1)
    validate_phantom(
        data,
        config=config,
        require_exact_count=True,
        require_y_bounds=True,
    )
    return {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "minimum": [float(value) for value in np.min(data, axis=0)],
        "maximum": [float(value) for value in np.max(data, axis=0)],
        "zero_energy_rows": int(np.count_nonzero(data[:, 3] == 0.0)),
    }


def _write_reproducible_zip(
    archive: Path,
    selected: list[tuple[str, Path]],
    expected: dict[str, dict[str, object]],
) -> None:
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as handle:
            for archive_name, source in sorted(selected):
                info = zipfile.ZipInfo(archive_name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                # ZipFile.open() reads the per-member level from ZipInfo in
                # Python 3.12; otherwise it silently falls back to zlib level 6.
                info._compresslevel = 9
                info.external_attr = 0o100644 << 16
                info.create_system = 3
                with source.open("rb") as source_handle, handle.open(info, "w") as target:
                    shutil.copyfileobj(source_handle, target, length=1024 * 1024)
        _verify_archive_copy(temporary, expected)
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_archive_copy(
    archive: Path,
    expected: dict[str, dict[str, object]],
) -> None:
    with zipfile.ZipFile(archive) as handle:
        bad = handle.testzip()
        if bad is not None:
            raise ValueError(f"ZIP CRC validation failed for {bad}")
        infos = handle.infolist()
        names = [info.filename for info in infos]
        _validate_archive_names(names)
        if set(names) != set(expected):
            raise ValueError("ZIP member set differs from the validated selection")
        for info in infos:
            if info.is_dir():
                raise ValueError(f"ZIP members must be files: {info.filename}")
            digest = hashlib.sha256()
            with handle.open(info) as member:
                for chunk in iter(lambda: member.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected[info.filename]["sha256"]:
                raise ValueError(f"ZIP content hash mismatch: {info.filename}")


def generate_phantom_batch(
    reference_root: str | Path,
    output_dir: str | Path,
    *,
    inverse: HolographicInverseConfig | None = None,
    seed: int = 7,
    scatterers_count: int = 300_000,
    limit: int | None = None,
    retain_diagnostics: bool = True,
    scanner: ExperimentConfig | None = None,
) -> tuple[Path, Path]:
    """Generate and retain one scanner-format phantom per reference B-scan.

    Every reference PNG below ``reference_root`` is inverted with the fixed
    holographic method and written to ``output_dir`` under the flattened
    ``sex_age_site_name.txt`` name produced by :func:`flatten_reference`.  The
    resulting directory is therefore a drop-in ``--phantom-dir`` for
    :func:`build_preliminary_submission`, and the same command regenerates every
    phantom for a fresh (for example, hidden-test) reference set.

    Unlike :func:`synthoct.benchmark.run_local_benchmark`, which renders and
    scores in a throwaway directory, this keeps the phantom text files and writes
    a fail-closed ``generation_summary.json`` provenance manifest.  Each retained
    phantom is re-parsed with :func:`audit_phantom_file`, so a completed run
    guarantees the exact 300k-row / four-column scanner contract on every file.
    Returns ``(output_dir, summary_path)``.
    """
    from dataclasses import asdict, replace

    from .holographic_inverse import HolographicInverseConfig, holographic_inverse_phantom
    from .provenance import dependency_versions, git_provenance

    reference_root = Path(reference_root).resolve()
    output_dir = Path(output_dir).resolve()
    inverse = inverse or HolographicInverseConfig()
    inverse.validate()
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    contract = (
        replace(scanner, scatterers_count=scatterers_count)
        if scanner is not None
        else ExperimentConfig(scatterers_count=scatterers_count)
    )
    if inverse.phase_encoding == "dispersion-canceling-pair":
        method = "holographic-inverse-v3-phase-pair"
    elif inverse.phase_iterations:
        method = "holographic-inverse-v2-phase-retrieval"
    else:
        method = "holographic-inverse-v1"

    references = _reference_inventory(reference_root)
    if limit is not None:
        references = references[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    if retain_diagnostics:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "generation_summary.json"
    partial_summary_path = output_dir / "generation_summary.partial.json"
    summary_path.unlink(missing_ok=True)
    partial_summary_path.unlink(missing_ok=True)

    def _payload(records: list[dict[str, object]], *, complete: bool, elapsed: float) -> dict[str, object]:
        return {
            "schema_version": SUBMISSION_SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "status": "generation_complete" if complete else "generation_partial",
            "method": method,
            "reference_root": str(reference_root),
            "n": len(records),
            "requested_limit": limit,
            "seed": seed,
            "parameters": asdict(inverse),
            "scanner_contract": {
                "rows_per_phantom": scatterers_count,
                "columns": ["X_micrometer", "Y_micrometer", "Z_micrometer", "Energy_percent"],
                "x_bounds": [-contract.x_max / 2.0, contract.x_max / 2.0],
                "y_bounds": [-contract.beam_radius, contract.beam_radius],
                "z_bounds": [0.0, contract.z_max],
                "energy_bounds": [0.0, 100.0],
            },
            "environment": {
                "dependencies": dependency_versions(),
                "git": git_provenance(),
            },
            "elapsed_seconds": elapsed,
            "files": records,
        }

    def _atomic_write(path: Path, payload: dict[str, object]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    records: list[dict[str, object]] = []
    started = time.perf_counter()
    for index, reference in enumerate(references, start=1):
        archive_name = flatten_reference(reference, reference_root)
        phantom_path = output_dir / archive_name
        diagnostics_path = (
            diagnostics_dir / f"{Path(archive_name).stem}.diagnostics.json"
            if retain_diagnostics
            else None
        )
        generation_start = time.perf_counter()
        holographic_inverse_phantom(
            reference,
            phantom_path,
            seed=seed,
            scatterers_count=scatterers_count,
            inverse=inverse,
            diagnostics_path=diagnostics_path,
            scanner=scanner,
        )
        generation_seconds = time.perf_counter() - generation_start
        audit = audit_phantom_file(phantom_path, config=contract)
        records.append(
            {
                "archive_name": archive_name,
                "reference": reference.relative_to(reference_root).as_posix(),
                "reference_sha256": sha256_file(reference),
                "generation_seconds": generation_seconds,
                **audit,
            }
        )
        _atomic_write(
            partial_summary_path,
            _payload(records, complete=False, elapsed=time.perf_counter() - started),
        )
        print(
            f"[{index}/{len(references)}] {archive_name} rows={audit['rows']}",
            flush=True,
        )

    _atomic_write(
        summary_path,
        _payload(records, complete=True, elapsed=time.perf_counter() - started),
    )
    partial_summary_path.unlink(missing_ok=True)
    return output_dir, summary_path


def build_preliminary_submission(
    reference_root: str | Path,
    phantom_dir: str | Path,
    archive: str | Path,
    *,
    count: int = DEFAULT_PRELIMINARY_COUNT,
    seed: int = 2026,
    config: ExperimentConfig = ExperimentConfig(),
    manifest_path: str | Path | None = None,
) -> tuple[Path, Path]:
    reference_root = Path(reference_root).resolve()
    phantom_dir = Path(phantom_dir).resolve()
    archive = Path(archive).resolve()
    if archive.suffix.lower() != ".zip":
        raise ValueError("submission archive must use a .zip suffix")
    if not phantom_dir.is_dir():
        raise ValueError(f"phantom directory does not exist: {phantom_dir}")
    manifest = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else archive.with_suffix(".manifest.json")
    )
    if manifest == archive:
        raise ValueError("submission archive and manifest paths must differ")

    references = _reference_inventory(reference_root)
    expected_all = {flatten_reference(path, reference_root): path for path in references}
    actual_files = sorted(phantom_dir.glob("*.txt"))
    unknown = sorted(path.name for path in actual_files if path.name not in expected_all)
    if unknown:
        raise ValueError(f"phantom files do not map to official references: {unknown}")

    selected_references = select_balanced_references(
        reference_root,
        count=count,
        seed=seed,
    )
    records: list[dict[str, object]] = []
    zip_inputs: list[tuple[str, Path]] = []
    expected_zip: dict[str, dict[str, object]] = {}
    for reference in selected_references:
        archive_name = flatten_reference(reference, reference_root)
        phantom = phantom_dir / archive_name
        if not phantom.is_file():
            raise ValueError(f"missing selected phantom: {phantom}")
        if phantom.is_symlink() or not phantom.resolve().is_relative_to(phantom_dir):
            raise ValueError(f"phantom must stay below the source directory: {phantom}")
        audit = audit_phantom_file(phantom, config=config)
        stratum = _stratum(reference, reference_root)
        record = {
            "archive_name": archive_name,
            "reference": reference.relative_to(reference_root).as_posix(),
            "reference_sha256": sha256_file(reference),
            "stratum": {"sex": stratum[0], "age": stratum[1], "site": stratum[2]},
            **audit,
        }
        records.append(record)
        zip_inputs.append((archive_name, phantom))
        expected_zip[archive_name] = audit

    _write_reproducible_zip(archive, zip_inputs, expected_zip)
    _verify_archive_copy(archive, expected_zip)
    strata = Counter(
        (record["stratum"]["sex"], record["stratum"]["age"], record["stratum"]["site"])
        for record in records
    )
    payload = {
        "schema_version": SUBMISSION_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "validated",
        "archive": {
            "path": archive.name,
            "sha256": sha256_file(archive),
            "size_bytes": archive.stat().st_size,
            "members": len(records),
            "layout": "root-flat .txt files only",
        },
        "selection": {
            "requested_count": count,
            "seed": seed,
            "reference_inventory": len(references),
            "strategy": "joint sex-age-site marginal balance with deterministic within-stratum ranking",
            "strata": {"/".join(key): value for key, value in sorted(strata.items())},
        },
        "scanner_contract": {
            "rows_per_phantom": config.scatterers_count,
            "columns": ["X_micrometer", "Y_micrometer", "Z_micrometer", "Energy_percent"],
            "x_bounds": [-config.x_max / 2.0, config.x_max / 2.0],
            "y_bounds": [-config.beam_radius, config.beam_radius],
            "z_bounds": [0.0, config.z_max],
            "energy_bounds": [0.0, 100.0],
        },
        "files": records,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest)
    return archive, manifest


def verify_submission_archive(
    archive: str | Path,
    *,
    expected_count: int = DEFAULT_PRELIMINARY_COUNT,
    config: ExperimentConfig = ExperimentConfig(),
    manifest_path: str | Path | None = None,
    reference_root: str | Path | None = None,
) -> dict[str, object]:
    """Parse and validate an existing root-flat phantom ZIP without extraction.

    Contract validation checks the archive structure and numeric scanner
    bounds.  Supplying a manifest also verifies member hashes and the archive
    hash; adding ``reference_root`` verifies the selected official inputs.
    """
    archive = Path(archive).resolve()
    if reference_root is not None and manifest_path is None:
        raise ValueError("reference_root verification requires a manifest")
    manifest: dict[str, object] | None = None
    manifest_files: dict[str, dict[str, object]] = {}
    if manifest_path is not None:
        manifest_file = Path(manifest_path).resolve()
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        manifest_files = {
            str(record["archive_name"]): record
            for record in manifest.get("files", [])
        }
        if manifest.get("archive", {}).get("sha256") != sha256_file(archive):
            raise ValueError("archive SHA-256 differs from the manifest")
    with zipfile.ZipFile(archive) as handle:
        bad = handle.testzip()
        if bad is not None:
            raise ValueError(f"ZIP CRC validation failed for {bad}")
        infos = handle.infolist()
        if len(infos) != expected_count:
            raise ValueError(f"expected {expected_count} ZIP members, found {len(infos)}")
        names = [info.filename for info in infos]
        _validate_archive_names(names)
        if manifest is not None and set(names) != set(manifest_files):
            raise ValueError("ZIP member set differs from the manifest")
        file_records = []
        for info in infos:
            if info.is_dir():
                raise ValueError(f"ZIP members must be files: {info.filename}")
            digest = hashlib.sha256()
            with handle.open(info) as member:
                for chunk in iter(lambda: member.read(1024 * 1024), b""):
                    digest.update(chunk)
            member_sha256 = digest.hexdigest()
            if (
                manifest is not None
                and member_sha256 != manifest_files[info.filename].get("sha256")
            ):
                raise ValueError(f"member SHA-256 differs from manifest: {info.filename}")
            with handle.open(info) as member:
                try:
                    data = np.loadtxt(io.BytesIO(member.read()), dtype=np.float64)
                except Exception as exc:
                    raise ValueError(f"could not parse {info.filename}") from exc
            if data.ndim == 1:
                data = data.reshape(1, -1)
            validate_phantom(
                data,
                config=config,
                require_exact_count=True,
                require_y_bounds=True,
            )
            file_records.append(
                {
                    "name": info.filename,
                    "rows": int(data.shape[0]),
                    "size_bytes": info.file_size,
                    "compressed_size_bytes": info.compress_size,
                    "sha256": member_sha256,
                }
            )
    status = "contract_validated"
    if manifest is not None:
        status = "manifest_validated"
    if reference_root is not None:
        root = Path(reference_root).resolve()
        for name, record in manifest_files.items():
            reference = (root / str(record["reference"])).resolve()
            if not reference.is_relative_to(root) or not reference.is_file():
                raise ValueError(f"manifest reference is missing or unsafe: {name}")
            if flatten_reference(reference, root) != name:
                raise ValueError(f"manifest reference/member mapping differs: {name}")
            if sha256_file(reference) != record.get("reference_sha256"):
                raise ValueError(f"reference SHA-256 differs from manifest: {reference}")
        status = "provenance_validated"
    return {
        "schema_version": SUBMISSION_SCHEMA_VERSION,
        "status": status,
        "archive": str(archive),
        "archive_sha256": sha256_file(archive),
        "members": len(file_records),
        "files": file_records,
    }
