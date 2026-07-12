from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _retained_files(root: Path, manifest: Path) -> list[Path]:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path != manifest)
    links = [path for path in files if path.is_symlink()]
    if links:
        names = ", ".join(str(path.relative_to(root)) for path in links)
        raise ValueError(f"refusing external or mutable symlink artifacts: {names}")
    partials = [path for path in files if ".partial." in path.name or path.name.endswith(".partial")]
    if partials:
        names = ", ".join(str(path.relative_to(root)) for path in partials)
        raise ValueError(f"refusing to manifest incomplete artifacts: {names}")
    return files


def _relative_artifact_path(path: str | Path) -> Path:
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"artifact path must stay below the evidence root: {path}")
    return relative


def build_evidence_manifest(
    root: str | Path,
    *,
    hosted_request_id: str,
    claims: dict[str, Any],
    phantom: str | Path = "phantom.txt",
) -> Path:
    root = Path(root)
    manifest = root / "evidence_manifest.json"
    phantom_relative = _relative_artifact_path(phantom)
    phantom_path = root / phantom_relative
    if not root.is_dir():
        raise ValueError(f"evidence root is not a directory: {root}")
    if not phantom_path.is_file():
        raise ValueError(f"phantom is missing: {phantom_path}")
    if phantom_path.is_symlink() or not phantom_path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"phantom must be a regular file below the evidence root: {phantom_path}")

    files = {
        path.relative_to(root).as_posix(): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in _retained_files(root, manifest)
    }
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "evidence_root": root.as_posix(),
        "hosted_request_id": hosted_request_id,
        "phantom_path": phantom_relative.as_posix(),
        "phantom_sha256": sha256_file(phantom_path),
        "official_or_hidden_score": False,
        "claims": claims,
        "files": files,
    }
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_evidence_manifest(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    manifest = root / "evidence_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    expected = payload.get("files", {})
    actual_paths = _retained_files(root, manifest)
    actual_names = {path.relative_to(root).as_posix() for path in actual_paths}
    expected_names = set(expected)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(f"manifest file-set mismatch: missing={missing}, extra={extra}")
    for path in actual_paths:
        name = path.relative_to(root).as_posix()
        record = expected[name]
        if path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"size mismatch: {name}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {name}")
    phantom = root / _relative_artifact_path(payload.get("phantom_path", "phantom.txt"))
    if sha256_file(phantom) != payload.get("phantom_sha256"):
        raise ValueError("top-level phantom SHA-256 mismatch")
    if payload.get("official_or_hidden_score") is not False:
        raise ValueError("official_or_hidden_score must be false")
    return payload
