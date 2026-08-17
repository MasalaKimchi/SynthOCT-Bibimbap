from __future__ import annotations

import hashlib
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

DEPENDENCY_PACKAGES = (
    "numpy",
    "scipy",
    "scikit-image",
    "sewar",
    "matplotlib",
    "lpips",
    "torch",
)
SOURCE_TREE_PATHS = (
    ".gitignore",
    "CITATION.cff",
    "README.md",
    "baseline_format",
    "docs",
    "environment.yml",
    "experiments",
    "pyproject.toml",
    "src",
    "tests",
    "tools",
)


def dependency_versions() -> dict[str, str]:
    resolved = {}
    for package in DEPENDENCY_PACKAGES:
        try:
            resolved[package] = version(package)
        except PackageNotFoundError:
            resolved[package] = "not-installed"
    return resolved


def _source_tree_digest(root: Path) -> tuple[str, int]:
    listed = subprocess.run(
        (
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            *SOURCE_TREE_PATHS,
        ),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    paths = sorted(path for path in listed.decode().split("\0") if path)
    digest = hashlib.sha256()
    count = 0
    for relative in paths:
        path = root / relative
        if not path.is_file():
            continue
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        digest.update(b"\n")
        count += 1
    return digest.hexdigest(), count


def git_provenance(root: str | Path | None = None) -> dict[str, object]:
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]

    def run(*command: str) -> bytes:
        return subprocess.run(
            command,
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout

    try:
        commit = run("git", "rev-parse", "HEAD").decode().strip()
        status = run("git", "status", "--short")
        tracked_diff = run("git", "diff", "--binary", "HEAD")
        source_tree_sha256, source_tree_file_count = _source_tree_digest(root)
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError):
        return {
            "commit": "unavailable",
            "dirty": True,
            "tracked_diff_sha256": "unavailable",
            "source_tree_sha256": "unavailable",
            "source_tree_file_count": 0,
            "source_tree_scope": list(SOURCE_TREE_PATHS),
        }
    return {
        "commit": commit,
        "dirty": bool(status),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "source_tree_sha256": source_tree_sha256,
        "source_tree_file_count": source_tree_file_count,
        "source_tree_scope": list(SOURCE_TREE_PATHS),
    }
