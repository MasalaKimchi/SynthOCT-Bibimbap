from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .phantom import ExperimentConfig


class ScannerError(RuntimeError):
    pass


def run_scanner(
    phantom_path: str | Path,
    output_path: str | Path,
    mode: str = "real",
    scanner_exe: str | Path = "Part2_Scanner.exe",
    config: ExperimentConfig = ExperimentConfig(),
    precomputed_path: str | Path | None = None,
) -> Path:
    phantom_path = Path(phantom_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "precomputed":
        if precomputed_path is None:
            raise ScannerError("--precomputed is required when mode=precomputed.")
        shutil.copyfile(precomputed_path, output_path)
        return output_path
    if mode == "real":
        scanner_exe = Path(scanner_exe)
        if not scanner_exe.exists():
            raise ScannerError(f"Scanner executable not found: {scanner_exe}")
        ini_path = output_path.with_suffix(".ini")
        config.write_ini(ini_path, phantom_path, output_path)
        result = subprocess.run(
            [str(scanner_exe), str(ini_path), str(phantom_path), str(output_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise ScannerError(result.stderr or result.stdout or "Scanner failed.")
        return output_path
    raise ScannerError(f"Unknown scanner mode: {mode}")
