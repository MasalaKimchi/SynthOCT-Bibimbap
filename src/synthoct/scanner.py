from __future__ import annotations

import shutil
from pathlib import Path

from .phantom import ExperimentConfig
from .scanners import ScannerError, WindowsExecutableScanner, render_phantom


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
        ini_path = output_path.with_suffix(".ini")
        return render_phantom(
            phantom_path,
            ini_path,
            output_path,
            backend=WindowsExecutableScanner(scanner_exe=scanner_exe, config=config),
        )
    raise ScannerError(f"Unknown scanner mode: {mode}")
