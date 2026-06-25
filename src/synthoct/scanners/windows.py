from __future__ import annotations

import subprocess
from pathlib import Path

from synthoct.phantom import ExperimentConfig

from .core import ScannerError
from .config import write_scanner_config


class WindowsExecutableScanner:
    """Official local validation backend for Windows Part2_Scanner.exe."""

    def __init__(
        self,
        scanner_exe: str | Path = "Part2_Scanner.exe",
        config: ExperimentConfig = ExperimentConfig(),
    ) -> None:
        self.scanner_exe = Path(scanner_exe)
        self.config = config

    def render(self, phantom_path: str | Path, config_path: str | Path, output_png: str | Path) -> Path:
        phantom_path = Path(phantom_path)
        config_path = Path(config_path)
        output_png = Path(output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        if not self.scanner_exe.exists():
            raise ScannerError(f"Scanner executable not found: {self.scanner_exe}")
        write_scanner_config(config_path, phantom_path, output_png, config=self.config)
        result = subprocess.run(
            [str(self.scanner_exe), str(config_path), str(phantom_path), str(output_png)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise ScannerError(result.stderr or result.stdout or "Scanner failed.")
        return output_png
