from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from skimage import io

from .phantom import ExperimentConfig, load_phantom


class ScannerError(RuntimeError):
    pass


def run_scanner(
    phantom_path: str | Path,
    output_path: str | Path,
    mode: str = "stub",
    scanner_exe: str | Path = "Part2_Scanner.exe",
    config: ExperimentConfig = ExperimentConfig(),
    precomputed_path: str | Path | None = None,
) -> Path:
    phantom_path = Path(phantom_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "stub":
        return stub_scan(phantom_path, output_path, config=config)
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


def stub_scan(
    phantom_path: str | Path,
    output_path: str | Path,
    config: ExperimentConfig = ExperimentConfig(),
) -> Path:
    data = load_phantom(phantom_path)
    image = np.zeros((config.n_depth, config.n_lateral), dtype=np.float64)
    x_idx = np.clip(((data[:, 0] + config.x_max / 2) / config.x_max * config.n_lateral).astype(int), 0, config.n_lateral - 1)
    z_idx = np.clip((data[:, 2] / config.z_max * config.n_depth).astype(int), 0, config.n_depth - 1)
    np.add.at(image, (z_idx, x_idx), np.sqrt(np.clip(data[:, 3], 0, 100) / 100.0))

    # A tiny depth attenuation and log compression make the stub visually OCT-like.
    attenuation = np.exp(-np.linspace(0, 4, config.n_depth))[:, None]
    image = image * attenuation
    image = np.log1p(image)
    image = image / (image.max() + 1e-10)
    io.imsave(output_path, (image * 255).astype(np.uint8))
    return Path(output_path)
