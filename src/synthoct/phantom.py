from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ExperimentConfig:
    n_depth: int = 256
    n_lateral: int = 512
    pixel_size_z: float = 6.0
    pixel_size_x: float = 6.0
    wavelength: float = 1.3
    beam_radius: float = 10.0
    b_scans_count: int = 1
    scatterers_count: int = 300_000
    config_filename: str = "Configuration.ini"
    scan_filename: str = "Scan_Raw.bin"

    @property
    def z_max(self) -> float:
        return self.n_depth * self.pixel_size_z

    @property
    def x_max(self) -> float:
        return self.n_lateral * self.pixel_size_x

    def write_ini(self, path: str | Path, scatterers_path: str | Path, output_path: str | Path) -> Path:
        import configparser

        path = Path(path)
        cfg = configparser.ConfigParser()
        cfg["Parameters"] = {
            "Scan Filename": self.scan_filename,
            "Scatterers coordinates File": str(scatterers_path),
            "A-scan pixel numbers": str(self.n_depth),
            "Vertical pixel size mcm": str(self.pixel_size_z),
            "Central wavelength mcm": str(self.wavelength),
            "Number of A-scans in B-scan": str(self.n_lateral),
            "Xmax mcm": str(self.x_max),
            "Number of B-scans": str(self.b_scans_count),
            "Ymax mcm": "0.0",
            "Beam Radius mcm": str(self.beam_radius),
            "Number of scatterers in B-scan": str(self.scatterers_count),
            "Output filename": str(output_path),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            cfg.write(handle)
        return path


def validate_phantom(data: np.ndarray, config: ExperimentConfig = ExperimentConfig()) -> None:
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError(f"Expected phantom array with shape (N, 4), got {data.shape}.")
    if not np.isfinite(data).all():
        raise ValueError("Phantom contains NaN or infinite values.")
    x, _y, z, energy = data.T
    if x.min() < -config.x_max / 2 or x.max() > config.x_max / 2:
        raise ValueError("X coordinates exceed official scanner bounds.")
    if z.min() < 0 or z.max() > config.z_max:
        raise ValueError("Z coordinates exceed official scanner bounds.")
    if energy.min() < 0 or energy.max() > 100:
        raise ValueError("Energy values must be in [0, 100].")


def save_phantom(data: np.ndarray, path: str | Path, config: ExperimentConfig = ExperimentConfig()) -> Path:
    validate_phantom(data, config=config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data, fmt="%.6e")
    return path


def load_phantom(path: str | Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    validate_phantom(data)
    return data
