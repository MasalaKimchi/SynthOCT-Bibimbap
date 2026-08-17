from __future__ import annotations

from pathlib import Path

from synthoct.phantom import ExperimentConfig


def write_api_config(path: str | Path, scatterers_count: int = 300_000) -> Path:
    """Write the hosted SynthOCT API Configuration.ini payload."""
    config = ExperimentConfig(scatterers_count=scatterers_count)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "[Parameters]",
                "scan filename = Scan_Raw.bin",
                "scatterers coordinates file = Scatterers.txt",
                f"a-scan pixel numbers = {config.n_depth}",
                f"vertical pixel size mcm = {config.pixel_size_z}",
                f"central wavelength mcm = {config.wavelength}",
                f"number of a-scans in b-scan = {config.n_lateral}",
                f"xmax mcm = {config.x_max}",
                f"number of b-scans = {config.b_scans_count}",
                "ymax mcm = 0.0",
                f"beam radius mcm = {config.beam_radius}",
                f"number of scatterers in b-scan = {scatterers_count}",
                "output filename = Output.png",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path
