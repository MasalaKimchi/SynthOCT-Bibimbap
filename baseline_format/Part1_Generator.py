"""Part1 (Generator) — the editable model, in the baseline's flat-script shape.

Baseline role (``SynthOCT_Baseline/Part1_Generator.py``): "Generate (X, Y, Z,
Energy%) scatterers, write a .txt phantom. The component you aim to improve."

In this submission the model that fills that slot is **phase-pair holographic
inversion**: instead of drawing scatterers from scratch, it solves the fixed
scanner's inverse problem so that the rendered B-scan reproduces a *reference*
OCT scan (the challenge's actual per-scan task).  The implementation is the
tested ``synthoct`` package; this file is a thin, baseline-shaped façade over it
so behaviour is identical to ``synthoct baseline holographic-inverse``.

Interface parity with the baseline is preserved:

* ``ExperimentConfig``            — same scanner contract (256x512, 6 µm, 1.3 µm,
                                    beam radius 10 µm), plus ``write_ini``.
* ``ScattererGenerator``          — ``save_to_file`` and the from-scratch demo
                                    generators (``generate_uniform`` /
                                    ``generate_two_layers``) used by the
                                    self-consistency check, **plus** the headline
                                    ``generate_from_reference`` inverse model.

Fixed, non-per-image parameters (axial 0.02, lateral 0.05, 200 iterations,
momentum 1.0, dispersion-canceling pair, max reflection amplitude 0.001, 51 dB
target) are the defaults of ``HolographicInverseConfig`` — the same values the
``synthoct`` CLI uses.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (must precede any `synthoct` import; sets sys.path)

import argparse
from pathlib import Path

import numpy as np

from synthoct.holographic_inverse import (
    HolographicInverseConfig,
    holographic_inverse_phantom,
)
from synthoct.phantom import ExperimentConfig as _SynthExperimentConfig
from synthoct.phantom import save_phantom

# Re-exported so callers can `from Part1_Generator import HolographicInverseConfig`.
__all__ = [
    "ExperimentConfig",
    "HolographicInverseConfig",
    "ScattererGenerator",
    "FIXED_INVERSE",
    "DEFAULT_SEED",
]

# The fixed method: dataclass defaults == the challenge configuration.
FIXED_INVERSE = HolographicInverseConfig()
DEFAULT_SEED = 7


class ExperimentConfig:
    """Scanner + phantom contract, exposing the baseline's attribute names.

    Values are identical to the official baseline's ``ExperimentConfig`` and to
    ``synthoct.phantom.ExperimentConfig``.  Kept as a light façade so baseline
    code that reads ``cfg.N_depth`` / ``cfg.x_max`` keeps working, while the
    numbers come from one authoritative place.
    """

    def __init__(
        self,
        *,
        n_depth: int = 256,
        n_lateral: int = 512,
        pixel_size_z: float = 6.0,
        pixel_size_x: float = 6.0,
        wavelength: float = 1.3,
        beam_radius: float = 10.0,
        b_scans_count: int = 1,
        scatterers_count: int = 300_000,
        config_filename: str = "Configuration.ini",
        scatterers_filename: str = "Scatterers_Exp.txt",
        output_filename: str = "Scan_Raw.bin",
    ) -> None:
        # Baseline-compatible attribute names.
        self.N_depth = n_depth
        self.N_lateral = n_lateral
        self.pixel_size_z = pixel_size_z
        self.pixel_size_x = pixel_size_x
        self.wavelength = wavelength
        self.beam_radius = beam_radius
        self.beam_diameter = beam_radius * 2.0
        self.b_scans_count = b_scans_count
        self.scatterers_count = scatterers_count
        self.config_filename = config_filename
        self.scatterers_filename = scatterers_filename
        self.output_filename = output_filename

    @property
    def z_max(self) -> float:
        return self.N_depth * self.pixel_size_z

    @property
    def x_max(self) -> float:
        return self.N_lateral * self.pixel_size_x

    def to_synthoct(self) -> _SynthExperimentConfig:
        """Return the authoritative ``synthoct`` scanner config."""
        return _SynthExperimentConfig(
            n_depth=self.N_depth,
            n_lateral=self.N_lateral,
            pixel_size_z=self.pixel_size_z,
            pixel_size_x=self.pixel_size_x,
            wavelength=self.wavelength,
            beam_radius=self.beam_radius,
            b_scans_count=self.b_scans_count,
            scatterers_count=self.scatterers_count,
            config_filename=self.config_filename,
            scan_filename=self.output_filename,
        )

    def write_ini(self, path: str | Path | None = None) -> Path:
        """Write the hosted-scanner ``Configuration.ini``.

        Part2 in this submission is the organizer-hosted challenge service, so
        the config that is actually sent is the one ``synthoct`` posts to the
        API. Delegating keeps this file byte-identical to the real request
        payload.
        """
        from synthoct.scanners import write_api_config

        target = Path(path) if path is not None else Path(self.config_filename)
        result = write_api_config(target, scatterers_count=self.scatterers_count)
        print(f"[Config] {result} generated (hosted-scanner payload).")
        return result


class ScattererGenerator:
    """Digital-phantom generator.

    ``generate_from_reference`` is the challenge model (phase-pair holographic
    inversion).  ``generate_uniform`` / ``generate_two_layers`` are the
    baseline's from-scratch demo generators, retained verbatim so the
    self-consistency check in ``Orchestrator.py`` still works.
    """

    def __init__(
        self,
        config: ExperimentConfig | None = None,
        inverse: HolographicInverseConfig | None = None,
    ) -> None:
        self.cfg = config or ExperimentConfig()
        self.inverse = inverse or FIXED_INVERSE
        self.inverse.validate()

    # --- Challenge model: reference-conditioned inverse synthesis -----------

    def generate_from_reference(
        self,
        reference_path: str | Path,
        output_path: str | Path,
        *,
        seed: int = DEFAULT_SEED,
        diagnostics_path: str | Path | None = None,
    ) -> Path:
        """Invert the fixed scanner so its render reproduces ``reference_path``.

        Delegates to ``synthoct.holographic_inverse.holographic_inverse_phantom``
        with the fixed challenge configuration; the written phantom is exactly
        the one produced by ``synthoct baseline holographic-inverse``.
        Returns the path of the saved 300,000-row ``X Y Z Energy(%)`` phantom.
        """
        return holographic_inverse_phantom(
            reference_path,
            output_path,
            seed=seed,
            scatterers_count=self.cfg.scatterers_count,
            inverse=self.inverse,
            diagnostics_path=diagnostics_path,
        )

    # --- Baseline demo generators (from-scratch), for self-consistency ------

    def generate_uniform(self, seed: int | None = None, amp: float = 1.0) -> np.ndarray:
        """Uniform scatterer cloud. ``amp`` is Backscattering Energy in % (0..100)."""
        if seed is not None:
            np.random.seed(seed)
        count = self.cfg.scatterers_count
        xs = (np.random.rand(count) - 0.5) * self.cfg.x_max
        ys = (np.random.rand(count) - 0.5) * (2 * self.cfg.beam_radius)
        zs = np.random.rand(count) * self.cfg.z_max
        amps = np.ones(count) * amp
        return np.column_stack((xs, ys, zs, amps))

    def generate_two_layers(
        self,
        seed: int | None = None,
        boundary_z_mcm: float = 500.0,
        amp_top: float = 1.0,
        amp_bottom: float = 5.0,
    ) -> np.ndarray:
        """Two-layer phantom split at ``boundary_z_mcm`` (energies in %)."""
        if seed is not None:
            np.random.seed(seed)
        data = self.generate_uniform(seed, amp=amp_top)
        zs = data[:, 2]
        data[zs > boundary_z_mcm, 3] = amp_bottom
        return data

    # --- Shared I/O ---------------------------------------------------------

    def save_to_file(self, data: np.ndarray, filename: str | Path) -> Path:
        """Write a phantom array, validating it against the scanner contract.

        Uses ``synthoct.phantom.save_phantom`` (``%.6e``; the scanner parses
        this and the baseline's ``%.4e`` identically).
        """
        return save_phantom(data, filename, config=self.cfg.to_synthoct())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="Part1_Generator",
        description=(
            "Generate a challenge digital phantom by inverting the fixed scanner "
            "against a reference OCT B-scan (phase-pair holographic inversion)."
        ),
    )
    parser.add_argument("--input", required=True, type=Path, help="Reference OCT B-scan (PNG/NPY).")
    parser.add_argument("--out", required=True, type=Path, help="Output phantom .txt path.")
    parser.add_argument("--diagnostics", type=Path, help="Optional generation diagnostics JSON.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Zero-energy filler seed.")
    parser.add_argument("--scatterers-count", type=int, default=300_000)
    args = parser.parse_args(argv)

    cfg = ExperimentConfig(scatterers_count=args.scatterers_count)
    generator = ScattererGenerator(cfg)
    result = generator.generate_from_reference(
        args.input,
        args.out,
        seed=args.seed,
        diagnostics_path=args.diagnostics,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
