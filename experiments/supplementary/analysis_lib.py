"""Reviewer-response analyses for the SynthOCT camera-ready revision.

Everything here reuses the shipped pipeline operators so the numbers agree with
Tables 1-3 of the manuscript.  Nothing in this module writes into ``src/``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

# Make the repository's ``src`` layout importable when an experiment is run
# directly from any working directory, without requiring ``PYTHONPATH`` or an
# editable installation.
SCRIPT_DIR = Path(__file__).resolve().parent


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "synthoct"
        ).is_dir():
            return candidate
    raise RuntimeError(f"could not locate repository root above {start}")


REPO = _find_repo_root(SCRIPT_DIR)
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthoct.features import load_scan  # noqa: E402
from synthoct.holographic_inverse import (  # noqa: E402
    HolographicInverseConfig,
    _forward_operators,
    _inverse_operators,
)
from synthoct.phantom import ExperimentConfig  # noqa: E402
from synthoct.scanners.reference import render_reference_array  # noqa: E402

DATASET = REPO / "DATASET" / "DATASET_PNG"
OUT = SCRIPT_DIR / "results"


def scan_paths() -> list[Path]:
    return sorted(DATASET.rglob("*.png"))


def series_key(path: Path) -> str:
    """Group the three frames (50/250/450) of one acquisition series."""
    stem = path.stem
    for frame in ("_frame50", "_frame250", "_frame450"):
        if stem.endswith(frame):
            stem = stem[: -len(frame)]
            break
    return f"{path.parent.relative_to(DATASET)}/{stem}"


def one_frame_per_series() -> list[Path]:
    """A 40-scan subset with one frame per acquisition series (frame 250)."""
    by_series: dict[str, Path] = {}
    for path in scan_paths():
        key = series_key(path)
        if key not in by_series or "_frame250" in path.stem:
            by_series.setdefault(key, path)
            if "_frame250" in path.stem:
                by_series[key] = path
    return [by_series[k] for k in sorted(by_series)]


def target_magnitude(path: Path, dynamic_range_db: float = 51.0) -> np.ndarray:
    scan = np.asarray(load_scan(path), dtype=np.float64)
    return scan, np.power(10.0, (scan * dynamic_range_db - dynamic_range_db) / 20.0)


# --------------------------------------------------------------------------
# operators
# --------------------------------------------------------------------------


def operators(scanner: ExperimentConfig, inverse: HolographicInverseConfig):
    axial, lateral = _forward_operators(
        scanner.n_depth,
        scanner.n_lateral,
        scanner.pixel_size_z,
        scanner.pixel_size_x,
        scanner.wavelength,
        scanner.beam_radius,
    )
    axial_inv, lateral_inv = _inverse_operators(
        scanner.n_depth,
        scanner.n_lateral,
        scanner.pixel_size_z,
        scanner.pixel_size_x,
        scanner.wavelength,
        scanner.beam_radius,
        inverse.axial_regularization,
        inverse.lateral_regularization,
    )
    return axial, lateral, axial_inv, lateral_inv


# --------------------------------------------------------------------------
# phase initialisation (Reviewer 3, comment 1)
# --------------------------------------------------------------------------


def initial_phase(kind: str, magnitude: np.ndarray, seed: int = 0) -> np.ndarray:
    """Initial field phase for the alternating-projection iteration.

    ``zero``      the shipped neutral start.
    ``hilbert``   argument of the axial analytic signal of the field magnitude,
                  i.e. the reviewer's literal suggestion.
    ``minphase``  minus the axial Hilbert transform of log-magnitude, i.e. the
                  Kramers-Kronig / minimum-phase construction, which is the
                  physically meaningful form of the same idea.
    ``random``    control.
    """
    if kind == "zero":
        return np.zeros_like(magnitude)
    if kind == "hilbert":
        return np.angle(hilbert(magnitude, axis=0))
    if kind == "minphase":
        log_mag = np.log(np.maximum(magnitude, 1e-12))
        return -np.imag(hilbert(log_mag, axis=0))
    if kind == "minphase_neg":
        log_mag = np.log(np.maximum(magnitude, 1e-12))
        return np.imag(hilbert(log_mag, axis=0))
    if kind == "random":
        rng = np.random.default_rng(seed)
        return rng.uniform(-np.pi, np.pi, magnitude.shape)
    raise ValueError(f"unknown initialisation {kind!r}")


def solve_with_trace(
    magnitude: np.ndarray,
    scanner: ExperimentConfig,
    inverse: HolographicInverseConfig,
    *,
    init: str = "zero",
    checkpoints: tuple[int, ...] = (),
    seed: int = 0,
):
    """Run the alternating projection, recording the residual at every step.

    Returns ``(final_coefficients, residual_trace, checkpoint_coefficients)``.
    ``residual_trace[t]`` is the mean absolute magnitude error after ``t``
    completed iterations, so index 0 is the initialisation itself.
    """
    axial, lateral, axial_inv, lateral_inv = operators(scanner, inverse)
    field = magnitude * np.exp(1j * initial_phase(init, magnitude, seed=seed))
    previous_phase: np.ndarray | None = None
    residuals: list[float] = []
    saved: dict[int, np.ndarray] = {}

    def record(step: int, current_field: np.ndarray) -> np.ndarray:
        coefficients = axial_inv @ current_field @ lateral_inv.T
        projected = np.abs(axial @ coefficients @ lateral.T)
        residuals.append(float(np.mean(np.abs(projected - magnitude))))
        if step in checkpoints:
            saved[step] = coefficients
        return coefficients

    coefficients = record(0, field)
    for step in range(1, inverse.phase_iterations + 1):
        projected_field = axial @ coefficients @ lateral.T
        projected_phase = np.angle(projected_field)
        if previous_phase is not None and inverse.phase_momentum:
            delta = np.angle(np.exp(1j * (projected_phase - previous_phase)))
            target_phase = projected_phase + inverse.phase_momentum * delta
        else:
            target_phase = projected_phase
        previous_phase = projected_phase
        field = magnitude * np.exp(1j * target_phase)
        coefficients = record(step, field)
    return coefficients, np.asarray(residuals), saved


# --------------------------------------------------------------------------
# nonnegative encodings (Reviewer 3, comment 4)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EncodedPhantom:
    rows: np.ndarray
    n_rows: int
    label: str


def _voxel_centres(scanner: ExperimentConfig):
    x = (np.arange(scanner.n_lateral, dtype=np.float64) - scanner.n_lateral / 2.0) * scanner.pixel_size_x
    z = (np.arange(scanner.n_depth, dtype=np.float64) + 0.5) * scanner.pixel_size_z
    return np.tile(x, scanner.n_depth), np.repeat(z, scanner.n_lateral)


def _scale_rows(rows: np.ndarray, max_amplitude: float) -> np.ndarray:
    peak = max(float(np.abs(rows[:, 3]).max()), 1e-30)
    scale = max_amplitude / peak
    out = rows.copy()
    out[:, 3] = 100.0 * (rows[:, 3] * scale) ** 2
    return out


def encode(
    coefficients: np.ndarray,
    scanner: ExperimentConfig,
    *,
    mode: str = "pair",
    levels: int = 0,
    budget: int = 300_000,
    max_amplitude: float = 1e-3,
) -> EncodedPhantom:
    """Nonnegative encodings of the complex grid.

    ``pair``          shipped analytic phase pair with free amplitudes.
    ``single``        one amplitude/depth-adjusted scatterer per coefficient.
    ``pair_equal``    both pair members forced to amplitude a/2 at d0 and d1.
    ``pair_quantized``pair amplitudes rounded to ``levels`` log-spaced values.
    ``equal_count``   every scatterer has one global amplitude; the analytic
                      weights are approximated by integer counts under a fixed
                      row budget.
    """
    magnitude = np.abs(coefficients).ravel()
    phase = np.angle(coefficients).ravel()
    base_x, base_z = _voxel_centres(scanner)
    d0 = -phase * scanner.wavelength / (4.0 * np.pi)

    if mode == "single":
        rows = np.column_stack((base_x, np.zeros_like(base_x), base_z + d0, magnitude))
        return EncodedPhantom(_scale_rows(rows, max_amplitude), len(rows), "single")

    sign = np.where(d0 >= 0.0, 1.0, -1.0)
    d1 = d0 - sign * scanner.wavelength / 2.0
    separation = d0 - d1
    w0 = -d1 / separation
    w1 = d0 / separation

    if mode == "pair":
        a0, a1 = magnitude * w0, magnitude * w1
    elif mode == "pair_equal":
        a0 = a1 = magnitude / 2.0
    elif mode == "pair_quantized":
        a0, a1 = magnitude * w0, magnitude * w1
        both = np.concatenate([a0, a1])
        positive = both[both > 0]
        if positive.size and levels > 1:
            lo, hi = np.log10(np.quantile(positive, 1e-4)), np.log10(positive.max())
            grid = np.concatenate([[0.0], np.logspace(lo, hi, levels)])
            a0 = grid[np.abs(a0[:, None] - grid[None, :]).argmin(axis=1)]
            a1 = grid[np.abs(a1[:, None] - grid[None, :]).argmin(axis=1)]
    elif mode == "equal_count":
        return _encode_equal_count(
            magnitude, w0, w1, base_x, base_z, d0, d1, budget, max_amplitude
        )
    else:
        raise ValueError(f"unknown encoding {mode!r}")

    rows = np.empty((2 * len(magnitude), 4), dtype=np.float64)
    rows[0::2, 0] = base_x
    rows[1::2, 0] = base_x
    rows[:, 1] = 0.0
    rows[0::2, 2] = base_z + d0
    rows[1::2, 2] = base_z + d1
    rows[0::2, 3] = a0
    rows[1::2, 3] = a1
    label = mode if mode != "pair_quantized" else f"pair_quantized_{levels}"
    return EncodedPhantom(_scale_rows(rows, max_amplitude), len(rows), label)


def _encode_equal_count(
    magnitude, w0, w1, base_x, base_z, d0, d1, budget, max_amplitude
):
    """One global scatterer amplitude; analytic weights become integer counts.

    Every emitted scatterer carries the same reflection amplitude, so a voxel
    coefficient is represented by how many scatterers sit at ``d0`` and how many
    at ``d1``.  The unit amplitude is chosen so the total count matches the row
    budget.
    """
    target0 = magnitude * w0
    target1 = magnitude * w1
    total = float(target0.sum() + target1.sum())
    unit = total / budget
    n0 = np.floor(target0 / unit + 0.5).astype(np.int64)
    n1 = np.floor(target1 / unit + 0.5).astype(np.int64)
    # Spend or reclaim the rounding slack on the largest residuals.
    slack = budget - int(n0.sum() + n1.sum())
    if slack:
        residual = np.concatenate([target0 / unit - n0, target1 / unit - n1])
        order = np.argsort(-residual) if slack > 0 else np.argsort(residual)
        take = order[: abs(slack)]
        delta = 1 if slack > 0 else -1
        half = len(n0)
        for idx in take:
            if idx < half:
                if n0[idx] + delta >= 0:
                    n0[idx] += delta
            elif n1[idx - half] + delta >= 0:
                n1[idx - half] += delta

    rows = []
    for counts, depth_offset in ((n0, d0), (n1, d1)):
        keep = counts > 0
        rows.append(
            np.column_stack(
                (
                    np.repeat(base_x[keep], counts[keep]),
                    np.zeros(int(counts[keep].sum())),
                    np.repeat(base_z[keep] + depth_offset[keep], counts[keep]),
                    np.full(int(counts[keep].sum()), unit),
                )
            )
        )
    stacked = np.vstack(rows)
    return EncodedPhantom(_scale_rows(stacked, max_amplitude), len(stacked), "equal_count")


# --------------------------------------------------------------------------
# rendering / metric
# --------------------------------------------------------------------------


def render_and_score(rows: np.ndarray, reference: np.ndarray) -> tuple[float, np.ndarray]:
    from sewar.full_ref import msssim

    config = ExperimentConfig(scatterers_count=len(rows))
    image = render_reference_array(rows, config=config)
    score = float(
        np.real(msssim((reference * 255).astype(np.uint8), (image * 255).astype(np.uint8)))
    )
    return score, image


def dump(name: str, payload) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    return path
