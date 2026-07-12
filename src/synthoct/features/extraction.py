from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage import io

HPX = 6.0
WINDOW_SIZE = 20
ORGANIZER_MAP_MODE = "organizer-compatible-v1"
SCIENTIFIC_MAP_MODE = "scientific-v1"
MapMode = Literal["organizer-compatible-v1", "scientific-v1"]


@dataclass(frozen=True)
class ScientificMapConfig:
    """Calibration and display contract for physically interpretable map audits.

    The organizer's published Part3 processor is intentionally preserved in a
    separate compatibility mode.  This configuration follows the scanner's
    documented 51 dB range, retains float maps and explicit validity masks,
    and uses fixed display scales so reference and prediction images remain
    comparable.
    """

    dynamic_range_db: float = 51.0
    oac_display_min: float = 0.0
    oac_display_max: float = 0.1
    sc_display_min: float = 0.5
    sc_display_max: float = 5.0
    rsc_display_min: float = 0.5
    rsc_display_max: float = 5.0
    oac_invalid_tail_rows: int = 1
    tissue_display_threshold: float = 0.1
    tissue_smoothing_sigma: float = 1.25
    oac_min_denominator_fraction: float = 1e-6

    def validate(self) -> None:
        values = np.asarray(
            [
                self.dynamic_range_db,
                self.oac_display_min,
                self.oac_display_max,
                self.sc_display_min,
                self.sc_display_max,
                self.rsc_display_min,
                self.rsc_display_max,
                self.tissue_display_threshold,
                self.tissue_smoothing_sigma,
                self.oac_min_denominator_fraction,
            ],
            dtype=float,
        )
        if not np.isfinite(values).all():
            raise ValueError("scientific map controls must be finite")
        if self.dynamic_range_db <= 0:
            raise ValueError("dynamic_range_db must be positive")
        if self.oac_display_max <= self.oac_display_min:
            raise ValueError("invalid OAC display range")
        if self.sc_display_max <= self.sc_display_min:
            raise ValueError("invalid SC display range")
        if self.rsc_display_max <= self.rsc_display_min:
            raise ValueError("invalid RSC display range")
        if not 0.0 <= self.tissue_display_threshold <= 1.0:
            raise ValueError("tissue_display_threshold must be in [0, 1]")
        if self.tissue_smoothing_sigma < 0:
            raise ValueError("tissue_smoothing_sigma must be non-negative")
        if not 0.0 < self.oac_min_denominator_fraction < 1.0:
            raise ValueError("oac_min_denominator_fraction must be in (0, 1)")
        if (
            isinstance(self.oac_invalid_tail_rows, bool)
            or not isinstance(self.oac_invalid_tail_rows, int)
            or self.oac_invalid_tail_rows < 1
        ):
            raise ValueError("oac_invalid_tail_rows must be a positive integer")


def load_scan(path: str | Path) -> np.ndarray:
    """Load a real or rendered OCT B-scan as a normalized grayscale image."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        arr = io.imread(path, as_gray=True)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.max(initial=0) > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def load_and_linearize_image(
    path: str | Path,
    *,
    dynamic_range_db: float = 40.0,
) -> np.ndarray:
    """Load a display PNG and invert its logarithmic intensity encoding.

    The default is 40 dB solely because it reproduces the published Part3
    evaluator.  Scientific processing passes 51 dB explicitly, matching the
    scanner and dataset documentation.
    """
    img = np.asarray(plt.imread(path), dtype=np.float32)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = np.mean(img, axis=2)
    if img.max(initial=0) > 1.0:
        img = img / 255.0
    return 10.0 ** (img * (dynamic_range_db / 10.0))


def calculate_oac(intensity: np.ndarray, h_px: float = HPX) -> np.ndarray:
    """Estimate the optical attenuation coefficient map from a reference scan."""
    epsilon = 1e-10
    cumsum_from_bottom = np.cumsum(intensity[::-1, :], axis=0)[::-1, :]
    denom = cumsum_from_bottom - 0.5 * intensity
    return intensity / (2 * h_px * (denom + epsilon))


def calculate_tissue_support(
    scan: np.ndarray,
    config: ScientificMapConfig,
) -> np.ndarray:
    """Estimate a reference-defined contiguous signal support per A-scan.

    The mask is intentionally conservative and deterministic.  It is not a
    biological tissue segmentation: it only prevents clipped/no-signal rows
    from dominating quantitative comparisons of derived maps.
    """
    smoothed = gaussian_filter(
        np.asarray(scan, dtype=np.float32),
        sigma=config.tissue_smoothing_sigma,
    )
    above = smoothed >= config.tissue_display_threshold
    support = np.zeros(above.shape, dtype=bool)
    for lateral in range(above.shape[1]):
        depth = np.flatnonzero(above[:, lateral])
        if depth.size:
            support[depth[0] : depth[-1] + 1, lateral] = True
    return support


def _window_contained(mask: np.ndarray, window_size: int) -> np.ndarray:
    """Return pixels whose complete square filter footprint is valid."""
    footprint_fraction = uniform_filter(
        np.asarray(mask, dtype=np.float32),
        size=window_size,
        mode="constant",
        cval=0.0,
    )
    return footprint_fraction >= 1.0 - 1e-6


def calculate_speckle_contrast_map(data: np.ndarray, window_size: int = WINDOW_SIZE) -> np.ndarray:
    """Estimate local speckle contrast over a sliding window."""
    mean_val = uniform_filter(data, size=window_size, mode="reflect")
    mean_sq_val = uniform_filter(data**2, size=window_size, mode="reflect")
    var_val = np.maximum(mean_sq_val - mean_val**2, 0)
    sc_map = np.sqrt(var_val) / (mean_val + 1e-10)
    border = window_size // 2
    if min(sc_map.shape) <= 2 * border:
        return sc_map
    cropped = sc_map[border:-border, border:-border]
    return np.pad(cropped, border, mode="edge")


def calculate_speckle_contrast_float(
    data: np.ndarray,
    window_size: int = WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an uncropped speckle map and a mask for filter-valid pixels."""
    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1:
        raise ValueError("window_size must be a positive integer")
    mean_val = uniform_filter(data, size=window_size, mode="reflect")
    mean_sq_val = uniform_filter(data**2, size=window_size, mode="reflect")
    var_val = np.maximum(mean_sq_val - mean_val**2, 0)
    result = np.sqrt(var_val) / (mean_val + 1e-10)
    valid = np.ones(result.shape, dtype=bool)
    border = window_size // 2
    if border:
        valid[:border, :] = False
        valid[-border:, :] = False
        valid[:, :border] = False
        valid[:, -border:] = False
    return np.asarray(result, dtype=np.float32), valid


def estimate_layer_boundary(scan: np.ndarray) -> int:
    """Locate the strongest smoothed depth-gradient boundary in a reference B-scan."""
    profile = scan.mean(axis=1)
    smooth = gaussian_filter(profile, sigma=max(1, len(profile) // 80))
    grad = np.abs(np.gradient(smooth))
    return int(np.argmax(grad))


def normalize_map(data: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    if vmin is None:
        vmin = float(np.min(data))
    if vmax is None:
        vmax = float(np.max(data))
    return np.clip((data - vmin) / (vmax - vmin + 1e-10), 0, 1)


def _save_fixed_scale_map(
    path: Path,
    data: np.ndarray,
    valid_mask: np.ndarray,
    *,
    vmin: float,
    vmax: float,
) -> Path:
    """Save a preview without Matplotlib's accidental second normalization."""
    preview = normalize_map(np.where(valid_mask, data, vmin), vmin=vmin, vmax=vmax)
    plt.imsave(path, preview, cmap="gray", vmin=0.0, vmax=1.0)
    return path


def calculate_scientific_map_arrays(
    input_image_path: str | Path,
    config: ScientificMapConfig | None = None,
) -> dict[str, np.ndarray]:
    """Calculate calibrated float maps and their explicit validity masks."""
    input_image_path = Path(input_image_path)
    config = config or ScientificMapConfig()
    config.validate()
    scan = load_scan(input_image_path)
    intensity = load_and_linearize_image(
        input_image_path,
        dynamic_range_db=config.dynamic_range_db,
    )
    mu_map = np.asarray(calculate_oac(intensity), dtype=np.float32)
    sc_map, sc_valid = calculate_speckle_contrast_float(intensity)
    rsc_map, rsc_valid = calculate_speckle_contrast_float(mu_map)

    tissue_support = calculate_tissue_support(scan, config)
    cumsum_from_bottom = np.cumsum(intensity[::-1, :], axis=0)[::-1, :]
    denominator = cumsum_from_bottom - 0.5 * intensity
    column_total = np.maximum(cumsum_from_bottom[0, :], 1e-10)
    denominator_reliable = (
        denominator
        >= config.oac_min_denominator_fraction * column_total[np.newaxis, :]
    )
    oac_valid = np.isfinite(mu_map) & tissue_support & denominator_reliable
    tail = config.oac_invalid_tail_rows
    if tail >= mu_map.shape[0]:
        raise ValueError("oac_invalid_tail_rows must be smaller than map height")
    oac_valid[-tail:, :] = False
    sc_valid &= np.isfinite(sc_map) & _window_contained(
        tissue_support,
        WINDOW_SIZE,
    )
    rsc_valid &= np.isfinite(rsc_map) & _window_contained(
        oac_valid,
        WINDOW_SIZE,
    )
    return {
        "OAC": mu_map,
        "SC": sc_map,
        "RSC": rsc_map,
        "TissueSupport": tissue_support,
        "OAC_valid": oac_valid,
        "SC_valid": sc_valid,
        "RSC_valid": rsc_valid,
    }


def _generate_organizer_maps(input_image_path: Path, output_dir: Path) -> dict[str, Path]:
    """Freeze the public Part3 behavior byte-for-byte for challenge estimates."""
    input_image_path = Path(input_image_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / input_image_path.stem

    intensity = load_and_linearize_image(input_image_path)
    mu_map = calculate_oac(intensity)
    sc_map = calculate_speckle_contrast_map(intensity)
    rsc_map = calculate_speckle_contrast_map(mu_map)

    paths = {
        "Struct": input_image_path,
        "OAC": base.with_name(base.name + "_OAC.png"),
        "SC": base.with_name(base.name + "_SC.png"),
        "RSC": base.with_name(base.name + "_RSC.png"),
    }
    # The published evaluator writes RGBA PNGs via Matplotlib.  This encoding is
    # metric-significant at the 1e-4 scale, so do not substitute skimage here.
    plt.imsave(paths["OAC"], normalize_map(mu_map, vmax=np.percentile(mu_map, 99)), cmap="gray")
    plt.imsave(paths["SC"], normalize_map(sc_map, vmin=0.5, vmax=5.0), cmap="gray")
    plt.imsave(paths["RSC"], normalize_map(rsc_map, vmin=0.5, vmax=5.0), cmap="gray")
    return paths


def _generate_scientific_maps(
    input_image_path: Path,
    output_dir: Path,
    config: ScientificMapConfig,
) -> dict[str, Path]:
    config.validate()
    # Never share filenames with the byte-frozen organizer path.  Keeping the
    # mode in the directory also makes visual artifacts self-identifying.
    output_dir = output_dir / SCIENTIFIC_MAP_MODE
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / input_image_path.stem
    arrays = calculate_scientific_map_arrays(input_image_path, config)
    mu_map = arrays["OAC"]
    sc_map = arrays["SC"]
    rsc_map = arrays["RSC"]
    oac_valid = arrays["OAC_valid"]
    sc_valid = arrays["SC_valid"]
    rsc_valid = arrays["RSC_valid"]
    tail = config.oac_invalid_tail_rows

    paths = {
        "Struct": input_image_path,
        "OAC": base.with_name(base.name + "_OAC.png"),
        "SC": base.with_name(base.name + "_SC.png"),
        "RSC": base.with_name(base.name + "_RSC.png"),
        "FloatMaps": base.with_name(base.name + "_scientific_maps.npz"),
        "Metadata": base.with_name(base.name + "_scientific_maps.json"),
    }
    _save_fixed_scale_map(
        paths["OAC"],
        mu_map,
        oac_valid,
        vmin=config.oac_display_min,
        vmax=config.oac_display_max,
    )
    _save_fixed_scale_map(
        paths["SC"],
        sc_map,
        sc_valid,
        vmin=config.sc_display_min,
        vmax=config.sc_display_max,
    )
    _save_fixed_scale_map(
        paths["RSC"],
        rsc_map,
        rsc_valid,
        vmin=config.rsc_display_min,
        vmax=config.rsc_display_max,
    )
    np.savez_compressed(
        paths["FloatMaps"],
        **arrays,
    )
    metadata = {
        "schema_version": 1,
        "mode": SCIENTIFIC_MAP_MODE,
        "config": asdict(config),
        "float_archive": paths["FloatMaps"].name,
        "units": {"OAC": "1/micrometer", "SC": "dimensionless", "RSC": "dimensionless"},
        "validity": {
            "TissueSupport": "contiguous per-column support above the smoothed display threshold",
            "OAC": f"tissue support, reliable denominator, and no final {tail} estimator row(s)",
            "SC": "complete local window inside tissue support",
            "RSC": "complete local window inside the valid OAC support",
        },
        "note": (
            "PNG files are fixed-scale previews; quantitative analysis must "
            "use the float archive and reference-defined masks. These are "
            "derived views, not independent physical ground truth."
        ),
    }
    paths["Metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def generate_maps(
    input_image_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    mode: MapMode = ORGANIZER_MAP_MODE,
    scientific_config: ScientificMapConfig | None = None,
) -> dict[str, Path]:
    """Generate organizer-compatible or scientifically calibrated feature maps.

    ``organizer-compatible-v1`` is the default and intentionally retains the
    public evaluator's 40 dB transform and save-time autoscaling.  Use
    ``scientific-v1`` for calibrated float maps, masks, and fixed-scale PNG
    previews; never mix the two modes in one score.
    """
    input_path = Path(input_image_path)
    target_dir = Path(output_dir) if output_dir else input_path.parent
    if mode == ORGANIZER_MAP_MODE:
        if scientific_config is not None:
            raise ValueError("scientific_config is valid only in scientific-v1 mode")
        return _generate_organizer_maps(input_path, target_dir)
    if mode == SCIENTIFIC_MAP_MODE:
        return _generate_scientific_maps(
            input_path,
            target_dir,
            scientific_config or ScientificMapConfig(),
        )
    raise ValueError(f"unsupported map mode: {mode}")
