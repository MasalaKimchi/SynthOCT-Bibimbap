from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from skimage import io
from skimage.transform import resize

from .evaluation import calculate_metrics
from .features import load_scan
from .phantom import ExperimentConfig, load_phantom, save_phantom


@dataclass(frozen=True)
class ScannerPair:
    label: str
    phantom_path: Path
    rendered_gray_path: Path
    ssim: float


def discover_scanner_pairs(outputs_dir: str | Path = "outputs", limit: int = 64) -> list[ScannerPair]:
    """Collect existing hosted-scanner phantom/render pairs from metrics CSVs."""
    pairs: dict[Path, ScannerPair] = {}
    csv_paths = sorted(set(Path(outputs_dir).glob("**/*metrics*.csv")) | set(Path(outputs_dir).glob("**/*validation_detail.csv")))
    for metrics_path in csv_paths:
        try:
            rows = list(csv.DictReader(metrics_path.open()))
        except Exception:
            continue
        for row in rows:
            phantom = row.get("phantom_path", "")
            rendered = row.get("synthetic_gray_png", "")
            if not phantom or not rendered:
                continue
            phantom_path = Path(phantom)
            rendered_path = Path(rendered)
            if not phantom_path.exists() or not rendered_path.exists():
                continue
            try:
                ssim = float(row.get("SSIM") or row.get("Struct_SSIM") or "nan")
            except ValueError:
                continue
            if not np.isfinite(ssim):
                continue
            pairs[phantom_path.resolve()] = ScannerPair(
                label=row.get("method") or row.get("label") or metrics_path.stem,
                phantom_path=phantom_path.resolve(),
                rendered_gray_path=rendered_path.resolve(),
                ssim=ssim,
            )
    return sorted(pairs.values(), key=lambda pair: pair.ssim, reverse=True)[:limit]


def phantom_to_field(
    phantom_path: str | Path,
    shape: tuple[int, int] = (128, 256),
    config: ExperimentConfig | None = None,
) -> np.ndarray:
    """Convert sparse scatterers into normalized density/energy/cumulative fields."""
    config = config or ExperimentConfig()
    data = load_phantom(phantom_path)
    rows, cols = shape
    xb = np.clip(((data[:, 0] / config.x_max) + 0.5) * cols, 0, cols - 1).astype(int)
    zb = np.clip((data[:, 2] / config.z_max) * rows, 0, rows - 1).astype(int)
    counts = np.zeros(shape, dtype=np.float32)
    energy = np.zeros(shape, dtype=np.float32)
    np.add.at(counts, (zb, xb), 1.0)
    np.add.at(energy, (zb, xb), data[:, 3].astype(np.float32))
    counts = gaussian_filter(counts, sigma=(0.5, 0.8))
    energy = gaussian_filter(energy, sigma=(0.5, 0.8))
    mean_energy = energy / np.maximum(counts, 1e-6)
    cumulative = np.cumsum(energy, axis=0)
    channels = [_log_normalize(counts), _log_normalize(energy), _log_normalize(mean_energy), _log_normalize(cumulative)]
    return np.stack(channels, axis=0).astype(np.float32)


def field_to_phantom(
    density: np.ndarray,
    energy: np.ndarray,
    output_path: str | Path,
    scatterers_count: int = 900_000,
    seed: int = 7,
    total_energy: float | None = None,
    energy_floor: float = 0.002,
    energy_ceiling: float = 0.075,
) -> Path:
    """Sample scanner-compatible scatterers from optimized learned fields."""
    config = ExperimentConfig(scatterers_count=scatterers_count)
    rng = np.random.default_rng(seed)
    density = np.asarray(density, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64)
    density = np.clip(density, 1e-8, None)
    density = gaussian_filter(density, sigma=(0.25, 0.4))
    density = density / density.sum()
    flat = rng.choice(density.size, size=scatterers_count, replace=True, p=density.ravel())
    z_bin, x_bin = np.divmod(flat, density.shape[1])
    xs = ((x_bin + rng.random(scatterers_count)) / density.shape[1] - 0.5) * config.x_max
    ys = (rng.random(scatterers_count) - 0.5) * (2 * config.beam_radius)
    zs = ((z_bin + rng.random(scatterers_count)) / density.shape[0]) * config.z_max
    energy_map = np.clip(energy_floor + (energy_ceiling - energy_floor) * energy, energy_floor, energy_ceiling)
    energies = energy_map[z_bin, x_bin] * rng.lognormal(mean=0.0, sigma=0.055, size=scatterers_count)
    if total_energy is not None:
        energies *= total_energy / max(energies.sum(), 1e-12)
    data = np.column_stack((xs, ys, zs, np.clip(energies, 0.001, 100.0)))
    return save_phantom(data, output_path, config=config)


def run_learned_surrogate_refinement(
    reference_path: str | Path,
    out_dir: str | Path,
    outputs_dir: str | Path = "outputs",
    base_phantom_path: str | Path | None = None,
    base_rendered_gray_path: str | Path | None = None,
    shape: tuple[int, int] = (128, 256),
    train_limit: int = 64,
    epochs: int = 160,
    optimize_steps: int = 220,
    seed: int = 23,
    scatterers_count: int = 900_000,
    energy_ratios: Iterable[float] = (0.78, 0.88, 1.0),
    texture_strengths: Iterable[float] = (0.0, 0.45),
    holdout_fraction: float = 0.2,
    anchored_residual: bool = False,
    density_residual_scale: float = 0.18,
    energy_residual_scale: float = 0.22,
    anchor_weight: float = 0.35,
    residual_kernel: int = 15,
) -> Path:
    """Train a CNN scanner surrogate and emit scanner-ready inverse candidates."""
    try:
        import torch
        from torch.nn import functional as F
    except Exception as exc:  # pragma: no cover - depends on optional environment package.
        raise RuntimeError("The learned surrogate optimizer requires torch.") from exc

    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir = Path(out_dir)
    phantom_dir = out_dir / "phantoms"
    preview_dir = out_dir / "surrogate_preview"
    phantom_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_scanner_pairs(outputs_dir, limit=train_limit)
    if not pairs:
        raise RuntimeError(f"No scanner-rendered training pairs found under {outputs_dir}.")
    train_pairs, holdout_pairs = _split_train_holdout(pairs, holdout_fraction=holdout_fraction, seed=seed)

    x_np = np.stack([phantom_to_field(pair.phantom_path, shape=shape) for pair in train_pairs], axis=0)
    y_np = np.stack([_load_resized(pair.rendered_gray_path, shape) for pair in train_pairs], axis=0)[:, None, :, :]
    target_np = _load_resized(reference_path, shape)[None, None, :, :]

    device = torch.device("cpu")
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np.astype(np.float32)).to(device)
    target = torch.from_numpy(target_np.astype(np.float32)).to(device)

    model = _ScannerSurrogate().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    weights = torch.linspace(1.0, 1.6, steps=len(train_pairs), device=device).flip(0).view(-1, 1, 1, 1)
    for _epoch in range(max(1, epochs)):
        pred = model(x)
        loss = ((pred - y).abs() * weights).mean() + 0.35 * _gradient_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    calibration_path = _write_surrogate_calibration(model, holdout_pairs, out_dir, shape=shape)
    holdout_summary = _summarize_calibration(calibration_path)

    base_pair = train_pairs[0]
    if base_phantom_path is not None:
        base_field = phantom_to_field(base_phantom_path, shape=shape)
    else:
        base_field = phantom_to_field(base_pair.phantom_path, shape=shape)
    density0 = np.clip(base_field[0], 1e-4, 0.999)
    energy0 = np.clip(base_field[1], 1e-4, 0.999)
    z_prior = torch.linspace(0.0, 1.0, steps=shape[0], device=device).view(1, 1, shape[0], 1)
    air_gate = torch.sigmoid((z_prior - 0.11) / 0.025)
    base_density_t = torch.from_numpy(density0[None, None].astype(np.float32)).to(device) * air_gate
    base_energy_t = torch.from_numpy(energy0[None, None].astype(np.float32)).to(device) * air_gate

    def lowpass(delta):
        kernel = int(residual_kernel)
        if kernel <= 1:
            return delta
        if kernel % 2 == 0:
            kernel += 1
        pad = kernel // 2
        return F.avg_pool2d(F.pad(delta, (pad, pad, pad, pad), mode="reflect"), kernel_size=kernel, stride=1)

    if anchored_residual:
        density_delta = torch.zeros_like(base_density_t, requires_grad=True)
        energy_delta = torch.zeros_like(base_energy_t, requires_grad=True)
        inv_opt = torch.optim.AdamW([density_delta, energy_delta], lr=5e-2, weight_decay=1e-4)
    else:
        density_var = torch.logit(torch.from_numpy(density0[None, None].astype(np.float32))).to(device).requires_grad_(True)
        energy_var = torch.logit(torch.from_numpy(energy0[None, None].astype(np.float32))).to(device).requires_grad_(True)
        inv_opt = torch.optim.AdamW([density_var, energy_var], lr=7e-2, weight_decay=1e-4)

    for _step in range(max(1, optimize_steps)):
        if anchored_residual:
            density_residual = lowpass(density_delta)
            energy_residual = lowpass(energy_delta)
            density = (base_density_t * torch.exp(float(density_residual_scale) * density_residual)).clamp(0.0, 1.0)
            energy = (base_energy_t * torch.exp(float(energy_residual_scale) * energy_residual)).clamp(0.0, 1.0)
            anchor_loss = (density - base_density_t).abs().mean() + (energy - base_energy_t).abs().mean()
            residual_loss = _smoothness(density_residual) + _smoothness(energy_residual)
        else:
            density = torch.sigmoid(density_var) * air_gate
            energy = torch.sigmoid(energy_var) * air_gate
            anchor_loss = torch.zeros((), device=device)
            residual_loss = torch.zeros((), device=device)
        cumulative = torch.cumsum(energy, dim=2)
        cumulative = cumulative / (cumulative.amax(dim=(2, 3), keepdim=True) + 1e-6)
        candidate = torch.cat([density, energy, energy / (density + 1e-3), cumulative], dim=1).clamp(0.0, 1.0)
        pred = model(candidate)
        loss = (
            (pred - target).abs().mean()
            + 0.40 * _gradient_loss(pred, target)
            + 0.18 * (pred.mean(dim=3) - target.mean(dim=3)).abs().mean()
            + 0.015 * _smoothness(density)
            + 0.010 * _smoothness(energy)
            + float(anchor_weight) * anchor_loss
            + 0.020 * residual_loss
        )
        inv_opt.zero_grad()
        loss.backward()
        inv_opt.step()

    target_texture = _reference_texture(reference_path, shape)
    target_texture_full = _reference_texture(reference_path, (256, 512))

    with torch.no_grad():
        if anchored_residual:
            density = (base_density_t * torch.exp(float(density_residual_scale) * lowpass(density_delta))).clamp(0.0, 1.0).cpu().numpy()[0, 0]
            energy = (base_energy_t * torch.exp(float(energy_residual_scale) * lowpass(energy_delta))).clamp(0.0, 1.0).cpu().numpy()[0, 0]
        else:
            density = (torch.sigmoid(density_var) * air_gate).cpu().numpy()[0, 0]
            energy = (torch.sigmoid(energy_var) * air_gate).cpu().numpy()[0, 0]

    reference_full = Path(reference_path)
    base_data = load_phantom(base_phantom_path or base_pair.phantom_path)
    base_total = float(base_data[:, 3].sum())
    rows = []
    candidate_idx = 0
    for texture_strength in texture_strengths:
        mixed_density = _blend_texture(density, target_texture, texture_strength)
        mixed_energy = _blend_texture(energy, target_texture, texture_strength * 0.55)
        cumulative = np.cumsum(mixed_energy, axis=0)
        cumulative /= cumulative.max() + 1e-6
        candidate = torch.from_numpy(
            np.stack([mixed_density, mixed_energy, mixed_energy / (mixed_density + 1e-3), cumulative], axis=0)[None].astype(np.float32)
        ).to(device)
        with torch.no_grad():
            preview = model(candidate).cpu().numpy()[0, 0]
        up_density_base = _blend_texture(_upsample(density, (256, 512)), target_texture_full, texture_strength)
        up_energy_base = _blend_texture(_upsample(energy, (256, 512)), target_texture_full, texture_strength * 0.55)
        for ratio in energy_ratios:
            candidate_idx += 1
            mode = "anchored_surrogate" if anchored_residual else "learned_surrogate"
            label = f"{mode}_t{texture_strength:.2f}_r{ratio:.2f}".replace(".", "p")
            phantom_path = phantom_dir / f"{candidate_idx:02d}_{label}.txt"
            preview_path = preview_dir / f"{candidate_idx:02d}_{label}_surrogate.png"
            field_to_phantom(
                up_density_base,
                up_energy_base,
                phantom_path,
                scatterers_count=scatterers_count,
                seed=seed + candidate_idx,
                total_energy=base_total * ratio,
            )
            _save_preview(preview, preview_path)
            preview_full = preview_dir / f"{candidate_idx:02d}_{label}_surrogate_full.png"
            _save_preview(_upsample(preview, (256, 512)), preview_full)
            metrics = calculate_metrics(reference_full, preview_full, include_lpips=False)
            rows.append(
                {
                    "method": label,
                    "status": "surrogate_unverified",
                    "evidence_source": "learned_surrogate_preview",
                    "evidence_scope": "not_challenge_evidence",
                    "phantom_path": str(phantom_path.resolve()),
                    "surrogate_preview_png": str(preview_full.resolve()),
                    "energy_ratio": ratio,
                    "texture_strength": texture_strength,
                    "anchored_residual": int(anchored_residual),
                    "density_residual_scale": density_residual_scale if anchored_residual else "",
                    "energy_residual_scale": energy_residual_scale if anchored_residual else "",
                    "anchor_weight": anchor_weight if anchored_residual else "",
                    "residual_kernel": residual_kernel if anchored_residual else "",
                    "train_pairs": len(train_pairs),
                    "holdout_pairs": len(holdout_pairs),
                    "surrogate_calibration_csv": str(calibration_path.resolve()) if calibration_path is not None else "",
                    "best_training_ssim": train_pairs[0].ssim,
                    **holdout_summary,
                    **{f"surrogate_{key}": value for key, value in metrics.items()},
                }
            )

    metrics_path = out_dir / "learned_surrogate_metrics.csv"
    with metrics_path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return metrics_path


def _split_train_holdout(
    pairs: list[ScannerPair],
    *,
    holdout_fraction: float = 0.2,
    seed: int = 23,
) -> tuple[list[ScannerPair], list[ScannerPair]]:
    if len(pairs) < 2 or holdout_fraction <= 0:
        return pairs, []
    rng = np.random.default_rng(seed)
    order = np.arange(len(pairs))
    rng.shuffle(order)
    holdout_count = int(round(len(pairs) * holdout_fraction))
    holdout_count = max(1, min(len(pairs) - 1, holdout_count))
    holdout_idx = set(order[:holdout_count].tolist())
    train_pairs = [pair for idx, pair in enumerate(pairs) if idx not in holdout_idx]
    holdout_pairs = [pair for idx, pair in enumerate(pairs) if idx in holdout_idx]
    return train_pairs, holdout_pairs


def _write_surrogate_calibration(model, holdout_pairs: list[ScannerPair], out_dir: Path, shape: tuple[int, int]) -> Path | None:
    if not holdout_pairs:
        return None

    import torch

    calibration_dir = out_dir / "surrogate_calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []
    with torch.no_grad():
        for idx, pair in enumerate(holdout_pairs, start=1):
            field = torch.from_numpy(phantom_to_field(pair.phantom_path, shape=shape)[None].astype(np.float32))
            preview = model(field).cpu().numpy()[0, 0]
            preview_path = calibration_dir / f"{idx:02d}_{_safe_label(pair.label)}_surrogate.png"
            preview_full = calibration_dir / f"{idx:02d}_{_safe_label(pair.label)}_surrogate_full.png"
            _save_preview(preview, preview_path)
            _save_preview(_upsample(preview, (256, 512)), preview_full)
            metrics = calculate_metrics(pair.rendered_gray_path, preview_full, include_lpips=False)
            rows.append(
                {
                    "method": pair.label,
                    "status": "surrogate_holdout_calibration",
                    "evidence_source": "learned_surrogate_holdout",
                    "evidence_scope": "not_challenge_evidence",
                    "phantom_path": str(pair.phantom_path),
                    "true_scanner_render_png": str(pair.rendered_gray_path),
                    "surrogate_preview_png": str(preview_full.resolve()),
                    "training_row_ssim": pair.ssim,
                    **{f"surrogate_{key}": value for key, value in metrics.items()},
                }
            )

    calibration_path = out_dir / "surrogate_calibration_metrics.csv"
    with calibration_path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return calibration_path


def _summarize_calibration(calibration_path: Path | None) -> dict[str, float | str]:
    if calibration_path is None or not calibration_path.exists():
        return {
            "surrogate_holdout_MS-SSIM_mean": "",
            "surrogate_holdout_SSIM_mean": "",
            "surrogate_holdout_LPIPS_PROXY_mean": "",
        }
    rows = list(csv.DictReader(calibration_path.open()))
    summary: dict[str, float | str] = {}
    for key in ("surrogate_MS-SSIM", "surrogate_SSIM", "surrogate_LPIPS_PROXY"):
        values = np.array([float(row[key]) for row in rows if row.get(key, "") not in {"", "nan"}], dtype=float)
        values = values[np.isfinite(values)]
        summary[f"{key.replace('surrogate_', 'surrogate_holdout_')}_mean"] = float(values.mean()) if values.size else ""
    return summary


class _ScannerSurrogate:
    def __new__(cls):
        from torch import nn

        return nn.Sequential(
            nn.Conv2d(4, 20, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(20, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 20, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(20, 1, kernel_size=1),
            nn.Sigmoid(),
        )


def _log_normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.log1p(np.maximum(arr, 0.0))
    peak = float(arr.max())
    if peak > 1e-8:
        arr = arr / peak
    return arr.astype(np.float32)


def _load_resized(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    image = load_scan(path)
    return np.asarray(resize(image, shape, anti_aliasing=True, preserve_range=True), dtype=np.float32)


def _reference_texture(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    image = _load_resized(path, shape)
    smooth = gaussian_filter(image, sigma=(2.0, 4.0))
    detail = np.maximum(image - gaussian_filter(image, sigma=(0.8, 1.2)), 0.0)
    rows = np.arange(shape[0], dtype=np.float32)[:, None]
    surface_mask = smooth > max(0.035, float(np.percentile(smooth, 72)))
    surface = np.argmax(surface_mask, axis=0)
    surface[~surface_mask.any(axis=0)] = int(shape[0] * 0.12)
    surface = gaussian_filter(surface.astype(np.float32), sigma=3.0)[None, :]
    tissue_gate = 1.0 / (1.0 + np.exp(-(rows - surface - 2.0) / 2.5))
    attenuation = np.exp(-np.maximum(rows - surface, 0.0) / max(12.0, shape[0] * 0.22))
    texture = (0.75 * _normalize01(smooth) + 0.25 * _normalize01(detail)) * tissue_gate * attenuation
    return np.clip(texture, 0.0, 1.0).astype(np.float32)


def _blend_texture(field: np.ndarray, texture: np.ndarray, strength: float) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    blended = (1.0 - strength) * field + strength * texture
    return np.clip(blended, 0.0, 1.0).astype(np.float32)


def _normalize01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr - float(arr.min())
    peak = float(arr.max())
    if peak > 1e-8:
        arr = arr / peak
    return arr


def _upsample(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    factors = (shape[0] / arr.shape[0], shape[1] / arr.shape[1])
    return np.clip(zoom(arr, factors, order=1), 0.0, 1.0)


def _safe_label(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:80] or "holdout"


def _save_preview(image: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(path, np.clip(image * 255.0, 0, 255).astype(np.uint8))


def _gradient_loss(pred, target) -> object:
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_target = target[:, :, 1:, :] - target[:, :, :-1, :]
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_target = target[:, :, :, 1:] - target[:, :, :, :-1]
    return (dy_pred - dy_target).abs().mean() + (dx_pred - dx_target).abs().mean()


def _smoothness(field) -> object:
    dy = field[:, :, 1:, :] - field[:, :, :-1, :]
    dx = field[:, :, :, 1:] - field[:, :, :, :-1]
    return dy.abs().mean() + dx.abs().mean()
