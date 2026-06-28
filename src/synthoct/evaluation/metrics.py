from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage import transform
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

from synthoct.features import load_scan


def _same_shape(ref: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if ref.shape != pred.shape:
        pred = transform.resize(pred, ref.shape, anti_aliasing=True)
    return ref, pred


def metric_evaluation_metadata(ref_path: str | Path, pred_path: str | Path) -> dict[str, str]:
    """Describe the image extent used by full-reference metric calculations."""
    ref = load_scan(ref_path)
    pred = load_scan(pred_path)
    return {
        "evaluation_region": "full_frame",
        "reference_shape": _shape_label(ref.shape),
        "prediction_shape": _shape_label(pred.shape),
        "evaluated_shape": _shape_label(ref.shape),
        "prediction_resized_to_reference": str(ref.shape != pred.shape).lower(),
    }


def calculate_metrics(ref_path: str | Path, pred_path: str | Path, include_lpips: bool = True) -> dict[str, float]:
    """Compare two rendered OCT PNGs or derived map PNGs."""
    ref, pred = _same_shape(load_scan(ref_path), load_scan(pred_path))
    results: dict[str, float] = {
        "MSE": float(mean_squared_error(ref, pred)),
        "PSNR": float(peak_signal_noise_ratio(ref, pred, data_range=1.0)),
        "SSIM": float(structural_similarity(ref, pred, data_range=1.0)),
    }

    try:
        from sewar.full_ref import msssim, vifp

        ref_u8 = (ref * 255).astype(np.uint8)
        pred_u8 = (pred * 255).astype(np.uint8)
        results["MS-SSIM"] = float(np.real(msssim(ref_u8, pred_u8)))
        results["VIF"] = float(vifp(ref_u8, pred_u8))
    except Exception:
        results["MS-SSIM"] = multiscale_ssim_fallback(ref, pred)
        results["VIF"] = float("nan")

    results["LPIPS"] = float("nan")
    results["LPIPS_PROXY"] = lpips_proxy(ref, pred)
    if include_lpips:
        try:
            import lpips
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            loss_fn = lpips.LPIPS(net="alex", verbose=False).to(device)
            t_ref = torch.from_numpy(ref).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
            t_pred = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
            t_ref = (t_ref * 2 - 1).to(device)
            t_pred = (t_pred * 2 - 1).to(device)
            with torch.no_grad():
                results["LPIPS"] = float(loss_fn(t_ref, t_pred).item())
        except Exception:
            pass
    return results


def _shape_label(shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape)


def multiscale_ssim_fallback(ref: np.ndarray, pred: np.ndarray) -> float:
    values = []
    weights = np.array([0.15, 0.25, 0.30, 0.30], dtype=float)
    cur_ref = ref
    cur_pred = pred
    for _ in range(len(weights)):
        if min(cur_ref.shape) < 8:
            break
        values.append(structural_similarity(cur_ref, cur_pred, data_range=1.0))
        if min(cur_ref.shape) < 16:
            break
        cur_ref = transform.resize(cur_ref, (cur_ref.shape[0] // 2, cur_ref.shape[1] // 2), anti_aliasing=True)
        cur_pred = transform.resize(cur_pred, (cur_pred.shape[0] // 2, cur_pred.shape[1] // 2), anti_aliasing=True)
    if not values:
        return float("nan")
    use_weights = weights[: len(values)]
    use_weights = use_weights / use_weights.sum()
    return float(np.average(values, weights=use_weights))


def lpips_proxy(ref: np.ndarray, pred: np.ndarray) -> float:
    """Fallback perceptual distance used only when real LPIPS is unavailable."""
    ref_gz, ref_gx = np.gradient(ref)
    pred_gz, pred_gx = np.gradient(pred)
    grad_err = np.mean(np.abs(ref_gz - pred_gz)) + np.mean(np.abs(ref_gx - pred_gx))
    hist_ref, _ = np.histogram(ref, bins=32, range=(0, 1), density=True)
    hist_pred, _ = np.histogram(pred, bins=32, range=(0, 1), density=True)
    hist_err = np.mean(np.abs(hist_ref - hist_pred)) / 32.0
    return float(np.clip(0.65 * grad_err + 0.35 * hist_err, 0.0, 1.0))
