from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage import transform
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

from .processor import load_scan


def _same_shape(ref: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if ref.shape != pred.shape:
        pred = transform.resize(pred, ref.shape, anti_aliasing=True)
    return ref, pred


def calculate_metrics(ref_path: str | Path, pred_path: str | Path, include_lpips: bool = True) -> dict[str, float]:
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
        results["MS-SSIM"] = float("nan")
        results["VIF"] = float("nan")

    results["LPIPS"] = float("nan")
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
