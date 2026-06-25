"""Backward-compatible evaluation metrics facade.

New code should import from :mod:`synthoct.evaluation`.
"""

from .evaluation import calculate_metrics, lpips_proxy, multiscale_ssim_fallback

__all__ = ["calculate_metrics", "lpips_proxy", "multiscale_ssim_fallback"]
