"""Evaluation and metric helpers for rendered OCT PNGs."""

from .maps import (
    competition_proxy_score,
    composite_score,
    evaluate_feature_map_metrics,
    profile_scores,
    resize_like,
    safe_corr,
)
from .metrics import calculate_metrics, lpips_proxy, metric_evaluation_metadata, multiscale_ssim_fallback
from .evidence import audit_challenge_evidence, decide_candidate_promotion, select_best_candidate
from .summaries import (
    challenge_lpips_key,
    finite_float,
    summarize_challenge_metrics,
    summarize_rows,
    summarize_sample_wins,
    write_rows,
)

__all__ = [
    "calculate_metrics",
    "competition_proxy_score",
    "composite_score",
    "evaluate_feature_map_metrics",
    "lpips_proxy",
    "metric_evaluation_metadata",
    "multiscale_ssim_fallback",
    "audit_challenge_evidence",
    "decide_candidate_promotion",
    "select_best_candidate",
    "profile_scores",
    "resize_like",
    "safe_corr",
    "challenge_lpips_key",
    "finite_float",
    "summarize_challenge_metrics",
    "summarize_rows",
    "summarize_sample_wins",
    "write_rows",
]
