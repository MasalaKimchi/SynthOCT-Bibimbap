from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .learned_surrogate import discover_scanner_pairs, phantom_to_field


LEARNED_PRIOR_CONFIGS: dict[str, dict[str, float]] = {
    "learned-prior-conservative": {
        "target_blend": 0.38,
        "prior_blend": 0.62,
        "texture_weight": 0.10,
    },
    "learned-prior-balanced": {
        "target_blend": 0.50,
        "prior_blend": 0.50,
        "texture_weight": 0.16,
    },
    "learned-prior-structural": {
        "target_blend": 0.62,
        "prior_blend": 0.38,
        "texture_weight": 0.24,
    },
    "learned-prior-dim-e12": {
        "target_blend": 0.62,
        "prior_blend": 0.38,
        "texture_weight": 0.24,
        "energy_scale": 0.12,
    },
    "learned-prior-dim-e18": {
        "target_blend": 0.62,
        "prior_blend": 0.38,
        "texture_weight": 0.24,
        "energy_scale": 0.18,
    },
    "learned-prior-dim-e25": {
        "target_blend": 0.62,
        "prior_blend": 0.38,
        "texture_weight": 0.24,
        "energy_scale": 0.25,
    },
    "learned-prior-sparse-p15": {
        "target_blend": 0.62,
        "prior_blend": 0.38,
        "texture_weight": 0.24,
        "density_power": 1.5,
    },
    "learned-prior-sparse-p22": {
        "target_blend": 0.62,
        "prior_blend": 0.38,
        "texture_weight": 0.24,
        "density_power": 2.2,
    },
    "learned-prior-sparse-p32": {
        "target_blend": 0.62,
        "prior_blend": 0.38,
        "texture_weight": 0.24,
        "density_power": 3.2,
    },
    "learned-prior-sparse-p55": {
        "target_blend": 0.62,
        "prior_blend": 0.38,
        "texture_weight": 0.24,
        "density_power": 5.5,
    },
    "learned-prior-sparse-p90": {
        "target_blend": 0.62,
        "prior_blend": 0.38,
        "texture_weight": 0.24,
        "density_power": 9.0,
    },
    "learned-prior-sparse-p120": {
        "target_blend": 0.62,
        "prior_blend": 0.38,
        "texture_weight": 0.24,
        "density_power": 12.0,
    },
    "learned-prior-sparse-p140": {
        "target_blend": 0.62,
        "prior_blend": 0.38,
        "texture_weight": 0.24,
        "density_power": 14.0,
    },
    "learned-prior-sparse-p160": {
        "target_blend": 0.62,
        "prior_blend": 0.38,
        "texture_weight": 0.24,
        "density_power": 16.0,
    },
    "learned-prior-sparse-p140-t32": {
        "target_blend": 0.62,
        "prior_blend": 0.38,
        "texture_weight": 0.32,
        "density_power": 14.0,
    },
    "learned-prior-sparse-p140-sigma16": {
        "target_blend": 0.62,
        "prior_blend": 0.38,
        "texture_weight": 0.24,
        "density_power": 14.0,
        "energy_sigma": 0.16,
    },
    "learned-prior-sparse-p140-sigma22": {
        "target_blend": 0.62,
        "prior_blend": 0.38,
        "texture_weight": 0.24,
        "density_power": 14.0,
        "energy_sigma": 0.22,
    },
}


def train_empirical_phantom_prior(
    outputs_dir: str | Path,
    out_path: str | Path,
    *,
    shape: tuple[int, int] = (128, 256),
    pair_limit: int = 64,
    temperature: float = 0.08,
) -> Path:
    """Fit a scanner-pair empirical prior over phantom density and energy fields.

    The artifact is not challenge evidence by itself. It distills previously
    rendered true-scanner phantom/render pairs into a reusable prior that a
    scanner-compatible generator can condition on later.
    """
    pairs = discover_scanner_pairs(outputs_dir, limit=pair_limit)
    if not pairs:
        raise RuntimeError(f"No scanner-rendered training pairs found under {outputs_dir}.")

    fields = np.stack([phantom_to_field(pair.phantom_path, shape=shape) for pair in pairs], axis=0)
    scores = np.array([pair.ssim for pair in pairs], dtype=np.float64)
    weights = _softmax_weights(scores, temperature=temperature)
    density_prior = np.tensordot(weights, fields[:, 0], axes=(0, 0))
    energy_prior = np.tensordot(weights, fields[:, 1], axes=(0, 0))
    mean_energy_prior = np.tensordot(weights, fields[:, 2], axes=(0, 0))
    cumulative_prior = np.tensordot(weights, fields[:, 3], axes=(0, 0))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        density_prior=density_prior.astype(np.float32),
        energy_prior=energy_prior.astype(np.float32),
        mean_energy_prior=mean_energy_prior.astype(np.float32),
        cumulative_prior=cumulative_prior.astype(np.float32),
        weights=weights.astype(np.float32),
        scores=scores.astype(np.float32),
        labels=np.array([pair.label for pair in pairs], dtype=np.str_),
        phantom_paths=np.array([str(pair.phantom_path) for pair in pairs], dtype=np.str_),
        rendered_gray_paths=np.array([str(pair.rendered_gray_path) for pair in pairs], dtype=np.str_),
        shape=np.array(shape, dtype=np.int32),
        evidence_source="empirical_true_scanner_pair_prior",
        evidence_scope="not_challenge_evidence",
    )
    metadata_path = out_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "artifact": str(out_path),
                "evidence_source": "empirical_true_scanner_pair_prior",
                "evidence_scope": "not_challenge_evidence",
                "pair_count": len(pairs),
                "shape": list(shape),
                "temperature": temperature,
                "labels": [pair.label for pair in pairs],
                "scores": [float(pair.ssim) for pair in pairs],
                "interpretation": "Empirical phantom prior for scanner-compatible generation; render generated phantoms through the true scanner before promotion.",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return out_path


def _softmax_weights(scores: np.ndarray, *, temperature: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Prior scores must be finite.")
    scaled = (scores - scores.max()) / max(float(temperature), 1e-6)
    weights = np.exp(scaled)
    total = float(weights.sum())
    if total <= 0:
        return np.full(scores.shape, 1.0 / max(1, scores.size), dtype=np.float64)
    return weights / total
