# Best Algorithm Report

The best current algorithmic direction is summarized in [current_findings.md](current_findings.md). This file keeps the decision in report form for quick review.

## Best General Method

`learned-prior-sparse-p140-t32` is the best general local generator. It uses:

```text
artifact=outputs/learned_priors/goal_h_candidates_prior.npz
package=outputs/submission_ready_learned_prior_sparse_p140_t32_full_corrected
evidence=outputs/api_validation_goal_t32_p140_h61_offset1_5x1/challenge_metrics_summary.csv
```

Why it wins locally:

- preserves density topology and physical-map behavior;
- beats H61 and other API-facing baselines on grouped true-scanner evidence;
- has real LPIPS and Struct/OAC/SC/RSC map evaluation;
- remains within the challenge runtime/package workflow.

## Best Public-Set Rescue

The strongest measured public-set artifact is:

```text
artifact=outputs/submission_ready_p140_t32_flow_energy_rank120_adaptive_rank36_patch
evidence=outputs/api_preliminary_p140_t32_flow_energy_rank120_adaptive_rank36_patch/challenge_metrics_summary.csv
official_score=0.6939983205067786
```

This is not the same as a proven hidden-holdout winner. It patches public validation failures and should be used as teacher data for Stage 2 residual learning.

## What Failed

- Direct visual inversion and direct lattice copying did not survive the true scanner.
- Standalone neural prior improved Structural LPIPS but damaged map metrics.
- Neural energy blends produced small perceptual gains but did not robustly beat `p140-t32`.
- Learned surrogate inverse optimization overfit surrogate artifacts.
- Flow/energy patches are helpful but incremental.

## Current Research Direction

Freeze Stage 1 as `p140-t32`. Learn Stage 2 residual controls for geometry, attenuation, energy, and texture from true-scanner teacher rows. Reject candidates whose uncertainty bounds predict Struct/OAC/SC/RSC harm.

## Promotion Rule

Do not replace `learned-prior-sparse-p140-t32` unless grouped true-scanner evidence improves the official aggregate with real LPIPS and no unacceptable map regression.
