# Winning Strategy

This file is now a compact strategy pointer. The canonical current state is [current_findings.md](current_findings.md); the command plan is [next_approach.md](next_approach.md).

## Strategy In One Page

The best current route is:

1. Keep `learned-prior-sparse-p140-t32` as the stable Stage 1 topology.
2. Use flow/energy/texture changes only as bounded residual controls.
3. Train or select controls from true-scanner teacher rows.
4. Gate every candidate with Struct/OAC/SC/RSC lower-confidence bounds.
5. Spend hosted API calls only on candidates expected to be map-safe.
6. Promote only grouped true-scanner results that improve the official aggregate.

## Current Candidate

```text
generator=learned-prior-sparse-p140-t32
prior=outputs/learned_priors/goal_h_candidates_prior.npz
evidence=outputs/api_validation_goal_t32_p140_h61_offset1_5x1/challenge_metrics_summary.csv
package=outputs/submission_ready_learned_prior_sparse_p140_t32_full_corrected
hidden_holdout_final_score=false
```

Best measured public rescue:

```text
package=outputs/submission_ready_p140_t32_flow_energy_rank120_adaptive_rank36_patch
evidence=outputs/api_preliminary_p140_t32_flow_energy_rank120_adaptive_rank36_patch/challenge_metrics_summary.csv
official_score=0.6939983205067786
```

Use the public rescue artifact as teacher data and public evidence, not as proof of hidden-holdout superiority.

## Evidence Rules

- True-scanner evidence means hosted SynthOCT API or official Windows `Part2_Scanner.exe`.
- Local surrogates, previews, and image-space metrics are search aids only.
- Real LPIPS is required for promotion.
- Single-reference wins are not enough.
- Public-set patches do not establish hidden-holdout generalization.

## Main Bottleneck

Structural LPIPS remains the limiting metric. OAC/SC/RSC maps are comparatively strong for `p140-t32`; attempts that improve appearance while damaging maps are rejected.

## Next Move

Run the feedback15 map-safe queue when the hosted scanner is reachable. If it is flat, build a topology-preserving residual-control trainer using the true-scanner teacher rows instead of replacing the density field.

## References

- Current findings: [current_findings.md](current_findings.md)
- Stage 2 resume: [stage2_oac_residual_journey.md](stage2_oac_residual_journey.md)
- Commands and gates: [next_approach.md](next_approach.md)
- Provenance policy: [rules_provenance.md](rules_provenance.md)
