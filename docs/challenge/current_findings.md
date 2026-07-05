# Current Findings And Journey

Last compacted: 2026-07-05.

This is the canonical short status for the SynthOCT-Bibimbap challenge work. Older experiment details are summarized in [../history/experiments.md](../history/experiments.md); operational commands live in [next_approach.md](next_approach.md).

## Current Judgment

`learned-prior-sparse-p140-t32` remains the best general local generator. It preserves the `p140-t32` density topology and has the strongest grouped true-scanner evidence. The best measured full-public-set rescue artifact adds flow/energy patches and improves public evidence, but it is still public-set rescue, not hidden-holdout proof.

The main unsolved bottleneck is Structural LPIPS. Flow, energy, neural, surrogate, and texture experiments can move parts of the score, but none has simultaneously solved structural perceptual similarity and preserved OAC/SC/RSC map behavior.

## Active Artifacts

| Purpose | Path | Status |
|---|---|---|
| Learned prior | `outputs/learned_priors/goal_h_candidates_prior.npz` | Required for current generator |
| Conservative full package | `outputs/submission_ready_learned_prior_sparse_p140_t32_full_corrected` | Current general package |
| Conservative evidence | `outputs/api_validation_goal_t32_p140_h61_offset1_5x1/challenge_metrics_summary.csv` | Grouped true-scanner evidence |
| Full public-set base evidence | `outputs/api_preliminary_learned_prior_sparse_p140_t32_120_concurrent/challenge_metrics_summary.csv` | 120 public pairs |
| Best public rescue package | `outputs/submission_ready_p140_t32_flow_energy_rank120_adaptive_rank36_patch` | Public-set rescue only |
| Best public rescue evidence | `outputs/api_preliminary_p140_t32_flow_energy_rank120_adaptive_rank36_patch/challenge_metrics_summary.csv` | Best measured public score |
| Stage 2 selector | `outputs/residual_selector_public_teacher/selector_oac_map_safe_feedback15.json` | Next feedback model |
| Stage 2 next queue | `outputs/residual_selector_public_teacher/two_stage_api_budget_120_queue_oac_map_safe_feedback15_strict_repeat_source_rowwise_diverse.csv` | Next hosted run |

## Best Evidence

| Candidate | Evidence | Takeaway |
|---|---|---|
| `learned-prior-sparse-p140-t32` | grouped offset validation, real LPIPS, maps | Promoted local general generator |
| Full public-set `p140-t32` render | `official_score=0.68479` | Map-competitive, Structural LPIPS fails |
| Flow/energy adaptive rank36 patch | `official_score=0.6939983205067786` | Best public-set rescue, not hidden-holdout proof |
| Best single residual probe | `MS-SSIM=0.734145597958592` | Encouraging single-reference result, not grouped evidence |

Promotion requires grouped true-scanner or official Windows evidence, real LPIPS, no unacceptable Struct/OAC/SC/RSC regression, runtime under the challenge budget, and `hidden_holdout_final_score=false` unless organizer results say otherwise.

## Rejected Or Limited Paths

- Offline H0-H60 proxy rankings are historical only; the renderer was invalidated.
- H61/H67/H68 were superseded by learned-prior evidence.
- Visual inversion and direct lattice methods did not survive the true scanner.
- Standalone neural prior improved Structural LPIPS but damaged map metrics and official score.
- Neural energy blends improved Structural LPIPS slightly but did not robustly beat `p140-t32` on grouped evidence.
- Learned surrogate inverse optimization overfit surrogate errors and failed to transfer.
- Flow/energy patches raise the public-set floor but do not solve the Structural LPIPS gate.

## Stage 2 Direction

Freeze Stage 1 as `p140-t32` topology. Treat flow, energy, and texture changes as bounded residual controls around that topology, not as density replacement. Use true-scanner teacher rows to train or select residual controls, then gate candidates with Struct/OAC/SC/RSC lower-confidence bounds before spending hosted API budget.

Current selector facts:

```text
selector=outputs/residual_selector_public_teacher/selector_oac_map_safe_feedback15.json
teacher_examples=362
strict_map_safe_rows=34
budget_queue_rows=80
expected_map_delta_lcb_min_mean=0.0
positive_map_gate_rows=0
```

Interpretation: the current selector supports a non-harm retest, not confident positive map improvement.

## Hosted API Status

On 2026-06-30, hosted validation to `synthoct.com:443` timed out before request IDs were assigned. The failed feedback15 folders were pruned. As of the 2026-07-05 cleanup, the retry output folders had not been created, and the 80-row map-safe budget queue remained the resume point.

## Cleanup Status

Deep local cleanup retained compact evidence and package archives, but pruned expanded phantoms, rendered PNGs, maps, transient probe renders, old selector iterations, caches, and bytecode. The retained output tree is intentionally compact: keep CSV/JSON evidence, prior artifacts, package zips, manifests, and the current selector queue.

## Next Action

When the hosted scanner is reachable, run the smoke command in [next_approach.md](next_approach.md). Promote nothing unless grouped true-scanner evidence improves the official aggregate without map harm. If results are flat, train a richer topology-preserving residual-control model rather than widening scalar sweeps.
