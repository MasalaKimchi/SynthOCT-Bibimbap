# Next Approach: Stage 2 Residual Learning And Evidence Gates

This document turns the current experimental audit into the next execution plan. It should guide the next scanner-available session and the next modeling pass.

## Current Judgment

The current direction is right, but the expected near-term gains are incremental. `learned-prior-sparse-p140-t32` remains the best general generator because it preserves density topology and physical-map behavior. Flow and energy patches should be treated as teacher data for residual control, not as proof of a hidden-general breakthrough.

The best measured public-set rescue artifact is:

```text
outputs/submission_ready_p140_t32_flow_energy_rank120_adaptive_rank36_patch
official_score=0.6939983205067786
MS-SSIM_mean=0.6653683801220828
Struct_MS-SSIM_median=0.6605330710784837
Struct_LPIPS_median=0.5885020792484283
```

The best single hosted true-scanner residual probe remains:

```text
MS-SSIM=0.734145597958592
metrics=outputs/residual_selector_public_teacher/probe_batch_feedback11_branch_aware_smoke4_concurrency4/residual_selector_probe_metrics.csv
```

Neither result is evidence that `0.85` or `0.90` MS-SSIM is reachable with the current selector alone. The current OAC-aware selector should be viewed as a low-risk API-budgeting tool, not the full breakthrough model.

## Mathematical Objective

Do not optimize a single structural number in isolation. The challenge-facing aggregate used locally is:

```text
mean(
  median(Struct_MS-SSIM),
  median(1 - Struct_LPIPS),
  median(OAC_MS-SSIM),
  median(1 - OAC_LPIPS),
  median(SC_MS-SSIM),
  median(1 - SC_LPIPS),
  median(RSC_MS-SSIM),
  median(1 - RSC_LPIPS)
)
```

The next model must therefore improve official aggregate score while protecting every physical map channel. A candidate that improves image-space MS-SSIM or Structural LPIPS but damages OAC, SC, or RSC is not a win.

Useful local targets:

- raise grouped official score over `learned-prior-sparse-p140-t32`;
- reduce Structural LPIPS without lowering Struct/OAC/SC/RSC MS-SSIM medians;
- show per-sample wins on a grouped split, not only a single-reference maximum;
- keep all evidence labeled as true scanner, surrogate, or preview.

## Physics-Informed Modeling Position

Freeze Stage 1 as the `p140-t32` topology. Density replacement has repeatedly been fragile: neural density blends, direct texture overlays, and target-locked visual inversions improved isolated appearance signals while damaging OAC/SC/RSC behavior.

Stage 2 should learn only residual controls around that topology:

- low-frequency geometry and flow parameters;
- attenuation-normalized energy residuals;
- speckle texture statistics;
- per-row and per-source correction strength;
- confidence estimates for map-safe promotion.

This keeps the scatterer field scanner-compatible and respects the coupling between density, attenuation, speckle contrast, and refined speckle contrast.

## Near-Term API Plan

When the hosted scanner is reachable, run the current map-safe selector queue as a controlled retest:

```bash
PYTHONPATH=src python -m synthoct.cli residual-selector-flow-batch \
  --selector-queue outputs/residual_selector_public_teacher/two_stage_api_budget_120_queue_oac_map_safe_feedback15_strict_repeat_source_rowwise_diverse.csv \
  --out outputs/residual_selector_public_teacher/probe_batch_feedback15_oac_map_safe_smoke4_retry \
  --max-candidates 4 \
  --max-energy-followups 0 \
  --api-key-file ~/.config/synthoct/api_key \
  --api-concurrency 1 \
  --poll-interval-seconds 3 \
  --max-polls 60
```

If the smoke produces valid request IDs and completed renders, run:

```bash
PYTHONPATH=src python -m synthoct.cli residual-selector-flow-batch \
  --selector-queue outputs/residual_selector_public_teacher/two_stage_api_budget_120_queue_oac_map_safe_feedback15_strict_repeat_source_rowwise_diverse.csv \
  --out outputs/residual_selector_public_teacher/probe_batch_feedback15_oac_map_safe_concurrency4_retry \
  --max-candidates 80 \
  --max-energy-followups 40 \
  --min-energy-flow-delta -0.003715494159916566 \
  --api-key-file ~/.config/synthoct/api_key \
  --api-concurrency 4 \
  --poll-interval-seconds 3 \
  --max-polls 100
```

Interpretation rules:

- Treat the current selector queue as a non-harm test. Its strict map-safe rows have `expected_map_delta_lcb_min=0.0`, not confident positive map improvement.
- Promote nothing from this run unless grouped true-scanner evidence improves the official aggregate.
- If results are flat, use the feedback to train a richer residual selector rather than widening scalar sweeps.

## Breakthrough Modeling Plan

The next high-ceiling model should be a two-stage residual trainer:

1. Generate or load `p140-t32` base phantoms.
2. Encode the reference B-scan into low-frequency anatomy, attenuation, speckle, and depth-profile features.
3. Predict residual controls rather than full density replacement.
4. Train against teacher deltas from flow+energy rows, with losses aligned to Struct/OAC/SC/RSC MS-SSIM and LPIPS.
5. Use grouped holdout by source traits such as sex, age band, body site, and frame to reduce leakage.
6. Calibrate uncertainty before sending candidates to the hosted scanner.
7. Reject candidates whose lower-confidence bounds indicate likely map harm.

The surrogate scanner, if added, is only a search accelerator. It must have held-out true-scanner calibration and uncertainty gating before any surrogate-selected candidate is promoted to hosted API calls.

## Promotion Gate

A candidate can replace the current local candidate only if all are true:

- rendered by hosted API or official Windows true scanner;
- evaluated full-frame with real LPIPS;
- grouped validation, not only public manifest rescue or a single-reference probe;
- official aggregate beats `learned-prior-sparse-p140-t32`;
- no unacceptable Struct/OAC/SC/RSC regression;
- generation remains within the `600` second challenge budget;
- artifact labels say `hidden_holdout_final_score=false`.

Public-set rescue rows can inform training, but they do not establish hidden-holdout performance.

## What Not To Do Next

- Do not spend API calls on direct texture overlays unless a map-aware model first predicts why they should preserve SC/RSC.
- Do not chase preview-only values such as lattice `0.99` MS-SSIM.
- Do not replace density topology with a neural field without a strict topology-preservation constraint.
- Do not call a full-public public-set patch the final general model.
- Do not optimize Structural LPIPS alone; it must be coupled to official aggregate and physical-map safety.
