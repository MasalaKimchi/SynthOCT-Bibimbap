# Next Approach

This is the compact execution plan. See [current_findings.md](current_findings.md) for the current evidence and [stage2_oac_residual_journey.md](stage2_oac_residual_journey.md) for the Stage 2 resume state.

## Objective

Improve the official-style aggregate while preserving Struct/OAC/SC/RSC behavior. Do not optimize a single structural metric in isolation.

Local aggregate:

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

## Current Position

`learned-prior-sparse-p140-t32` remains the general model. The adaptive flow/energy rank36 patch is the best measured public-set rescue, but it is public evidence only. The next run is a controlled feedback15 selector retest, not a final promotion attempt.

## Executable Residual-Control Model

The topology-preserving model path now predicts bounded residual controls rather than replacing the phantom density field:

```bash
PYTHONPATH=src python -m synthoct.cli train-topology-residual-control-model \
  --base-api-metrics outputs/api_preliminary_p140_t32_flow_energy_rank120_adaptive_rank36_patch/api_metrics.csv \
  --teacher-metrics outputs/residual_selector_public_teacher/probe_batch_feedback11_branch_aware_smoke4_concurrency4/residual_selector_probe_metrics.csv \
  --out outputs/residual_selector_public_teacher/topology_residual_control_model_v1.json \
  --holdout-fraction 0

PYTHONPATH=src python -m synthoct.cli plan-topology-residual-control-queue \
  --base-api-metrics outputs/api_preliminary_p140_t32_flow_energy_rank120_adaptive_rank36_patch/api_metrics.csv \
  --model outputs/residual_selector_public_teacher/topology_residual_control_model_v1.json \
  --out outputs/residual_selector_public_teacher/topology_residual_control_model_v1_queue120.csv
```

The first artifact observes aggregate map-objective teacher deltas, but not per-channel Struct/OAC/SC/RSC deltas:

```text
map_objective_target_observed=true
map_target_observed=false
queue_map_safety=map_safety_not_trained
```

Use it as a control-model scaffold. Retrain with map-enriched teacher rows before treating it as map-safe or spending broad API budget.

The 120-row MS-SSIM comparison is reproducible with:

```bash
PYTHONPATH=src python -m synthoct.cli compare-api-ms-ssim \
  --base-metrics outputs/api_preliminary_learned_prior_sparse_p140_t32_120_concurrent/api_metrics.csv \
  --candidate-metrics outputs/api_preliminary_p140_t32_flow_energy_rank120_adaptive_rank36_patch/api_metrics.csv \
  --out outputs/residual_selector_public_teacher/ms_ssim_120_comparison_base_vs_adaptive_rank36_patch.csv \
  --base-method learned-prior-sparse-p140-t32 \
  --candidate-method p140-t32-flow-energy-rank120-adaptive-rank36-patch \
  --model-queue outputs/residual_selector_public_teacher/topology_residual_control_model_v1_queue120.csv
```

Current summary: candidate mean MS-SSIM `0.6653683801220828` vs base `0.6555505534821618`, with `118` candidate wins, `0` base wins, and `2` ties.

## Hosted Scanner Smoke

Run this first when `synthoct.com` is reachable:

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

## Full Retest

If the smoke produces valid request IDs and completed renders:

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

## Interpretation Rules

- Treat the queue as a map-safe non-harm test.
- Promote nothing from a single probe.
- Promote nothing unless grouped true-scanner evidence improves the official aggregate.
- If results are flat, train a richer topology-preserving residual-control model rather than widening scalar sweeps.
- Keep evidence labels explicit: true scanner, surrogate, preview, public-set rescue, or hidden-holdout final.

## Promotion Gate

A candidate can replace the current local candidate only if all are true:

- rendered by hosted API or official Windows true scanner;
- evaluated full-frame with real LPIPS;
- grouped validation, not only public rescue or a single-reference probe;
- official aggregate beats `learned-prior-sparse-p140-t32`;
- no unacceptable Struct/OAC/SC/RSC regression;
- generation remains within the `600` second challenge budget;
- artifact labels say `hidden_holdout_final_score=false`.

## Do Not Do Next

- Do not spend API calls on direct texture overlays without a map-aware model.
- Do not chase preview-only lattice or surrogate scores.
- Do not replace density topology without strict topology-preservation evidence.
- Do not treat full-public-set rescue as hidden-holdout proof.
