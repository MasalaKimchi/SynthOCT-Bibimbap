# Stage 2 OAC-Aware Residual Journey

This note records the current state of the push from the stable `p140-t32` learned-prior topology toward a Stage 2 residual selector/trainer. It is written as a resume point for the next scanner-available session.

## Current Position

Stage 1 should stay frozen as the `learned-prior-sparse-p140-t32` density topology. The best general generator remains `learned-prior-sparse-p140-t32`, while flow and energy patches are public-set rescue layers. They are useful teacher data, but they are not hidden-holdout breakthroughs.

The best measured public-set package is currently:

```text
outputs/submission_ready_p140_t32_flow_energy_rank120_adaptive_rank36_patch
```

with hosted true-scanner evidence at:

```text
outputs/api_preliminary_p140_t32_flow_energy_rank120_adaptive_rank36_patch/challenge_metrics_summary.csv
```

That summary reports:

```text
official_score=0.6939983205067786
MS-SSIM_mean=0.6653683801220828
Struct_MS-SSIM_median=0.6605330710784837
Struct_LPIPS_median=0.5885020792484283
preliminary_threshold_pass=0
failure=Struct_LPIPS>=0.4
```

The best single hosted true-scanner residual probe found so far is:

```text
MS-SSIM=0.734145597958592
source=DATASET_PNG/Female/1950-1960/Cheek/l__shcheka_frame250.png
method=selector_energy_basecurrent_e0p798_sig1p80_tex0p26_deep0p66_s0p380_p4_rank120_row0_l_shcheka_frame250
metrics=outputs/residual_selector_public_teacher/probe_batch_feedback11_branch_aware_smoke4_concurrency4/residual_selector_probe_metrics.csv
```

This is a hosted true-scanner single-reference probe, not grouped or hidden-holdout evidence. The target of `0.85` or `0.90` MS-SSIM has not been achieved.

## What We Learned

Stage 1 residual modeling produced stable but insufficient gains. The best no-training Stage 1 validation rows reached `0.7334089110214908` MS-SSIM, but strict map guardrails did not pass broadly. Depth and local profile preservation reduced damage but also reduced upside.

Target-guided Stage 2 texture overlay was a clear negative result. The top-four overlay validation produced `24` hosted rows with mean MS-SSIM around `0.66419`, maximum `0.71459`, and substantial SC/RSC regressions. Direct target-locked texture injection is too disruptive for this scanner.

Preview-only direct lattice rows reached very high local image-space values, including `0.9949915270369818`, but true-scanner transfer failed. Treat those previews as diagnostic mirages, not evidence.

The better Stage 2 direction is parameter selection and residual control distillation:

- predict per-row and per-source residual strength;
- train against Struct/OAC/SC/RSC deltas, not just aggregate MS-SSIM;
- use flow and energy patches as teacher data;
- reject candidates unless uncertainty-gated and map-safe;
- require hosted true-scanner validation before promotion.

## Current Stage 2 Selector State

The OAC-aware residual selector now stores per-map teacher deltas and predicts lower-confidence bounds for:

```text
Struct_MS-SSIM delta
OAC_MS-SSIM delta
SC_MS-SSIM delta
RSC_MS-SSIM delta
```

The planning CLI supports:

```bash
synthoct plan-residual-selector-queue \
  --require-map-safe \
  --min-map-delta-lcb 0 \
  --min-map-safe-win-rate 0.5
```

The current OAC-aware selector artifact is:

```text
outputs/residual_selector_public_teacher/selector_oac_map_safe_feedback15.json
```

It was trained from `362` teacher examples with `166` train examples and `196` held-out examples. A strict non-harm map gate retained `34` rows:

```text
outputs/residual_selector_public_teacher/next_probe_queue_oac_map_safe_feedback15_strict_rowwise_limit120.csv
rows=34
map_safety_status=map_safe_pass for all rows
expected_delta_lcb mean=0.0025712735633801388
expected_map_delta_lcb_min mean=0.0
```

A stricter positive-map gate with `--min-map-delta-lcb 0.00025` retained `0` rows. That means the current teacher evidence supports non-harmful map-safe candidates, but not confident positive OAC/SC/RSC improvement.

The budgeted candidate queue is:

```text
outputs/residual_selector_public_teacher/two_stage_api_budget_120_queue_oac_map_safe_feedback15_strict_repeat_source_rowwise_diverse.csv
rows=80
map_safety_status=map_safe_pass for all rows
exact parameter overlap with recent comparable budgets=0
```

Exact-source exclusion produced no rows because the strict map-safe sources had already appeared in feedback data. The repeat-source queue still contains new parameterizations, so it is valid for controlled retesting once the hosted scanner is back.

## API Interruption

On 2026-06-30, hosted validation could not continue because `synthoct.com:443` timed out from this machine. Both concurrency `4` and concurrency `1` failed before request IDs were assigned, and direct `curl` checks to `https://synthoct.com/` also timed out.

Partial failed-output folders from that outage are:

```text
outputs/residual_selector_public_teacher/probe_batch_feedback15_oac_map_safe_concurrency4
outputs/residual_selector_public_teacher/probe_batch_feedback15_oac_map_safe_smoke4_concurrency1
```

Rows in those partial metrics have `request_id=failed`, so they do not appear to represent spent hosted scanner jobs.

## Resume Plan

When the website is reachable again, first run a small smoke:

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

If the smoke succeeds, run the full map-safe batch:

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

After the hosted run completes, enrich or inspect map deltas before promoting anything:

```bash
PYTHONPATH=src pytest tests/test_dataset_cli_scanner.py tests/test_submission_api.py -q --tb=short
ruff check src/synthoct/residual_selector.py src/synthoct/cli.py tests/test_dataset_cli_scanner.py
```

Promotion remains blocked until a grouped true-scanner result beats the current `p140-t32` family without degrading Struct/OAC/SC/RSC or LPIPS. Do not claim hidden-holdout success from any public-set rescue row.
