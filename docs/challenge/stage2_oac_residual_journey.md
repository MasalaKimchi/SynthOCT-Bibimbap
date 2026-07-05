# Stage 2 Residual Journey

This note is the compact resume record for the current Stage 2 work. The full current-state summary is [current_findings.md](current_findings.md), and the executable plan is [next_approach.md](next_approach.md).

## Decision

Freeze Stage 1 as `learned-prior-sparse-p140-t32`. It is the best general local generator because it preserves density topology and physical-map behavior. Stage 2 should learn bounded residual controls around that topology:

- flow and low-frequency geometry;
- energy and attenuation controls;
- speckle texture statistics;
- per-row correction strength;
- uncertainty and map-safety gates.

Do not promote full density replacement, visual inversion, direct lattice copying, or surrogate-ranked candidates without grouped true-scanner evidence.

## Current Evidence

Best full-public-set rescue:

```text
outputs/submission_ready_p140_t32_flow_energy_rank120_adaptive_rank36_patch
official_score=0.6939983205067786
MS-SSIM_mean=0.6653683801220828
Struct_MS-SSIM_median=0.6605330710784837
Struct_LPIPS_median=0.5885020792484283
```

Best single residual probe:

```text
MS-SSIM=0.734145597958592
metrics=outputs/residual_selector_public_teacher/probe_batch_feedback11_branch_aware_smoke4_concurrency4/residual_selector_probe_metrics.csv
```

Interpretation: useful public evidence and teacher data, but not proof that the current selector reaches `0.85` or `0.90` MS-SSIM.

## Feedback15 Selector State

```text
selector=outputs/residual_selector_public_teacher/selector_oac_map_safe_feedback15.json
teacher_examples=362
strict_queue=outputs/residual_selector_public_teacher/next_probe_queue_oac_map_safe_feedback15_strict_rowwise_limit120.csv
strict_rows=34
map_safety_status=map_safe_pass for all rows
expected_delta_lcb mean=0.0025712735633801388
expected_map_delta_lcb_min mean=0.0
```

A stricter positive-map gate with `--min-map-delta-lcb 0.00025` retained `0` rows. The selector supports a non-harm retest, not confident positive map improvement.

The budgeted resume queue is:

```text
outputs/residual_selector_public_teacher/two_stage_api_budget_120_queue_oac_map_safe_feedback15_strict_repeat_source_rowwise_diverse.csv
rows=80
map_safety_status=map_safe_pass for all rows
exact parameter overlap with recent comparable budgets=0
```

## API Interruption

On 2026-06-30, hosted validation could not continue because `synthoct.com:443` timed out from this machine. Both concurrency `4` and concurrency `1` failed before request IDs were assigned, and direct `curl` checks to `https://synthoct.com/` also timed out.

The failed folders were pruned:

```text
outputs/residual_selector_public_teacher/probe_batch_feedback15_oac_map_safe_concurrency4
outputs/residual_selector_public_teacher/probe_batch_feedback15_oac_map_safe_smoke4_concurrency1
```

Rows in those partial metrics had `request_id=failed`, so they are not interpreted as spent hosted scanner jobs.

## Cleanup Status

Cleanup verification on 2026-07-05 confirmed:

- failed feedback15 outage folders remain pruned;
- smoke/full retry folders have not been created yet;
- the 80-row map-safe budget queue remains the hosted-scanner resume point;
- expanded phantoms, rendered PNGs, maps, transient probe render folders, old selector iterations, caches, and bytecode were pruned;
- retained files are compact CSV/JSON evidence, learned prior artifacts, submission manifests, challenge-format phantom zips, code zips, and current queues.

## Resume

When the hosted scanner is reachable, run the smoke command in [next_approach.md](next_approach.md). If smoke succeeds, run the full map-safe batch. Promote nothing unless grouped true-scanner evidence improves the official aggregate without map harm.
