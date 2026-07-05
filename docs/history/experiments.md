# Compact Experiment History

This file replaces the long running experiment diary. It preserves decisions that still matter and points to the canonical current status in [../challenge/current_findings.md](../challenge/current_findings.md).

## Current State

- Current general generator: `learned-prior-sparse-p140-t32`.
- Current learned prior: `outputs/learned_priors/goal_h_candidates_prior.npz`.
- Current grouped evidence: `outputs/api_validation_goal_t32_p140_h61_offset1_5x1/challenge_metrics_summary.csv`.
- Best measured public-set rescue artifact: `outputs/submission_ready_p140_t32_flow_energy_rank120_adaptive_rank36_patch`.
- Next Stage 2 queue: `outputs/residual_selector_public_teacher/two_stage_api_budget_120_queue_oac_map_safe_feedback15_strict_repeat_source_rowwise_diverse.csv`.

## Compressed Timeline

### Offline Proxy Era

H0-H60 experiments used an internal offline proxy. They produced useful intuition about low depth compensation, OAC weighting, sparse density, and inhomogeneous point processes, but they are not current ranking evidence. H41/H56 and the subagent council studies are historical only because the proxy renderer was invalidated.

Decision retained: direct image-style tricks and excessive density replacement are risky; physics-map behavior matters.

### Hosted API Baselines

H61/H67/H68 were compared with hosted true-scanner renders. H61 was the conservative default until learned-prior methods beat it on grouped, offset validation.

Decision retained: use hosted API or official Windows scanner evidence for promotion; small proxy-only wins are not enough.

### Learned Prior Promotion

True-scanner rows were distilled into `goal_h_candidates_prior.npz`. Sparse learned-prior variants beat H61 on grouped hosted evidence with real LPIPS and Struct/OAC/SC/RSC maps.

Best promoted variant:

```text
learned-prior-sparse-p140-t32
official_score=0.67703 on grouped 5x1 offset evidence
corrected_package=outputs/submission_ready_learned_prior_sparse_p140_t32_full_corrected
```

The fixed packaging path corrected a reference PNG scale bug in the earlier full package. Do not upload the old uncorrected package if it exists locally.

### Neural And Hybrid Attempts

Standalone neural prior reduced Structural LPIPS but hurt OAC/SC/RSC behavior and official score. Energy-only neural blends lowered Structural LPIPS slightly while preserving density topology, but broader grouped summaries still favored `p140-t32`.

Decision retained: neural signals are useful as residual controls or losses, not as full density replacement without topology and map constraints.

### Surrogate And Direct-Lattice Attempts

Learned-surrogate inverse optimization and direct lattice image-copying produced appealing previews but failed under the hosted true scanner. Anchoring reduced catastrophic failures but still did not beat the base.

Decision retained: preview metrics and local surrogate scores are not promotion evidence. True-scanner transfer is the gate.

### Full Public-Set Validation

The corrected `p140-t32` package was rendered over all 120 public pairs:

```text
official_score=0.68479
Struct_MS-SSIM_median=0.65296
Struct_LPIPS_median=0.58736
failure=Struct_LPIPS>=0.4
```

Interpretation: the generator is map-competitive but structurally/perceptually weak.

### Flow/Energy Public Rescue

Flow plus energy corrections improved many public rows. Positive-only and adaptive promotion eventually produced:

```text
artifact=outputs/submission_ready_p140_t32_flow_energy_rank120_adaptive_rank36_patch
official_score=0.6939983205067786
Struct_MS-SSIM_median=0.6605330710784837
Struct_LPIPS_median=0.5885020792484283
```

Decision retained: flow/energy is a tactical public-set rescue layer and teacher-data source. It is not hidden-holdout proof and does not solve Structural LPIPS.

### Stage 2 Residual Selector

Stage 1 is frozen as `p140-t32` topology. Stage 2 predicts bounded residual controls using true-scanner teacher rows and map-safety lower-confidence bounds.

Current selector:

```text
selector=outputs/residual_selector_public_teacher/selector_oac_map_safe_feedback15.json
strict_queue=outputs/residual_selector_public_teacher/next_probe_queue_oac_map_safe_feedback15_strict_rowwise_limit120.csv
budget_queue=outputs/residual_selector_public_teacher/two_stage_api_budget_120_queue_oac_map_safe_feedback15_strict_repeat_source_rowwise_diverse.csv
budget_rows=80
```

The hosted API was unreachable during the feedback15 attempt on 2026-06-30. No request IDs were assigned, so those rows are not counted as spent hosted jobs.

## 2026-07-05 Topology-Preserving Residual Control Model Scaffold

The Stage 2 plan now has an executable bounded control-model path:

```bash
synthoct train-topology-residual-control-model
synthoct plan-topology-residual-control-queue
synthoct compare-api-ms-ssim
```

Artifacts:

```text
model=outputs/residual_selector_public_teacher/topology_residual_control_model_v1.json
queue=outputs/residual_selector_public_teacher/topology_residual_control_model_v1_queue120.csv
```

The artifact records:

```text
model_type=topology_preserving_residual_control_model_v1
base_topology_preserved=true
predicts_full_density_replacement=false
promotion_allowed_without_true_scanner=false
hidden_holdout_final_score=false
```

It predicts flow/geometry, energy, and texture controls while preserving the current phantom topology. The first artifact trained from the surviving feedback11 probe metrics only:

```text
teacher_example_count=3
base_row_count=120
holdout_example_count=0
map_objective_target_observed=true
map_target_observed=false
```

This first model is a scaffold, not a promotion candidate. The surviving teacher rows include aggregate map-objective deltas, but not complete per-channel Struct/OAC/SC/RSC map-delta supervision, so the generated queue labels map safety as `map_safety_not_trained`.

The explicit 120-row MS-SSIM comparison between base `learned-prior-sparse-p140-t32` and the best adaptive public rescue patch was written to:

```text
outputs/residual_selector_public_teacher/ms_ssim_120_comparison_base_vs_adaptive_rank36_patch.csv
outputs/residual_selector_public_teacher/ms_ssim_120_comparison_base_vs_adaptive_rank36_patch_summary.json
```

Summary:

```text
base MS-SSIM mean      = 0.6555505534821618
candidate MS-SSIM mean = 0.6653683801220829
mean delta             = +0.009817826639921047
median delta           = +0.008736216730646451
candidate wins         = 118
base wins              = 0
ties                   = 2
```

Interpretation: the 120-row comparison supports the core premise that residual flow+energy controls produce broad public-set MS-SSIM gains. It does not solve Structural LPIPS or hidden-holdout generalization. The next meaningful improvement is to retrain the control model with richer map-enriched teacher rows and validate a small smoke queue through the hosted true scanner.

## Current Lessons

- Preserve topology first; replace density only with strong evidence.
- Optimize the official aggregate, not Structural MS-SSIM alone.
- Treat Structural LPIPS as the bottleneck.
- Use true-scanner evidence for promotion; label surrogate and preview results clearly.
- Use residual controls with uncertainty and map-safety gates for the next attempt.

## Archive Policy

Ignored `outputs/` paths are machine-local evidence. After the 2026-07-05 cleanup, expanded phantoms, PNG renders, maps, transient probe folders, and old selector iterations were pruned. Keep compact CSV/JSON summaries, prior artifacts, package zips, manifests, and current queues.
