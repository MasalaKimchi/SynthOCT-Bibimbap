# Winning Strategy Reset

This document records the current interpretation of the SynthOCT Challenge rules and what counts as real evidence for winning.

## What Winning Means

The official task is to submit a generator, code, or model that produces scanner-compatible digital phantoms from real OCT B-scans. The organizers run the submitted approach on a hidden hold-out test dataset to determine final score and ranking.

Winning therefore means:

- generate valid `X Y Z Energy` scatterer tables for each input B-scan;
- keep phantom generation within `600` seconds per B-scan;
- run on the stated target environment: Windows 10/11, `16GB RAM`, `RTX 3060 12GB`, Python `3.12.12`, and the recommended PyTorch version when a model is used;
- render through the fixed SynthOCT Virtual Scanner, either the official `Part2_Scanner.exe` or the organizers' hosted scanner/API;
- rank well on the official challenge-facing metrics: median `MS-SSIM` and median inverted `LPIPS` (`1 - LPIPS`) across Structural intensity, OAC, SC, and RSC maps. Depth-profile diagnostics remain useful local guardrails but are not the handout's main leaderboard formula.

The project should not claim competition progress from direct image synthesis, preview renders, local surrogate predictions, or single-reference manual tuning unless those candidates are then rendered by the fixed scanner and validated on a representative multi-sample split.

## Scanner Vocabulary

- **True scanner**: the official fixed physics scanner. Locally this is `Part2_Scanner.exe`; remotely this is the hosted SynthOCT API endpoint used by the organizers for portal/API renders. These are the only scanner paths that should count as challenge evidence.
- **Surrogate scanner**: any learned or approximate local model trained to predict scanner output. It is useful for cheap search and gradients, but it is not the official scanner and must never be treated as proof of leaderboard performance.
- **Preview renderer**: any direct image proxy, feature-field preview, deterministic lattice preview, or local non-scanner visualization. It is only a triage tool.

The local source trail and machine-readable evidence policy are summarized in [rules_provenance.md](rules_provenance.md).

## Status Of The `0.773537950274` Result

The `0.773537950274` MS-SSIM result was a true hosted-API render for a single visible/reference B-scan, and the local metric function compared the full image array after resizing the prediction to the reference shape. It was not a cropped-patch-only score.

However, it was not a competition-wide result:

- it was not evaluated on the full public dataset;
- it was not evaluated on hidden hold-out data;
- it did not include a robust mean/std/win-rate summary across grouped folds;
- it was produced by local single-reference optimization artifacts, not necessarily by a fully reproducible generator that generalizes to arbitrary hidden B-scans;
- it did not establish the final `LPIPS` tradeoff, and in many runs only `LPIPS_PROXY` was available.

The number is useful as proof that scanner-in-loop amplitude/energy correction can reach a local high score on one case. It is not evidence that the method would win.

Metric CSVs now include evaluation metadata such as `evaluation_region=full_frame`, `reference_shape`, `prediction_shape`, `evaluated_shape`, and `prediction_resized_to_reference` so full-image scores cannot be confused with crop or patch scores.

## Current True-Scanner Candidate Status

Current local status:

- `learned-prior-sparse-p140-t32` is the conservative local submission candidate.
- Required artifact: `outputs/learned_priors/goal_h_candidates_prior.npz`.
- Current evidence file: `outputs/api_validation_goal_t32_p140_h61_offset1_5x1/challenge_metrics_summary.csv`.
- Current corrected full package: `outputs/submission_ready_learned_prior_sparse_p140_t32_full_corrected`.
- Best measured full-public-set rescue artifact: `outputs/submission_ready_p140_t32_flow_energy_rank120_adaptive_rank36_patch`, with evidence at `outputs/api_preliminary_p140_t32_flow_energy_rank120_adaptive_rank36_patch/challenge_metrics_summary.csv`.
- `H61_api_low_depth_prelim` remains the fallback if learned-prior packaging or evidence checks fail.

```text
public-set p140-t32 flow+energy adaptive rank36 patch   official_score=0.69400  Struct_LPIPS=0.58850  threshold_pass=0
public-set p140-t32 flow+energy adaptive weak12 patch   official_score=0.69292  Struct_LPIPS=0.58804  threshold_pass=0
public-set p140-t32 flow+energy rank120 positive patch  official_score=0.69245  Struct_LPIPS=0.58813  threshold_pass=0
learned-prior-sparse-p140-t32  official_score=0.67703  Struct_LPIPS=0.57616  threshold_pass=0
learned-prior-sparse-p140      official_score=0.67190  Struct_LPIPS=0.58624  threshold_pass=0
H61_api_low_depth_prelim       official_score=0.52492  Struct_LPIPS=0.66842  threshold_pass=0
```

The flow+energy patch is selected from public-set worst cases and improves the measured public-set floor; the adaptive weak-row sweep shows that per-sample parameter selection can add small gains. It is not yet a hidden-general generator. Use it as a rescue artifact and as evidence for a future learned residual/parameter-selection module, not as proof that the operator solves the challenge.

This is enough for local submission preparation under strict real-LPIPS hosted true-scanner evidence. It is not hidden-holdout proof, and it still fails the preliminary Structural LPIPS gate.

Recent rejected or exploratory branches:

- H67/H68 lost to H61 on the initial small hosted-API gate.
- Visual inverse pipelines (`P06`-`P09`) did not beat p140-t32 under full Struct/OAC/SC/RSC metrics.
- Standalone `neural-prior` improved Structural LPIPS but lost too much Structural MS-SSIM and SC/RSC map quality.
- Hybrid neural energy blends (`e20`, `e60`) reduced Structural LPIPS, but broader offset-1 plus offset-2 grouped summaries still slightly favor p140-t32 on official aggregate.

See [../history/experiments.md](../history/experiments.md) for the dated evidence trail and [ml_dl_expert_review.md](ml_dl_expert_review.md) for the current ML/DL assessment.

The latest Stage 2 residual-selector work is summarized in [stage2_oac_residual_journey.md](stage2_oac_residual_journey.md). That note records the decision to freeze Stage 1 as `p140-t32` topology, use flow+energy patches as teacher data, and require uncertainty-gated Struct/OAC/SC/RSC map safety before spending hosted API calls.

The next execution plan is [next_approach.md](next_approach.md). It makes the current strategy explicit: run the map-safe selector queue as an incremental true-scanner feedback pass, then build a topology-preserving residual trainer rather than replacing the `p140-t32` density field.

## Why Prior Optimization Was Far From Competition-Optimal

The strongest branch so far was coordinate-preserving energy-ratio feedback plus tiny axial and global amplitude calibration. That is a local correction method, not a general learned inverse model. Its limitations are structural:

- It depends on a known reference/render pair and repeatedly queries the true scanner, which is expensive and brittle.
- It optimized one visible case more than population-level generalization.
- It discovered that the 900k scatterer topology was fragile: coordinate jitter, added scatterers, residual injection, and optical-flow transport hurt MS-SSIM.
- It had limited degrees of freedom after finding a basin; most remaining scalar knobs saturated.
- It did not learn anatomical or tissue priors from the dataset.
- It was vulnerable to metric mismatch: MSE/SSIM/texture suppression could improve while MS-SSIM worsened.
- It could not use gradients through the true scanner.
- API outages made broad scanner-in-loop search unreliable.
- Current learned-prior scalar variants still leave Structural LPIPS around `0.56-0.57`, far above the preliminary `<0.4` gate, which points to a missing learned texture/anatomy model rather than a simple energy-noise calibration issue.
- Current visual inverse recipes can chase structural appearance but collapse the physical map metrics, so they do not satisfy the challenge objective even when they seem intuitively closer to the target image.

## Fair Competitive Direction

The most credible path is a hybrid approach:

1. Build a representative local validation split from the public Zenodo dataset, grouped by subject/site/frame to reduce leakage.
2. Generate baseline phantoms and render them through the true scanner on that split.
3. Train a surrogate scanner only as a search accelerator, with strict hold-out validation against true scanner renders.
4. Train or fit a phantom generator that predicts layered density, attenuation, speckle statistics, and energy fields from the input B-scan, then samples valid scatterers.
5. Use scanner-in-loop refinement only as a final polishing stage, constrained to runtime and reproducibility.
6. Evaluate every promoted candidate through the true scanner across the validation split and summarize medians for `MS-SSIM` and `1 - LPIPS` across Struct/OAC/SC/RSC, plus mean/std and per-sample wins for debugging.
7. Package the final generator, not a hand-tuned output folder, and verify it runs within the challenge hardware/runtime envelope.

When complete official Struct/OAC/SC/RSC medians are available, local promotion should be based primarily on the official aggregate score and runtime. Structural-only mean MS-SSIM/LPIPS and internal profile guardrails remain debugging signals, but they should not veto a higher complete official score.

The official-style objective should be treated as the mean of eight terms: median `MS-SSIM` and median `1 - LPIPS` for `Struct`, `OAC`, `SC`, and `RSC`. This makes single-metric optimization unsafe. A candidate that lowers Structural LPIPS but damages SC/RSC can lose even if it looks visually better.

## Promotion Gate

A method should become the new final candidate only if it beats the current final generator under true-scanner validation on a representative split by at least one of:

- higher mean `MS-SSIM` without worse `LPIPS`;
- lower mean `LPIPS` without worse `MS-SSIM`;
- more per-sample wins on challenge-facing metrics;
- similar challenge metrics with better OAC/SC/RSC guardrails and no runtime penalty.

Single-image improvements, surrogate-only improvements, and preview-only improvements are not enough.

Use the executable evidence audit before promoting a candidate:

```bash
synthoct audit-evidence \
  --metrics outputs/api_validation/challenge_metrics_summary.csv \
  --strict \
  --require-real-lpips \
  --max-generation-seconds 600
```

`promotion_ready=true` means the CSV is local grouped true-scanner evidence that is fair to use for candidate comparison. It still does not mean hidden-holdout victory; only organizer execution on the hidden test set can establish the final competition result.

Strict promotion also requires `evaluation_region=full_frame` and `generation_seconds_max <= 600`, because crop-only, unlabeled, or over-budget metric rows are not strong enough evidence for final-candidate promotion.

Then compare the candidate against the current baseline or conservative fallback:

```bash
synthoct decide-promotion \
  --metrics outputs/api_validation/challenge_metrics_summary.csv \
  --candidate H_new_candidate \
  --baseline H61_api_low_depth_prelim \
  --strict \
  --require-real-lpips \
  --max-generation-seconds 600
```

The decision requires grouped true-scanner full-frame evidence, generation within the `600` second challenge budget, higher mean `MS-SSIM`, no worse `LPIPS`/proxy, no worse per-sample win counts, and no regression in available physical guardrails such as depth correlation, OAC profile correlation, speckle-contrast error, and OAC/SC/RSC map similarity. Passing this gate means “promote locally for submission preparation,” not “claim hidden-holdout victory.”

To rank all validated methods at once, use:

```bash
synthoct select-best \
  --metrics outputs/api_validation/challenge_metrics_summary.csv \
  --baseline H61_api_low_depth_prelim \
  --strict \
  --require-real-lpips \
  --max-generation-seconds 600
```

This returns the best candidate that clears the same fair-evidence gate. If no method clears the gate, keep the baseline rather than promoting a weaker or less fairly evaluated method.

If a small physical-guardrail tolerance is intentionally needed, pass `--max-guardrail-regression`, but treat that as a documented risk rather than a default path.

For a single local readiness answer, run:

```bash
synthoct challenge-readiness \
  --metrics outputs/api_validation/challenge_metrics_summary.csv \
  --baseline H61_api_low_depth_prelim \
  --method H_new_candidate \
  --strict \
  --require-real-lpips \
  --max-generation-seconds 600
```

This command combines the evidence audit and candidate selection checks. It can return `local_candidate_ready=true` only for grouped true-scanner full-frame evidence where the named method is the selected promoted candidate. It always reports `hidden_holdout_final_score=false`, because no local command can prove the official final ranking.

When creating final artifacts, pass the same metrics file to `prepare-submission`:

```bash
synthoct prepare-submission \
  --zip 18095266.zip \
  --out outputs/submission_ready_selected \
  --method H_new_candidate \
  --evidence-metrics outputs/api_validation/challenge_metrics_summary.csv \
  --baseline H61_api_low_depth_prelim \
  --strict-evidence \
  --max-generation-seconds 600
```

This writes `submission_readiness_report.json` and refuses to package an unselected method in strict mode.

After packaging, include the submission directory in the same readiness check:

```bash
synthoct challenge-readiness \
  --metrics outputs/api_validation/challenge_metrics_summary.csv \
  --baseline H61_api_low_depth_prelim \
  --method H_new_candidate \
  --submission-dir outputs/submission_ready_selected \
  --strict \
  --max-generation-seconds 600
```

## Evidence Labels In Artifacts

Metric CSVs should carry explicit evidence labels so results cannot be misread:

- `evidence_source=hosted_api_true_scanner`: rendered by the hosted SynthOCT API, which is challenge-style true-scanner evidence.
- `evidence_source=learned_surrogate_preview`: predicted by a local learned surrogate; useful for candidate generation only.
- `evidence_scope=single_reference_candidate_queue`: one visible/reference B-scan, not a competition-wide result.
- `evidence_scope=grouped_validation_*`: public grouped validation over multiple samples.
- `evidence_scope=submission_manifest_render`: a submission manifest rendered through the hosted API.
- `evidence_scope=not_challenge_evidence`: preview-only evidence that must not be promoted without true-scanner rendering.

Surrogate preview metrics should use `surrogate_*` column names rather than bare challenge metric names such as `MS-SSIM`, so downstream ranking code does not accidentally treat them as official evidence.

Metric rows should also carry `evaluation_region=full_frame` for structural challenge metrics. Missing evaluation-region labels are treated as insufficient promotion evidence.

## ML/DL Use Boundary

The fair ML/DL path is not a direct image generator. It is a hybrid inverse system:

1. train a surrogate scanner on true-scanner phantom/render pairs;
2. reserve held-out true-scanner renders and record `surrogate_calibration_metrics.csv`;
3. use the surrogate only to search phantom density/energy fields or initialize a learned phantom generator;
4. render selected phantoms through the hosted API or official Windows scanner;
5. promote only after `synthoct audit-evidence --strict` passes on grouped true-scanner metrics.

The `optimize-learned-surrogate` command writes candidate preview rows with `evidence_source=learned_surrogate_preview` and writes holdout calibration rows with `evidence_source=learned_surrogate_holdout`. Both are intentionally `evidence_scope=not_challenge_evidence`.

The `train-phantom-prior` plus `baseline learned-prior` path gives learned or empirical phantom-field artifacts a scanner-compatible generator interface. It is still candidate generation, not scoring: prior artifacts are `not_challenge_evidence`, and their generated phantoms need true-scanner grouped validation before promotion.

The executable ML/DL generator lane is now `train-neural-phantom-prior` plus `baseline neural-prior`. It trains a small CNN from true-scanner validation rows that contain `reference_png`, `phantom_path`, and `synthetic_gray_png`, then predicts density/energy fields for new reference scans and samples valid scatterer tables. This is the right challenge shape for ML/DL because it outputs phantoms, not final OCT images. It does not change the evidence rule: the `.pt` model artifact is `not_challenge_evidence`, and `neural-prior` can only replace `learned-prior-sparse-p140-t32` after grouped hosted-API or official-Windows validation beats it under real Struct/OAC/SC/RSC MS-SSIM and LPIPS.

The next ML/DL direction is narrower than full neural replacement: keep `p140-t32` density topology fixed, then learn residual energy, low-frequency geometry, attenuation, and speckle-statistic controls with uncertainty-gated promotion. The model may use surrogate predictions for search, but only held-out true-scanner calibration and grouped hosted validation can make it promotable.
