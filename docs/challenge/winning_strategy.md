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

On `2026-06-28`, a small matched hosted-API run compared the active H-series candidates on `2` grouped public samples at `300000` scatterers:

```text
H61_api_low_depth_prelim  MS-SSIM_mean=0.03080  LPIPS_PROXY_mean=0.05651  MS-SSIM_wins=2  LPIPS_wins=2
H67_coarse_to_fine_crisp  MS-SSIM_mean=0.02665  LPIPS_PROXY_mean=0.05773  MS-SSIM_wins=0  LPIPS_wins=0
H68_layer_map_prior       MS-SSIM_mean=0.01337  LPIPS_PROXY_mean=0.05681  MS-SSIM_wins=0  LPIPS_wins=0
```

The evidence audit passed as grouped, full-frame hosted true-scanner evidence, but only with `LPIPS_PROXY`. The promotion selector rejected H67 and H68, so `H61_api_low_depth_prelim` remained the then-current default at that point in the experiment log. This was not a hidden-holdout result; it was a small local gate that prevented promoting weaker candidates.

A follow-up learned-prior offset-holdout run trained an empirical prior from those true-scanner pairs and validated on later samples using `--sample-offset 1`. `learned-prior-structural` reached `MS-SSIM_mean=0.11676` versus H61 at `0.02946`, with better physical guardrails, but its `LPIPS_PROXY_mean=0.10594` was much worse than H61 at `0.05800`. The learned-prior branch is therefore the best current MS-SSIM direction, not a promoted final method. The immediate target is reducing learned-prior LPIPS/proxy distance while retaining its structural gains.

Density sparsification fixed that proxy failure, making `learned-prior-sparse-p140` the first strong learned-prior local candidate validated through the hosted true scanner on a `5`-fold x `1` sample offset split. After installing the real metric stack (`sewar`, `lpips`, and compatible `torchvision`), the saved true-scanner renders were rescored with real LPIPS and the `sewar` MS-SSIM backend:

```text
learned-prior-sparse-p140  MS-SSIM_mean=0.67634  LPIPS_mean=0.58467  MS-SSIM_wins=5  LPIPS_wins=5
H61_api_low_depth_prelim   MS-SSIM_mean=0.42519  LPIPS_mean=0.68546  MS-SSIM_wins=0  LPIPS_wins=0
```

`synthoct audit-evidence --strict --require-real-lpips`, `synthoct select-best --strict --require-real-lpips`, and `synthoct challenge-readiness --strict --require-real-lpips` pass for `learned-prior-sparse-p140` when using this metrics file. This is a local promotion for submission preparation, not proof of hidden-holdout victory.

The saved `5`-sample true-scanner renders were then rescored with Struct/OAC/SC/RSC map metrics and real LPIPS. The handout-style official aggregate score is:

```text
learned-prior-sparse-p140  official_score=0.67190  threshold_pass=0  threshold_failure=Struct_LPIPS>=0.4
H61_api_low_depth_prelim   official_score=0.52492  threshold_pass=0
```

This proves a stronger local public-split candidate than H61 under the full local metric stack, but it also exposes the next optimization target: reduce Structural LPIPS below the preliminary threshold without giving up the OAC/SC/RSC gains.

A follow-up `2`-fold hosted-API LPIPS variant check around `learned-prior-sparse-p140` found `learned-prior-sparse-p140-t32` slightly ahead on official score (`0.67839` versus `0.67403` for `p140`) and `learned-prior-sparse-p120` slightly lower on Structural LPIPS (`0.56568` versus `0.57119`). All variants still failed `Struct_LPIPS>=0.4`. This makes the current limitation sharper: small density-power, texture-weight, and energy-noise tweaks do not close the perceptual structural gap.

The `p140-t32` variant was then rerun on the same `5`-fold x `1` sample offset pattern as the current `p140` evidence. It scored `0.67703` under the official eight-median aggregate, compared with `0.67190` for `p140` and `0.52492` for H61. `p140-t32` is therefore the current local promoted package candidate, but it still fails the preliminary gate because Structural LPIPS remains `0.57616`, above `<0.4`.

The same `2`-fold offset split was used to recheck visual inverse pipelines (`P06`-`P09`) against `learned-prior-sparse-p140-t32`. The best visual recipe was `P09_gamma_sparse_lowfloor_ssim` with official score `0.38542` and Structural LPIPS `0.58958`, far below `p140-t32` at `0.67839`. Visual recipes remained target-locked enough to look conceptually tempting, but true-scanner map metrics show they damage OAC/SC/RSC too much to be competitive.

An initial scanner-compatible neural phantom-prior model was trained from `17` available true-scanner rows and evaluated on a separate offset-2 `2`-sample hosted-API split. It was not promoted: `neural-prior` scored `0.56240` versus `0.72217` for `learned-prior-sparse-p140-t32`. The important signal is narrower: neural-prior reduced Structural LPIPS to `0.45202` versus `0.56202` for p140-t32, but gave up Structural MS-SSIM and SC/RSC map LPIPS. This confirms that ML/DL can attack the right bottleneck, but the current model is not yet a winning generator. The next target is a hybrid that preserves p140-t32's OAC/SC/RSC behavior while importing neural-prior's lower perceptual structural distance.

The first hybrid attempt confirmed that density topology is fragile: small neural density blends damaged MS-SSIM and SC/RSC enough to lose badly. Energy-only blending was safer. `hybrid-neural-p140-t32-e20` beat `learned-prior-sparse-p140-t32` on a matched offset-2 `5`-sample hosted-API run (`official_score=0.70240` versus `0.70051`) and slightly lowered Structural LPIPS (`0.59613` versus `0.59778`). But an offset-1 `5`-sample check reversed the official-score ordering (`0.67652` versus `0.67703`), and the merged offset-1 plus offset-2 diagnostic summary leaves `p140-t32` slightly ahead (`0.68760` versus `0.68674`).

A stronger energy-only blend, `hybrid-neural-p140-t32-e60`, improved Structural LPIPS more consistently and beat `p140-t32` on separate offset-1 and offset-2 `5`-sample runs. The combined offset-1 plus offset-2 `10`-sample grouped diagnostic still did not promote it: `p140-t32` scored `0.68760`, while `e60` scored `0.68710`, despite better Structural LPIPS (`0.58103` versus `0.58697`). The hybrid remains useful evidence that neural energy shaping attacks the right bottleneck, not a robust replacement for p140-t32 yet.

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
