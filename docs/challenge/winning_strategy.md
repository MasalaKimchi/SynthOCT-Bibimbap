# Winning Strategy Reset

This document records the current interpretation of the SynthOCT Challenge rules and what counts as real evidence for winning.

## What Winning Means

The official task is to submit a generator, code, or model that produces scanner-compatible digital phantoms from real OCT B-scans. The organizers run the submitted approach on a hidden hold-out test dataset to determine final score and ranking.

Winning therefore means:

- generate valid `X Y Z Energy` scatterer tables for each input B-scan;
- keep phantom generation within `600` seconds per B-scan;
- run on the stated target environment: Windows 10/11, `16GB RAM`, `RTX 3060 12GB`, Python `3.12.12`, and the recommended PyTorch version when a model is used;
- render through the fixed SynthOCT Virtual Scanner, either the official `Part2_Scanner.exe` or the organizers' hosted scanner/API;
- rank well on the official challenge-facing metrics, especially `MS-SSIM` and `LPIPS`, with OAC, speckle contrast, refined speckle contrast, and depth-profile diagnostics used as physical guardrails.

The project should not claim competition progress from direct image synthesis, preview renders, local surrogate predictions, or single-reference manual tuning unless those candidates are then rendered by the fixed scanner and validated on a representative multi-sample split.

## Scanner Vocabulary

- **True scanner**: the official fixed physics scanner. Locally this is `Part2_Scanner.exe`; remotely this is the hosted SynthOCT API endpoint used by the organizers for portal/API renders. These are the only scanner paths that should count as challenge evidence.
- **Surrogate scanner**: any learned or approximate local model trained to predict scanner output. It is useful for cheap search and gradients, but it is not the official scanner and must never be treated as proof of leaderboard performance.
- **Preview renderer**: any direct image proxy, feature-field preview, deterministic lattice preview, or local non-scanner visualization. It is only a triage tool.

## Status Of The `0.773537950274` Result

The `0.773537950274` MS-SSIM result was a true hosted-API render for a single visible/reference B-scan, and the local metric function compared the full image array after resizing the prediction to the reference shape. It was not a cropped-patch-only score.

However, it was not a competition-wide result:

- it was not evaluated on the full public dataset;
- it was not evaluated on hidden hold-out data;
- it did not include a robust mean/std/win-rate summary across grouped folds;
- it was produced by local single-reference optimization artifacts, not necessarily by a fully reproducible generator that generalizes to arbitrary hidden B-scans;
- it did not establish the final `LPIPS` tradeoff, and in many runs only `LPIPS_PROXY` was available.

The number is useful as proof that scanner-in-loop amplitude/energy correction can reach a local high score on one case. It is not evidence that the method would win.

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

## Fair Competitive Direction

The most credible path is a hybrid approach:

1. Build a representative local validation split from the public Zenodo dataset, grouped by subject/site/frame to reduce leakage.
2. Generate baseline phantoms and render them through the true scanner on that split.
3. Train a surrogate scanner only as a search accelerator, with strict hold-out validation against true scanner renders.
4. Train or fit a phantom generator that predicts layered density, attenuation, speckle statistics, and energy fields from the input B-scan, then samples valid scatterers.
5. Use scanner-in-loop refinement only as a final polishing stage, constrained to runtime and reproducibility.
6. Evaluate every promoted candidate through the true scanner across the validation split and summarize mean, standard deviation, and per-sample wins for `MS-SSIM` and `LPIPS`.
7. Package the final generator, not a hand-tuned output folder, and verify it runs within the challenge hardware/runtime envelope.

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
  --require-real-lpips
```

`promotion_ready=true` means the CSV is local grouped true-scanner evidence that is fair to use for candidate comparison. It still does not mean hidden-holdout victory; only organizer execution on the hidden test set can establish the final competition result.

## Evidence Labels In Artifacts

Metric CSVs should carry explicit evidence labels so results cannot be misread:

- `evidence_source=hosted_api_true_scanner`: rendered by the hosted SynthOCT API, which is challenge-style true-scanner evidence.
- `evidence_source=learned_surrogate_preview`: predicted by a local learned surrogate; useful for candidate generation only.
- `evidence_scope=single_reference_candidate_queue`: one visible/reference B-scan, not a competition-wide result.
- `evidence_scope=grouped_validation_*`: public grouped validation over multiple samples.
- `evidence_scope=submission_manifest_render`: a submission manifest rendered through the hosted API.
- `evidence_scope=not_challenge_evidence`: preview-only evidence that must not be promoted without true-scanner rendering.

Surrogate preview metrics should use `surrogate_*` column names rather than bare challenge metric names such as `MS-SSIM`, so downstream ranking code does not accidentally treat them as official evidence.
