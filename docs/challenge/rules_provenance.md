# Challenge Evidence Provenance

This file records what the repository can prove locally about SynthOCT challenge evidence, and what still depends on the organizers.

## Local Source Status

- `18095266.zip` contains `DATASET.zip` only; no embedded scanner executable was found in the local dataset archive.
- The local challenge handout `314-SynthOCT_2026_Digital_Phantoms_Simulation_for_Physics-Based_Scans_2026-04-22T16-37-10.pdf` describes `120` public training/validation B-scans, `60` hidden final test B-scans, the hosted platform at `http://synthoct.com/`, and the baseline/evaluation repository at `https://github.com/SynthOCTChallenge/SynthOCT_Baseline`.
- The code has two true-scanner paths: hosted API rendering through `https://synthoct.com/process_oct`, and the Windows `Part2_Scanner.exe` wrapper.
- Learned surrogate predictions, preview renderers, direct lattices, feature-map previews, and single-reference candidate queues are not official scanner evidence.

## Scanner Evidence Policy

The evidence audit treats only these `evidence_source` values as true-scanner evidence:

```text
hosted_api_true_scanner
official_windows_true_scanner
```

The audit deliberately reports:

```text
surrogate_scanner_is_true_scanner = false
```

That means a surrogate scanner can help search or debug candidates, but it does not indicate that the hosted API or official scanner was used. A candidate must be rendered by the hosted API or `Part2_Scanner.exe` before its metrics can support promotion.

## Current Local Candidate Evidence

The current strongest local candidate is `learned-prior-sparse-p140-t32`, using:

- prior artifact: `outputs/learned_priors/goal_h_candidates_prior.npz`;
- true-scanner metrics: `outputs/api_validation_goal_t32_p140_h61_offset1_5x1/challenge_metrics_summary.csv`;
- full package: `outputs/submission_ready_learned_prior_sparse_p140_t32_full_corrected`.

The current real-LPIPS, full-frame, hosted true-scanner comparison is:

```text
learned-prior-sparse-p140-t32  n=5  MS-SSIM=0.67705  LPIPS=0.58603  MS-SSIM_wins=5  LPIPS_wins=5
learned-prior-sparse-p140      n=5  MS-SSIM=0.67634  LPIPS=0.58467  MS-SSIM_wins=5  LPIPS_wins=5
H61_api_low_depth_prelim       n=5  MS-SSIM=0.42519  LPIPS=0.68546  MS-SSIM_wins=0  LPIPS_wins=0
```

This is strong enough for local submission preparation under the repo's fair-evidence gate. It is not proof of competition victory.

Under the handout-style mean of eight medians, `learned-prior-sparse-p140-t32` scores `0.67703` versus `0.67190` for `p140` and `0.52492` for H61 on this `5`-sample public split. It still does not pass the preliminary threshold gate because Structural LPIPS is `0.57616`, above the `<0.4` threshold.

The earlier uncorrected full package path was retired after a reference-PNG scale bug was found in submission packaging. Use only the `_corrected` package for local upload preparation.

## What Winning Still Requires

Official winning requires organizer-side execution of the submitted generator/model on the hidden hold-out dataset and final ranking by the challenge metrics. The handout describes the main leaderboard as the mean of median `MS-SSIM` and median inverted `LPIPS` (`1 - LPIPS`) across four categories: Structural intensity, OAC, SC, and RSC. Local commands therefore always report:

```text
hidden_holdout_final_score = false
official_final_ranking_proven = false
```

until an organizer-issued hidden-holdout result is added as evidence.
