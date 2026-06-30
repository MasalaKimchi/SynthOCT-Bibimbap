# Brief Manuscript Report: Current Best SynthOCT Algorithm

## Abstract

The best general submission algorithm identified so far is `learned-prior-sparse-p140-t32`, an empirical learned-prior digital phantom generator. It does not synthesize OCT images directly; instead, it converts each input B-scan into a challenge-valid scatterer table with `X Y Z Energy` columns, which must then be rendered by the SynthOCT true scanner. On grouped hosted true-scanner validation with real LPIPS and Struct/OAC/SC/RSC maps, it outperformed both the earlier `learned-prior-sparse-p140` variant and the conservative `H61_api_low_depth_prelim` baseline. The highest measured public-set artifact is a tactical extension, `p140-t32-flow-energy-rank120-adaptive-rank36-patch`, which applies optical-flow and energy-ratio corrections to selected public-set rows. That patch improves measured public-set score but should be interpreted as public-set rescue evidence, not hidden-holdout proof.

## Introduction

The SynthOCT challenge requires a reproducible generator that outputs scanner-compatible digital phantoms, not final rendered OCT images. The local project therefore treats hosted SynthOCT API or official Windows scanner renders as the only meaningful challenge evidence. Earlier H-series physics priors, visual inverse recipes, and standalone neural-prior experiments were useful for exploration, but the current evidence favors an empirical prior that preserves physical-map behavior while improving structural similarity.

## Methods

The promoted generator uses the artifact `outputs/learned_priors/goal_h_candidates_prior.npz`, built by distilling prior true-scanner phantom/render pairs into empirical density and energy fields. The artifact is not itself challenge evidence; it is a reusable prior that conditions later phantom generation.

For each target B-scan, `learned-prior-sparse-p140-t32` computes scanner-relevant features from the real OCT reference: normalized intensity, linearized optical attenuation coefficient (OAC), speckle contrast, local detail, and an estimated tissue boundary. These target-derived fields are blended with the empirical prior fields using:

- `target_blend = 0.62`
- `prior_blend = 0.38`
- `texture_weight = 0.32`
- `density_power = 14.0`
- `energy_sigma = 0.10`

The high density power sparsifies the sampling distribution, addressing overly bright learned-prior renders observed in earlier variants. The `t32` texture setting modestly increases target-detail contribution relative to `p140`. The final density and energy maps are sampled into `300000` scatterers, with coordinates clipped to scanner bounds and energy clipped to the valid challenge range. The output remains a plain text `X Y Z Energy` phantom file.

Validation used full-frame hosted true-scanner renders with real LPIPS enabled. The primary local score is the handout-style aggregate using median MS-SSIM and median inverted LPIPS across Structural intensity, OAC, speckle contrast (SC), and refined speckle contrast (RSC) maps.

The public-set rescue extension starts from the packaged `p140-t32` phantoms, applies optical-flow coordinate transport plus energy-ratio feedback to API-identified weak rows, and promotes only rows that improved in public-set true-scanner renders. The adaptive weak-row version retests alternative flow strengths on the weakest remaining rows.

## Results

On the grouped 5-fold, 1-sample-per-fold offset validation, `learned-prior-sparse-p140-t32` achieved the best local official-style score:

| Method | n | Official score | Struct MS-SSIM median | Struct LPIPS median |
|---|---:|---:|---:|---:|
| `learned-prior-sparse-p140-t32` | 5 | 0.67703 | 0.70000 | 0.57616 |
| `learned-prior-sparse-p140` | 5 | 0.67190 | 0.70040 | 0.58624 |
| `H61_api_low_depth_prelim` | 5 | 0.52492 | 0.43925 | 0.66842 |

On the full 120-pair public-set hosted API render, the base `p140-t32` package scored:

| Artifact | n | Official score | MS-SSIM mean | Struct MS-SSIM median | Struct LPIPS median |
|---|---:|---:|---:|---:|---:|
| `learned-prior-sparse-p140-t32` | 120 | 0.68479 | 0.65555 | 0.65296 | 0.58736 |

The best measured public-set rescue artifact improved the public-set aggregate:

| Artifact | n | Official score | MS-SSIM mean | Struct MS-SSIM median | Struct LPIPS median |
|---|---:|---:|---:|---:|---:|
| `p140-t32-flow-energy-rank120-adaptive-rank36-patch` | 120 | 0.69400 | 0.66537 | 0.66053 | 0.58850 |

Both the base and patched artifacts still fail the preliminary threshold because Structural LPIPS remains above `0.4`.

## Discussion

The evidence supports `learned-prior-sparse-p140-t32` as the best general algorithm so far because it is a reproducible generator, runs through the challenge-compatible phantom interface, and wins the grouped true-scanner comparison against the previous learned-prior and H61 candidates. It also preserves strong OAC/SC/RSC map behavior compared with visual inverse and standalone neural-prior alternatives.

The flow+energy adaptive patch has the highest measured public-set score, but it is selected using public validation failures and per-row true-scanner feedback. That makes it valuable as a rescue layer and as evidence that residual alignment can improve many cases, but weaker as a hidden-holdout generalization claim. The conservative final-generator interpretation should therefore remain `learned-prior-sparse-p140-t32`, with the patch described separately as a public-set postprocessing extension.

The main unresolved bottleneck is perceptual structural similarity. Neural-prior experiments reduced Structural LPIPS in some splits, but they degraded MS-SSIM and physical-map metrics enough that they did not replace `p140-t32`. A next-generation method should learn residual density/energy or parameter selection from true-scanner evidence while constraining OAC, SC, and RSC map preservation.

## Conclusion

The current best general SynthOCT-Bibimbap method is an empirical learned-prior sparse phantom generator, `learned-prior-sparse-p140-t32`, packaged with `outputs/learned_priors/goal_h_candidates_prior.npz`. The strongest measured public-set artifact adds a flow+energy rescue patch and reaches an official-style score of `0.6939983205067786`, but hidden-holdout performance remains unproven. The remaining challenge is to lower Structural LPIPS without sacrificing the physical-map gains that made the learned-prior method successful.

## Evidence Files

- Grouped validation: `outputs/api_validation_goal_t32_p140_h61_offset1_5x1/challenge_metrics_summary.csv`
- Full public-set base render: `outputs/api_preliminary_learned_prior_sparse_p140_t32_120_concurrent/challenge_metrics_summary.csv`
- Full public-set rescue patch render: `outputs/api_preliminary_p140_t32_flow_energy_rank120_adaptive_rank36_patch/challenge_metrics_summary.csv`
- Conservative full package: `outputs/submission_ready_learned_prior_sparse_p140_t32_full_corrected`
- Best measured public-set rescue package: `outputs/submission_ready_p140_t32_flow_energy_rank120_adaptive_rank36_patch`
- Current Stage 2 residual journey and resume note: `docs/challenge/stage2_oac_residual_journey.md`
