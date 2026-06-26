# H41 Council Combination Study

This study is historical and was invalidated by hosted API renders. Do not use these rankings for submission decisions; the current final method is `H61_api_low_depth_prelim`, and submission candidates must be rerendered through the hosted SynthOCT API or official Windows `Part2_Scanner.exe`.

## Objective

H41 was the strongest internal method after broad physics optimization. The agent council suggested several orthogonal directions: inverse PSF physics, speckle moment matching, anatomical layer priors, perceptual retrieval, and robust method selection. This study tested whether those directions should be combined with H41 rather than only compared against it.

Validation used an offline renderer that has been removed from the codebase. The target score was `CompetitionProxy`, a leaderboard-oriented internal proxy that weights structural MS-SSIM, LPIPS proxy, and physics-map agreement.

## Combination Wave 1: Direct Blends

Tested methods:

- `H42_h41_anatomy_blend`
- `H43_h41_oac_guarded`
- `H44_h41_perceptual_blend`
- `H45_h41_speckle_calibrated`
- `H46_h41_inverse_structure`
- `H47_h41_anatomy_perceptual`
- `H48_h41_council_balanced`
- `H49_h41_structural_guarded`
- `H50_h41_texture_guarded`
- `council-combo`
- `council-combo-conservative`

Result: direct averaging mostly diluted H41. H44 was closest but still below H41:

```text
H41_final_optimized          proxy=0.4207
portfolio                    proxy=0.4190
H44_h41_perceptual_blend     proxy=0.4185
agent_anatomical_boundary    proxy=0.4176
```

Decision: reject direct council averaging as the final method.

## Combination Wave 2: Micro-Blends And Anti-Blends

The first wave implied that the council directions were often downhill from H41 under the internal competition proxy. The second wave therefore tested tiny blends and bounded extrapolations away from harmful directions:

- `H51_h41_micro_perceptual`
- `H52_h41_micro_anatomy`
- `H53_h41_micro_inverse`
- `H54_h41_micro_speckle`
- `H55_h41_micro_anatomy_perceptual`
- `H56_h41_anti_anatomy`
- `H57_h41_anti_speckle`
- `H58_h41_anti_inverse`
- `H59_h41_anti_oac_guard`
- `H60_h41_micro_portfolio_centroid`

Initial 3 folds x 2 samples result:

```text
H56_h41_anti_anatomy         proxy=0.4216
H59_h41_anti_oac_guard       proxy=0.4209
H41_final_optimized          proxy=0.4207
H57_h41_anti_speckle         proxy=0.4203
H58_h41_anti_inverse         proxy=0.4202
```

Confirmatory 3 folds x 4 samples result:

```text
H56_h41_anti_anatomy         proxy=0.4244  MS=0.1922  LPIPS_PROXY=0.1127  OAC_MS=0.4625  DepthCorr=0.8037
H59_h41_anti_oac_guard       proxy=0.4237  MS=0.1911  LPIPS_PROXY=0.1124  OAC_MS=0.4636  DepthCorr=0.8030
H58_h41_anti_inverse         proxy=0.4237  MS=0.1908  LPIPS_PROXY=0.1124  OAC_MS=0.4628  DepthCorr=0.8022
H41_final_optimized          proxy=0.4230  MS=0.1897  LPIPS_PROXY=0.1131  OAC_MS=0.4624  DepthCorr=0.8020
portfolio                    proxy=0.4223  MS=0.1857  LPIPS_PROXY=0.1146  OAC_MS=0.4802  DepthCorr=0.8137
```

Per-sample wins in the confirmatory run:

```text
H56_h41_anti_anatomy         5 / 12
H58_h41_anti_inverse         2 / 12
H59_h41_anti_oac_guard       2 / 12
H55_h41_micro_anatomy_perceptual 1 / 12
H41_final_optimized          1 / 12
agent_anatomical_boundary    1 / 12
```

## Interpretation

The surprising outcome is that the best council combination is an anti-blend, not an ensemble. The anatomical agent itself improved OAC and depth guardrails, but it reduced structural MS-SSIM. Extrapolating slightly away from that anatomical parameter direction increased the metrics closest to the leaderboard target while preserving H41's physical behavior.

H56 changes H41 only modestly. It slightly increases density power, reduces boundary boost, lowers lateral smoothing, raises OAC percentile, and nudges energy statistics away from the anatomy-heavy prior. The likely effect is crisper scanner-space structure with less over-smoothed boundary emphasis.

## Historical Decision

At the time of this offline study, `H56_h41_anti_anatomy` replaced H41 as the best internal offline method, with `portfolio` kept as a conservative backup because it had better physical guardrails but lower competition proxy.

That decision has since been superseded. `synthoct baseline final` now targets `H61_api_low_depth_prelim`, and preliminary portal upload should use rendered synthetic/reference PNG pairs, not raw phantom zips.

## Caveat

This result is invalidated for ranking. Hosted API or Windows `Part2_Scanner.exe` renders should decide any submission candidate.

Artifacts:

- `outputs/council_combinations/internal_validation_summary.csv`
- `outputs/council_combinations/council_combinations_progress.png`
- `outputs/council_micro_combinations/internal_validation_summary.csv`
- `outputs/council_micro_combinations/council_micro_progress.png`
- `outputs/council_micro_confirm/internal_validation_summary.csv`
- `outputs/council_micro_confirm/council_micro_confirm_progress.png`
