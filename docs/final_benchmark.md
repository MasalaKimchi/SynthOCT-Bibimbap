# Final Internal Benchmark

## Groundbreaking Direction

The best next idea is not another hand-written layer rule. It is **physics-constrained hyperparameter search over a differentiable-in-spirit phantom parameterization**:

1. Represent tissue as an inhomogeneous scatterer point process.
2. Estimate density from raw intensity, OAC, speckle contrast, and Beer-Lambert depth compensation.
3. Keep the output challenge-compliant: only `X, Y, Z, Energy` scatterers.
4. Search physically meaningful knobs: depth compensation, OAC percentile, density exponent, energy mixing, texture variance, boundary boost, lateral smoothing.
5. Select by cross-fold competition proxy: MS-SSIM up, LPIPS/proxy down, with OAC/SC/RSC as guardrails.

This is a small, honest surrogate for a future scanner-in-the-loop optimizer.

## Optimizer Candidate

`H41_final_optimized` was compiled from the optimizer winner `O_random_68`.

Configuration:

```json
{
  "density_power": 1.1950643094718914,
  "depth_compensation": 1.232449572578015,
  "oac_weight": 2.1664221698464448,
  "texture_weight": 0.3718513055230929,
  "energy_oac_scale": 28.504499297377823,
  "band_boost": 0.32309648245405514,
  "lateral_smooth": 0.5263087292628759,
  "oac_percentile": 81.52821510765463,
  "base_energy_mix": 0.7099850978498786,
  "texture_sigma_scale": 1.2453729328071115
}
```

## H41 Confirmatory Comparison

Validation: 3 folds x 2 samples/fold, maps enabled, 6000 scatterers per generated phantom.

```text
H41_final_optimized     proxy=0.4207  MS=0.1862  LPIPS_PROXY=0.1139  OAC_MS=0.4663  DepthCorr=0.8071
H11_low_depth_comp      proxy=0.4152  MS=0.1674  LPIPS_PROXY=0.1198  OAC_MS=0.5266  DepthCorr=0.8518
H34_epidermal_emphasis  proxy=0.4142  MS=0.1670  LPIPS_PROXY=0.1190  OAC_MS=0.5209  DepthCorr=0.8293
H1_attenuation_density  proxy=0.4072  MS=0.1523  LPIPS_PROXY=0.1245  OAC_MS=0.5259  DepthCorr=0.8466
H0_official             proxy=0.3173  MS=-0.0035 LPIPS_PROXY=0.1471  OAC_MS=0.2266  DepthCorr=0.0683
```

Improvement vs official baseline:

- Competition proxy: `+32.6%` relative improvement.
- Structural MS-SSIM proxy: from approximately `-0.0035` to `0.1862`.
- LPIPS proxy: `22.6%` lower.
- OAC MS-SSIM: `105.8%` higher.

Tradeoff:

- H41 is best for the competition-weighted surrogate.
- H11 remains the conservative physics candidate because it keeps better OAC and depth-profile agreement.

## Current Final Candidate

The council-combination pass promoted `H56_h41_anti_anatomy`, a small bounded extrapolation away from the anatomy-heavy council direction. This keeps H41's core physics while reducing the boundary/anatomy smoothing that hurt structural MS-SSIM.

Configuration:

```json
{
  "density_power": 1.2074694542296427,
  "depth_compensation": 1.2254455383842562,
  "oac_weight": 2.1717359434341605,
  "texture_weight": 0.37759940996494035,
  "energy_oac_scale": 27.82485924116805,
  "band_boost": 0.29294420105037955,
  "lateral_smooth": 0.4764134276039059,
  "oac_percentile": 82.290472316267,
  "base_energy_mix": 0.7203839056778689,
  "texture_sigma_scale": 1.2690027674316804
}
```

Confirmatory validation: 3 folds x 4 samples/fold, maps enabled, 6000 scatterers.

```text
H56_h41_anti_anatomy    proxy=0.4244  MS=0.1922  LPIPS_PROXY=0.1127  OAC_MS=0.4625  DepthCorr=0.8037
H59_h41_anti_oac_guard  proxy=0.4237  MS=0.1911  LPIPS_PROXY=0.1124  OAC_MS=0.4636  DepthCorr=0.8030
H58_h41_anti_inverse    proxy=0.4237  MS=0.1908  LPIPS_PROXY=0.1124  OAC_MS=0.4628  DepthCorr=0.8022
H41_final_optimized     proxy=0.4230  MS=0.1897  LPIPS_PROXY=0.1131  OAC_MS=0.4624  DepthCorr=0.8020
portfolio               proxy=0.4223  MS=0.1857  LPIPS_PROXY=0.1146  OAC_MS=0.4802  DepthCorr=0.8137
```

Decision: `synthoct baseline final` now uses H56. H41 remains the previous incumbent and should still be checked on Windows because H56's margin is small.

## Artifacts

- Optimizer summary: `outputs/optimizer_broad/optimizer_summary.csv`
- Optimizer top-20 plot: `outputs/optimizer_broad/optimizer_top20.png`
- Final comparison summary: `outputs/final_comparison/internal_validation_summary.csv`
- Final comparison figure: `outputs/final_comparison/final_comparison.png`
- Council combination summary: `outputs/council_micro_confirm/internal_validation_summary.csv`
- Council combination figure: `outputs/council_micro_confirm/council_micro_confirm_progress.png`
- H41 example submission artifact: `outputs/synthoct_h41_final_submission.zip`

## Next Groundbreaking Step

Move from surrogate search to **official scanner-in-the-loop Bayesian optimization** on Windows:

1. Use H41 and H11 as priors.
2. Generate candidate phantoms.
3. Run `Part2_Scanner.exe`.
4. Score real MS-SSIM/LPIPS on Struct/OAC/SC/RSC.
5. Update the candidate distribution.

That is the highest-leverage path because it optimizes the actual hidden objective rather than our macOS surrogate.
