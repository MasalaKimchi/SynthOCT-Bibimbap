# H0-H10 Internal Validation Progress

Official metric check: the SynthOCT baseline README states that leaderboard ranking uses **MS-SSIM** and **LPIPS**. The challenge page also emphasizes physical consistency via OAC and speckle statistics. Our internal y-axis therefore uses `CompetitionProxy`: structural MS-SSIM, LPIPS or LPIPS proxy, and physics-map MS-SSIM/SSIM.

This is a macOS surrogate validation curve, not an official scanner score. Promote only hypotheses that survive cross-fold validation and later confirm on Windows with `Part2_Scanner.exe`.

## Current Results

Run:

```bash
PYTHONPATH=src python -m synthoct.cli validate-internal \
  --zip 18095266.zip \
  --out outputs/hypothesis_progress \
  --methods H0_official H1_attenuation_density H2_depth_compensated \
    H3_speckle_matched H4_oac_dominant H5_boundary_band \
    H6_void_inclusions H7_lateral_coherence H8_multilayer \
    H9_log_energy H10_refined_attenuation_band \
  --folds 3 \
  --max-per-fold 2 \
  --scatterers-count 6000 \
  --plot outputs/hypothesis_progress/hypothesis_progress.png
```

## Hypothesis Decisions

- H1 attenuation-density sampling: accepted as the current winner. It best preserves depth attenuation and OAC structure.
- H2 aggressive depth compensation: rejected; overcompensates depth and hurts MS-SSIM.
- H3 speckle-matched texture: promising but behind H1; keep as a component.
- H4 OAC-dominant sampling: promising, but too much OAC weighting reduces depth profile robustness.
- H5 boundary band: accepted as near-winner; strongest structural MS-SSIM/LPIPS proxy tradeoff.
- H6 void inclusions: rejected for now; void prior likely needs learned placement.
- H7 lateral coherence: neutral; smooths texture but does not beat H1.
- H8 multilayer prior: rejected; too many priors degrade the surrogate score.
- H9 log-energy compression: neutral; improves stability but loses structural MS-SSIM.
- H10 refined attenuation band: rejected as a winner; confirms H1 is a cleaner current direction than H1 plus boundary prior.

## Next Iteration

Start from H1, not H10. Sweep only one axis at a time:

1. OAC weight around `1.8-2.4`.
2. Depth compensation around `1.5-2.1`.
3. Texture weight around `0.25-0.45`.
4. Optional weak boundary band around `0.0-0.25`.

Then confirm the top two with the official Windows scanner.

## H11-H40 Expansion

A second theory-driven screen added 30 hypotheses, documented in [theory_context.md](theory_context.md). The broad screen used 3 folds x 1 sample/fold, then the top candidates were confirmed on 3 folds x 2 samples/fold.

Confirmatory ranking:

```text
H11_low_depth_comp               proxy=0.4158
H34_epidermal_emphasis           proxy=0.4129
H29_low_depth_high_oac           proxy=0.4126
H17_superlinear_density          proxy=0.4120
H32_superlinear_boundary         proxy=0.4109
H40_conservative_winner          proxy=0.4109
H12_mid_depth_comp               proxy=0.4107
H37_multi_layer_smooth           proxy=0.4100
H36_multi_layer_light            proxy=0.4084
H1_attenuation_density           proxy=0.4073
H0_official                      proxy=0.3198
```

Decision: H11 replaces H1 as the current base. The accepted theory is lower Beer-Lambert depth compensation with OAC-guided inhomogeneous point-process sampling. The rejected theory is that added layer/void/multilayer complexity automatically improves the competition proxy.

## H41-H60 Council Combination Update

H41 later replaced H11 after broad random/local optimization. A follow-up council study tested direct H41 blends with inverse-physics, speckle, anatomy, perceptual, and portfolio hypotheses. Direct blends H42-H50 did not beat H41; they generally diluted the structural metric gains.

The successful refinement was H56, a small bounded extrapolation away from the anatomy-heavy council direction. On the 3 folds x 4 samples confirmation:

```text
H56_h41_anti_anatomy         proxy=0.4244
H59_h41_anti_oac_guard       proxy=0.4237
H58_h41_anti_inverse         proxy=0.4237
H41_final_optimized          proxy=0.4230
portfolio                    proxy=0.4223
```

Decision: promote `H56_h41_anti_anatomy` as the current internal final method, with H41 retained as the previous incumbent and Windows scanner confirmation still required.
