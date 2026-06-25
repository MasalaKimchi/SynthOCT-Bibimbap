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
