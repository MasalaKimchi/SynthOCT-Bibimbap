# Subagent Council Synthesis

## Setup

H41 remains the incumbent. Five subagents attacked the challenge from different perspectives:

1. Inverse physics / OCT forward model.
2. Statistical speckle / stochastic processes.
3. Anatomical / dermatology layer model.
4. Perceptual / CNN representation learning.
5. Optimization / ensemble / benchmarking.

Each proposed a fundamentally different strategy. I converted the compact, immediately testable parts into executable representatives:

- `agent_inverse_psf`
- `agent_speckle_moment`
- `agent_anatomical_boundary`
- `agent_perceptual_retrieval`
- `portfolio`

These were compared against `H41_final_optimized`, `H11_low_depth_comp`, and `H0_official`.

## Benchmark Result

Validation: 3 folds x 2 samples/fold, 6000 scatterers, maps enabled.

```text
H41_final_optimized          proxy=0.4207  MS=0.1862  LPIPS_PROXY=0.1139  OAC_MS=0.4663  DepthCorr=0.8071
portfolio                    proxy=0.4190  MS=0.1806  LPIPS_PROXY=0.1156  OAC_MS=0.4816  DepthCorr=0.8184
agent_anatomical_boundary    proxy=0.4176  MS=0.1750  LPIPS_PROXY=0.1157  OAC_MS=0.5130  DepthCorr=0.8222
agent_perceptual_retrieval   proxy=0.4165  MS=0.1779  LPIPS_PROXY=0.1152  OAC_MS=0.4686  DepthCorr=0.8027
H11_low_depth_comp           proxy=0.4152  MS=0.1674  LPIPS_PROXY=0.1198  OAC_MS=0.5266  DepthCorr=0.8518
agent_inverse_psf            proxy=0.4090  MS=0.1645  LPIPS_PROXY=0.1222  OAC_MS=0.4575  DepthCorr=0.7989
agent_speckle_moment         proxy=0.4073  MS=0.1544  LPIPS_PROXY=0.1204  OAC_MS=0.5126  DepthCorr=0.8155
H0_official                  proxy=0.3173  MS=-0.0035 LPIPS_PROXY=0.1471  OAC_MS=0.2266  DepthCorr=0.0683
```

Decision: keep H41 as the current final submission method. Keep `portfolio` and `agent_anatomical_boundary` as serious backups because they are close to H41 and better on physical guardrails.

## Perspective Summaries

### A. Inverse Physics

Core idea: approximate the scanner inverse. Undo log compression and attenuation, deconvolve a PSF-like blur, place scatterers so the expected scanner field matches the observed B-scan.

Most promising future variant: alternating inverse rendering with the official Windows scanner in the loop.

Current executable proxy, `agent_inverse_psf`, did not beat H41 in the invalidated offline run. The concept must be retested with hosted API renders.

### B. Speckle Statistics

Core idea: treat speckle as the inverse target. Match local mean, variance, contrast, Gamma/Nakagami-like shape, and autocorrelation using marked point processes.

Current executable proxy, `agent_speckle_moment`, improved physical texture guardrails relative to official baseline but did not beat H41. The concept remains promising, especially as a post-H41 calibration layer.

### C. Anatomical Layer Model

Core idea: skin is layered tissue, not just an attenuation field. Model surface, epidermis, DEJ boundary, dermis, and sparse inclusions with site/age-aware priors.

Current executable proxy, `agent_anatomical_boundary`, came closest among non-H41 single-perspective variants and improved OAC/depth guardrails. This is the strongest non-H41 scientific direction.

### D. Perceptual / CNN

Core idea: use learned feature geometry to select or predict phantom parameters, while still outputting scatterers. Best near-term idea is retrieval-augmented H41.

Current executable proxy, `agent_perceptual_retrieval`, is competitive but not better than H41. A real version would need a candidate library and embeddings from actual rendered phantoms.

### E. Meta-Optimization / Ensemble

Core idea: stop looking for one universal generator. Route each scan to the best hypothesis or candidate family using input-derived descriptors.

Current executable proxy, `portfolio`, is second-best and improves OAC/depth guardrails. It did not beat H41 on the competition proxy, but it is the most robust-looking backup.

## Expanded Understanding Of The Problem

The main nuisance is non-identifiability: many scatterer distributions can produce similar OCT scans after blur, attenuation, speckle, and log compression. A method can improve physical maps while hurting MS-SSIM/LPIPS, or improve perceptual proxy while weakening OAC/depth plausibility.

Other nuisance factors:

- mismatch between any offline approximation and the official scanner;
- unknown hidden-test distribution;
- skin site and age/sex heterogeneity;
- layer-boundary ambiguity in weak-contrast scans;
- speckle randomness and seed sensitivity;
- metric conflict between structure, perceptual texture, OAC, SC, and RSC;
- large text phantoms and runtime constraints for 300k scatterers;
- risk of optimizing public/reference dataset structure instead of hidden challenge data.

## Recommendations

1. Submit H56 as current best after the council-combination update.
2. Keep H41 as the previous incumbent and H11/anatomical-boundary as conservative physics backups.
3. Next true breakthrough: official scanner-in-the-loop alternating inverse rendering.
4. Next practical improvement: build a candidate library and per-scan retrieval/router, but train/select it with nested folds to avoid winner's curse.
5. Add site-stratified reporting before trusting any anatomy-heavy method.

## Update: H41 Combination Pass

After this council comparison, H42-H60 tested direct blends, micro-blends, and anti-blends between H41 and the council directions. Direct averaging did not beat H41. The best refinement was `H56_h41_anti_anatomy`, a small bounded extrapolation away from the anatomy-heavy parameter direction:

```text
H56_h41_anti_anatomy         proxy=0.4244
H41_final_optimized          proxy=0.4230
```

Current final method: `H56_h41_anti_anatomy`. See `docs/council_combinations.md`.

## Artifacts

- Comparison CSV: `outputs/agent_council/internal_validation_summary.csv`
- Figure: `outputs/agent_council/agent_council.png`
