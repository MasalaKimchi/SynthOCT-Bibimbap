# Historical Experiment Notes

These notes preserve useful hypothesis context from older internal runs. They are not current ranking evidence. The offline renderer that produced the H0-H60 rankings was removed from the executable code, and submission decisions should use hosted SynthOCT API or official Windows `Part2_Scanner.exe` renders.

Current final method: `H61_api_low_depth_prelim`.

## H0-H40 Offline Screen

The first screens used an internal proxy that combined structural MS-SSIM, LPIPS proxy, and physics-map agreement. These runs suggested that simple inverse-physics sampling was more useful than blind phantoms or direct image-style tricks.

Key historical decisions:

- `H1_attenuation_density` beat the official-style random baseline in the invalidated offline renderer.
- Aggressive depth compensation and naive void inclusions were rejected.
- `H11_low_depth_comp` later replaced H1 as the offline base.
- The accepted theory was lower Beer-Lambert depth compensation with OAC-guided inhomogeneous point-process sampling.
- Added layer, void, or multilayer complexity did not automatically improve the proxy.

The H11-H40 expansion kept several concepts alive as priors: low depth compensation, OAC weighting, superlinear density, weak boundary priors, epidermal emphasis, and smooth multilayer variants.

## H41 Optimizer Candidate

`H41_final_optimized` came from a broad physics-parameter search. Its configuration explored:

- density exponent;
- depth compensation;
- OAC weight;
- texture weight;
- energy/OAC scaling;
- layer-boundary boost;
- lateral smoothing;
- OAC percentile;
- base energy mix;
- texture variance.

Historical 3 folds x 2 samples comparison:

```text
H41_final_optimized     proxy=0.4207  MS=0.1862  LPIPS_PROXY=0.1139  OAC_MS=0.4663  DepthCorr=0.8071
H11_low_depth_comp      proxy=0.4152  MS=0.1674  LPIPS_PROXY=0.1198  OAC_MS=0.5266  DepthCorr=0.8518
H0_official             proxy=0.3173  MS=-0.0035 LPIPS_PROXY=0.1471  OAC_MS=0.2266  DepthCorr=0.0683
```

Interpretation: H41 was best under the invalidated offline renderer, while H11 kept better OAC and depth-profile agreement.

## Subagent Council Study

Five strategy perspectives were converted into executable representatives:

- inverse PSF physics: `agent_inverse_psf`;
- speckle moments: `agent_speckle_moment`;
- anatomical boundaries: `agent_anatomical_boundary`;
- perceptual retrieval: `agent_perceptual_retrieval`;
- robust method selection: `portfolio`.

Historical 3 folds x 2 samples comparison:

```text
H41_final_optimized          proxy=0.4207  MS=0.1862  LPIPS_PROXY=0.1139  OAC_MS=0.4663  DepthCorr=0.8071
portfolio                    proxy=0.4190  MS=0.1806  LPIPS_PROXY=0.1156  OAC_MS=0.4816  DepthCorr=0.8184
agent_anatomical_boundary    proxy=0.4176  MS=0.1750  LPIPS_PROXY=0.1157  OAC_MS=0.5130  DepthCorr=0.8222
H11_low_depth_comp           proxy=0.4152  MS=0.1674  LPIPS_PROXY=0.1198  OAC_MS=0.5266  DepthCorr=0.8518
```

The council clarified the tradeoff: anatomy-heavy and guardrail-heavy methods could improve physical maps while hurting structural metrics. A future router or retrieval library may still be useful, but only if selected with nested folds and scanner-in-the-loop renders.

## H42-H60 Combination Pass

Direct blends between H41 and council directions mostly diluted H41. The best refinement was a bounded anti-blend:

```text
H56_h41_anti_anatomy         proxy=0.4244  MS=0.1922  LPIPS_PROXY=0.1127  OAC_MS=0.4625  DepthCorr=0.8037
H59_h41_anti_oac_guard       proxy=0.4237  MS=0.1911  LPIPS_PROXY=0.1124  OAC_MS=0.4636  DepthCorr=0.8030
H58_h41_anti_inverse         proxy=0.4237  MS=0.1908  LPIPS_PROXY=0.1124  OAC_MS=0.4628  DepthCorr=0.8022
H41_final_optimized          proxy=0.4230  MS=0.1897  LPIPS_PROXY=0.1131  OAC_MS=0.4624  DepthCorr=0.8020
```

Historical decision: `H56_h41_anti_anatomy` replaced H41 in the invalidated offline benchmark. That decision is superseded by `H61_api_low_depth_prelim`.

## API-Facing Candidates

Current API-facing methods are the only candidates that should be considered active:

```text
H61_api_low_depth_prelim  current final/default
H67_coarse_to_fine_crisp  structural candidate
H68_layer_map_prior       physical-map candidate
H11_low_depth_comp        conservative historical backup
```

Hosted API triage reported H67 as a promising MS-SSIM candidate and H68 as a physical-map candidate, but H61 remains the default until matched official scanner or hosted API runs justify promotion.

## Archived Artifact Pointers

Historical outputs may exist under ignored `outputs/` paths:

- `outputs/optimizer_broad/optimizer_summary.csv`
- `outputs/final_comparison/internal_validation_summary.csv`
- `outputs/agent_council/internal_validation_summary.csv`
- `outputs/council_micro_confirm/internal_validation_summary.csv`
- `outputs/submission_ready_h67_full/SUBMISSION_README.md`

Because `outputs/` is ignored and machine-local, treat these as breadcrumbs rather than required repository files.
