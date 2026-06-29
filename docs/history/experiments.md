# Historical Experiment Notes

These notes preserve useful hypothesis context from older internal runs. They are not current ranking evidence. The offline renderer that produced the H0-H60 rankings was removed from the executable code, and submission decisions should use hosted SynthOCT API or official Windows `Part2_Scanner.exe` renders.

Current promoted local submission candidate: `learned-prior-sparse-p140-t32`, using `outputs/learned_priors/goal_h_candidates_prior.npz` and real-LPIPS evidence from `outputs/api_validation_goal_t32_p140_h61_offset1_5x1/challenge_metrics_summary.csv`.

Conservative legacy fallback: `H61_api_low_depth_prelim`.

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
H61_api_low_depth_prelim  then-current default before learned-prior promotion
H67_coarse_to_fine_crisp  structural candidate
H68_layer_map_prior       physical-map candidate
H11_low_depth_comp        conservative historical backup
```

Hosted API triage reported H67 as a promising MS-SSIM candidate and H68 as a physical-map candidate, but H61 remains the default until matched official scanner or hosted API runs justify promotion.

## 2026-06-28 Matched Hosted-API Check

A matched hosted-API run compared the active API-facing candidates on a small grouped split:

```bash
synthoct validate-internal \
  --zip 18095266.zip \
  --out outputs/api_validation_goal_h_candidates_2x1 \
  --methods H61_api_low_depth_prelim H67_coarse_to_fine_crisp H68_layer_map_prior \
  --folds 2 \
  --max-per-fold 1 \
  --scatterers-count 300000 \
  --no-maps \
  --api-key-file ~/.config/synthoct/api_key
```

Challenge summary:

```text
H61_api_low_depth_prelim  n=2  MS-SSIM=0.03080  LPIPS_PROXY=0.05651  MS-SSIM_wins=2  LPIPS_wins=2
H67_coarse_to_fine_crisp  n=2  MS-SSIM=0.02665  LPIPS_PROXY=0.05773  MS-SSIM_wins=0  LPIPS_wins=0
H68_layer_map_prior       n=2  MS-SSIM=0.01337  LPIPS_PROXY=0.05681  MS-SSIM_wins=0  LPIPS_wins=0
```

`synthoct audit-evidence --strict` passed because the rows were grouped, full-frame, hosted true-scanner evidence with generation time below `600` seconds. `synthoct select-best --baseline H61_api_low_depth_prelim --strict` correctly rejected H67 and H68: neither beat H61 on mean MS-SSIM, LPIPS proxy, per-sample wins, or available physical guardrails.

Interpretation: this run is real local challenge-style evidence but still very small and proxy-LPIPS only. It strengthened the decision to keep H61 as the default until a candidate beat it on a larger grouped true-scanner split with real LPIPS.

## 2026-06-28 Learned-Prior Offset Holdout

The H-candidate true-scanner pairs above were distilled into an empirical phantom prior:

```bash
synthoct train-phantom-prior \
  --outputs-dir outputs/api_validation_goal_h_candidates_2x1 \
  --out outputs/learned_priors/goal_h_candidates_prior.npz \
  --shape 128 256 \
  --pair-limit 6
```

To reduce leakage, learned-prior candidates were validated on `--sample-offset 1`, not the first-per-fold samples used to build the prior:

```bash
synthoct validate-internal \
  --zip 18095266.zip \
  --out outputs/api_validation_goal_learned_prior_variants_offset1_2x1 \
  --methods H61_api_low_depth_prelim learned-prior-conservative learned-prior-balanced learned-prior-structural \
  --learned-prior-artifact outputs/learned_priors/goal_h_candidates_prior.npz \
  --folds 2 \
  --max-per-fold 1 \
  --sample-offset 1 \
  --scatterers-count 300000 \
  --no-maps \
  --api-key-file ~/.config/synthoct/api_key
```

Challenge summary:

```text
learned-prior-structural     n=2  MS-SSIM=0.11676  LPIPS_PROXY=0.10594  MS-SSIM_wins=2  LPIPS_wins=0
learned-prior-balanced       n=2  MS-SSIM=0.11043  LPIPS_PROXY=0.10591  MS-SSIM_wins=0  LPIPS_wins=0
learned-prior-conservative   n=2  MS-SSIM=0.10026  LPIPS_PROXY=0.10639  MS-SSIM_wins=0  LPIPS_wins=0
H61_api_low_depth_prelim     n=2  MS-SSIM=0.02946  LPIPS_PROXY=0.05800  MS-SSIM_wins=0  LPIPS_wins=2
```

Interpretation: the learned prior is the strongest true-scanner MS-SSIM direction found so far on an offset holdout, and it greatly improves depth/OAC/speckle guardrails. It is not promotable under the current strict gate because LPIPS proxy is much worse and real LPIPS is not available in this environment. The next optimization target is preserving the learned-prior structural gain while reducing perceptual/proxy distance.

## 2026-06-28 Ultrasparse Learned-Prior Promotion

The LPIPS proxy failure above came from overly bright learned-prior renders. Rendered means were around `0.59-0.63` while the references and H61 renders were near `0.10`. Energy scaling had little effect, so the useful control was density sparsification.

On the same `2`-fold offset split, `learned-prior-sparse-p140` became the first candidate to pass the local fair-evidence promotion gate:

```text
learned-prior-sparse-p140  n=2  MS-SSIM=0.21349  LPIPS_PROXY=0.05357  MS-SSIM_delta=+0.18403  LPIPS_PROXY_delta=-0.00443
H61_api_low_depth_prelim   n=2  MS-SSIM=0.02946  LPIPS_PROXY=0.05800
```

A larger `5`-fold x `1` sample offset validation confirmed the direction:

```bash
synthoct validate-internal \
  --zip 18095266.zip \
  --out outputs/api_validation_goal_p140_vs_h61_offset1_5x1 \
  --methods H61_api_low_depth_prelim learned-prior-sparse-p140 \
  --learned-prior-artifact outputs/learned_priors/goal_h_candidates_prior.npz \
  --folds 5 \
  --max-per-fold 1 \
  --sample-offset 1 \
  --scatterers-count 300000 \
  --no-maps \
  --api-key-file ~/.config/synthoct/api_key
```

Initial proxy-only challenge summary:

```text
learned-prior-sparse-p140  n=5  MS-SSIM=0.19766  LPIPS_PROXY=0.05626  MS-SSIM_wins=5  LPIPS_wins=5
H61_api_low_depth_prelim   n=5  MS-SSIM=0.01931  LPIPS_PROXY=0.06015  MS-SSIM_wins=0  LPIPS_wins=0
```

After installing the real metric stack (`sewar==0.4.6`, `lpips==0.1.4`, `torchvision==0.21.0` with `torch==2.6.0`), the same saved hosted true-scanner renders were rescored with `--include-lpips`. The absolute MS-SSIM values changed because the evaluator now uses `sewar.full_ref.msssim` instead of the fallback implementation, but both methods were recomputed under the same backend:

```text
learned-prior-sparse-p140  n=5  MS-SSIM=0.67634  LPIPS=0.58467  MS-SSIM_wins=5  LPIPS_wins=5
H61_api_low_depth_prelim   n=5  MS-SSIM=0.42519  LPIPS=0.68546  MS-SSIM_wins=0  LPIPS_wins=0
```

`synthoct audit-evidence --strict --require-real-lpips`, `synthoct select-best --strict --require-real-lpips`, and `synthoct challenge-readiness --strict --require-real-lpips` selected `learned-prior-sparse-p140` at this stage. The strict smoke package at `outputs/submission_ready_learned_prior_sparse_p140_5x1_smoke` and the full public-set package at `outputs/submission_ready_learned_prior_sparse_p140_full` were generated with `--require-real-lpips`, and their `submission_readiness_report.json` files were marked `ready`. That package covered all `120` public PNG B-scans. It was later superseded by `learned-prior-sparse-p140-t32`; final competition ranking still requires organizer hidden-holdout execution.

The saved `5`-sample API renders were later rescored with Struct/OAC/SC/RSC map LPIPS to match the handout's ranking formula more closely:

```text
learned-prior-sparse-p140  official_score=0.67190  preliminary_threshold_pass=0  failure=Struct_LPIPS>=0.4
H61_api_low_depth_prelim   official_score=0.52492  preliminary_threshold_pass=0
```

This keeps P140 as the best local candidate, while making Structural LPIPS the next competitive bottleneck.

## 2026-06-29 Learned-Prior LPIPS Variant Check

A targeted hosted-API run tested small learned-prior variants around `learned-prior-sparse-p140` on the same `2`-fold x `1` sample offset split style, this time with real LPIPS and Struct/OAC/SC/RSC maps enabled:

```bash
synthoct validate-internal \
  --zip 18095266.zip \
  --out outputs/api_validation_goal_lpips_variants_offset1_2x1 \
  --methods learned-prior-sparse-p120 learned-prior-sparse-p140 learned-prior-sparse-p160 learned-prior-sparse-p140-t32 learned-prior-sparse-p140-sigma16 learned-prior-sparse-p140-sigma22 \
  --learned-prior-artifact outputs/learned_priors/goal_h_candidates_prior.npz \
  --folds 2 \
  --max-per-fold 1 \
  --sample-offset 1 \
  --scatterers-count 300000 \
  --include-lpips \
  --api-key-file ~/.config/synthoct/api_key
```

Official-score ranking on this small split:

```text
learned-prior-sparse-p140-t32     official_score=0.67839  Struct_LPIPS=0.56779  threshold_pass=0
learned-prior-sparse-p140         official_score=0.67403  Struct_LPIPS=0.57119  threshold_pass=0
learned-prior-sparse-p140-sigma16 official_score=0.67391  Struct_LPIPS=0.57047  threshold_pass=0
learned-prior-sparse-p140-sigma22 official_score=0.67377  Struct_LPIPS=0.56976  threshold_pass=0
learned-prior-sparse-p120         official_score=0.67198  Struct_LPIPS=0.56568  threshold_pass=0
learned-prior-sparse-p160         official_score=0.67016  Struct_LPIPS=0.57103  threshold_pass=0
```

Interpretation: modest texture increase (`p140-t32`) improved official score on the tiny split, and lower density power (`p120`) slightly lowered Structural LPIPS. Neither direction comes close to the `<0.4` Structural LPIPS gate. Increasing scatterer energy lognormal variance (`sigma16`/`sigma22`) did not materially solve perceptual texture. This suggests the remaining gap is not a one-knob sparsity or energy-noise issue; it likely requires a stronger learned inverse generator or calibrated surrogate-guided field optimization.

## 2026-06-29 Visual-Inversion Recheck

The visual inverse pipelines were retested against the learned-prior leader on the same `2`-fold x `1` sample offset pattern with real LPIPS and full Struct/OAC/SC/RSC maps:

```bash
synthoct validate-internal \
  --zip 18095266.zip \
  --out outputs/api_validation_goal_visual_vs_learned_offset1_2x1 \
  --methods learned-prior-sparse-p140-t32 P06_visual_surface_dark_body P07_surface_cutoff_broad_mix P08_sparse_top_texture_ssim P09_gamma_sparse_lowfloor_ssim H61_api_low_depth_prelim \
  --learned-prior-artifact outputs/learned_priors/goal_h_candidates_prior.npz \
  --folds 2 \
  --max-per-fold 1 \
  --sample-offset 1 \
  --scatterers-count 300000 \
  --include-lpips \
  --api-key-file ~/.config/synthoct/api_key
```

Official-score ranking:

```text
learned-prior-sparse-p140-t32  official_score=0.67839  Struct_LPIPS=0.56779
H61_api_low_depth_prelim       official_score=0.54207  Struct_LPIPS=0.66816
P09_gamma_sparse_lowfloor_ssim official_score=0.38542  Struct_LPIPS=0.58958
P07_surface_cutoff_broad_mix   official_score=0.35054  Struct_LPIPS=0.61825
P08_sparse_top_texture_ssim    official_score=0.33268  Struct_LPIPS=0.59932
P06_visual_surface_dark_body   official_score=0.32690  Struct_LPIPS=0.62031
```

Interpretation: the target-locked visual recipes did not solve Structural LPIPS, and they heavily regressed OAC/SC/RSC metrics. This rejects the current visual-inversion family as a competitive direction unless it is rebuilt with map-aware training or a calibrated surrogate; it should not displace learned-prior methods.

## 2026-06-29 P140-T32 Five-Fold Promotion

The `p140-t32` learned-prior variant was rerun on the `5`-fold x `1` sample offset pattern used by the previous `p140` evidence:

```bash
synthoct validate-internal \
  --zip 18095266.zip \
  --out outputs/api_validation_goal_p140_t32_offset1_5x1 \
  --methods learned-prior-sparse-p140-t32 \
  --learned-prior-artifact outputs/learned_priors/goal_h_candidates_prior.npz \
  --folds 5 \
  --max-per-fold 1 \
  --sample-offset 1 \
  --scatterers-count 300000 \
  --include-lpips \
  --api-key-file ~/.config/synthoct/api_key
```

Its summary was merged with the existing matched `p140` and H61 rows at `outputs/api_validation_goal_t32_p140_h61_offset1_5x1/challenge_metrics_summary.csv`:

```text
learned-prior-sparse-p140-t32  official_score=0.67703  Struct_LPIPS=0.57616  threshold_pass=0
learned-prior-sparse-p140      official_score=0.67190  Struct_LPIPS=0.58624  threshold_pass=0
H61_api_low_depth_prelim       official_score=0.52492  Struct_LPIPS=0.66842  threshold_pass=0
```

Because the complete handout-style Struct/OAC/SC/RSC metric set is available, the local promotion gate now treats the official aggregate as the primary challenge metric instead of letting structural-only mean LPIPS or internal profile guardrails veto a higher official score. `learned-prior-sparse-p140-t32` is therefore the current local package candidate. It is still not hidden-holdout proof, and Structural LPIPS remains the primary unsolved failure.

The corrected full public-set package is `outputs/submission_ready_learned_prior_sparse_p140_t32_full_corrected`, with `120` manifest rows, `120` validation rows, strict evidence readiness, and a code zip that includes `artifacts/goal_h_candidates_prior.npz`.

## 2026-06-29 Submission Packaging Scale Bug

A visual inspection of the preliminary PNG pairs showed that the synthetic and real scans were not plausibly aligned. The root cause was in submission packaging, not candidate selection: loaded dataset PNGs can arrive as floats in `[0, 1]`, and one packaging path wrote them directly to `uint8` without multiplying by `255`. That made temporary reference PNGs nearly black before phantom generation. The validation path was already scaling correctly, so the validated candidate and the packaged candidate had diverged.

The bad full package at `outputs/submission_ready_learned_prior_sparse_p140_t32_full` should not be uploaded. The corrected package was regenerated at:

```text
outputs/submission_ready_learned_prior_sparse_p140_t32_full_corrected
```

The corrected 10-pair hosted-API check is in `outputs/api_preliminary_learned_prior_sparse_p140_t32_10_corrected/api_metrics.csv`. The bad package averaged `MS-SSIM=0.4745` and `LPIPS=0.6324` on those 10 API-rendered pairs; the corrected package improved to `MS-SSIM=0.6821` and `LPIPS=0.5771`.

The preliminary portal single-pair evaluation for `004_l__shcheka_frame450_reference.png` reported:

```text
Struct  MS-SSIM=0.7005  LPIPS=0.5810
OAC     MS-SSIM=0.7986  LPIPS=0.3100
SC      MS-SSIM=0.7290  LPIPS=0.3589
RSC     MS-SSIM=0.7623  LPIPS=0.3548
Average MS-SSIM=0.7476  LPIPS=0.4012
```

This is useful official-style single-pair evidence for the corrected package, but it is still not a grouped or hidden-holdout result. Structural LPIPS remains the main weakness.

## 2026-06-29 Neural Phantom-Prior Scaffold

The current results suggest the remaining gap is not a scalar density, energy-noise, or visual-inversion issue. Structural LPIPS remains around `0.56-0.58`, while the preliminary gate requires `<0.4`. To open the higher-ceiling ML/DL route without violating the challenge boundary, the repo now has an executable neural phantom-prior path:

```bash
synthoct train-neural-phantom-prior \
  --outputs-dir outputs \
  --out outputs/neural_priors/phantom_field_prior.pt \
  --epochs 80

synthoct baseline neural-prior \
  --input data/DATASET_PNG/Female/1950-1960/Cheek/l__shcheka_frame250.png \
  --artifact outputs/neural_priors/phantom_field_prior.pt \
  --out outputs/neural_prior.txt
```

This trains a small CNN from existing true-scanner rows with reference images and phantom files, then predicts scanner-compatible density/energy fields for new references. The output is still a four-column scatterer table, so it is a fair candidate-generator architecture rather than direct image synthesis.

An initial artifact was trained from the currently discoverable `17` true-scanner rows at `outputs/neural_priors/goal_true_scanner_prior.pt` with metadata at `outputs/neural_priors/goal_true_scanner_prior.json`. A smoke generation from this artifact produced a finite `4096 x 4` scanner-bounds phantom.

The first hosted true-scanner check used an offset-2 `2`-fold x `1` split:

```bash
synthoct validate-internal \
  --zip 18095266.zip \
  --out outputs/api_validation_neural_prior_offset2_2x1 \
  --methods neural-prior learned-prior-sparse-p140-t32 H61_api_low_depth_prelim \
  --neural-prior-artifact outputs/neural_priors/goal_true_scanner_prior.pt \
  --learned-prior-artifact outputs/learned_priors/goal_h_candidates_prior.npz \
  --folds 2 \
  --max-per-fold 1 \
  --sample-offset 2 \
  --scatterers-count 300000 \
  --include-lpips \
  --api-key-file ~/.config/synthoct/api_key
```

Official-score ranking:

```text
learned-prior-sparse-p140-t32  official_score=0.72217  Struct_MS=0.69877  Struct_LPIPS=0.56202
H61_api_low_depth_prelim       official_score=0.58020  Struct_MS=0.48262  Struct_LPIPS=0.65599
neural-prior                   official_score=0.56240  Struct_MS=0.51412  Struct_LPIPS=0.45202
```

Interpretation: `neural-prior` is not promoted. It trails `p140-t32` by `-0.15977` official score and regresses OAC/SC/RSC map metrics, especially SC/RSC LPIPS around `0.57`. But it is the first direction that materially improves Structural LPIPS, cutting it from `0.56202` to `0.45202` on this split and moving much closer to the `<0.4` preliminary gate. The next competitive direction is therefore a hybrid: preserve p140-t32 density/physical-map behavior while using neural predictions or losses to reduce perceptual structural distance.

## 2026-06-29 Hybrid Neural Energy Blend

Two hybrid families were tested against `learned-prior-sparse-p140-t32` on the same offset-2 validation pattern:

- density blends `hybrid-neural-p140-t32-b05/b10/b15/b20`;
- energy-only blends `hybrid-neural-p140-t32-e05/e10/e20`.

The density blends were rejected. Even a `5%` neural density contribution reduced Structural LPIPS slightly but collapsed MS-SSIM and SC/RSC map quality:

```text
learned-prior-sparse-p140-t32  official_score=0.72217  Struct_MS=0.69877  Struct_LPIPS=0.56202
hybrid-neural-p140-t32-b05     official_score=0.47606  Struct_MS=0.43270  Struct_LPIPS=0.53572
```

The energy-only blend preserved p140-t32's density behavior and gave the first matched true-scanner improvement over the current promoted candidate. On a `5`-fold x `1` offset-2 hosted-API run:

```text
hybrid-neural-p140-t32-e20     official_score=0.70240  Struct_MS=0.62916  Struct_LPIPS=0.59613
learned-prior-sparse-p140-t32  official_score=0.70051  Struct_MS=0.62997  Struct_LPIPS=0.59778
```

`audit-evidence --strict --require-real-lpips` passed, and `decide-promotion --candidate hybrid-neural-p140-t32-e20 --baseline learned-prior-sparse-p140-t32` returned `promote=true` on this matched split. The gain is small (`+0.00189` official score) and Structural LPIPS remains far above `<0.4`, so this is a cautious local package candidate rather than a solved competition entry.

A strict full public-set package was generated at `outputs/submission_ready_hybrid_neural_p140_t32_e20_full`. It contains `120` manifest rows, `120` validation rows, includes both `artifacts/goal_h_candidates_prior.npz` and `artifacts/goal_true_scanner_prior.pt` in the code zip, passes zip integrity checks, and `challenge-readiness` reports `local_candidate_ready=true`, `hidden_holdout_final_score=false`.

A follow-up matched offset-1 `5`-sample run reversed the official-score ordering:

```text
learned-prior-sparse-p140-t32  official_score=0.67703  Struct_MS=0.70000  Struct_LPIPS=0.57616
hybrid-neural-p140-t32-e20     official_score=0.67652  Struct_MS=0.70099  Struct_LPIPS=0.57141
```

Combining the offset-1 and offset-2 detail rows into a `10`-sample grouped diagnostic summary at `outputs/api_validation_hybrid_energy_e20_offsets1_2_10x1_grouped/challenge_metrics_summary.csv` gives:

```text
learned-prior-sparse-p140-t32  official_score=0.68760  Struct_LPIPS=0.58697
hybrid-neural-p140-t32-e20     official_score=0.68674  Struct_LPIPS=0.58377
```

Interpretation: `e20` consistently lowers Structural LPIPS a little, but it is not a robust official-score promotion over `p140-t32` across the broader public split. Treat `outputs/submission_ready_hybrid_neural_p140_t32_e20_full` as an exploratory package that proves the hybrid can be packaged, not as the current final candidate. The conservative current local candidate remains `learned-prior-sparse-p140-t32` until a hybrid beats it on broader grouped true-scanner evidence.

## 2026-06-29 Extended Neural Energy Blend Check

Because `e20` lowered Structural LPIPS but failed to robustly promote, stronger energy-only neural blends were added while keeping learned-prior density fixed:

```text
hybrid-neural-p140-t32-e30
hybrid-neural-p140-t32-e40
hybrid-neural-p140-t32-e60
```

A small offset-2 `2`-fold hosted true-scanner sweep selected `e60`:

```text
hybrid-neural-p140-t32-e60     official_score=0.72671  Struct_LPIPS=0.55787
hybrid-neural-p140-t32-e40     official_score=0.72591  Struct_LPIPS=0.55971
hybrid-neural-p140-t32-e30     official_score=0.72561  Struct_LPIPS=0.56045
hybrid-neural-p140-t32-e20     official_score=0.72540  Struct_LPIPS=0.56116
learned-prior-sparse-p140-t32  official_score=0.72217  Struct_LPIPS=0.56202
```

An offset-1 `2`-fold confirmation also selected `e60`:

```text
hybrid-neural-p140-t32-e60     official_score=0.68045  Struct_LPIPS=0.55990
learned-prior-sparse-p140-t32  official_score=0.67839  Struct_LPIPS=0.56779
hybrid-neural-p140-t32-e20     official_score=0.67681  Struct_LPIPS=0.56573
```

`e60` was therefore escalated to matched `5`-fold checks. It won each offset separately:

```text
offset-1:
hybrid-neural-p140-t32-e60     official_score=0.67783  Struct_LPIPS=0.56686
learned-prior-sparse-p140-t32  official_score=0.67703  Struct_LPIPS=0.57616

offset-2:
hybrid-neural-p140-t32-e60     official_score=0.70513  Struct_LPIPS=0.59520
learned-prior-sparse-p140-t32  official_score=0.70051  Struct_LPIPS=0.59778
```

However, merging the two `5`-fold detail files into a `10`-sample grouped diagnostic at `outputs/api_validation_hybrid_energy_e60_offsets1_2_10x1_grouped/challenge_metrics_summary.csv` did not promote `e60`:

```text
learned-prior-sparse-p140-t32  official_score=0.68760  Struct_LPIPS=0.58697
hybrid-neural-p140-t32-e60     official_score=0.68710  Struct_LPIPS=0.58103
```

`audit-evidence --strict --require-real-lpips` passed on the grouped diagnostic, but `decide-promotion --candidate hybrid-neural-p140-t32-e60 --baseline learned-prior-sparse-p140-t32` returned `promote=false` because the official score delta was `-0.00049`. Interpretation: stronger neural energy blending is a real Structural LPIPS improvement, but it trades away enough Structural MS-SSIM and SC/RSC map similarity that the current official aggregate still favors `p140-t32` on the broader grouped summary. The next hybrid target should constrain MS-SSIM and SC/RSC while borrowing the `e60` perceptual benefit.

## Archived Artifact Pointers

Historical outputs may exist under ignored `outputs/` paths:

- `outputs/optimizer_broad/optimizer_summary.csv`
- `outputs/final_comparison/internal_validation_summary.csv`
- `outputs/agent_council/internal_validation_summary.csv`
- `outputs/council_micro_confirm/internal_validation_summary.csv`
- `outputs/submission_ready_h67_full/SUBMISSION_README.md`

Because `outputs/` is ignored and machine-local, treat these as breadcrumbs rather than required repository files.
