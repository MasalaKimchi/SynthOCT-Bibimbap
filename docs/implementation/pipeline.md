# Implementation Pipeline

The package mirrors the SynthOCT challenge stages: load reference scans, extract physical features, generate scatterer phantoms, render with a scanner backend, evaluate rendered PNGs, then package submission artifacts.

## Package Layout

- `synthoct.dataset`: load/list Zenodo OCT reference scans.
- `synthoct.features`: compute OAC, speckle contrast, refined speckle contrast, depth profiles, and boundary estimates.
- `synthoct.generators`: generate official, heuristic, physics-guided, named-hypothesis, and final phantoms.
- `synthoct.scanners`: render phantoms through the hosted SynthOCT API, Windows `Part2_Scanner.exe`, or precomputed fixture mode.
- `synthoct.evaluation`: compute MS-SSIM, LPIPS/fallback metrics, and Struct/OAC/SC/RSC comparisons.
- `synthoct.submission`: write manifests, phantom zips, code zips, API render results, upload plans, and portal-ready PNG-pair folders.

Legacy direct-synthesis and facade modules were removed so the source surface stays aligned with inverse-physics phantom generation.

## Generating Phantoms

Prepare or inspect the dataset:

```bash
synthoct data prepare --zip 18095266.zip --out data
synthoct data list --zip 18095266.zip | head
```

Generate challenge-compatible phantoms:

```bash
synthoct baseline official --out outputs/official.txt

synthoct baseline heuristic \
  --input data/DATASET_PNG/Female/1950-1960/Cheek/l__shcheka_frame250.png \
  --out outputs/heuristic.txt

synthoct baseline hypothesis \
  --name H61_api_low_depth_prelim \
  --input data/DATASET_PNG/Female/1950-1960/Cheek/l__shcheka_frame250.png \
  --out outputs/h61.txt

synthoct baseline pipeline \
  --name P06_visual_surface_dark_body \
  --input data/DATASET_PNG/Female/1950-1960/Cheek/l__shcheka_frame250.png \
  --out outputs/p06.txt

synthoct baseline final \
  --input data/DATASET_PNG/Female/1950-1960/Cheek/l__shcheka_frame250.png \
  --out outputs/final.txt
```

The CLI subcommand is still named `baseline` for compatibility, but these commands write phantom scatterer files, not final OCT images.

The P-series promising pipelines translate the inverse-problem strategy review and scanner-in-the-loop sweeps into nine scanner-compatible phantom generators:

- `P01_simulator_constrained_prior`: balanced simulator-constrained warm start with layer and feature-map priors.
- `P02_unrolled_feature_consistency`: crisper multi-map consistency candidate with stronger OAC and boundary weighting.
- `P03_speckle_preserving_texture`: texture-heavy candidate that preserves local speckle variation.
- `P04_bayesian_posterior_sample`: stochastic posterior-style candidate with void sampling and compressed energies.
- `P05_attenuation_layer_map`: attenuation-forward layer prior with high OAC and layer-band weighting.
- `P06_visual_surface_dark_body`: hosted-scanner-calibrated broad superficial scattering with dark body preservation.
- `P07_surface_cutoff_broad_mix`: explicit air/tissue cutoff with broad superficial scattering and faint body speckle.
- `P08_sparse_top_texture_ssim`: sparse-top high-texture recipe tuned for plain SSIM; the current best single-scan setting uses `--scatterers-count 900000`.
- `P09_gamma_sparse_lowfloor_ssim`: gamma-emulation recipe that suppresses low-intensity dots and adds a faint tissue floor; the current best single-scan setting uses `--scatterers-count 900000`.

For scanner-in-loop refinement on one reference scan, use `optimize-correction`. It starts from the P09 density/energy field, renders through the hosted API, applies smoothed `reference / rendered` density corrections, and writes a ranked CSV so the best iteration can be selected:

```bash
synthoct optimize-correction \
  --input data/DATASET_PNG/Female/1950-1960/Cheek/l__shcheka_frame250.png \
  --out outputs/correction_refinement \
  --scatterers-count 900000 \
  --api-key-file ~/.config/synthoct/api_key
```

This command is intentionally separate from `baseline pipeline`: its method depends on the previous API render, so it is an optimizer rather than a pure one-shot generator.

When a strong phantom/render pair already exists, `optimize-transfer` performs a second scanner-in-loop pass by using the rendered grayscale image to apply mild gamma-like energy shaping in phantom space:

```bash
synthoct optimize-transfer \
  --ref outputs/api_all_methods_render/reference.png \
  --phantom outputs/api_correction_iter4_sweep/phantoms/I4_03_density_r4_090.txt \
  --rendered-gray outputs/api_correction_iter4_sweep/synthetic_gray/I4_03_density_r4_090_gray.png \
  --out outputs/transfer_refinement \
  --exponents 0.04 0.10 0.18 0.24 \
  --api-key-file ~/.config/synthoct/api_key
```

The first successful transfer sweep used the current best density-corrected phantom and improved hosted-API SSIM from `0.2651` to `0.2724` at exponent `0.18`; a recursive micro-step from that result reached `0.2726` at exponent `0.04`. This remains far below the target, but it is the best verified scanner-rendered direction so far.

The follow-on `optimize-energy-ratio` command preserves scatterer coordinates and applies smoothed `reference / rendered` energy feedback:

```bash
synthoct optimize-energy-ratio \
  --ref outputs/api_all_methods_render/reference.png \
  --phantom outputs/api_transfer_gen2_from_sgf02/phantoms/01_transfer_e0p040.txt \
  --rendered-gray outputs/api_transfer_gen2_from_sgf02/synthetic_gray/01_transfer_e0p040_gray.png \
  --out outputs/energy_ratio_refinement \
  --exponents 0.008 0.012 0.024 0.04 0.08 \
  --api-key-file ~/.config/synthoct/api_key
```

Seven recursive passes of this operator raised the first reference scan to hosted-API SSIM `0.2912`, the best verified scanner-rendered score so far. The next recursive pass generated valid phantom files, but the hosted API rejected those requests at POST with 500/400 responses, so they are not counted as scored candidates.

## Scanner Backends

Use the scanner abstraction from Python:

```python
from synthoct.scanners import render_phantom

render_phantom("phantom.txt", "Configuration.ini", "synthetic.png", backend="api")
render_phantom("phantom.txt", "Configuration.ini", "synthetic.png", backend="windows")
```

The default CLI scanner mode is the hosted SynthOCT API, which is the correct macOS path because `Part2_Scanner.exe` is a Windows executable:

```bash
synthoct scan \
  --phantom outputs/final.txt \
  --out outputs/final_api_scan.png \
  --api-key-file ~/.config/synthoct/api_key
```

API keys are read from `SYNTHOCT_API_KEY`, `SYNTHOCT_CHALLENGE_API_KEY`, or an untracked file passed with `--api-key-file`.

For official local Windows validation:

1. Create the Python `3.12.12` environment from `environment.yml`.
2. Download `Part2_Scanner.exe` from the official baseline instructions.
3. Place `Part2_Scanner.exe` at the repository root or pass `--scanner-exe`.
4. Generate a four-column phantom.
5. Render with Windows mode.
6. Evaluate the rendered PNG.

```bash
synthoct baseline final \
  --input data/DATASET_PNG/Female/1950-1960/Cheek/l__shcheka_frame250.png \
  --out outputs/final.txt

synthoct scan \
  --phantom outputs/final.txt \
  --out outputs/final_windows_scan.png \
  --mode windows \
  --scanner-exe Part2_Scanner.exe

synthoct evaluate \
  --ref data/DATASET_PNG/Female/1950-1960/Cheek/l__shcheka_frame250.png \
  --pred outputs/final_windows_scan.png \
  --maps \
  --metrics \
  --out-csv outputs/metrics.csv
```

Keep generation plus rendering within the challenge target of 600 seconds per B-scan on the stated hardware class.

## Internal Validation

Use hosted-API validation as the private leaderboard:

```bash
synthoct validate-internal \
  --zip 18095266.zip \
  --out outputs/api_validation \
  --wave promising-pipelines \
  --folds 5 \
  --max-per-fold 2 \
  --include-lpips \
  --api-key-file ~/.config/synthoct/api_key
```

Important outputs:

- `challenge_metrics_summary.csv`: challenge-facing MS-SSIM and LPIPS or LPIPS proxy summary.
- `internal_validation_summary.csv`: full diagnostic table.
- `hypothesis_wins.csv`: per-sample win counts.
- `internal_validation_detail.csv`: sample-level diagnostics.

Use `--rerun-existing` only when cached API renders should be ignored.

Before treating a candidate as promoted, audit the evidence labels and scope:

```bash
synthoct audit-evidence \
  --metrics outputs/api_validation/challenge_metrics_summary.csv \
  --strict \
  --require-real-lpips \
  --max-generation-seconds 600
```

The audit rejects surrogate-preview evidence and treats single-reference true-scanner renders as limited evidence rather than final-candidate proof.

Challenge-facing metric rows include `evaluation_region=full_frame`, `reference_shape`, `prediction_shape`, `evaluated_shape`, and `prediction_resized_to_reference`. The metric implementation resizes a prediction to the reference dimensions when needed, then computes metrics over the full reference-shaped array rather than a crop.

After the evidence audit passes, compare a candidate to the current final method:

```bash
synthoct decide-promotion \
  --metrics outputs/api_validation/challenge_metrics_summary.csv \
  --candidate H_new_candidate \
  --baseline H61_api_low_depth_prelim \
  --strict \
  --max-generation-seconds 600
```

This command promotes only when the candidate stays inside the generation runtime budget, improves mean MS-SSIM, does not worsen LPIPS/proxy, has no worse per-sample win counts, and does not regress available physical guardrails such as depth/OAC/speckle-map agreement under the same fair evidence gate.

To select the best promoted method automatically:

```bash
synthoct select-best \
  --metrics outputs/api_validation/challenge_metrics_summary.csv \
  --baseline H61_api_low_depth_prelim \
  --strict \
  --max-generation-seconds 600
```

If this exits nonzero, no validated method currently deserves to replace the baseline.

Use `--max-guardrail-regression` only when accepting a small physical-diagnostic tradeoff is deliberate and documented.

## Learned Surrogate Acceleration

The learned surrogate command trains a local CNN approximation of the scanner from existing hosted-scanner phantom/render pairs. It is an accelerator for candidate generation, not a challenge scorer:

```bash
synthoct optimize-learned-surrogate \
  --ref data/DATASET_PNG/Female/1950-1960/Cheek/l__shcheka_frame250.png \
  --outputs-dir outputs \
  --out outputs/learned_surrogate_refinement \
  --holdout-fraction 0.2
```

It writes:

- `learned_surrogate_metrics.csv`: surrogate-preview candidate rows, labeled `evidence_source=learned_surrogate_preview`.
- `surrogate_calibration_metrics.csv`: held-out true-scanner render versus surrogate prediction rows, labeled `evidence_source=learned_surrogate_holdout`.

Neither file is promotion-ready evidence. Render the emitted phantoms through `render-candidate-queue`, then validate the best candidates on a grouped split with `validate-internal`.
