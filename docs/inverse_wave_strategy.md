# Inverse Optimization Hypothesis Wave

## Goal

This wave turns inverse-scattering and inverse-rendering ideas into challenge-safe phantom generators. Each method still emits only `X, Y, Z, Energy` scatterers, but the parameter choices are inspired by analysis-by-synthesis inversion rather than direct image generation.

See [challenge_baseline_comparison.md](challenge_baseline_comparison.md) for the explicit comparison against the official challenge baseline generator.

## First Triage Wave

Run the inverse-inspired wave against the current incumbents:

```bash
PYTHONPATH=src python -m synthoct.cli validate-internal \
  --zip 18095266.zip \
  --out outputs/inverse_wave_1 \
  --wave inverse-wave-1 \
  --folds 5 \
  --max-per-fold 2 \
  --scatterers-count 20000
```

This gives 10 reference samples. Treat it as triage, not proof.

Important outputs:

- `challenge_metrics_summary.csv`: challenge-facing MS-SSIM and LPIPS/LPIPS_PROXY means, standard deviations, and per-sample wins.
- `internal_validation_summary.csv`: full diagnostic mean/stdev table.
- `hypothesis_wins.csv`: MS-SSIM per-sample win counts and margin to sample best.
- `internal_validation_detail.csv`: sample-level diagnostics.

## Promotion Rules

Promote a candidate to a larger run when it has one of:

- higher mean MS-SSIM than `H61_api_low_depth_prelim` without worse LPIPS;
- lower mean LPIPS than `H61_api_low_depth_prelim` without worse MS-SSIM;
- more per-sample wins on either MS-SSIM or LPIPS;
- similar score but better physical guardrails;
- strong wins on a coherent scan subset that could justify a router.

Kill a candidate when it loses clearly on both mean score and per-sample wins. If all candidates are within noise, keep only conceptually distinct methods for the expanded run.

## API Call Discipline

Use hosted API calls in successive stages:

1. 10 samples for broad triage.
2. 25-50 samples for the top 2-4 survivors.
3. largest feasible run for the final 1-2 candidates.

The current inverse wave methods are H62-H70. They cover linearized Born-style inversion, Bayesian/OAC smoothing, posterior speckle calibration, depth-histogram matching, patch-retrieval proxy behavior, coarse-to-fine crispness, layer-map priors, and historical H56-centered blends.

## First Results

Fast 10-sample triage without maps:

```text
H67_coarse_to_fine_crisp  MS-SSIM=0.1936  LPIPS_PROXY=0.1056
H68_layer_map_prior       MS-SSIM=0.1842  LPIPS_PROXY=0.1078
H64_speckle_posterior     MS-SSIM=0.1789  LPIPS_PROXY=0.1057
H56_h41_anti_anatomy      MS-SSIM=0.1775  LPIPS_PROXY=0.1076
```

Map-inclusive 10-sample expansion:

```text
H67_coarse_to_fine_crisp  MS-SSIM=0.1936  LPIPS_PROXY=0.1056  OAC_MS=0.4841
H68_layer_map_prior       MS-SSIM=0.1842  LPIPS_PROXY=0.1078  OAC_MS=0.5816
H64_speckle_posterior     MS-SSIM=0.1789  LPIPS_PROXY=0.1057  OAC_MS=0.5685
H56_h41_anti_anatomy      MS-SSIM=0.1775  LPIPS_PROXY=0.1076  OAC_MS=0.5100
```

Map-inclusive 25-sample confirmation:

```text
H67_coarse_to_fine_crisp  MS-SSIM=0.1958  LPIPS_PROXY=0.1046  OAC_MS=0.4780
H68_layer_map_prior       MS-SSIM=0.1845  LPIPS_PROXY=0.1075  OAC_MS=0.5794
H56_h41_anti_anatomy      MS-SSIM=0.1789  LPIPS_PROXY=0.1068  OAC_MS=0.5030
H64_speckle_posterior     MS-SSIM=0.1761  LPIPS_PROXY=0.1055  OAC_MS=0.5657
```

Interpretation: keep `H61_api_low_depth_prelim` as the final default. Promote `H67_coarse_to_fine_crisp` as the MS-SSIM candidate and `H68_layer_map_prior` as the physical-map candidate only for hosted API / Windows scanner comparison. Do not replace H61 until official scanner renders report stronger challenge metrics, especially actual LPIPS.

## API Candidate Bundles

Prepare 10-sample candidate bundles:

```bash
PYTHONPATH=src python -m synthoct.cli prepare-submission \
  --zip 18095266.zip \
  --out outputs/submission_ready_h68_10 \
  --method H68_layer_map_prior \
  --limit 10

PYTHONPATH=src python -m synthoct.cli prepare-submission \
  --zip 18095266.zip \
  --out outputs/submission_ready_h67_10 \
  --method H67_coarse_to_fine_crisp \
  --limit 10
```

Then render each bundle with the hosted API when an API key is available:

```bash
PYTHONPATH=src python -m synthoct.cli api-evaluate-submission \
  --zip 18095266.zip \
  --submission-dir outputs/submission_ready_h68_10 \
  --out outputs/api_preliminary_h68_10 \
  --limit 10 \
  --include-lpips \
  --api-key-file ~/.config/synthoct/api_key
```
