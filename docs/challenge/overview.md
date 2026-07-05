# SynthOCT Challenge Overview

SynthOCT-Bibimbap is an inverse-physics digital phantom generator. The primary algorithmic output is not a direct synthetic OCT image; it is a scanner-compatible point-scatterer table:

```text
X Y Z Energy
```

The SynthOCT Virtual Scanner renders those phantoms into OCT B-scans for scoring. This means useful work in this repository should preserve the official `Part1_Generator.py` contract: generate physically plausible scatterers from a real reference B-scan, then evaluate only after hosted API or official Windows scanner rendering.

## Official Contract

Every generated phantom must match the baseline format:

- plain text with four numeric columns: `X`, `Y`, `Z`, `Energy`;
- coordinates in micrometers;
- `X` within `[-1536, 1536]`;
- `Z` within `[0, 1536]`;
- `Energy` within `[0, 100]`, interpreted by the scanner as reflection amplitude `sqrt(Energy / 100)`;
- default package size of `300000` scatterers per B-scan unless a smoke-test count is explicitly requested.

The challenge-facing metrics are MS-SSIM and LPIPS after scanner rendering. The local challenge handout describes the main final score as the mean of median MS-SSIM and median inverted LPIPS (`1 - LPIPS`) across four categories: Structural intensity, OAC, SC, and RSC. Depth-profile agreement and related profile diagnostics are internal guardrails for physical plausibility.

## Baseline Ladder

The repository keeps several challenge-safe generator families:

1. Official-style random phantoms: uniform or two-layer scatterer distributions that do not inspect the target scan.
2. Heuristic layer phantoms: estimate simple layer parameters from the reference B-scan.
3. Physics-guided phantoms: sample density and energy fields from intensity, OAC, speckle, layer boundary, and smoothing features.
4. API-facing hypotheses: selected H-series parameterizations that still emit only `X Y Z Energy` scatterers.

The legacy `baseline final` method is `H61_api_low_depth_prelim`. The current promoted local submission candidate is `learned-prior-sparse-p140-t32`, which packages an empirical prior artifact and has beaten H61 on grouped hosted true-scanner validation with real LPIPS. `H67_coarse_to_fine_crisp` and `H68_layer_map_prior` remain candidate methods for hosted API or Windows scanner comparison, not automatic replacements.

## Physics Model

The generation problem is underdetermined: many scatterer fields can produce similar OCT scans after blur, attenuation, coherent speckle, and log compression. The practical objective is scanner-consistent synthesis, not unique recovery of tissue microstructure.

The active approach models tissue as an inhomogeneous point process:

- density comes from image intensity, OAC, speckle contrast, and depth compensation;
- energy comes from OAC-weighted and texture-weighted reflectivity maps;
- layer boundaries can boost or shape sampling fields;
- lateral smoothing controls tissue continuity across neighboring A-scans;
- all methods finish by writing challenge-format scatterer rows.

## Validation Discipline

Offline ranking was useful for brainstorming, but it is not a trustworthy final signal. Promote a method only when scanner renders improve challenge-facing metrics across grouped folds or a matched official-scanner subset.

The official final ranking is stronger than local validation: the organizers run the submitted code/model on a hidden hold-out test dataset. A local hosted-API score on one public B-scan, even if produced by the true scanner, is not a competition-wide score. Local surrogate scanners and preview renderers are useful only for triage; they are not proof of challenge performance.

The compact current status is [current_findings.md](current_findings.md). In short: `learned-prior-sparse-p140-t32` remains the best general generator, adaptive flow/energy is the best measured public-set rescue, and Structural LPIPS remains the main bottleneck.

Recommended hosted API triage:

```bash
synthoct validate-internal \
  --zip 18095266.zip \
  --out outputs/api_validation_quick \
  --methods H61_api_low_depth_prelim H67_coarse_to_fine_crisp H68_layer_map_prior \
  --folds 3 \
  --max-per-fold 1 \
  --api-key-file ~/.config/synthoct/api_key
```

For a fair official-baseline comparison, render both methods on the same reference subset:

```bash
synthoct validate-internal \
  --zip 18095266.zip \
  --out outputs/official_vs_h67_api_10 \
  --methods official H67_coarse_to_fine_crisp \
  --folds 5 \
  --max-per-fold 2 \
  --include-lpips \
  --api-key-file ~/.config/synthoct/api_key
```

Report real LPIPS when available. If a fallback is used, label it explicitly as `LPIPS_PROXY`.

## Promotion Rules

Promote a candidate only when it has one of:

- higher challenge-facing score than the current local candidate without worse evidence quality;
- lower LPIPS or higher MS-SSIM without regressing the official aggregate or physical maps;
- more per-sample wins on challenge metrics;
- similar challenge score with stronger physical guardrails;
- coherent subset wins that justify a later scan router.

Kill a candidate when it loses clearly on both mean score and per-sample wins. If results are within noise, keep only conceptually distinct methods for larger runs.

## Current Decision

Use `learned-prior-sparse-p140-t32` as the current local submission candidate when its prior artifact and real-LPIPS evidence file are included. Use `H61_api_low_depth_prelim` as the conservative fallback if learned-prior packaging or evidence validation fails. Treat older H11/H41/H56 results as historical priors only. The current promoted evidence includes complete Struct/OAC/SC/RSC true-scanner metrics, but it still fails the preliminary LPIPS threshold and is not hidden-holdout proof.

See [current_findings.md](current_findings.md) for active artifacts, current evidence, and the resume point. See [rules_provenance.md](rules_provenance.md) for the local evidence sources and the explicit `surrogate_scanner_is_true_scanner=false` policy.
