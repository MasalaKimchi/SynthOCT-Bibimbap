# SynthOCT Challenge Strategy

## Objective

SynthOCT asks participants to generate digital phantoms: four-column scatterer distributions `(X, Y, Z, Energy)`. The organizers run those phantoms through a fixed Virtual Scanner to create OCT B-scans, then compare them against reference scans. The challenge is therefore inverse physics, not direct image synthesis.

The practical target is high MS-SSIM and low LPIPS after scanning, with runtime below 600 seconds per B-scan on the stated RTX 3060 12GB / 16GB RAM class machine.

## API-First Baseline Ladder

1. Official baseline: uniform and two-layer scatterer distributions copied into a package-friendly form.
2. Heuristic inverse baseline: estimate a layer boundary and energy levels from the input scan, OAC map, and speckle contrast map.
3. Physics-guided hypotheses: generate scatterer density and energy maps from intensity, OAC, and speckle features.
4. Hosted-API candidate search: render phantoms with the challenge scanner and promote only candidates that improve API-rendered metrics.

## Removed Offline Surrogates

The previous CNN, portfolio, council, and offline-surrogate branches are intentionally removed from the executable code. They were useful for brainstorming, but they can give a false impression of challenge performance because only the hosted API or official Windows scanner measures the physics-rendered B-scan.

## Current Direction

The best route is still inverse physics, but with scanner-in-the-loop verification:

- invert obvious macrostructure using depth profiles and OAC;
- represent tissue as layered/patchwise parameter maps rather than raw pixels;
- refine candidates with hosted API renders, not local surrogates;
- keep hard validation on runtime, bounds, and scanner compatibility.

## Hosted API Validation Protocol

Use `synthoct validate-internal` as the private leaderboard before submitting. It creates grouped folds from the Zenodo archive, generates phantoms for each method, renders each phantom with the hosted SynthOCT scanner API, and reports structural plus physics-map metrics.

Recommended small API loop:

```bash
PYTHONPATH=src python -m synthoct.cli validate-internal \
  --zip 18095266.zip \
  --out outputs/api_validation_quick \
  --methods H61_api_low_depth_prelim H67_coarse_to_fine_crisp H68_layer_map_prior \
  --folds 3 \
  --max-per-fold 1 \
  --api-key-file ~/.config/synthoct/api_key
```

Recommended physics-map loop:

```bash
PYTHONPATH=src python -m synthoct.cli validate-internal \
  --zip 18095266.zip \
  --out outputs/api_validation_maps \
  --methods H61_api_low_depth_prelim H67_coarse_to_fine_crisp H68_layer_map_prior \
  --folds 3 \
  --max-per-fold 1 \
  --api-key-file ~/.config/synthoct/api_key
```

Treat single-sample results as unstable. A method should only be promoted when hosted API renders improve across folds, not just on one scan. The most important signals are MS-SSIM, LPIPS/LPIPS_PROXY, OAC/RSC agreement, depth-profile correlation, OAC-profile correlation, generation time, and rendered visual sanity.
