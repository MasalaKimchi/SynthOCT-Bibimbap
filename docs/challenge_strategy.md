# SynthOCT Challenge Strategy

## Objective

SynthOCT asks participants to generate digital phantoms: four-column scatterer distributions `(X, Y, Z, Energy)`. The organizers run those phantoms through a fixed Virtual Scanner to create OCT B-scans, then compare them against reference scans. The challenge is therefore inverse physics, not direct image synthesis.

The practical target is high MS-SSIM and low LPIPS after scanning, with runtime below 600 seconds per B-scan on the stated RTX 3060 12GB / 16GB RAM class machine.

## Baseline Ladder

1. Official baseline: uniform and two-layer scatterer distributions copied into a package-friendly form.
2. Heuristic inverse baseline: estimate a layer boundary and energy levels from the input scan, OAC map, and speckle contrast map.
3. Parameter-search baseline: initialize from the heuristic and prepare a small scanner-in-the-loop grid search for Windows validation.
4. Pretrained CNN baseline: use frozen ImageNet CNNs for embeddings and parameter regression, while still emitting phantoms as the deliverable.
5. Hybrid physics+ML: train a lightweight head on top of a pretrained encoder using synthetic scanner-generated pairs.

## Pretrained CNN Rationale

Pretrained CNNs should not generate final OCT images for submission. Their useful role is feature extraction: retrieval, perceptual scoring, and prediction of low-dimensional phantom parameters such as layer boundary, energy statistics, and heterogeneity. This keeps the method compatible with the challenge rule that outputs must be digital phantoms.

Use `resnet50` first because it is stable and well supported, then compare `efficientnet_b0` for speed and `convnext_tiny` for stronger features.

## Likely Winning Direction

The best route is probably hybrid:

- invert obvious macrostructure using depth profiles and OAC;
- represent tissue as layered/patchwise parameter maps rather than raw pixels;
- generate many synthetic phantoms, scan them on Windows, and train a CNN to map real scans to those parameters;
- refine each sample with a short local search using MS-SSIM and LPIPS where scanner access is available;
- keep hard validation on runtime, bounds, and scanner compatibility.

## Internal Validation Protocol

Use `synthoct validate-internal` as the private leaderboard before submitting. It creates grouped folds from the Zenodo archive, generates phantoms for each method, renders deterministic surrogate scans on macOS, and reports structural plus physics-map metrics.

Recommended fast loop:

```bash
PYTHONPATH=src python -m synthoct.cli validate-internal \
  --zip 18095266.zip \
  --out outputs/internal_validation_quick \
  --folds 3 \
  --max-per-fold 2 \
  --scatterers-count 8000 \
  --no-maps
```

Recommended physics-map loop:

```bash
PYTHONPATH=src python -m synthoct.cli validate-internal \
  --zip 18095266.zip \
  --out outputs/internal_validation_maps \
  --methods official heuristic physics-guided \
  --folds 3 \
  --max-per-fold 1 \
  --scatterers-count 6000
```

Treat this as a ranking proxy, not a challenge score. A method should only be promoted when it improves across folds, not just on one scan. The most important proxy signals are Composite, OAC/RSC SSIM, depth-profile correlation, OAC-profile correlation, and generation time. Final confirmation still requires the official Windows scanner.
