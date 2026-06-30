# ML/DL Expert Review

This note records the current expert judgment on whether a machine-learning or deep-learning phantom generator should beat the hand-built learned-prior methods.

## Decision

A stronger ML/DL approach is the most credible route to a competition-winning model, but the ML/DL branch tested so far is not yet better than `learned-prior-sparse-p140-t32`.

The current conservative local candidate remains:

```text
outputs/submission_ready_learned_prior_sparse_p140_t32_full_corrected
```

The exploratory hybrid package remains useful for follow-up testing, but should not be treated as the final candidate:

```text
outputs/submission_ready_hybrid_neural_p140_t32_e20_full
```

## Evidence

Standalone neural prior, offset-2 2-sample true-scanner split:

```text
learned-prior-sparse-p140-t32  official_score=0.72217  Struct_LPIPS=0.56202
neural-prior                   official_score=0.56240  Struct_LPIPS=0.45202
```

Interpretation: the neural model found a path toward the main perceptual bottleneck, but it lost too much Structural MS-SSIM and SC/RSC map quality to promote.

Hybrid neural energy blend, combined offset-1 plus offset-2 10-sample grouped true-scanner split:

```text
learned-prior-sparse-p140-t32  official_score=0.68760  Struct_LPIPS=0.58697
hybrid-neural-p140-t32-e20     official_score=0.68674  Struct_LPIPS=0.58377
```

Interpretation: neural energy shaping consistently lowers Structural LPIPS slightly, but the broader official-score comparison still favors `p140-t32` by `0.00086`.

An extended energy-only sweep added `e30`, `e40`, and `e60`. `e60` beat `p140-t32` on separate offset-1 and offset-2 5-fold checks, but still did not promote on the combined 10-sample grouped diagnostic:

```text
learned-prior-sparse-p140-t32  official_score=0.68760  Struct_LPIPS=0.58697
hybrid-neural-p140-t32-e60     official_score=0.68710  Struct_LPIPS=0.58103
```

Interpretation: stronger neural energy blending buys more Structural LPIPS improvement than `e20`, but the current official median aggregate still favors `p140-t32` once offsets are combined.

The strict readiness check for `learned-prior-sparse-p140-t32` reports:

```text
local_candidate_ready=true
official_final_ranking_proven=false
surrogate_scanner_is_true_scanner=false
hidden_holdout_final_score=false
```

## Why ML/DL Has Higher Ceiling

The challenge is a conditional inverse-generation problem: infer a valid `X Y Z Energy` scatterer field from a real OCT B-scan such that the fixed scanner render matches Structural intensity, OAC, SC, and RSC maps. Hand-designed priors can encode sparsity, depth decay, and energy shaping, but they cannot reliably learn population-level tissue appearance or scanner-specific perceptual texture.

LPIPS is especially likely to reward learned representations. The current scalar and visual-inversion variants can move MS-SSIM and brightness, but Structural LPIPS remains around `0.56-0.59`, far above the preliminary `<0.4` gate. That gap looks like a missing learned anatomy/texture model, not a single density, energy, or noise knob.

## Why The Current ML/DL Model Underperformed

- It trained from only about `17` discoverable true-scanner rows, which is very small for a robust conditional generator.
- It learned from generated phantom/render pairs rather than ground-truth phantoms for real scans, so its labels inherit bias from earlier generators.
- The current CNN predicts low-resolution density and energy fields, not a rich layered tissue, attenuation, and speckle process.
- Density topology is extremely fragile: a `5%` neural density blend reduced Structural LPIPS slightly but dropped official score from `0.72217` to `0.47606` on the matched offset-2 split.
- The standalone neural prior improved Structural LPIPS but regressed SC/RSC LPIPS, showing that visual similarity alone is not aligned with the full challenge objective.
- Stronger energy-only blends now lower Structural LPIPS more reliably, but still trade away enough Structural MS-SSIM or SC/RSC map quality to lose the broader official-score gate.
- Surrogate or preview quality is not true-scanner evidence. Every promoted model must be rendered by the hosted API or official Windows scanner.
- The current validation splits are still public-data local evidence, not hidden-holdout proof.

## Next Fair Experiment

The next ML/DL attempt should preserve `p140-t32` density stability while learning only the residual components that the hand-built prior cannot express:

1. Train on all available true-scanner render/phantom pairs, with grouped holdout splits.
2. Predict residual energy, layer modulation, or texture statistics rather than replacing density topology outright.
3. Use losses aligned to Structural, OAC, SC, and RSC maps, not only image-space reconstruction.
4. Gate all candidates through true-scanner grouped validation with real LPIPS.
5. Promote only if the official aggregate beats `learned-prior-sparse-p140-t32` on a broader matched split and does not merely improve Structural LPIPS in isolation.

The concrete execution plan is in [next_approach.md](next_approach.md).

## Residual Trainer Shape

The next model should not be a direct OCT image generator and should not replace the entire density field. It should learn a compact residual policy around `p140-t32`:

- input features: normalized reference intensity, OAC proxy, speckle contrast, refined speckle contrast, depth profile, body-site traits, frame traits, and current `p140-t32` metric context when available;
- outputs: residual strength, low-frequency flow parameters, energy exponent/sigma, attenuation-normalized row energy correction, and speckle texture controls;
- losses: official-style Struct/OAC/SC/RSC MS-SSIM and LPIPS terms, plus penalties for map-regression lower-confidence bounds;
- validation: grouped source holdout first, hosted true-scanner confirmation second.

The current OAC-aware residual selector is a teacher-data distillation tool. It can choose API probes, but the expected gain is small. If the next hosted run is flat, the right response is to train a richer residual policy from the accumulated feedback rather than broaden hand-tuned scalar sweeps.
