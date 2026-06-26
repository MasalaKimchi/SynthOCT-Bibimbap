# Challenge Baseline Comparison

## Official Challenge Baseline

The challenge baseline's editable component is `Part1_Generator.py` from `SynthOCTChallenge/SynthOCT_Baseline`.

It generates a digital phantom as a random point-scatterer table with four columns:

```text
X, Y, Z, Energy
```

The two provided generator modes are:

- uniform scatterers with one constant energy;
- two-layer scatterers with one constant shallow energy and one constant deeper energy.

The baseline is not image-conditioned. It does not inspect the target OCT B-scan, estimate tissue structure, estimate attenuation, or optimize against the target scan. The orchestrator example uses a layered phantom with `boundary_z_mcm=400.0`, `amp_top=0.01`, and `amp_bottom=2.5`.

## H67 Candidate

The current final method is `H61_api_low_depth_prelim`. `H67_coarse_to_fine_crisp` is an experimental image-conditioned candidate. It reads the reference scan and derives a density/energy sampling field from:

- scan intensity;
- optical attenuation coefficient estimate;
- speckle contrast;
- estimated layer boundary;
- lateral/depth smoothing and density-sharpening parameters.

It still emits the same challenge-compatible `X, Y, Z, Energy` phantom format, but it is an inverse-generation method rather than a blind random phantom.

## Available Metric Evidence

The only existing hosted-API comparison that includes the challenge baseline is a single-sample probe:

```text
official_two_layer  MS-SSIM=0.0439  LPIPS_PROXY=0.0701
```

That probe does not include H67, so it is not a fair matched comparison against H67.

The current 25-sample H67 candidate run reports:

```text
H67_coarse_to_fine_crisp  MS-SSIM=0.1958  LPIPS_PROXY=0.1046
```

Those H67 numbers are useful for candidate triage, but they should not be quoted as a direct improvement over the challenge baseline until both methods are rendered on the same reference scans with the hosted API or official Windows scanner.

## Required Fair Comparison

Run the challenge baseline and H67 on the exact same reference subset:

```bash
PYTHONPATH=src python -m synthoct.cli validate-internal \
  --zip 18095266.zip \
  --out outputs/official_vs_h67_api_10 \
  --methods official H67_coarse_to_fine_crisp \
  --folds 5 \
  --max-per-fold 2 \
  --include-lpips \
  --api-key-file ~/.config/synthoct/api_key
```

Report only:

- MS-SSIM, higher is better;
- LPIPS, lower is better.

If real LPIPS is unavailable, label the fallback explicitly as `LPIPS_PROXY`.

## Current Interpretation

Conceptually, H67 is a stronger approach than the official baseline because it conditions the phantom on the input scan and estimates latent tissue structure. Operationally, H61 remains the final method. Empirically, the repo still needs a matched hosted-API or Windows-scanner run of `official` versus `H67_coarse_to_fine_crisp` before claiming a challenge-metric win over the baseline or promoting H67.
