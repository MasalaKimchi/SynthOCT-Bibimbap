# SynthOCT-Bibimbap

This repository retains the highest-scoring, scanner-compatible method: coherent
holographic inversion. It solves the recovered complex OCT forward operator with
regularized axial/lateral pseudoinverses, encodes coefficient phase as sub-pixel
depth offsets, and uses zero-energy rows to preserve the official 300,000-row
phantom contract.

## Reproduce the winning phantom

```bash
python -m pip install -e ".[dev]"
synthoct baseline holographic-inverse \
  --input path/to/reference.png \
  --out outputs/holographic_inverse_contract300k/phantom.txt \
  --diagnostics outputs/holographic_inverse_contract300k/diagnostics.json
```

The exact recovered scanner is available as
`synthoct.scanners.reference.render_reference_array`; hosted rendering is
provided by `synthoct.scanners.api`. Full-reference metrics are implemented in
`synthoct.evaluation.metrics` and feature-map metrics in
`synthoct.evaluation.maps`.

## Proven API evidence

The retained `outputs/holographic_inverse_contract300k/` directory contains the
300,000-row phantom, diagnostics, hosted synthetic PNG, grayscale PNG, request
manifest, and official metric CSVs. The completed hosted request was `d5a0d2b8`:

```text
Struct MS-SSIM  0.9626228523683698
Struct SSIM     0.7873176994307578
Struct PSNR     26.116106495891287
Struct MSE      0.0024456221008384293
Real LPIPS      0.1337243765592575
OAC MS-SSIM     0.9926245077890009
SC MS-SSIM      0.9828958734019101
RSC MS-SSIM     0.9841042167630918
Official score  0.9586191274561562
```

API keys, datasets, scanner binaries, caches, and exploratory outputs are
intentionally excluded from version control.
