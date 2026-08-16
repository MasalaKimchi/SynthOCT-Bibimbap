# Data

## Public dataset

Experiments use version 1 of Zenodo record
[`18095266`](https://zenodo.org/records/18095266), *In vivo Human Skin Optical
Coherence Tomography (OCT) Dataset for Classification Benchmarking and Digital
Phantom Generation* (CC BY 4.0).

The record contains 120 in-vivo human-skin B-scans at 256×512 pixels, 6 µm per
pixel, approximately 51 dB dynamic range, 1.3 µm center wavelength, and 20 µm
beam FWHM. The scans form 40 three-frame acquisition series; filenames identify
frames 50, 250, and 450. They are therefore 120 scans, not 120 independent
subjects.

The challenge uses this public set for development and a separate organizer-held
test set. Hidden scans are not part of the Zenodo archive or this repository.

## Download and verification

1. Download `18095266.zip` from the Zenodo record. It is a one-file wrapper
   containing `DATASET.zip`.
2. Verify the inner archive:

   ```text
   file                   DATASET.zip
   size                   38,210,250 bytes
   Zenodo/local MD5       6c461f35a74b768b2960e0391687c10f
   local SHA-256          9d9d114b83d21488ca36383c27d4c77168c39e517eb6b1225bfbe59e5b784f01
   ZIP integrity test     passed
   ```

3. Extract `DATASET.zip`. Point CLI commands at the extracted `DATASET_PNG`
   directory.

`DATASET/` and downloaded archives are ignored by Git.

## Contents

Extraction yields 120 PNG images and 120 corresponding NPY arrays. The set is
stratified across eight sex/age/site groups with 15 scans each:
Female/Male × 1950–1960/1990–2000 × Cheek/Eye_corner.

- Every array is 256×512.
- NPY files are `float32` arrays with integer-valued samples in `[0, 255]`.
- PNG files are opaque RGBA grayscale with identical RGB channels.
- PNG and NPY representations differ by at most one intensity level because of
  display encoding.

## Use in this repository

Challenge-compatible comparisons use the PNG references. This follows the
published `Part3_Processor.py`/`Orchestrator.py` image path and avoids a
one-level representation mismatch. The generator also accepts NPY input.

Evidence scopes are kept separate:

| Analysis | Reference set | Rendering scope |
|---|---|---|
| Final 200-iteration phase-pair evaluation | All 120 public PNGs | Organizer-hosted challenge service; descriptive public-set results |
| Main fixed-regularization ablation | All 120 public PNGs | Local implementation of the published forward model |
| Supplementary sensitivity analyses | Defined 14- or 40-scan series subsets | Local implementation of the published forward model |

The local experiments support controlled method comparisons, not claims about
an organizer-issued score. The organizer-hosted evaluation checks the complete
final method on the public set, not hidden-set generalization. Reproduction
scripts and compact results are in [`experiments/`](../experiments/).

The synthetic self-consistency images in the baseline repository are not used as
real targets. All promoted comparisons pair a Zenodo in-vivo reference with a
render of the phantom generated from that reference.

Scanner dimensions, pixel sizes, wavelength, beam radius, row count, and energy
interpretation are checked against the official
[`SynthOCTChallenge/SynthOCT_Baseline`](https://github.com/SynthOCTChallenge/SynthOCT_Baseline)
at commit `9508e27c4b0bd54e0d76172a1c85af329472f2fb`.
