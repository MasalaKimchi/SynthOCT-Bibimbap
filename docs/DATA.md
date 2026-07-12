# Data

> This document consolidates the former `data_provenance.md` and `LABEL_CORRECTION.md` notes into a single reference.

## Obtaining the dataset

The raw `DATASET/` files are **not** committed to this repository — they are downloaded from Zenodo. To reproduce the reference set:

1. Download `18095266.zip` from <https://zenodo.org/records/18095266>. It is a one-file wrapper containing `DATASET.zip`.
2. Verify the inner `DATASET.zip` against the checksums in [Archive verification](#archive-verification) below.
3. Extract `DATASET.zip` to obtain the 120 PNG + 120 NPY files.

## 1. Dataset provenance

The public reference set is Zenodo record
[`18095266`](https://zenodo.org/records/18095266), **In vivo Human Skin Optical
Coherence Tomography (OCT) Dataset for Classification Benchmarking and Digital
Phantom Generation**. Its metadata describes 120 in-vivo human-skin B-scans,
256x512 pixels, 6 micrometers per pixel, approximately 51 dB dynamic range,
1.3-micrometer center wavelength, and 20-micrometer beam FWHM.

The evaluator/scanner contract is checked against the official
[`SynthOCTChallenge/SynthOCT_Baseline`](https://github.com/SynthOCTChallenge/SynthOCT_Baseline)
repository at commit `9508e27c4b0bd54e0d76172a1c85af329472f2fb`.

> **Challenge context:** the SynthOCT challenge uses these 120 public B-scans plus 60 hidden organizer-held test scans. Only the 120 public scans are contained in this Zenodo record; the 60 test scans are held out by the organizers and are not part of this archive.

### Archive verification

The download `18095266.zip` is a one-file wrapper containing `DATASET.zip`. The
inner archive is the authoritative Zenodo payload:

```text
Zenodo key             DATASET.zip
size                   38,210,250 bytes
Zenodo/local MD5       6c461f35a74b768b2960e0391687c10f
local SHA-256          9d9d114b83d21488ca36383c27d4c77168c39e517eb6b1225bfbe59e5b784f01
ZIP integrity test     passed
```

### Contents and structure

Extraction contains exactly **120 PNGs and 120 corresponding NPY arrays**. The
set is perfectly stratified into eight sex/age/site groups with 15 scans each:
Female/Male x 1950-1960/1990-2000 x Cheek/Eye_corner.

- All arrays are 256x512.
- The NPY files are `float32` arrays whose values are integer-valued in `[0,255]`.
- Each PNG is opaque RGBA grayscale with identical RGB channels.
- PNG and NPY values differ by at most one intensity level because of PNG
  visualization encoding; they are therefore paired representations of the same
  measured B-scan, not separate synthetic data.

## 2. Use in evaluation (reference pairing)

Competition-style comparisons use the **official PNG references**. This matches
the official `Part3_Processor.py`/`Orchestrator.py` path, which loads image
files, generates OAC/SC/RSC maps through Matplotlib, and evaluates grayscale
images with SEWAR MS-SSIM and AlexNet LPIPS. The generator also accepts NPY
input, but PNG remains the scoring reference to avoid a one-level encoding
mismatch with the image-based evaluator. Scanner dimensions, pixel sizes,
wavelength, beam radius, row count, and energy interpretation are aligned to
`Part1_Generator.py`.

The reference side of every promoted comparison is real in-vivo data from the
official archive:

| Evidence | Reference | Prediction | True-scanner status |
|---|---|---|---|
| Hosted request `69bc223f` (v3-200) | Official Zenodo in-vivo PNG | Hosted API render of our phantom | Current true hosted scanner, `n=1` |
| Hosted request `b166d529` (v3-50) | Official Zenodo in-vivo PNG | Hosted API render of our phantom | Historical true hosted scanner, `n=1` |
| All-120 v1/v3-50/v3-200 benchmark | All 120 official Zenodo in-vivo PNGs | Source-equivalent local scanner renders | Local validation, not 120 hosted calls |
| Four-case map/LPIPS panels | Four official Zenodo in-vivo PNGs | Source-equivalent local scanner renders | Local validation, not hosted |

Each retained all-120 detail CSV contains exactly the Zenodo PNG path set and
retains a matching SHA-256 for every reference; the four-case panel resolves to
the same extracted files and its stored hashes also match.

The synthetic self-consistency scans committed to the baseline GitHub repository
are **not** used as real targets or as competition evidence. The official README
states that its default Orchestrator is a self-consistency demonstration and must
be reconfigured for real/synthetic challenge comparison; our evaluation directly
pairs Zenodo references with scanner renders. Local scanner renders are labeled
as local validation and are never presented as official-scanner evidence.

## 3. Historical label correction

Some retained CSVs predate the **2026-07-10** evidence-terminology audit. In
those files the `official_score` column is a **locally calculated,
single-reference formula value** from a hosted scanner image — it is **not** an
organizer-issued leaderboard or hidden-test score, and it also used a secondary
grayscale copy.

- The corrected organizer-compatible evaluation of the raw hosted PNG is in
  `../organizer_compatible_estimate.csv`.
- Current code and documentation use the name `competition_formula_estimate` for
  locally calculated values.
