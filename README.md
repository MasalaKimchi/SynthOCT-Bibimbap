# SynthOCT-Bibimbap

Physics-based OCT inversion for the [SynthOCT Challenge 2026](https://synthoct.com/)
(MICCAI / SASHIMI). Given a real 256×512 skin-OCT B-scan, the generator solves
the challenge's per-scan **inverse problem**: it produces a 300,000-row
`X Y Z Energy` digital phantom that, when rendered by the fixed virtual scanner,
reproduces the input B-scan.

The method is **phase-pair holographic inversion** — momentum-accelerated phase
retrieval against the coherent scanner operator, with each complex voxel
coefficient encoded as a dispersion-canceling pair of sub-resolution scatterers.
A single fixed set of global parameters is used for every scan (no per-image
tuning).

## Results at a glance

Across all 120 public references, the fixed v3-200 method improved
source-equivalent Structural MS-SSIM on **120/120** cases over the earlier v1 and
v3-50 variants:

| Structural MS-SSIM (n=120, local) | v1 | v3-200 |
|---|---:|---:|
| mean | 0.955954 | 0.994588 |
| median | 0.956021 | 0.994849 |
| minimum | 0.933827 | 0.985133 |

On the hosted true scanner (request `69bc223f`), the v3-200 phantom scored Struct
MS-SSIM `0.996817`, LPIPS `0.013080`, eight-term estimate `0.996514`.

> **These are local / hosted public-development estimates, explicitly *not*
> organizer-issued leaderboard or hidden-test scores.** The composite `0.994079`
> quoted elsewhere is a local organizer-compatible estimate. The method
> reconstructs a *scanner-equivalent* phantom, not unique tissue microstructure.
> See [docs/COMPLIANCE.md](docs/COMPLIANCE.md) for the full validity boundary.

## Repository layout

```
src/synthoct/          Installable package (the `synthoct` CLI)
  phantom.py             Scanner contract + phantom I/O
  holographic_inverse.py Generation  (baseline Part1)
  scanners/              Hosted API client + source-equivalent local scanner (Part2)
  features/extraction.py OAC / SC / RSC parametric maps (baseline Part3)
  evaluation/            MS-SSIM / LPIPS metrics + competition formula
  benchmark.py           Batch evaluation (baseline Orchestrator path)
  submission.py          Deterministic ZIP packaging + verification
  evidence.py            Evidence-manifest builder / verifier
  cli.py                 `synthoct` command surface
tests/                 Test suite (pytest)
tools/                 One-off audit / batch-run scripts
docs/                  METHODS.md, DATA.md, COMPLIANCE.md
paper/miccai2026/      LaTeX manuscript source + figure generation
```

The raw dataset, the hosted scanner binary, and all generated outputs are **not**
tracked in git — see [Data](#data) and [docs/DATA.md](docs/DATA.md).

## Installation

Requires Python 3.12. Install the package and its development / evaluation extras:

```bash
python -m pip install -e ".[dev,lpips]"
```

Or reproduce the pinned conda environment:

```bash
conda env create -f environment.yml   # creates env "synthoct-py312"
conda activate synthoct-py312
```

The `lpips` extra pulls in Torch and is needed only for LPIPS scoring. The fixed
scanner binary (`Part2_Scanner.exe`) is external and Windows-only; hosted
rendering is used instead via `synthoct scan`.

## Data

Reference scans come from Zenodo record
[`18095266`](https://zenodo.org/records/18095266) (120 in-vivo skin B-scans).
The archive is not committed; download and extract it as described in
[docs/DATA.md](docs/DATA.md), then point the CLI at the extracted `DATASET_PNG`
directory.

## Quickstart — invert one B-scan

```bash
synthoct baseline holographic-inverse \
  --input  path/to/reference.png \
  --out    outputs/candidate/phantom.txt \
  --diagnostics outputs/candidate/diagnostics.json

# Render locally and score with organizer-compatible map encoding
synthoct render-reference --phantom outputs/candidate/phantom.txt --out outputs/candidate/local.png
synthoct evaluate \
  --ref  path/to/reference.png \
  --pred outputs/candidate/local.png \
  --maps --include-lpips \
  --out-csv outputs/candidate/competition_estimate.csv
```

Median phantom generation is ~8 s per B-scan (slowest 19.3 s), well within the
challenge's 600 s limit.

## Reproducing a full submission

Generate and **retain** one submission-named phantom per reference (this is the
command to run on a fresh or hidden reference set), then package and verify a
deterministic, sex/age/site-balanced archive:

```bash
# 1. Invert every reference into retained, submission-named phantoms
synthoct generate-batch \
  --reference-root path/to/DATASET_PNG \
  --out-dir        outputs/hidden_test/phantoms

# 2. Build a deterministic 60-case submission ZIP + manifest
synthoct prepare-submission \
  --reference-root path/to/DATASET_PNG \
  --phantom-dir    outputs/hidden_test/phantoms \
  --out            outputs/hidden_test/submission.zip \
  --manifest       outputs/hidden_test/submission.manifest.json \
  --count 60 --seed 2026

# 3. Independently re-parse every member and bind it to the manifest
synthoct verify-submission \
  --archive        outputs/hidden_test/submission.zip \
  --manifest       outputs/hidden_test/submission.manifest.json \
  --reference-root path/to/DATASET_PNG \
  --expected-count 60
```

`generate-batch` writes a fail-closed `generation_summary.json` recording a
per-file SHA-256, row/column counts, bounds, and the fixed generation
parameters. Confirm the signed-in portal's exact filename/layout convention
before uploading.

Other commands: `synthoct scan` (hosted rendering, `--manifest` retains request
provenance), `synthoct benchmark-local` (batch scoring in a throwaway
directory), `synthoct evidence-manifest` (build / fail-closed verify the evidence
hash manifest). Run `synthoct <command> --help` for full options.

## Feature-map modes

The published `Part3` processor and the scanner documentation disagree on dynamic
range (40 dB vs 51 dB). The repository resolves this explicitly with two modes:

- **`organizer-compatible-v1`** (default) — freezes the published 40 dB OAC/SC/RSC
  behavior byte-for-byte; used for all competition-formula estimates.
- **`scientific-v1`** — uses the 51 dB calibration with float arrays and explicit
  validity masks; audit-only and never enters a competition score.

The OAC/SC/RSC "reference" maps are derived from the reference B-scan and are not
independently measured optical-property ground truth. Details and the full audit
are in [docs/METHODS.md](docs/METHODS.md#3-feature-maps-definitions--audit).

## Documentation

| Document | Contents |
|---|---|
| [docs/METHODS.md](docs/METHODS.md) | Phase-pair holographic inversion algorithm, mapping to the official baseline, feature-map definitions & audit, and the full experiment log. |
| [docs/DATA.md](docs/DATA.md) | Dataset provenance (Zenodo `18095266`), archive checksums, directory structure, evaluation pairing, and the historical label correction. |
| [docs/COMPLIANCE.md](docs/COMPLIANCE.md) | Code-structure compliance with the challenge baseline and the method-legitimacy assessment, with the validity boundary. |
| [paper/miccai2026/](paper/miccai2026/) | MICCAI/SASHIMI manuscript source and figure generation. |

## Validity boundary

- The challenge brief defines per-B-scan inverse reconstruction from the supplied
  OCT image, so reference-conditioned generation is the intended task.
- Local source-equivalent scores are search/validation evidence; hosted (or the
  organizers' Windows binary) renders are the true-scanner check.
- `official_score`, `winning`, and hidden-test claims are intentionally avoided
  until issued by the organizers.
- API keys, datasets, scanner binaries, caches, and exploratory outputs are
  excluded from version control.
