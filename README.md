# SynthOCT-Bibimbap

Physics-based OCT inversion for the [SynthOCT Challenge 2026](https://synthoct.com/)
(MICCAI / SASHIMI). Given a 256×512 skin-OCT B-scan, the method emits the
challenge-format digital phantom: exactly 300,000 rows of `X Y Z Energy` values
for rendering by the fixed virtual scanner.

The method, **phase-pair holographic inversion**, has two stages. Regularized
alternating projections (RAP) select a phase favored by the regularized scanner model,
and a closed-form phase pair represents each complex coefficient with two
nonnegative-energy entries half a wavelength apart. One fixed configuration is
applied to every scan; there is no per-image parameter tuning.

## Results at a glance

In the camera-ready evaluation, a fixed-regularization 2×2 ablation used the
local implementation of the published forward model on all 120 public scans.
The zero-phase, single-scatterer baseline reached mean MS-SSIM 0.82558; combining
50 RAP iterations with phase-pair encoding reached 0.99351. Every nonbaseline
cell improved all 120 scans. The tracked
[`ablation_results.json`](experiments/main_ablation/ablation_results.json)
records the full configuration and paired contrasts.

The final 200-iteration method was then rendered through the organizer's
challenge service for the same complete public set. It achieved mean structural
MS-SSIM 0.99428 and median structural LPIPS 0.02261. The tracked
[`hosted_results.json`](experiments/main_ablation/hosted_results.json) records
the aggregate and its historical-evidence limitation.

> These are descriptive results on the public dataset, not a leaderboard or
> hidden-test score. Local rendering is used for controlled configuration
> comparisons; organizer-hosted rendering evaluates the final method. The
> phantoms reproduce displayed B-scans but do not recover the acquired phase or
> tissue microstructure. See [docs/COMPLIANCE.md](docs/COMPLIANCE.md).

## Repository layout

```text
src/synthoct/       Installable package and `synthoct` CLI
baseline_format/    Flat Part1/Part2/Part3/Orchestrator compatibility entrypoints
experiments/        Published main-paper and supplementary analyses
tests/              Automated tests
tools/              Reproduction and audit utilities
docs/               Methods, data provenance, and validity boundary
```

Raw data, scanner binaries, generated runs, manuscript working files, and
camera-ready delivery archives are not tracked. Conference PDFs and LaTeX
sources are packaged separately from this software repository.

## Installation

Python 3.12 is required. Install the package with development and evaluation
extras:

```bash
python -m pip install -e ".[dev,lpips]"
```

Or create the conda environment described by `environment.yml`:

```bash
conda env create -f environment.yml
conda activate synthoct-py312
```

The `lpips` extra installs Torch and is needed only for LPIPS evaluation. The
organizers' fixed `Part2_Scanner.exe` is external and Windows-only; the CLI can
instead submit phantoms to the hosted service with `synthoct scan`.

## Data

The 120 public reference scans are version 1 of Zenodo record
[`18095266`](https://zenodo.org/records/18095266). Download and extract the
archive as described in [docs/DATA.md](docs/DATA.md), then pass the extracted
`DATASET_PNG` directory to the CLI. The dataset is intentionally excluded from
Git.

## Invert one B-scan

```bash
synthoct baseline holographic-inverse \
  --input path/to/reference.png \
  --out outputs/candidate/phantom.txt \
  --diagnostics outputs/candidate/diagnostics.json

# Controlled local rendering and organizer-compatible evaluation
synthoct render-reference \
  --phantom outputs/candidate/phantom.txt \
  --out outputs/candidate/local.png

synthoct evaluate \
  --ref path/to/reference.png \
  --pred outputs/candidate/local.png \
  --maps --include-lpips \
  --out-csv outputs/candidate/competition_estimate.csv
```

Mean phantom generation for the final method was 8.43 s per B-scan (maximum
19.28 s) on an Apple M3 Pro CPU without GPU acceleration, within the
challenge's 600 s limit.

## Build and verify a phantom submission

```bash
# Generate one validated, submission-named phantom per reference
synthoct generate-batch \
  --reference-root path/to/DATASET_PNG \
  --out-dir outputs/submission/phantoms

# Package a deterministic balanced archive and provenance manifest
synthoct prepare-submission \
  --reference-root path/to/DATASET_PNG \
  --phantom-dir outputs/submission/phantoms \
  --out outputs/submission/submission.zip \
  --manifest outputs/submission/submission.manifest.json \
  --count 60 --seed 2026

# Re-parse every member and bind it to the manifest
synthoct verify-submission \
  --archive outputs/submission/submission.zip \
  --manifest outputs/submission/submission.manifest.json \
  --reference-root path/to/DATASET_PNG \
  --expected-count 60
```

`generate-batch` records hashes, shape and bounds checks, fixed generation
parameters, and environment provenance. Confirm the signed-in challenge
portal's current filename and archive-layout requirements before upload.
Generated files belong under `outputs/<run-name>/`; the entire directory is
ignored by Git.

## Reproduce the published analyses

[experiments/](experiments/) contains the original Python runners and compact
numerical evidence for the main-paper comparisons and Supplementary Tables
S1--S4. Long per-scan records, generated phantoms, and rendered images are
written under ignored `outputs/` directories when the commands run. The raw
dataset and authenticated organizer-hosted service remain external.

## Feature-map modes

The published `Part3` processor and scanner metadata use different dynamic
ranges (40 dB and 51 dB). The repository keeps the two purposes explicit:

- `organizer-compatible-v1` (default) preserves the published 40 dB OAC/SC/RSC
  behavior for compatibility estimates.
- `scientific-v1` uses 51 dB float arrays and explicit validity masks for audit
  only; it cannot enter the competition formula.

OAC, SC, and RSC reference views are derived from the structural B-scan, not
independent optical-property ground truth. Details are in
[docs/METHODS.md](docs/METHODS.md#3-feature-maps-definitions--audit).

## Documentation

| Document | Contents |
|---|---|
| [docs/METHODS.md](docs/METHODS.md) | Algorithm, challenge-baseline mapping, feature-map audit, and final evidence map. |
| [docs/DATA.md](docs/DATA.md) | Dataset version, checksums, structure, and evaluation pairing. |
| [docs/COMPLIANCE.md](docs/COMPLIANCE.md) | Technical compliance and the scientific validity boundary. |

## Citation

If this software supports your work, cite:

> Justin Namuk Kim. “Phase-Pair Holographic Inversion for OCT Digital Phantom
> Synthesis.” SynthOCT Challenge, SASHIMI at MICCAI 2026.

Machine-readable software citation metadata is provided in [CITATION.cff](CITATION.cff).

## License

No open-source software license has been selected for this release. The code is
public for reproducibility, but reuse or redistribution requires permission from
the author.
