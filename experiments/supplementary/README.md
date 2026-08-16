# Supplementary post-review experiments

This reproducibility bundle contains the diagnostic scripts and exact JSON
evidence behind Supplementary Sections and Tables S1--S4 of the camera-ready
manuscript. These are sensitivity analyses, not part of the primary all-120
evaluation. Unless noted below, they use the middle acquisition (`frame250`)
from each of the 40 filename-defined acquisition series.

All rendering and scoring here use this repository's local implementation of
the published forward model. None of the JSON values is an organizer-hosted,
hidden-test, leaderboard, or official challenge score.
Supplementary Figure S1 is based on separate organizer-hosted renders and is
outside this bundle.

## Experiment-to-supplement map

| Supplement | Script | Retained evidence | Scope |
|---|---|---|---|
| S1: phase initialization | `exp_init.py` | `init_study.json`, `init_summary.json` | 14 scans: every third series; zero, axial analytic-signal, and minimum-phase starts at 0/10/25/50/200 iterations |
| S2: lateral regularization | `exp_regularization_matched.py` | `regularization_matched_factorial.json`, `regularization_matched_interpretation.json` | 40-scan matched phase-by-encoding factorial across five lateral regularizers |
| S2: supporting/legacy diagnostic | `exp_regularization.py` | `regularization_spectra.json`, `regularization_sweep.json`, `regularization_summary.json` | 40-scan mode spectrum plus an earlier confounded branch comparison retained for provenance |
| S3: finite-resolution overlap | `exp_operators.py` | `operators.json` | deterministic scanner geometry, conditioning, effective DOF, and nominal occupancy |
| S3: coefficient participation and exploratory artifact band | `exp_density.py` | `density_study.json`, `density_summary.json` | 40 scans; the artifact-band output is descriptive and does not identify the artifact mechanism |
| S4: amplitude constraints | `exp_encoding.py` | `encoding_study.json`, `encoding_summary.json` | 40 scans; continuous, quantized, equal-within-pair, single, and one-global-amplitude encodings |

`analysis_lib.py` contains the shared scanner, sampling, encoding, rendering,
and metric helpers. The retained study JSON files contain per-scan records;
the summary JSON files contain the corresponding aggregates.

## Dataset and environment

Use version 1 of the public in-vivo human-skin OCT dataset (Zenodo record
18095266). Follow [`../../docs/DATA.md`](../../docs/DATA.md) to verify and
extract it so the repository contains:

```text
DATASET/DATASET_PNG/<Sex>/<Age>/<Site>/*.png
```

The scripts select the 120 PNG references only; they do not use the held-out
organizer test set. Install this repository's Python 3.12 dependencies from the
root environment (for example, `python -m pip install -e .`). Direct script
execution discovers `src/` from the repository layout, so no external
`PYTHONPATH` is required.

## Run

Run from the repository root. Scripts write the standard result names beneath
`experiments/supplementary/results/`, except the matched factorial, whose output
path is explicit:

```bash
python experiments/supplementary/exp_operators.py
python experiments/supplementary/exp_encoding.py
python experiments/supplementary/exp_density.py
python experiments/supplementary/exp_regularization.py
python experiments/supplementary/exp_regularization_matched.py \
  --output experiments/supplementary/results/regularization_matched_factorial.json
python experiments/supplementary/exp_init.py
```

Approximate original runtimes on an Apple M3 Pro were seconds, 11 minutes,
10 minutes, 24 minutes, 26 minutes, and 8 minutes, respectively. The density
and encoding scripts accept `--full` to use all 120 images instead of the
standard 40-scan diagnostic subset; initialization accepts `--full` for all
120 images. Those full-run variants use `_full.json` filenames and are not part
of the retained camera-ready evidence. `exp_regularization.py --quick` uses the
first eight diagnostic scans for development only. The matched-factorial
script accepts `--limit N` for a development subset.

## Provenance and interpretation

- The checked-in JSON files are the camera-ready evidence. Rerunning a script
  overwrites its standard output, so use a temporary copy or compare hashes
  before replacing retained results.
- `regularization_matched_factorial.json` is the authoritative matched S2 sweep.
  Its SHA-256 is recorded unchanged in
  `regularization_matched_interpretation.json` as
  `836355b2cbbad0d82f677616a8dab84597419b70f02a77e1657d30cac1c3707f`.
- The `regularization_sweep.json` branch comparison is retained but superseded
  for causal interpretation because it changes axial regularization, phase
  selection, and encoding together. The matched factorial isolates those
  comparisons more cleanly; neither sweep establishes a universal optimum.
- The artifact-band fields in the density study are exploratory. They do not
  establish physical localization or a generating mechanism.
- The amplitude alphabets are per-phantom adaptive controls, not biological
  priors or upper bounds for optimized constrained encodings.
- MS-SSIM is computed after local rendering against displayed PNG targets. The
  results characterize fidelity under the local implementation of the
  published forward model, not recovery of measured OCT phase or tissue
  microstructure.
