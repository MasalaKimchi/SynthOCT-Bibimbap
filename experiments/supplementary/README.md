# Supplementary experiments

These deterministic local-model runners support Supplementary Sections and
Tables S1--S4. Unless noted otherwise, they use the middle acquisition
(`frame250`) from each of the 40 filename-defined public acquisition series.

| Section | Runner | Compact published summary |
|---|---|---|
| S1: initialization | `exp_init.py` | `results/init_summary.json` |
| S2: lateral regularization | `exp_regularization_matched.py` | `results/regularization_matched_summary.json` |
| S3: overlap geometry | `exp_operators.py` | `results/operators.json` |
| S3: coefficient participation | `exp_density.py` | `results/density_summary.json` |
| S4: amplitude constraints | `exp_encoding.py` | `results/encoding_summary.json` |

Install the repository environment, download the dataset described in
[`../../docs/DATA.md`](../../docs/DATA.md), and run from the repository root:

```bash
python experiments/supplementary/exp_operators.py
python experiments/supplementary/exp_init.py
python experiments/supplementary/exp_regularization_matched.py
python experiments/supplementary/exp_density.py
python experiments/supplementary/exp_encoding.py
```

The commands write raw per-scan records and regenerated summaries beneath the
ignored directory:

```text
outputs/experiments/supplementary/
  operators.json
  init_study.json
  init_summary.json
  regularization_matched_factorial.json
  regularization_matched_summary.json
  density_study.json
  density_summary.json
  encoding_study.json
  encoding_summary.json
```

Approximate original runtimes on an Apple M3 Pro were seconds, 8 minutes,
26 minutes, 10 minutes, and 11 minutes, respectively. `exp_init.py`,
`exp_density.py`, and `exp_encoding.py` accept `--full` for all 120 scans;
those optional `_full.json` outputs are also ignored. The matched runner accepts
`--limit N` for development checks.

The tracked `results/` directory contains only the small summaries quoted by
the supplement. Exact JSON bytes can vary with dependency/BLAS versions, while
the numerical conclusions should reproduce within floating-point tolerance.
Supplementary Figure S1 is organizer-hosted evidence and is outside this local
bundle.
