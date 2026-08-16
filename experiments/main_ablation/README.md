# Main-paper ablation

This bundle supports the primary fixed-regularization 2 x 2 ablation reported
in the main paper. It crosses:

- phase selection: zero iterations or 50 momentum iterations; and
- encoding: one shifted scatterer or a dispersion-canceling phase pair.

All 120 public B-scans were evaluated at fixed axial regularization 0.02,
lateral regularization 0.05, phase momentum 1.0, and 51 dB dynamic range using
the repository's local implementation of the published forward model. These
are local validation scores, not official, hosted, hidden-test, or leaderboard
scores.

## Files

- [`ablation_results.json`](ablation_results.json) records configuration
  summaries, paired contrasts, inference settings, the software environment,
  and the dataset-manifest SHA-256.
- [`evidence/ablation_fullset/`](evidence/ablation_fullset/) contains one
  `summary.json` and one per-image `detail.csv` for each of the four cells.

The CSV files are the complete per-image evidence currently available for this
ablation and must remain paired by reference path for the reported contrasts.
No additional per-case JSON, rendered images, phantom CSVs, or execution logs
are included or implied. The bundle records results rather than an independent
runner; the production CLI and method are documented in the repository root
and [`../../docs/METHODS.md`](../../docs/METHODS.md).

The machine-readable evidence retains the historical token
`source_equivalent_local_scanner` exactly as generated. In this release that
legacy token means only "local implementation of the published forward model";
it does not designate an official or organizer-hosted result.

## Dataset provenance

The references are version 1 of the public in-vivo skin OCT dataset, Zenodo
record 18095266, expected locally at `DATASET/DATASET_PNG/`. The dataset itself
is not tracked. The exact ordered reference manifest used by this run is bound
by the SHA-256 stored in `ablation_results.json`; archive checksums and extraction
instructions are in [`../../docs/DATA.md`](../../docs/DATA.md).
