# Full-set factorial ablation evidence

These published artifacts from the local implementation of the published
forward model support the primary ablation table in the camera-ready
manuscript.
Each directory contains the benchmark's per-image `detail.csv` and complete
`summary.json` for one cell of the fixed-regularization 2x2 design:

- phase iterations: 0 or 50;
- scatterer encoding: single or dispersion-canceling pair;
- references: all 120 public scans;
- fixed regularization: axial 0.02, lateral 0.05;
- fixed momentum: 1.0;
- evidence source: local implementation of the published forward model, not an
  official or hidden challenge score.

`../../ablation_results.json` records the cross-configuration paired contrasts,
40-series cluster-bootstrap intervals, inference seed, and dataset-manifest
hash. Keep the four CSV files together: paired conclusions require matching
reference paths across all configurations.

The JSON and CSV files retain the historical machine-readable token
`source_equivalent_local_scanner` exactly as generated. It denotes the local
implementation described above, not an organizer-hosted result.
