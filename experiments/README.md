# Experiments and evidence

This directory publishes the numerical evidence used by the camera-ready
manuscript and supplement. The two bundles have deliberately different scopes:

- [`main_ablation/`](main_ablation/) contains the primary fixed-regularization
  2 x 2 ablation on all 120 public B-scans.
- [`supplementary/`](supplementary/) contains post-review sensitivity analyses
  supporting Supplementary Sections and Tables S1--S4.

All scores in these bundles were computed with this repository's local
implementation of the published forward model. They are not hidden-test,
leaderboard, or organizer-issued scores. The supplement's Figure S1 uses
separate organizer-hosted renders and is therefore not reproduced by these
local diagnostic scripts.

The raw dataset is intentionally not tracked. See [`../docs/DATA.md`](../docs/DATA.md)
for download, checksum, and extraction instructions.
