# Experiments

This directory contains the Python runners and compact published summaries for
the camera-ready paper:

- [`main_ablation/`](main_ablation/) reproduces the all-120 fixed-regularization
  2 x 2 ablation.
- [`supplementary/`](supplementary/) reproduces the local S1--S4 diagnostic
  experiments.

Download the public dataset as described in [`../docs/DATA.md`](../docs/DATA.md)
and place it at `DATASET/DATASET_PNG/`. Generated per-scan CSV/JSON files are
written under `outputs/experiments/`; that directory is ignored by Git. Only
small summaries used in the manuscript are retained here.

All scores are local-model public-data evidence, not organizer-issued,
leaderboard, or hidden-test scores. Supplementary Figure S1 uses separate
organizer-hosted renders and is not reproduced by these local runners.
