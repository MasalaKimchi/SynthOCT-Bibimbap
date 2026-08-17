# Experiments

This directory contains the Python runners and compact published summaries for
the camera-ready paper:

- [`main_ablation/`](main_ablation/) reproduces both all-120 local-model
  comparisons and records the organizer-hosted final-method aggregate.
- [`supplementary/`](supplementary/) reproduces the local S1--S4 diagnostic
  experiments.

Download the public dataset as described in [`../docs/DATA.md`](../docs/DATA.md)
and place it at `DATASET/DATASET_PNG/`. Generated per-scan CSV/JSON files are
written under `outputs/experiments/`; that directory is ignored by Git. Only
small summaries used in the manuscript are retained here.

Evidence scope is labeled in each compact result. Local-model comparisons are
not organizer-issued, leaderboard, or hidden-test scores. Hosted reruns require
challenge-service credentials and create new request IDs; Supplementary Figure
S1 uses those external organizer-hosted renders.
