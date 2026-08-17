# Main-paper experiments

This directory keeps the compact numerical evidence used by the camera-ready
paper. Large per-scan results are deliberately regenerated below `outputs/`,
which is ignored by Git.

## Local-model comparisons

[`run.py`](run.py) reproduces two studies on the public reference scans:

- the fixed-regularization 2×2 ablation crossing zero/50 RAP iterations with
  single/phase-pair encoding at axial/lateral regularization 0.02/0.05; and
- the earlier method (0.03/0.20 regularization, zero phase, single scatterer)
  versus the final method (0.02/0.05, 200 RAP iterations, phase pair).

From the repository root, the default runs both studies:

```bash
python experiments/main_ablation/run.py
```

Use `--study ablation` or `--study full-comparison` to run one study. A quick
installation smoke test is:

```bash
python experiments/main_ablation/run.py \
  --study all --limit 1 --bootstrap-replicates 1000
```

`--limit` always labels the aggregates as development subsets; only a complete
120-scan run whose manifest matches the published public set is labeled
`full_public_120`. `--summarize-only` rebuilds the selected aggregate from
existing `detail.csv` and `summary.json` files without rerunning the scanner.

The default output inventory is:

```text
outputs/experiments/main_ablation/
  zero_single/{detail.csv,summary.json}
  momentum50_single/{detail.csv,summary.json}
  zero_pair/{detail.csv,summary.json}
  momentum50_pair/{detail.csv,summary.json}
  earlier_zero_single/{detail.csv,summary.json}
  final200_pair/{detail.csv,summary.json}
  ablation_results.json
  full_method_comparison.json
```

The tracked compact counterparts are
[`ablation_results.json`](ablation_results.json) and
[`full_method_comparison.json`](full_method_comparison.json). The full run is
CPU-only and hardware-dependent; expect tens of minutes rather than smoke-test
latency.

## Organizer-hosted final-method evaluation

Table 2 uses the external organizer-hosted challenge service, not the local
model. Install the `lpips` extra, generate the final phantoms, then run the
resumable hosted evaluator:

```bash
synthoct generate-batch \
  --reference-root DATASET/DATASET_PNG \
  --out-dir outputs/experiments/main_hosted/phantoms

SYNTHOCT_API_KEY="..." python tools/run_hosted_api_public120.py \
  --reference-root DATASET/DATASET_PNG \
  --phantom-dir outputs/experiments/main_hosted/phantoms \
  --out-dir outputs/experiments/main_hosted/evaluation
```

The API key is read from the environment and is not serialized. The hosted
runner writes resumable request state, request IDs, raw/grayscale renders,
feature maps, per-scan metrics, and a summary below the ignored output
directory. Service access and valid credentials are required, and a fresh run
receives fresh request IDs. The historical request ledger and raw hosted
renders were not retained in Git; [`hosted_results.json`](hosted_results.json)
therefore records the camera-ready aggregate and this evidence limitation
explicitly, rather than presenting it as a hidden or official score.

Dataset version, archive hashes, and extraction instructions are in
[`../../docs/DATA.md`](../../docs/DATA.md).
