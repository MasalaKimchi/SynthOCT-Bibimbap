# SynthOCT-Bibimbap

Mac-first baselines for the [SynthOCT Challenge 2026](https://synthoct.com/). The challenge output is a digital phantom: a text file of scatterers with columns `X, Y, Z, Energy`. The official Windows Virtual Scanner turns that phantom into an OCT B-scan for scoring.

## Setup

Target environment:

- Python `3.12.12`

```bash
conda env create -f environment.yml
conda activate synthoct-py312
```

or:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Dataset

The local `18095266.zip` is treated as the official Zenodo dataset archive.

```bash
synthoct data prepare --zip 18095266.zip --out data
synthoct data list --zip 18095266.zip | head
```

## Baselines

```bash
synthoct baseline official --out outputs/official.txt
synthoct baseline heuristic --input data/DATASET_PNG/Female/1950-1960/Cheek/l__shcheka_frame250.png --out outputs/heuristic.txt
synthoct baseline hypothesis --name H61_api_low_depth_prelim --input data/DATASET_PNG/Female/1950-1960/Cheek/l__shcheka_frame250.png --out outputs/h61.txt
synthoct baseline final --input data/DATASET_PNG/Female/1950-1960/Cheek/l__shcheka_frame250.png --out outputs/final.txt
```

All baselines emit valid four-column phantom files. Final ranking must use phantoms rendered through the hosted SynthOCT API or official Windows scanner.

## Scanner Modes

All ranking renders must use the hosted SynthOCT API or the official Windows `Part2_Scanner.exe`:

```bash
synthoct scan --phantom outputs/official.txt --out outputs/real_scan.png --mode real --scanner-exe Part2_Scanner.exe
```

See [docs/scanner_windows.md](docs/scanner_windows.md) for Windows validation.

## Evaluation

```bash
synthoct evaluate --ref reference.png --pred outputs/real_scan.png --maps --metrics --out-csv outputs/metrics.csv
```

The metric stack reports MSE, PSNR, SSIM, MS-SSIM, VIF, and an LPIPS proxy. Install `.[lpips]` only when real LPIPS is needed locally.

## Internal Validation Loop

Run a private multi-fold leaderboard through the hosted scanner API:

```bash
PYTHONPATH=src python -m synthoct.cli validate-internal \
  --zip 18095266.zip \
  --out outputs/api_validation \
  --methods H61_api_low_depth_prelim H67_coarse_to_fine_crisp H68_layer_map_prior \
  --folds 3 \
  --max-per-fold 1 \
  --api-key-file ~/.config/synthoct/api_key
```

This issues one hosted API render per method/sample. Use `--scatterers-count` below `300000` only for explicit smoke tests, not ranking.

## Research Notes

See [docs/challenge_strategy.md](docs/challenge_strategy.md) for the baseline ladder and competition strategy.

The current API-first strategy is in [docs/challenge_strategy.md](docs/challenge_strategy.md), with the theory context in [docs/theory_context.md](docs/theory_context.md). Older offline-proxy notes are kept as historical context only and should not be used to claim challenge performance.

The current API preliminary winner is `H61_api_low_depth_prelim`. Generate the default final phantom with:

```bash
synthoct baseline final --input reference.png --out outputs/Scatterers_H61_Final.txt
```

Submission packaging instructions are in [docs/submission_checklist.md](docs/submission_checklist.md), and the end-to-end hypothesis/verification/submission flowchart is in [docs/submission_flowchart.md](docs/submission_flowchart.md).
