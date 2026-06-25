# SynthOCT-Bibimbap

Mac-first baselines for the [SynthOCT Challenge 2026](https://synthoct.com/). The challenge output is a digital phantom: a text file of scatterers with columns `X, Y, Z, Energy`. The official Windows Virtual Scanner turns that phantom into an OCT B-scan for scoring.

## Setup

Target environment:

- Python `3.12.12`
- PyTorch `2.10.0`
- TorchVision `0.25.0`

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
synthoct baseline pretrained-cnn --input data/DATASET_PNG/Female/1950-1960/Cheek/l__shcheka_frame250.png --out outputs/cnn.txt --backbone resnet50
```

All baselines emit valid four-column phantom files. The pretrained-CNN baseline uses frozen CNN features when TorchVision weights are available and falls back to deterministic handcrafted features offline.

## Scanner Modes

The official scanner is `Part2_Scanner.exe`, so macOS development uses an adapter:

```bash
synthoct scan --phantom outputs/official.txt --out outputs/stub_scan.png --mode stub
synthoct scan --phantom outputs/official.txt --out outputs/real_scan.png --mode real --scanner-exe Part2_Scanner.exe
```

See [docs/scanner_windows.md](docs/scanner_windows.md) for Windows validation.

## Evaluation

```bash
synthoct evaluate --ref reference.png --pred outputs/stub_scan.png --maps --metrics --out-csv outputs/metrics.csv
```

The metric stack reports MSE, PSNR, SSIM, MS-SSIM, VIF, and LPIPS where optional dependencies are available.

## Internal Validation Loop

Run a private multi-fold leaderboard without submitting:

```bash
PYTHONPATH=src python -m synthoct.cli validate-internal \
  --zip 18095266.zip \
  --out outputs/internal_validation_quick \
  --folds 3 \
  --max-per-fold 2 \
  --scatterers-count 8000 \
  --no-maps
```

For a slower physics-focused pass, omit `--no-maps` and compare OAC/SC/RSC scores. This uses a deterministic surrogate scanner on macOS, so it is for iteration only; final ranking still needs the official Windows scanner.

## Research Notes

See [docs/challenge_strategy.md](docs/challenge_strategy.md) for the baseline ladder and competition strategy.

The current H0-H10 experiment report is in [docs/hypothesis_progress.md](docs/hypothesis_progress.md), with the generated figure at `outputs/hypothesis_progress/hypothesis_progress.png`.
