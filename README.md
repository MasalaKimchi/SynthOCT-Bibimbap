# SynthOCT-Bibimbap

Inverse-physics digital phantom generation for the [SynthOCT Challenge 2026](https://synthoct.com/).

This repo does **not** submit direct OCT image synthesis as its primary algorithmic output. The challenge contract is a scanner-compatible digital phantom: a plain text scatterer table with four columns:

```text
X Y Z Energy
```

The SynthOCT Virtual Scanner renders those phantoms into synthetic OCT PNGs for scoring.

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

Keep `data/`, `outputs/`, `secrets/`, scanner binaries, API keys, and downloaded challenge assets out of git.

## Repository Stages

- `synthoct.dataset`: load/list the Zenodo OCT reference scans.
- `synthoct.features`: extract OAC, speckle contrast, depth profiles, and boundary estimates from real scans.
- `synthoct.generators`: generate four-column phantom scatterer files from reference scans.
- `synthoct.scanners`: render phantoms with the hosted SynthOCT API or Windows `Part2_Scanner.exe`.
- `synthoct.evaluation`: compute MS-SSIM, LPIPS/fallback metrics, and Struct/OAC/SC/RSC comparisons on rendered PNGs.
- `synthoct.submission`: create manifest/phantom/code packages and preliminary portal PNG-pair folders.

Historical modules such as `synthoct.baselines`, `synthoct.processor`, `synthoct.metrics`, and `synthoct.api` remain compatibility facades.

## Workflow 1: Generate Digital Phantoms

Prepare or inspect the dataset:

```bash
synthoct data prepare --zip 18095266.zip --out data
synthoct data list --zip 18095266.zip | head
```

Generate scanner-compatible phantoms from real OCT reference scans:

```bash
synthoct baseline official --out outputs/official.txt
synthoct baseline heuristic --input data/DATASET_PNG/Female/1950-1960/Cheek/l__shcheka_frame250.png --out outputs/heuristic.txt
synthoct baseline hypothesis --name H61_api_low_depth_prelim --input data/DATASET_PNG/Female/1950-1960/Cheek/l__shcheka_frame250.png --out outputs/h61.txt
synthoct baseline final --input data/DATASET_PNG/Female/1950-1960/Cheek/l__shcheka_frame250.png --out outputs/final.txt
```

`baseline` is the legacy CLI name, but these commands generate phantoms, not final images. The current final method is still `H61_api_low_depth_prelim`, and H67/H68 remain available candidate phantom generators.

## Workflow 2: Render Phantoms To Synthetic OCT PNGs

macOS should use the hosted SynthOCT API by default because `Part2_Scanner.exe` is a Windows executable:

```bash
synthoct scan \
  --phantom outputs/final.txt \
  --out outputs/final_api_scan.png \
  --api-key-file ~/.config/synthoct/api_key
```

Windows official local validation remains available:

```bash
synthoct scan \
  --phantom outputs/final.txt \
  --out outputs/final_windows_scan.png \
  --mode windows \
  --scanner-exe Part2_Scanner.exe
```

The shared code path is `synthoct.scanners.render_phantom(phantom_path, config_path, output_png, backend=...)`.

Evaluate rendered PNGs against real references:

```bash
synthoct evaluate --ref reference.png --pred outputs/final_api_scan.png --maps --metrics --out-csv outputs/metrics.csv
```

## Workflow 3: Prepare Preliminary Portal PNG Pairs

The preliminary portal asks for rendered PNG pairs:

1. Synthetic OCT scan PNG rendered from the generated phantom.
2. Matching real reference scan PNG from the Zenodo dataset.

It does not ask for raw phantom files for this preliminary upload step.

Create a phantom package and manifest:

```bash
synthoct prepare-submission \
  --zip 18095266.zip \
  --out outputs/submission_ready_h61 \
  --scatterers-count 300000
```

Render those phantoms through the hosted API and rank the resulting pairs:

```bash
synthoct api-evaluate-submission \
  --zip 18095266.zip \
  --submission-dir outputs/submission_ready_h61 \
  --out outputs/api_preliminary_h61 \
  --api-key-file ~/.config/synthoct/api_key
```

Copy the selected PNG pairs into a clean upload folder:

```bash
synthoct prepare-png-pairs \
  --upload-plan outputs/api_preliminary_h61/preliminary_upload_plan.csv \
  --out outputs/preliminary_png_pairs_h61
```

Submission packaging details are in [docs/submission_checklist.md](docs/submission_checklist.md), and Windows scanner notes are in [docs/scanner_windows.md](docs/scanner_windows.md).
