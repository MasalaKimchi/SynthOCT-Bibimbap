# Implementation Pipeline

The package mirrors the SynthOCT challenge stages: load reference scans, extract physical features, generate scatterer phantoms, render with a scanner backend, evaluate rendered PNGs, then package submission artifacts.

## Package Layout

- `synthoct.dataset`: load/list Zenodo OCT reference scans.
- `synthoct.features`: compute OAC, speckle contrast, refined speckle contrast, depth profiles, and boundary estimates.
- `synthoct.generators`: generate official, heuristic, physics-guided, named-hypothesis, and final phantoms.
- `synthoct.scanners`: render phantoms through the hosted SynthOCT API, Windows `Part2_Scanner.exe`, or precomputed fixture mode.
- `synthoct.evaluation`: compute MS-SSIM, LPIPS/fallback metrics, and Struct/OAC/SC/RSC comparisons.
- `synthoct.submission`: write manifests, phantom zips, code zips, API render results, upload plans, and portal-ready PNG-pair folders.

Legacy direct-synthesis and facade modules were removed so the source surface stays aligned with inverse-physics phantom generation.

## Generating Phantoms

Prepare or inspect the dataset:

```bash
synthoct data prepare --zip 18095266.zip --out data
synthoct data list --zip 18095266.zip | head
```

Generate challenge-compatible phantoms:

```bash
synthoct baseline official --out outputs/official.txt

synthoct baseline heuristic \
  --input data/DATASET_PNG/Female/1950-1960/Cheek/l__shcheka_frame250.png \
  --out outputs/heuristic.txt

synthoct baseline hypothesis \
  --name H61_api_low_depth_prelim \
  --input data/DATASET_PNG/Female/1950-1960/Cheek/l__shcheka_frame250.png \
  --out outputs/h61.txt

synthoct baseline final \
  --input data/DATASET_PNG/Female/1950-1960/Cheek/l__shcheka_frame250.png \
  --out outputs/final.txt
```

The CLI subcommand is still named `baseline` for compatibility, but these commands write phantom scatterer files, not final OCT images.

## Scanner Backends

Use the scanner abstraction from Python:

```python
from synthoct.scanners import render_phantom

render_phantom("phantom.txt", "Configuration.ini", "synthetic.png", backend="api")
render_phantom("phantom.txt", "Configuration.ini", "synthetic.png", backend="windows")
```

The default CLI scanner mode is the hosted SynthOCT API, which is the correct macOS path because `Part2_Scanner.exe` is a Windows executable:

```bash
synthoct scan \
  --phantom outputs/final.txt \
  --out outputs/final_api_scan.png \
  --api-key-file ~/.config/synthoct/api_key
```

API keys are read from `SYNTHOCT_API_KEY`, `SYNTHOCT_CHALLENGE_API_KEY`, or an untracked file passed with `--api-key-file`.

For official local Windows validation:

1. Create the Python `3.12.12` environment from `environment.yml`.
2. Download `Part2_Scanner.exe` from the official baseline instructions.
3. Place `Part2_Scanner.exe` at the repository root or pass `--scanner-exe`.
4. Generate a four-column phantom.
5. Render with Windows mode.
6. Evaluate the rendered PNG.

```bash
synthoct baseline final \
  --input data/DATASET_PNG/Female/1950-1960/Cheek/l__shcheka_frame250.png \
  --out outputs/final.txt

synthoct scan \
  --phantom outputs/final.txt \
  --out outputs/final_windows_scan.png \
  --mode windows \
  --scanner-exe Part2_Scanner.exe

synthoct evaluate \
  --ref data/DATASET_PNG/Female/1950-1960/Cheek/l__shcheka_frame250.png \
  --pred outputs/final_windows_scan.png \
  --maps \
  --metrics \
  --out-csv outputs/metrics.csv
```

Keep generation plus rendering within the challenge target of 600 seconds per B-scan on the stated hardware class.

## Internal Validation

Use hosted-API validation as the private leaderboard:

```bash
synthoct validate-internal \
  --zip 18095266.zip \
  --out outputs/api_validation \
  --methods H61_api_low_depth_prelim H67_coarse_to_fine_crisp H68_layer_map_prior \
  --folds 5 \
  --max-per-fold 2 \
  --include-lpips \
  --api-key-file ~/.config/synthoct/api_key
```

Important outputs:

- `challenge_metrics_summary.csv`: challenge-facing MS-SSIM and LPIPS or LPIPS proxy summary.
- `internal_validation_summary.csv`: full diagnostic table.
- `hypothesis_wins.csv`: per-sample win counts.
- `internal_validation_detail.csv`: sample-level diagnostics.

Use `--rerun-existing` only when cached API renders should be ignored.
