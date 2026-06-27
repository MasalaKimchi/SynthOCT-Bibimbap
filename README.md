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

Legacy direct-synthesis and facade modules were removed from source so the package surface mirrors the challenge stages.

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
synthoct baseline pipeline --name P06_visual_surface_dark_body --input data/DATASET_PNG/Female/1950-1960/Cheek/l__shcheka_frame250.png --out outputs/p06.txt
synthoct baseline final --input data/DATASET_PNG/Female/1950-1960/Cheek/l__shcheka_frame250.png --out outputs/final.txt
```

`baseline` is the legacy CLI name, but these commands generate phantoms, not final images. The current final method is still `H61_api_low_depth_prelim`, and H67/H68 remain available candidate phantom generators.

Nine P-series promising pipelines are available for validation. `P01`-`P05` are literature-inspired physics priors; `P06_visual_surface_dark_body` and `P07_surface_cutoff_broad_mix` are scanner-calibrated visual inverse methods from hosted-API sweeps; `P08_sparse_top_texture_ssim` and `P09_gamma_sparse_lowfloor_ssim` target plain SSIM and can be tested with `--scatterers-count 900000`.

For a single reference scan, the strongest current direction is scanner-in-loop density correction rather than another one-pass phantom. It renders a P09-like seed through the hosted API, computes smoothed reference/rendered correction fields, and emits ranked iterations:

```bash
synthoct optimize-correction \
  --input data/DATASET_PNG/Female/1950-1960/Cheek/l__shcheka_frame250.png \
  --out outputs/correction_refinement \
  --scatterers-count 900000 \
  --api-key-file ~/.config/synthoct/api_key
```

After a good rendered phantom exists, use selective transfer refinement to apply mild scanner-observed energy shaping and render the candidates back through the API:

```bash
synthoct optimize-transfer \
  --ref outputs/api_all_methods_render/reference.png \
  --phantom outputs/api_correction_iter4_sweep/phantoms/I4_03_density_r4_090.txt \
  --rendered-gray outputs/api_correction_iter4_sweep/synthetic_gray/I4_03_density_r4_090_gray.png \
  --out outputs/transfer_refinement \
  --exponents 0.04 0.10 0.18 0.24 \
  --api-key-file ~/.config/synthoct/api_key
```

On the first reference scan, selective transfer raised the best hosted-API SSIM from `0.2651` to `0.2724` with exponent `0.18`; a recursive micro-step from that result reached `0.2726` with exponent `0.04`.

The current strongest refinement is coordinate-preserving energy-ratio feedback. Starting from the transfer result, it applies smoothed `reference / rendered` energy corrections while keeping every scatterer coordinate fixed:

```bash
synthoct optimize-energy-ratio \
  --ref outputs/api_all_methods_render/reference.png \
  --phantom outputs/api_transfer_gen2_from_sgf02/phantoms/01_transfer_e0p040.txt \
  --rendered-gray outputs/api_transfer_gen2_from_sgf02/synthetic_gray/01_transfer_e0p040_gray.png \
  --out outputs/energy_ratio_refinement \
  --exponents 0.008 0.012 0.024 0.04 0.08 \
  --api-key-file ~/.config/synthoct/api_key
```

Seven recursive energy-ratio passes improved the first reference scan to hosted-API SSIM `0.2912`. The next recursive pass generated valid phantoms but the hosted API rejected them at POST with 500/400 responses, so `0.2912` is the current verified scanner-rendered best.

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

Submission packaging details are in [docs/submission/guide.md](docs/submission/guide.md), scanner notes are in [docs/implementation/pipeline.md](docs/implementation/pipeline.md), and challenge strategy is in [docs/challenge/overview.md](docs/challenge/overview.md).

For the current competition-winning interpretation, including the distinction between the true scanner, hosted API, local surrogate scanners, and single-reference scores, see [docs/challenge/winning_strategy.md](docs/challenge/winning_strategy.md).
