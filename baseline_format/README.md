# baseline_format — SynthOCT submission in the official baseline's shape

This folder presents the SynthOCT-Bibimbap method in the **flat-script layout of
the official [`SynthOCT_Baseline`](https://github.com/SynthOCTChallenge/SynthOCT_Baseline)**
(`Part1_Generator.py` / `Part2_Scanner` / `Part3_Processor.py` / `Orchestrator.py`),
so it drops into a baseline-oriented workflow without reorganizing the rest of
the repository.

These are **thin façades** over the installable, tested `synthoct` package in
`../src`. There is one source of truth (the package); every file here just
re-exposes it under the baseline's names and interfaces, so behaviour is
identical to the `synthoct` CLI. Nothing here reimplements the method.

## File-by-file mapping

| Baseline file | This folder | Delegates to (`synthoct` …) | Role |
| :--- | :--- | :--- | :--- |
| `Part1_Generator.py` (Editable) | `Part1_Generator.py` | `holographic_inverse.holographic_inverse_phantom` | **The model.** `ScattererGenerator.generate_from_reference()` inverts the fixed scanner against a reference B-scan (phase-pair holographic inversion). Byte-identical to `synthoct baseline holographic-inverse`. |
| `Part2_Scanner.exe` (Fixed) | `Part2_Scanner.py` | `scanners.render_with_api` + `scanners.write_api_config` | **Fixed scanner.** Renders a phantom through the organizer-hosted challenge service at synthoct.com (the organizers' binary is Windows-only). |
| `Part3_Processor.py` (Fixed) | `Part3_Processor.py` | `features.extraction.generate_maps` (`organizer-compatible-v1`) | **Fixed maps.** OAC / SC / RSC, 40 dB linearization, byte-for-byte the published Part3. Same `{Struct, OAC, SC, RSC}` return + `_OAC/_SC/_RSC.png` filenames. |
| `Orchestrator.py` (Manager) | `Orchestrator.py` | `evaluation.*` (mirrors `benchmark.py`) | **Manager.** Default **Challenge mode** (synthetic-vs-real). Self-consistency check retained as `--mode self-consistency`. |
| — | `_bootstrap.py` | — | Puts `../src` on `sys.path` if `synthoct` isn't installed, so `python Orchestrator.py` just works. |

## Fixed method parameters (the challenge configuration)

Defaults of `synthoct.holographic_inverse.HolographicInverseConfig`, applied to
every scan (no per-image tuning):

- axial regularization `0.02`, lateral regularization `0.05`
- `200` phase-retrieval iterations, momentum `1.0`
- `dispersion-canceling-pair` encoding
- max reflection amplitude `0.001`, target dynamic range `51 dB`
- zero-energy filler seed `7`

Scanner contract: `256 × 512`, `6 µm/px`, `1.3 µm` wavelength, beam radius `10 µm`,
`300,000` scatterers. Phantom: `X Y Z Energy(%)`, `X ∈ [−1536, 1536] µm`,
`Z ∈ [0, 1536] µm`, `Energy ∈ [0, 100] %`.

## Feature-map note (40 dB vs 51 dB)

`Part3_Processor.py` uses `organizer-compatible-v1`, which freezes the published
Part3's 40 dB linearization (`I = 10**(4·P)`) and save-time autoscaling — this is
what every competition-formula estimate uses. The documented 51 dB scientific
map mode exists in `synthoct` but is audit-only and intentionally not surfaced
here.

## Requirements

Use the repository's environment (Python 3.12): `pip install -e ".[dev,lpips]"`
from the repo root, or the pinned `synthoct-py312` conda env. `_bootstrap.py`
otherwise falls back to `../src`. A SynthOCT **API key** is required for
rendering (`export SYNTHOCT_API_KEY=…`, or pass `--api-key-file`). `--include-lpips`
needs the `lpips`/`torch` extra.

## Usage

Generate one phantom from a reference B-scan (Part1 only):

```bash
cd baseline_format
python Part1_Generator.py --input /path/to/reference.png --out phantom.txt --diagnostics diag.json
```

Render a phantom through the organizer-hosted challenge service (Part2 only):

```bash
python Part2_Scanner.py phantom.txt synthetic.png --api-key-file ~/.synthoct_key
```

Maps from a scan (Part3 only) — writes `synthetic_OAC.png`, `_SC.png`, `_RSC.png`:

```python
import Part3_Processor as P; P.generate_maps("synthetic.png")
```

Full Challenge-mode evaluation over a reference directory:

```bash
python Orchestrator.py --mode challenge \
  --reference-root /path/to/DATASET_PNG \
  --out-dir outputs/baseline_format_run \
  --include-lpips --api-key-file ~/.synthoct_key --limit 5
```

Writes `Final_Metrics_Report.csv` + `summary.json` (with per-scan hosted request
IDs and, with `--include-lpips`, the eight-median competition-formula estimate).

Faithful port of the baseline's original A/B check:

```bash
python Orchestrator.py --mode self-consistency --out-dir outputs/self_consistency --api-key-file ~/.synthoct_key
```

## Scope of scores

Everything produced here is a **local / hosted development estimate**, explicitly
**not** an organizer-issued leaderboard or hidden-test score (`summary.json`
records `official_or_hidden_score: false`). The leaderboard submission itself is
a ZIP of phantom `.txt` files — build it with `synthoct generate-batch` →
`synthoct prepare-submission` → `synthoct verify-submission` (see the repo
README). This folder is the baseline-shaped view of the method, not a separate
implementation of it. See `../docs/COMPLIANCE.md` for the full validity boundary.
