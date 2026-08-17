# SynthOCT-Bibimbap — Methods

This document covers the phase-pair holographic-inversion algorithm, its mapping
to the challenge baseline, the Struct/OAC/SC/RSC feature-map audit, and the
final evidence behind the fixed method.

**Scope note on numbers.** Scores from different scopes or metric backends are not
interchangeable. Results labeled *local* use this repository's implementation of
the published forward model; results labeled *hosted* use the organizer's
challenge service. Both use public data and are **never** organizer-leaderboard
or hidden-test results.

---

## 1. Phase-Pair Holographic Inversion (core method)

The local published forward model is a **low-reflectivity coherent linear-field
approximation** followed by magnitude, log compression, and clipping. The method
recovers a 300,000-row phantom whose scanner render matches a real B-scan by
choosing a phase favored by the regularized model and encoding each field
coefficient as a dispersion-canceling pair of sub-resolution rows.

### 1.1 Why the earlier method stalled

The earlier method inverted a real, zero-phase target field and encoded each
complex coefficient with a single sub-wavelength depth shift. Two effects
limited it:

- The axial Hanning window leaves the 256×256 operator at **rank 254**, so forcing zero phase
  excites poorly conditioned modes and creates axial ringing.
- A single depth shift has the desired coefficient phase only at the center wavenumber; its phase
  drifts across the scanner bandwidth.

### 1.2 Momentum phase retrieval

Let `M` = target field magnitude, `A` = axial operator, `L` = lateral beam operator, and `A+`,
`L+` their Tikhonov pseudoinverses. Starting from `F = M`, each iteration computes:

```text
C = A+ F (L+)T
P = A C LT
q = angle(P)
q_target = q + beta * wrapped(q - q_previous)
F = M * exp(i * q_target)
```

This preserves the measured magnitude while choosing a phase favored by the
regularized model; it is not a hard feasible-set projection. **Fixed settings:**
200 iterations, `beta = 1`, Tikhonov regularization `alpha_z = 0.02`, `alpha_x = 0.05`.

### 1.3 Dispersion-canceling pair

For coefficient `C = |C| exp(i·phi)`, the single-row encoding used depth offset
`d0 = -phi·lambda/(4·pi)`. The final encoding adds a carrier-equivalent
companion half a wavelength away on the opposite side:

```text
d1 = d0 - sign(d0) * lambda/2
w0 = -d1 / (d0 - d1)
w1 =  d0 / (d0 - d1)
```

Both weights are nonnegative and sum to one. The pair has the exact desired phase at the center
wavenumber (offsets differ by `lambda/2`), while the weighted mean offset vanishes:

```text
w0*d0 + w1*d1 = 0.
```

The first derivative of phase with respect to wavenumber therefore cancels; the remaining
dispersion is second order. Energies are `100*(scale*|C|*w)^2`, with each reflection amplitude
capped at `0.001`.

### 1.4 Contract packing

At 256×512 the pair requires **262,144 active rows** and leaves **37,856 zero-energy fillers** in
the 300,000-row contract. Active depths remain within bounds, approximately **2.35 to 1533.65 µm**.

### 1.5 Measured effect (earlier method → final method)

The authoritative all-public local-model comparison used the current float64
metric loader on 120 paired scans:

| Structural MS-SSIM | Earlier zero-phase single | Final 200-iteration pair |
|---|---:|---:|
| Mean | 0.955954 | 0.994588 |
| Median | 0.956021 | 0.994849 |
| Minimum | 0.933827 | 0.985133 |

The final method improved all 120 scans and all 40 filename-defined acquisition
series. Its mean paired gain was `0.038634`; the 200,000-replicate series-cluster
95% CI was `[0.037530, 0.039790]`, and the one-sided exact Wilcoxon signed-rank
test gave `p=9.095e-13`. These are public-development results, not hidden-test
proof. The exact configurations, runner, and compact result are under
[`experiments/main_ablation/`](../experiments/main_ablation/).

---

## 2. Mapping to the Challenge Baseline

Maps SynthOCT-Bibimbap onto the official
[`SynthOCTChallenge/SynthOCT_Baseline`](https://github.com/SynthOCTChallenge/SynthOCT_Baseline)
pipeline (checked at commit `9508e27c4b0bd54e0d76172a1c85af329472f2fb`) so a reviewer familiar with
the baseline can locate every counterpart and confirm the fixed scanner contract is unchanged.

### 2.1 Component correspondence

| Baseline file | Role in baseline | Type | SynthOCT-Bibimbap counterpart |
| :--- | :--- | :--- | :--- |
| `Part1_Generator.py` | Produces `(X, Y, Z, Energy%)` scatterers and writes the `.txt` phantom. The component participants improve. | **Editable** | `src/synthoct/holographic_inverse.py` (`holographic_inverse_phantom`), driven by `synthoct baseline holographic-inverse` (one scan) and `synthoct generate-batch` (full reference set). Phantom I/O and the numeric contract live in `src/synthoct/phantom.py`. |
| `Part2_Scanner.exe` | Fixed coherent virtual scanner (external Windows binary); renders a phantom into a raw B-scan. | **Fixed** | Not reimplemented. The organizer-hosted scanner is called via `synthoct scan` (`src/synthoct/scanners/api.py`); `src/synthoct/scanners/reference.py` implements the published forward model for controlled local comparisons only. The organizer-hosted render remains authoritative. |
| `Part3_Processor.py` | Fixed OAC / SC / RSC parametric maps, 40 dB linearization. | **Fixed** | `src/synthoct/features/extraction.py`, `organizer-compatible-v1` mode. Reproduces the published 40 dB `10**(P*4)` linearization and OAC/SC/RSC encoding byte-for-byte; a separate `scientific-v1` mode is audit-only and never enters a competition score. |
| `Orchestrator.py` | Manager: `import Part1_Generator as Generator`, drives generation → scan → maps → metrics; ships in self-consistency mode. | **Manager** | `src/synthoct/cli.py` (`benchmark-local`, `evaluate`) plus `src/synthoct/benchmark.py`. The baseline's default Orchestrator is a self-consistency demo that "must be reconfigured for real/synthetic comparison"; Bibimbap pairs official Zenodo references with scanner renders directly. |

### 2.2 Scanner contract parity

Fixed scanner parameters are identical to the baseline `ExperimentConfig`
(`src/synthoct/phantom.py` vs. baseline `Part1_Generator.py`):

| Parameter | Value | Source |
| :--- | :--- | :--- |
| A-scan pixels (`N_depth`) | 256 | Zenodo dataset spec |
| B-scan A-scans (`N_lateral`) | 512 | Zenodo dataset spec |
| Pixel size (z, x) | 6.0 µm | Zenodo dataset spec |
| Central wavelength | 1.3 µm | Zenodo dataset spec |
| Beam radius | 10.0 µm (20 µm diameter) | Zenodo dataset spec |
| Scatterers per phantom | 300000 | Baseline contract |
| Phantom columns | `X Y Z Energy(%)` | Baseline contract |
| X bounds | [-1536, 1536] µm | `= ±x_max/2` |
| Z bounds | [0, 1536] µm | `= [0, z_max]` |
| Energy bounds | [0, 100] % | Baseline contract |

`validate_phantom` enforces every bound above and rejects any file that is not exactly
`300000 × 4` finite columns. The baseline writes `fmt='%.4e'`; Bibimbap writes `%.6e` (strictly
higher precision — the scanner parses either).

### 2.3 Dynamic-range note

The baseline `Part3_Processor.py` linearizes with `10**(P*4)` (40 dB) while the scanner
documentation states 51 dB. Bibimbap preserves the published 40 dB behavior in
`organizer-compatible-v1` for all competition-formula estimates, and isolates the documented 51 dB
calibration in a non-scoring `scientific-v1` audit mode (see §3).

### 2.4 Reproducing a full submission

The organizers hand participants reference B-scans (120 public; 60 hidden at final evaluation).
One command inverts every reference into a retained, submission-named phantom directory:

```bash
synthoct generate-batch \
  --reference-root path/to/DATASET_PNG \
  --out-dir outputs/hidden_test/phantoms
```

Each phantom is written as `sex_age_site_name.txt` (the flattened name the ZIP builder expects),
re-parsed against the scanner contract, and logged with a per-file SHA-256 in
`generation_summary.json`. The directory is then a drop-in `--phantom-dir` for packaging:

```bash
synthoct prepare-submission \
  --reference-root path/to/DATASET_PNG \
  --phantom-dir outputs/hidden_test/phantoms \
  --out outputs/hidden_test/submission.zip \
  --count 60 --seed 2026

synthoct verify-submission \
  --archive outputs/hidden_test/submission.zip \
  --manifest outputs/hidden_test/submission.manifest.json \
  --reference-root path/to/DATASET_PNG \
  --expected-count 60
```

`generate-batch` retains phantoms and provenance, unlike `benchmark-local`, which generates in a
temporary directory for scoring only. Confirm the signed-in portal's exact root/naming convention
before upload.

---

## 3. Feature Maps: Definitions & Audit

### 3.1 Two non-interchangeable modes

The corrected implementation keeps two modes because the published compatibility path contains not
only one genuine save-time bug but also calibration, estimator-validity, padding, clipping,
normalization, and metric-region issues:

- **`organizer-compatible-v1`** — byte-frozen published behavior, for challenge estimates only
  (40 dB linearization, the historical CSV schema, LPIPS scored on natural-image PNGs).
- **`scientific-v1`** — 51 dB float maps, explicit reference-defined masks, fixed preview scales,
  and no competition formula (rejects LPIPS, since masked float parameter arrays are not
  natural-image inputs).

Phantom **generation** always runs at 51 dB. Existing final-method phantoms do not need regeneration
because of the Part3 contradiction.

### 3.2 Folder semantics

A hosted reproduction writes beneath the selected `--out-dir` (the documented
example uses `outputs/experiments/main_hosted/evaluation/`) with this structure:

- `raw/` — PNG payload returned by the hosted virtual-scanner API (some stored as color/RGBA even
  though the signal is grayscale).
- `gray/` — deterministic 8-bit grayscale conversion used for structural (Struct) evaluation. All
  120 raw→gray conversions were regenerated and matched the retained files exactly.
- `maps/reference/` — OAC, SC, RSC images computed *from each reference B-scan* by the published
  Part3 processor.
- `maps/prediction/` — the same transformations computed from the paired hosted render.

Thus `maps/reference/` was **not** an independent parametric ground-truth dataset. Unusual terminal
rows, borders, saturated regions, and contrast differences were products of the map processor. The
large generated run was removed during final repository cleanup. The hosted runner can regenerate
the raw outputs and metrics, while the compact aggregate is tracked under
[`experiments/main_ablation/`](../experiments/main_ablation/). The assembled
manuscript figure itself is supplied separately with the paper.

### 3.3 All-120 integrity checks

One-to-one inventories were verified for 120 references, phantoms, API jobs, raw PNGs, grayscale
PNGs, and metric rows. Flattened names were reversed against reference-relative paths with no
collisions or cross-case pairings. All **720** organizer map PNGs (3 map types ×
reference/prediction × 120 cases) were regenerated from their claimed source. A deliberately
shifted-pairing negative control scored substantially worse, supporting the pairing audit.

Retained **hosted** structural results:

| Quantity (hosted, n=120) | Value |
| :--- | ---: |
| Mean MS-SSIM | 0.9942843141 |
| Median MS-SSIM | 0.9945532174 |
| Minimum MS-SSIM | 0.9857009208 |
| Median LPIPS | 0.0226107640 |
| Eight-median formula (byte-compatible path) | 0.9940789428 |

*(This hosted mean `0.99428` differs from the §1.5 local-model mean `0.994588` only by
scope — organizer challenge service vs local implementation; both are public-development estimates,
not a leaderboard/hidden-test score.)*

The `scientific-v1` pass on all 120 pairs produced diagnostic derived-map metrics only. The audit
tool writes these to `audit/scientific_v1_metrics.csv` and `audit/scientific_v1_summary.json`:

| scientific-v1 (masked) | OAC | SC | RSC |
| :--- | ---: | ---: | ---: |
| Mean valid fraction | 0.50598 | 0.19734 | 0.19734 |
| Mean float-map correlation | 0.99873 | 0.99869 | 0.99884 |
| Mean masked absolute error | 0.000185 | 0.04721 | 0.02666 |

### 3.4 Known defects in the published path

1. **Dynamic-range contradiction.** Metadata/scanner specify ~51 dB, but Part3 expands the PNG as
   `10 ** (pixel * 40 / 10)`. Using 40 dB changes derived arrays, especially speckle-derived
   quantities. The inverse scanner already used 51 dB; 40 dB is kept only inside the exact
   compatibility evaluator.
2. **RSC save-time autoscaling.** The processor normalizes RSC to nominal `[0.5, 5]` then calls
   Matplotlib without `vmin=0, vmax=1`, so Matplotlib can re-normalize the array. This occurred for
   14 of 120 reference/prediction pairs (28 PNGs), producing gain mismatches up to ~17%. Frozen in
   compatibility mode, removed from scientific previews.
3. **Invalid OAC terminal row.** The cumulative OAC estimator has an unreliable terminal
   denominator; percentile normalization then saturates that row white. Scientific mode masks
   configured terminal rows and unreliable denominators, and erodes RSC validity through the full
   local-window footprint (an RSC pixel cannot stay valid if it depends on invalid OAC input).
4. **Manufactured SC/RSC borders.** The published code computes a reflected-window filter, crops a
   10-pixel border, and pads back by duplicating edge values — creating constant borders.
   Scientific mode keeps the float filter output but marks any pixel whose full 20-pixel window is
   not valid.
5. **Clipping and independent normalization.** SC/RSC are clipped to fixed display ranges while each
   OAC image is normalized by its own 99th percentile. PNG similarity after these operations does
   not establish equal physical units. Scientific mode stores float arrays with common fixed
   preview scales; PNGs are explicitly non-quantitative previews.
6. **Background-weighted metrics.** Full-frame PNG metrics over-weight shared dark/clipped regions.
   Scientific mode builds a contiguous per-A-scan support from the smoothed reference display,
   requires reliable OAC denominators, and propagates complete filter footprints. Only the
   *reference* mask defines the comparison region, so a prediction cannot improve its score by
   declaring difficult pixels invalid.
7. **Cross-mode overwrite and mixed diagnostics.** An early two-mode implementation reused PNG
   filenames and left profile diagnostics fixed at 40 dB, risking overwrites or mixing 40 dB
   diagnostics into a 51 dB row. Scientific outputs now live under a mode-specific directory, and
   all profile calculations receive the selected map mode.

### 3.5 Compatibility guarantees & interpretation boundary

The organizer mode remains the default and preserves the historical CSV schema. A checked-in
deterministic input has fixed upstream-compatible OAC/SC/RSC pixel hashes; independent parity
against the current published processor was pixel-exact for all three maps. The competition formula
rejects scientific rows, and scientific evaluation rejects LPIPS.

**Interpretation boundary.** The corrected pipeline can test whether the hosted render and reference
induce similar *derived evaluation views*. It **cannot** prove recovery of the true tissue scatterer
distribution, quantitative optical attenuation, or biological microstructure — those claims require
independently calibrated raw signals and independent property measurements.

---

## 4. Final Evidence Map

Only evidence used by the camera-ready manuscript is part of the release:

| Claim | Scope | Reproduction entrypoint | Compact record |
|---|---|---|---|
| Fixed 0/50 phase × encoding ablation | Local model, 120 public scans | `experiments/main_ablation/run.py` | `experiments/main_ablation/ablation_results.json` |
| Earlier method vs final 200-iteration method | Local model, 120 public scans | `experiments/main_ablation/run.py` | `experiments/main_ablation/full_method_comparison.json` |
| Final method, organizer-hosted rendering | Hosted service, 120 public scans | `tools/run_hosted_api_public120.py` | `experiments/main_ablation/hosted_results.json` |
| Supplementary Tables S1--S4 | Local model, defined 14/40-scan subsets | `experiments/supplementary/` | `experiments/supplementary/results/` |

Generated per-scan CSV/JSON files, phantoms, hosted request state, and rendered
images live below ignored `outputs/`. The compact hosted record contains the
reported aggregate, while a rerun creates new service request IDs and a new
content-bound job manifest.

### 4.1 Evidence rules

- Use SEWAR MS-SSIM and real AlexNet LPIPS; never substitute a fallback MS-SSIM or `LPIPS_PROXY` in
  a competition claim.
- Generate OAC/SC/RSC maps with the organizer's Matplotlib encoding and score the raw scanner PNG,
  not a secondary rounded grayscale copy.
- Call the eight-median aggregation a `competition_formula_estimate` unless the organizer issued it.
- New hosted runs preserve request IDs, content hashes, dependency versions,
  git state, scope, and failure rows. Historical request state predating this
  release was not retained; only its reported aggregate is published.
- Include tracked and untracked source/config files in the source-tree digest; batch summaries
  retain all effective controls, scanner constants, dependency versions, and environment provenance.
