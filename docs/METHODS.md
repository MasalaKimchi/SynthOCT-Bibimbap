# SynthOCT-Bibimbap — Methods

> This document consolidates the former `phase_pair_method.md`, `baseline_mapping.md`,
> `feature_map_audit.md`, and `experiments.md` into a single reference. It covers the core
> phase-pair holographic-inversion algorithm, its mapping onto the official challenge baseline,
> the feature-map definitions and the audit of how Struct/OAC/SC/RSC maps are computed, and the
> experiment log behind the fixed method.

**Scope note on numbers.** Scores from different scopes or metric backends are not
interchangeable. Results labeled *local* (source-equivalent scanner) or *hosted* (official cloud
API) are public/development estimates, **never** organizer-leaderboard or hidden-test results.
Some v1/v3-50 aggregates were computed with an older float32 metric loader and some with the
official float64 loader; such comparisons are flagged as mixed-precision.

---

## 1. Phase-Pair Holographic Inversion (core method)

The fixed virtual scanner is a **coherent linear field operator** followed by magnitude, log
compression, and clipping. The method recovers a 300,000-scatterer phantom whose scanner render
matches a real B-scan by choosing a *scanner-feasible complex field* and encoding each field
coefficient as a dispersion-canceling pair of sub-resolution scatterers.

### 1.1 Why v1 stalled

v1 inverted a real, zero-phase target field and encoded each complex coefficient with a single
sub-wavelength depth shift. Two effects limited it:

- The axial Hanning window leaves the 256×256 operator at **rank 254**, so forcing zero phase
  excites poorly conditioned modes and creates axial ringing.
- A single depth shift has the desired coefficient phase only at the center wavenumber; its phase
  drifts across the scanner bandwidth.

The hosted/local scanner mismatch was **not** the bottleneck: raw hosted PNGs agree with their
source-equivalent local renders at MS-SSIM `0.999676` (v1), `0.999677` (v3-50), and `0.999672`
(selected v3-200). (The older v1 value `0.999313` came from a secondary grayscale copy and is not
used.)

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

This preserves the measured magnitude while choosing a scanner-feasible phase. **Fixed settings:**
200 iterations, `beta = 1`, Tikhonov regularization `alpha_z = 0.02`, `alpha_x = 0.05`.

### 1.3 Dispersion-canceling pair

For coefficient `C = |C| exp(i·phi)`, v1 used depth offset `d0 = -phi·lambda/(4·pi)`. v3 adds a
carrier-equivalent companion half a wavelength away on the opposite side:

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

### 1.5 Measured effect (v1 → v3-200)

On the retained hosted reference, organizer-compatible metrics changed:

| Metric | v1 | v3-200 | Delta |
|---|---:|---:|---:|
| Struct MS-SSIM | 0.962757 | 0.996817 | +0.034059 |
| Struct LPIPS | 0.134742 | 0.013080 | -0.121662 |
| OAC MS-SSIM | 0.992656 | 0.999701 | +0.007045 |
| SC MS-SSIM | 0.982782 | 0.998844 | +0.016062 |
| RSC MS-SSIM | 0.983875 | 0.999193 | +0.015318 |
| Eight-term local estimate | 0.958343 | 0.996514 | +0.038170 |

The same fixed method improved all eight metric components on all four independent local strata.
The machine-readable panel is retained under
`outputs/holographic_inverse_v3_phase_pair_iter200_contract300k/four_case_panel/`; its local
eight-term estimate rose from v1 `0.955140` → v3-50 `0.993074` → v3-200 `0.993781`.

The subsequent fixed **all-public source-equivalent local** evaluation (n=120):

| Public Structural MS-SSIM (n=120) | v1 | v3-200 |
|---|---:|---:|
| Mean | 0.955942 | 0.994588 |
| Median | 0.956044 | 0.994849 |
| Minimum | 0.933871 | 0.985133 |
| Maximum | 0.967150 | 0.998175 |

v3-200 won all 120 paired comparisons against both v3-50 and v1. Versus v1, mean paired gain was
`+0.038646` (smallest `+0.029806`). Versus v3-50, mean gain was `+0.001070` (smallest `+0.000299`),
and mean remaining error to one fell by `16.51%`. Every v3-200 gain exceeds the observed
`0.000015` float32/float64 precision difference; no global correction was applied. These are
strong public/development results, **not** hidden-test proof.

### 1.6 Remaining opportunities

- Use the 37,856 filler rows as sparse second-order correction atoms at surface/high-gradient
  residuals.
- The checkpoint sweep selected a fixed global budget of **200 iterations**. At 300 iterations the
  extra local gain was only about `+0.000052` on the same exploratory metric path, and generation
  time was anomalously `61.6 s`; the 300 checkpoint was not rescored with the official-aligned loader.
- Test larger physical reflection scales with transmission compensation.
- Seek organizer clarification that sub-resolution coherent pairs and very low energies satisfy the
  intended scientific-phantom interpretation.

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
| `Part2_Scanner.exe` | Fixed coherent virtual scanner (external Windows binary); renders a phantom into a raw B-scan. | **Fixed** | Not reimplemented. Hosted scanner called via `synthoct scan` (`src/synthoct/scanners/api.py`); a source-equivalent local scanner for dev/validation only is `src/synthoct/scanners/reference.py`. The hosted/official render remains authoritative. |
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

Phantom **generation** always runs at 51 dB. Existing v3-200 phantoms do not need regeneration
because of the Part3 contradiction.

### 3.2 Folder semantics

The retained all-120 hosted run is `outputs/hosted_api_public120_v3_200/`:

- `raw/` — PNG payload returned by the hosted virtual-scanner API (some stored as color/RGBA even
  though the signal is grayscale).
- `gray/` — deterministic 8-bit grayscale conversion used for structural (Struct) evaluation. All
  120 raw→gray conversions were regenerated and matched the retained files exactly.
- `maps/reference/` — OAC, SC, RSC images computed *from each reference B-scan* by the published
  Part3 processor.
- `maps/prediction/` — the same transformations computed from the paired hosted render.

Thus `maps/reference/` is **not** an independent parametric ground-truth dataset. Unusual terminal
rows, borders, saturated regions, and contrast differences are products of the map processor.

### 3.3 All-120 integrity checks

One-to-one inventories were verified for 120 references, phantoms, API jobs, raw PNGs, grayscale
PNGs, and metric rows. Flattened names were reversed against reference-relative paths with no
collisions or cross-case pairings. All **720** retained organizer map PNGs (3 map types ×
reference/prediction × 120 cases) were regenerable from their claimed source. A deliberately
shifted-pairing negative control scored substantially worse, supporting the pairing audit.

Retained **hosted** structural results:

| Quantity (hosted, n=120) | Value |
| :--- | ---: |
| Mean MS-SSIM | 0.9942843141 |
| Median MS-SSIM | 0.9945532174 |
| Minimum MS-SSIM | 0.9857009208 |
| Median LPIPS | 0.0226107640 |
| Eight-median formula (byte-compatible path) | 0.9940789428 |

*(This hosted mean `0.99428` differs from the §1.5 source-equivalent local mean `0.994588` only by
scope — hosted cloud API vs source-equivalent local scanner; both are public-development estimates,
not a leaderboard/hidden-test score.)*

The `scientific-v1` pass on all 120 pairs (diagnostic derived-map metrics only, stored in
`audit/scientific_v1_metrics.csv` and `audit/scientific_v1_summary.json`):

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

## 4. Experiment Log & Results

Every row is *local* unless a hosted request ID is given. A `local` row is never an organizer
leaderboard result.

### 4.1 Chronological ledger

**2026-07-11 — Float64-aligned v1 comparison rerun.** Reran the exact retained v1 config
(`alpha_z=0.03`, `alpha_x=0.20`, zero-phase iterations, momentum 0, single scatterer) through the
current source-equivalent scanner/evaluator → `outputs/holographic_inverse_v1_current_public120/`.
Structural MS-SSIM: mean `0.955953650`, median `0.956020936`, minimum `0.933827016` (120 paired
references). Against unchanged v3-200 rows, mean gain `0.038634109`; all 120 images and all 40
filename-defined series improve. A 200,000-replicate series-cluster bootstrap gives 95% CI
`[0.037530, 0.039790]`; the one-sided exact Wilcoxon signed-rank p-value is `9.095e-13`. This
supersedes the legacy-precision v1 aggregate for the paper comparison; the mean shift was only
`+1.15e-5`, so the conclusion is unchanged.

**2026-07-10 — Coherent v1 baseline audit.** Hypothesis: exact inversion of the recovered coherent
scanner can exceed older statistical learned-prior methods. Method: zero-phase target, Tikhonov
`alpha_z=0.03`, `alpha_x=0.20`, one shifted scatterer per voxel, 131,072 active rows. Hosted
request `d5a0d2b8`. Corrected organizer-compatible single-case result: Struct MS-SSIM `0.962757`,
real LPIPS `0.134742`, eight-term local formula estimate `0.958343`. Lesson: the approach is real
and generalizes locally, but the retained `official_score` label was too strong and the evidence
scope was only `n=1`.

**2026-07-10 — Lateral regularization sweep.** Hypothesis: stronger lateral regularization
suppresses ringing. Tested `alpha_z ∈ {.001,.005,.015,.03}`, `alpha_x ∈ {.005,.02,.07,.20,.50}` on
the retained reference. Zero-phase single-scatterer MS-SSIM improved from `0.962712` at `.03/.20`
to `0.969989` at `.03/.50`. Lesson: operator conditioning matters, but regularization alone leaves
a large phase-representation error.

**2026-07-10 — Three-phase conic encoding.** Hypothesis: two nonnegative weights from a
0/+120/−120° basis reduce the maximum depth shift. Result: `0.962712 → 0.962854` MS-SSIM without
phase retrieval; after phase retrieval it underperformed the single-scatterer encoding.
**Rejected** — reducing offset magnitude does not cancel broadband phase slope.

**2026-07-10 — Momentum phase retrieval.** Hypothesis: OCT intensity leaves phase free, so
alternating projections can avoid the axial null space. Single-scatterer retained-reference: plain
200 iterations `0.980110`; circular momentum `beta=1` reached `0.981525` in 50 iterations. Four
independent strata: v1 mean `0.952667`; momentum-50 mean `0.977167`. Lesson: phase freedom is a
high-value optimization dimension; 50 momentum iterations capture nearly all of the 100-iteration
gain for the single-scatterer branch.

**2026-07-10 — Dispersion-canceling phase pair (initial 50-iteration promotion → v3).** Hypothesis:
two carrier-equivalent depths with zero weighted mean offset cancel first-order broadband phase
error. Fixed method: `alpha_z=0.02`, `alpha_x=0.05`, 50 momentum iterations, two rows per voxel,
max reflection amplitude `0.001`. Serialized full local retained reference: Struct MS-SSIM
`0.995988`. Four independent serialized local strata: Struct mean `0.993497`, minimum `0.990091`;
every Struct/OAC/SC/RSC MS-SSIM and real-LPIPS component improved; the retained panel's local
formula estimate rose `0.955140 → 0.993074`. Fixed all-120 local structural comparison: v1
mean/median/min `0.955942/0.956044/0.933871`; v3-50 `0.993518/0.993899/0.984114`; v3 won `120/120`,
mean paired delta `+0.037576`, minimum delta `+0.029060`. Hosted request `b166d529`: Struct MS-SSIM
`0.996075`, Struct LPIPS `0.015910`, eight-term local estimate `0.995898`. **Promoted as v3** —
hosted output confirms the local prediction rather than exposing a surrogate gap.

**2026-07-10 — Iteration-budget refinement (200 iterations promoted).** Hypothesis: the phase-pair
encoder keeps benefiting from phase retrieval past the 50-iteration single-scatterer plateau. On the
same exploratory metric path, retained-reference checkpoints were ≈ `50: 0.995988`, `100: 0.996549`,
`150: 0.996756`, `200: 0.996841`, `300: 0.996894`; the selected 200 checkpoint rescored at
`0.996854` under the official-aligned loader (300 not rescored). **Decision:** fixed global budget of
200 — the same-path exploratory gain at 300 was only ≈ `+0.000052` while its generation time was
anomalously `61.6 s`. All-120 official-real-reference local: mean/median/min/max Structural MS-SSIM
`0.994588/0.994849/0.985133/0.998175`; v3-200 beat v3-50 and v1 on `120/120`, mean delta vs v3-50
`+0.001070`, cutting mean remaining error by `16.51%`. Four-case real-LPIPS/map panel: all eight
metrics improved on every case vs both v3-50 and v1; local formula estimate rose v3-50 `0.993074` →
v3-200 `0.993781`. Hosted request `69bc223f`: Struct MS-SSIM `0.996817`, Struct LPIPS `0.013080`,
eight-term local estimate `0.996514` (`+0.000615` over the v3-50 hosted result). Rejected outer
target correction: `eta=1` degraded; `eta=0.1` improved three of four cases but regressed one by
`0.000021` — no robust selection rule, not promoted. Precision note: this run uses the official
float64 `img_as_float` metric path; retained v1/v3-50 CSVs use the older, slightly conservative
float32 loader, so comparisons are labeled mixed precision with no blanket correction.

### 4.2 Prior families not to repeat (without a new rationale)

- H0–H70 density/depth/OAC/layer scalar sweeps and invalid offline-render ranks.
- P06–P09 visual recipes; P120/P140/P160/t32/sigma one-knob variants.
- Neural density blends and e05–e60 energy blends.
- Texture smoothing, coordinate jitter, scatterer injection, y-beam reshaping,
  gamma/high-frequency corrections.
- Learned-surrogate inversion, anchored surrogate, and direct lattice: preview/surrogate gains did
  not transfer to the true scanner.
- Public positive-only flow/energy patching as a hidden-general solution.

Historical best before coherent inversion was the full-public **P140-t32** family: official-style
public aggregate ≈ `0.68479`; public oracle rescue layers reached ≈ `0.693998` but were
selection-leaky for hidden evaluation.

The regularization, conic-encoding, and early-momentum ablations are session-recorded lessons; their
raw scratch outputs were not retained. V1/v3 hosted evidence, the all-120 comparison, and the
four-case real-LPIPS panel do have machine-readable CSV/JSON artifacts under `outputs/`.

### 4.3 Evidence rules

- Use SEWAR MS-SSIM and real AlexNet LPIPS; never substitute a fallback MS-SSIM or `LPIPS_PROXY` in
  a competition claim.
- Generate OAC/SC/RSC maps with the organizer's Matplotlib encoding and score the raw scanner PNG,
  not a secondary rounded grayscale copy.
- Call the eight-median aggregation a `competition_formula_estimate` unless the organizer issued it.
- Preserve request IDs, content hashes, dependency versions, git state, scope, and failure rows for
  every hosted experiment.
- Include tracked and untracked source/config files in the source-tree digest; batch summaries
  retain all effective controls, scanner constants, dependency versions, and environment provenance.
