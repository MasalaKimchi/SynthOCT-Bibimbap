# SynthOCT-Bibimbap — Compliance & Legitimacy

> This document consolidates the two former internal review docs —
> `CODE_STRUCTURE_COMPLIANCE.md` and `METHOD_LEGITIMACY_ASSESSMENT.md` — into a
> single reference. Both were honest, pre-submission static reviews of the
> repository against the official SynthOCT 2026 baseline and challenge brief;
> they are published here for transparency. Score figures are the authors'
> retained **local** estimates, explicitly non-official. The definitive
> signal — the organizer-run hidden-test score — was not available at review
> time and is not claimed here.

**Reference baseline:** [`SynthOCTChallenge/SynthOCT_Baseline`](https://github.com/SynthOCTChallenge/SynthOCT_Baseline) @ `9508e27c4b0bd54e0d76172a1c85af329472f2fb`

**Summary of the two verdicts:**

- **Code structure — Compliant** on the hard submission contract; two structural gaps were closed in place.
- **Method legitimacy — Legitimate.** Solves the challenge's defined per-B-scan inverse problem with a fixed algorithm, permitted data only, and honest score labels. One scientific caveat (phantom realism / map non-independence), which is inherent to the challenge design, not prohibited, and already disclosed by the authors.

---

# Part A — Code-Structure Compliance

**Scope:** whether the code is correctly structured against the official baseline's submission contract as the author prepares to submit.

**Verdict: Compliant on the hard contract; two structural gaps closed in place.** The phantom format, scanner parameters, and evaluator behavior match the baseline exactly. The package deviates from the baseline's flat-script *shape* (which is allowed), but before this review it lacked (1) a committed command to regenerate a full phantom set for the hidden-test phase and (2) a document mapping the package to the baseline's `Part1/Part2/Part3/Orchestrator` contract. Both were added without changing any existing behavior.

## A.1 What the baseline requires

The official baseline is four flat scripts with a deliberately narrow contract:

| File | Type | Role |
| :--- | :--- | :--- |
| `Part1_Generator.py` | **Editable** | Generate `(X, Y, Z, Energy%)` scatterers, write a `.txt` phantom. "The component you aim to improve." |
| `Part2_Scanner.exe` | **Fixed** | Coherent virtual scanner (external Windows binary). Phantom → raw B-scan. |
| `Part3_Processor.py` | **Fixed** | OAC / SC / RSC parametric maps, 40 dB linearization. |
| `Orchestrator.py` | **Manager** | `import Part1_Generator as Generator`; drives generate → scan → maps → metrics. |

The binding numeric contract for a submission is the **phantom file**: 300,000 rows of `X Y Z Energy(%)`, with `X ∈ [−1536, 1536] µm`, `Z ∈ [0, 1536] µm`, `Energy ∈ [0, 100] %`, from `ExperimentConfig` (256×512, 6 µm/px, 1.3 µm, 20 µm beam diameter).

Because participants are explicitly invited to *replace* `Part1`, and because the actual task (per-B-scan inversion) differs from the baseline's from-scratch self-consistency demo, there is **no rigid requirement to preserve the flat-script filenames**. The hard requirement is the phantom-format contract and the fixed downstream processing.

## A.2 Hard-contract compliance — PASS (verified empirically)

| Contract element | Baseline | SynthOCT-Bibimbap | Status |
| :--- | :--- | :--- | :--- |
| Rows per phantom | 300000 | `validate_phantom` enforces exactly 300000 | ✅ |
| Columns | `X Y Z Energy(%)` | 4 finite columns, same order | ✅ |
| X bounds | ±1536 µm | enforced; generated files pinned at ±1536 | ✅ |
| Z bounds | [0, 1536] µm | enforced | ✅ |
| Energy bounds | [0, 100] % | enforced | ✅ |
| Scanner params | 256×512, 6 µm, 1.3 µm, beam radius 10 µm | identical in `phantom.py::ExperimentConfig` | ✅ |
| Numeric format | `np.savetxt(fmt='%.4e')` | `%.6e` (higher precision; scanner parses either) | ✅ |
| Map linearization | `10**(img*4.0)` (40 dB), fixed `Part3` | `load_and_linearize_image(dynamic_range_db=40.0)` → `10**(img*4.0)`, byte-for-byte in `organizer-compatible-v1` mode | ✅ |
| OAC / SC / RSC | fixed `Part3` formulas | reproduced in `features/extraction.py`, organizer mode | ✅ |

**Empirical check.** A generated phantom is exactly 300000 × 4, X pinned at ±1536, energies within bounds. `DATASET/DATASET_PNG` is laid out `Sex/Age/Site/` exactly as `submission.py` expects, and a validated balanced-60 submission ZIP + manifest already pass `verify-submission` at `provenance_validated`.

The one intentional deviation — the 40 dB vs. 51 dB dynamic-range question — is handled correctly: the published 40 dB behavior is frozen for all competition-formula estimates (`organizer-compatible-v1`), and the documented 51 dB calibration is isolated in a non-scoring `scientific-v1` audit mode. This resolves a contradiction in the *baseline*, and is not a defect in this repo.

## A.3 Structural deviation — installable package (legitimate)

The repository is an installable Python package rather than four flat scripts:

```
src/synthoct/
  phantom.py             scanner contract + phantom I/O
  holographic_inverse.py generation (baseline Part1)
  scanners/reference.py  source-equivalent local scanner (dev/validation)
  scanners/api.py        hosted true scanner client (baseline Part2 path)
  features/extraction.py OAC/SC/RSC maps (baseline Part3)
  evaluation/            MS-SSIM / LPIPS metrics + competition formula
  benchmark.py           batch evaluation (baseline Orchestrator path)
  submission.py          deterministic ZIP packaging + verification
  cli.py                 `synthoct` command surface
```

This is a legitimate and arguably better structure: testable (27 passing tests), pip-installable, and it keeps the scanner contract in one validated place. The challenge does not require the flat-script layout. Two things were missing for a baseline-oriented reviewer, both now closed:

- **Gap A — no retained batch-generation command.** `benchmark-local` generated each phantom into a temporary directory and discarded it; `prepare-submission` required a pre-populated `--phantom-dir`. The 120 phantoms were produced by an *uncommitted* loop, so there was no committed, reproducible command to generate 60 phantoms for the hidden-test phase.
- **Gap B — no baseline mapping document.** Nothing connected the package modules to the baseline's `Part1/Part2/Part3/Orchestrator` contract or stated the single reproduction command.

## A.4 Fixes applied in place (additive, non-destructive)

All changes are additive; no existing function signature, default, or output changed, and the full pre-existing test suite still passes.

1. **`synthoct generate-batch`** (`cli.py`) + **`generate_phantom_batch()`** (`submission.py`): inverts every reference B-scan below `--reference-root` into `--out-dir` under the flattened `sex_age_site_name.txt` name the ZIP builder expects. Each file is re-parsed against the scanner contract via `audit_phantom_file`, and a fail-closed `generation_summary.json` records per-file SHA-256, row/column counts, bounds, fixed generation parameters, and environment/git provenance. Reuses the fixed `baseline holographic-inverse` defaults (axial 0.02, lateral 0.05, 200 iterations, momentum 1.0, dispersion-canceling pair).
2. **`docs/baseline_mapping.md`** — component- and parameter-level correspondence to the baseline, the 40 dB note, and the exact `generate-batch → prepare-submission → verify-submission` sequence.
3. **`README.md`** — a "Relationship to the official baseline" section plus a `generate-batch` usage block.
4. **Tests** (`tests/test_submission.py`) — `test_generate_batch_retains_named_phantoms_and_feeds_submission` and `test_generate_batch_limit_must_be_positive`.

**Verification performed:** `python -m pytest -q` → **27 passed** (25 pre-existing + 2 new); `ruff check` on touched files → clean; end-to-end on real dataset scans, `generate-batch → prepare-submission → verify-submission` returns **`provenance_validated`** with every phantom at exactly 300000 rows.

## A.5 Action items for the author (git hygiene / portal — not code)

These require the author's decision and were **not** committed automatically:

1. **Several core modules are currently untracked in git** (`submission.py`, `benchmark.py`, `evidence.py`, `provenance.py`, the `docs/` folder, `tests/test_submission.py`, `tests/test_feature_maps.py`, `tools/`). Run `git add` on the intended files, review the diff, and confirm the untracked state is not an accidental `.gitignore` over-match.
2. **Confirm the portal's exact phantom naming/root convention** before upload; `prepare-submission` emits root-flat `sex_age_site_name.txt`.
3. **`Part2_Scanner.exe` is intentionally absent** (external, Windows-only, git-ignored) — correct; do not commit it. The hidden-test render must go through the hosted API (`synthoct scan`) or an organizer-run scanner, since the local `reference.py` scanner is source-equivalent for development only.

**Bottom line:** the code is structurally sound and now maps cleanly onto the baseline contract with a reproducible full-submission path. Remaining items are git bookkeeping and portal confirmation, not code defects.

---

# Part B — Method-Legitimacy Assessment

**Question addressed:** Does the phase-pair holographic inversion method genuinely *synthesize* digital phantoms for the SynthOCT 2026 task, or does it *game* the challenge (exploit the scoring, the scanner, or the evaluation pipeline in a way that would not survive the organizers' intent or hidden test)? This is a code-and-design review, not an organizer-issued score.

**Verdict: LEGITIMATE.** The method solves the challenge's literally-defined problem (per–B-scan inverse reconstruction of a scatterer distribution) with a fixed, non-per-image algorithm, using only permitted data, and it reports scores with honest scope labels. It is **not** a metric hack, **not** a memorization/leakage attack, and **not** a submission-format exploit.

There is **one legitimate scientific caveat** — the phantom is a *mathematical holographic encoding* rather than a *biologically plausible tissue model*, and the four evaluation maps are not mutually independent — but this caveat is **inherent to how the organizers designed the challenge**, is **not prohibited by any rule**, and is **already disclosed** by the authors. It affects the paper's scientific-realism narrative, not the submission's validity (see §B.5).

| Dimension | Finding |
|---|---|
| Does the task the code performs match the task the brief defines? | **Yes** — per-case inverse reconstruction, verbatim. |
| Fixed method or per-image tuning / test-set fitting? | **Fixed** global parameters; generalizes across all 120. |
| Only permitted data / weights? | **Yes** — no external data, no pretrained weights, no hidden-test access. |
| Direct exploitation of MS-SSIM / LPIPS? | **No** — it matches the *image*, which is what those metrics reward. |
| Memorization / copying of references? | **No** in the gaming sense (see §B.4, 4.4/4.7). |
| Honest reporting of scores? | **Yes** — everything labeled non-official / local estimate. |
| Scientific realism of the phantom | **Weak, and disclosed** — the one caveat, §B.5. |

## B.1 What the challenge asks for

Grounded in the challenge brief (`314-SynthOCT_2026_...pdf`):

- **The task is an inverse problem, per scan.** Generate digital phantoms that, when virtually scanned, match the target real scans statistically, structurally, and physically (p. 1/6); "solving the inverse problem per case" (p. 17); "mapping the underlying scatterer distributions ... within target real OCT scans" (p. 24).
- **A wide range of methods is explicitly sanctioned** — structural masks seeded with scatterers, dense masks with parameter optimization, or end-to-end diffusion models (p. 1/6). A physics-based analytic inversion sits squarely inside this permitted space.
- **There is no scatterer-level ground truth — by the organizers' own admission.** OCT resolution cannot resolve individual scatterers, so "establishing a direct ground truth at the single-scatterer level is not feasible. Instead, validation relies on indirect parametric maps" (p. 17).
- **Scoring is pure synthetic-vs-real image agreement:** the arithmetic mean of median MS-SSIMs and inverted LPIPS (1 − LPIPS) across four categories — Structural Intensity, OAC, SC, RSC (p. 19). Qualifying thresholds: Struct > 0.3, OAC > 0.4, SC > 0.4, RSC > 0.5, LPIPS < 0.4 (p. 3/7). Winners are top-3 by SSIM on the hidden 60-scan test set. **No term rewards phantom realism.**
- **Data is fixed and leakage is organizer-controlled:** 120 public B-scans (Zenodo 18095266); 60 held-out test scans withheld until final validation.

**Consequence:** because the scored objective *is* synthetic-real image convergence and the task *is* per-case inversion, a method that inverts the scanner to reproduce the reference image is doing the defined task. The bar for "gaming" is therefore narrow: exploiting the *metric* rather than the image, the *scanner implementation* rather than its physics, disallowed data/test access, or the *submission/eval pipeline*. Each is checked in §B.4.

## B.2 What the method does

From `holographic_inverse.py`, `scanners/reference.py`, and `docs/phase_pair_method.md`:

1. **Forward model.** The fixed scanner is a coherent linear field operator — an axial operator `A` (windowed complex exponential in wavenumber) and a lateral Gaussian-beam operator `L` — followed by magnitude, log-compression, and clipping: `image = clip(20·log10|A · C · Lᵀ| ...)`, faithfully reproduced in `render_reference_array`.
2. **Inversion.** Given a reference B-scan, it converts to a target field magnitude and runs momentum-accelerated Gerchberg–Saxton phase retrieval (200 iterations, β = 1) using Tikhonov pseudo-inverses `A⁺`, `L⁺` (α_z = 0.02, α_x = 0.05), yielding a complex coefficient `C` at every 256×512 voxel.
3. **Encoding.** Each complex coefficient is realized as two non-negative scatterers half a wavelength apart (a "dispersion-canceling pair") whose weighted mean depth offset is zero. This uses 262,144 active rows; the remaining 37,856 rows are zero-energy fillers, giving exactly the 300,000-row contract.

**Verified against retained diagnostics** (`diagnostics.json`): `active_scatterers = 262144`, `zero_energy_fillers = 37856` (sum = 300,000); `projected_magnitude_mae = 3.69e-4` (the linear inversion reproduces the target field magnitude almost exactly); peak amplitude = √(energy/100) = `1.0e-3`, exactly the `max_reflection_amplitude = 0.001` cap. The method is deterministic (`seed=7` only affects zero-energy filler positions), the code matches the docs, no reference image is stored in the phantom, and no evaluation code is invoked inside the generator.

## B.3 Why the scores are so high (mechanistic explanation)

The near-perfect MS-SSIM (~0.995) is expected behavior of an accurate operator inverse:

> **The method drives the scanner into its linear regime and then inverts the linear operator analytically.**

The real forward model is coherent-linear **plus a nonlinear Beer–Lambert transmission term** (each scatterer attenuates everything below it by `1 − (beam·amplitude)²`). By capping every reflection amplitude at 0.001, each attenuation factor `≈ 1 − 1e-6 ≈ 1`, so the transmission cascade is effectively the identity and the scanner reduces to a purely linear map `magnitude(A · C · Lᵀ)`. A well-conditioned Tikhonov inverse then inverts it to near-machine precision — hence `projected_magnitude_mae ≈ 4e-4` and MS-SSIM ≈ 0.995. This is a valid exploitation of the *physics the organizers specified*, not of a bug or of the metric. (The physical unusualness of this operating point is the caveat in §B.5.)

## B.4 Challenge-gaming failure modes — audit results

| # | Failure mode | Result |
|---|---|---|
| 4.1 | **Metric exploitation** (fool MS-SSIM/LPIPS without matching the image) | **PASS** — matches the rendered image magnitude (MAE 4e-4); metrics score high *because* the image matches. No adversarial LPIPS pattern, no MS-SSIM blur trick. |
| 4.2 | **Derived-map shortcut** (optimize OAC/SC/RSC directly, decoupled from the scan) | **PASS** — maps are computed deterministically from the rendered intensity image, so matching the image ⇒ matching all four maps as an arithmetic consequence (with the §B.5 caveat that this makes the maps non-independent — a property of the *challenge*). |
| 4.3 | **Scanner-implementation exploit** (target a quirk of the local renderer the true scanner lacks) | **PASS** — raw hosted-scanner PNGs agree with local source-equivalent renders at MS-SSIM ≈ 0.9997; the gain is confirmed on the hosted true scanner, not only locally. |
| 4.4 | **Ground-truth-at-inference leakage** (phantom smuggles reference pixels into the render) | **PASS** — the phantom is a physical scatterer list scored only through the scanner; energies are physical amplitudes, not encoded pixels. |
| 4.5 | **Test-set / hidden-set fitting** | **PASS** — fixed global parameters, not selected per public image; the 60-scan test set is organizer-held. The ledger explicitly **rejected** "public oracle rescue layers" as "selection-leaky for hidden evaluation" (`docs/experiments.md`). |
| 4.6 | **Disallowed data / pretrained weights** | **PASS** — only the 120 public Zenodo scans; no external datasets, no pretrained networks (LPIPS AlexNet used for evaluation only, as organizers specify). |
| 4.7 | **Degenerate / mode-collapsed output** | **PASS** — a full 262,144-scatterer field derived per reference; each case yields a distinct phantom. A shifted-pairing negative control scored "substantially worse" (`docs/feature_map_audit.md`), confirming case specificity. |
| 4.8 | **Submission-format / pipeline exploit** | **PASS** — obeys the exact `300000 × 4` contract and bounds (`phantom.py::validate_phantom`); `submission.py` build+verify re-parses every member and binds to a hash manifest. |
| 4.9 | **Dishonest score reporting** | **PASS** — every aggregate labeled a "local competition-formula estimate," explicitly "not an organizer-issued leaderboard or hidden-test score." A prior over-strong `official_score` label was found and corrected down to its true n=1 scope. |

**No failure mode is triggered.** Several checks surfaced *positive* integrity signals: the negative-control pairing test (4.7), the hosted-scanner confirmation (4.3), the explicit rejection of leaky oracle layers (4.5), and the self-corrected score label (4.9).

**Two nuances stated plainly:**

- *Is inverting the scanner from the reference a form of copying?* No — the task *is* to reconstruct a phantom **from** the reference scan (an inverse problem). Conditioning on the reference is the input, not leakage; each phantom is derived analytically from *its own* input scan with no cross-case memory. The 60 hidden scans will each be inverted the same way.
- *The improvement is shown mostly locally.* The all-120 comparison and ablations are source-equivalent local results (labeled as such). The key transfer claim is backed by hosted-API renders (request IDs retained) agreeing with local renders at MS-SSIM ≈ 0.9997. The final hidden-test score remains unknown until the organizers run it.

## B.5 The one real caveat — map non-independence & phantom realism

This is the only substantive concern, and it is **scientific, not compliance-related**:

1. **The four evaluation maps are not independent.** OAC, SC, and RSC are all computed deterministically from the rendered intensity image (`evaluation/maps.py`, `features/extraction.py`). The authors state directly that the reference maps are "not an independent parametric ground-truth dataset" (`docs/feature_map_audit.md`). So "matched on all four maps" is closer to "the rendered image matches, applied four ways" than to four independent physical validations. This is a property of the challenge's own evaluation design (no scatterer ground truth exists, p. 17), not a manipulation — but the paper should not over-claim "physically validated across four independent axes."
2. **The phantom is a holographic encoding, not a tissue model.** Amplitudes are capped ~6 orders of magnitude below the allowed maximum to keep the scanner linear (§B.3), so the scatterers are near-transparent and arranged in sub-resolution dispersion-cancelling pairs — a construction chosen for *operator invertibility*, not biological plausibility. Real skin has strongly attenuating, multiply-scattering structure. The method reproduces the *scan* faithfully but does not claim to recover the *true* scatterer distribution. The authors already draw this boundary: the pipeline "cannot prove recovery of the true tissue scatterer distribution ... or biological microstructure" (`docs/feature_map_audit.md`), and the README flags that organizer clarification on phantom realism "would further reduce challenge-spirit ambiguity."

**Why this does not change the verdict:** the challenge scores synthetic-vs-real image convergence with no realism term, explicitly permits a wide range of phantom-construction methods, and concedes single-scatterer ground truth is unmeasurable. A submission cannot be disqualified for producing a non-biological phantom under the published rules. The honest move — already partly taken in the repo — is to frame the contribution as **"an accurate physics-based inverse-rendering solution to the challenge's scored objective,"** not **"recovery of true tissue microstructure."** The residual risk is a *spirit-of-the-challenge* judgment reserved to the organizers, and the repo already recommends seeking their clarification.

## B.6 What could and could not be tested here

| Tested empirically (from retained artifacts / code) | Not testable in this review |
|---|---|
| Phantom contract compliance (300k×4, bounds, finite) | The **official hidden-test score** (organizer-run on 60 withheld scans) |
| Inversion accuracy (`projected_magnitude_mae ≈ 4e-4`) | Independent re-rendering on the organizers' **exact Windows scanner binary** (only hosted-API evidence retained) |
| Amplitude/energy operating point (linearization argument) | Whether organizers will exercise **spirit-of-challenge** discretion on realism |
| Map derivation from intensity (non-independence) | Live re-execution of the full 120-case benchmark (relied on retained CSVs/JSON) |
| Fixed-method / no-per-image-tuning claim; score-honesty labeling | |

To convert the right column into evidence: (a) submit to synthoct.com for an official leaderboard
number, and (b) retain the resulting hosted-request manifest alongside the submission record. Large
historical API run directories were removed from the final repository because they are generated
artifacts; request IDs and aggregate public-development results remain documented.

## B.7 Recommendation

**Proceed to submission and paper.** The method is a legitimate, well-engineered, honestly-reported solution to the SynthOCT 2026 task as defined and scored. The single required change is one of **framing, not method**: ensure the manuscript claims *inverse-rendering fidelity to the scored objective* and explicitly states the map-non-independence and phantom-realism boundary (§B.5) rather than implying recovery of true tissue microstructure. No changes to the generation method are needed on legitimacy grounds.

---

*Both assessments were produced by static review of the repository and challenge brief. Score figures cited are the authors' retained local estimates, explicitly non-official. The definitive legitimacy signal — the organizer-run hidden-test score — was not available at review time and is not claimed here.*
