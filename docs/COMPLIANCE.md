# Compliance and validity boundary

This document states what the repository satisfies and what its evidence can
support. It is a technical assessment against the published SynthOCT baseline,
not an organizer ruling or an official score.

**Reference baseline:**
[`SynthOCTChallenge/SynthOCT_Baseline`](https://github.com/SynthOCTChallenge/SynthOCT_Baseline)
at commit `9508e27c4b0bd54e0d76172a1c85af329472f2fb`.

## Submission contract

The binding interface is the generated phantom consumed by the fixed virtual
scanner. `src/synthoct/phantom.py` validates that interface before packaging.

| Contract item | Required value | Repository behavior |
|---|---:|---|
| Rows | 300,000 | Exact count enforced |
| Columns | `X Y Z Energy(%)` | Four finite columns in this order |
| X range | −1536 to 1536 µm | Bounds enforced |
| Z range | 0 to 1536 µm | Bounds enforced |
| Energy range | 0 to 100% | Bounds enforced |
| Scan dimensions | 256 × 512 | Fixed in `ExperimentConfig` |
| Pixel pitch | 6 µm | Fixed in `ExperimentConfig` |
| Center wavelength | 1.3 µm | Fixed in `ExperimentConfig` |
| Beam radius | 10 µm | Fixed in `ExperimentConfig` |

The baseline writes four-column text at four-digit scientific precision; this
repository writes the same values at six-digit precision. The scanner parses
both. `prepare-submission` packages validated phantoms, while
`verify-submission` independently re-parses each archive member and checks its
manifest binding.

## Relationship to the baseline

The main implementation is an installable package, and `baseline_format/`
provides the familiar flat entrypoints.

| Baseline component | Repository counterpart |
|---|---|
| Editable `Part1_Generator.py` | `src/synthoct/holographic_inverse.py` and the `synthoct baseline holographic-inverse` command |
| Fixed `Part2_Scanner.exe` | External organizer binary or hosted challenge service; `src/synthoct/scanners/reference.py` is a local model for controlled comparisons only |
| Fixed `Part3_Processor.py` | `src/synthoct/features/extraction.py`, with published 40 dB behavior preserved in `organizer-compatible-v1` |
| `Orchestrator.py` | CLI generation, rendering, evaluation, and submission commands |

The package layout does not change the scanner-facing artifact. Automated tests
cover phantom validation, deterministic packaging, manifest verification,
feature-map compatibility, and CLI generation/evaluation paths.

## Evidence labels

Three evidence scopes must not be conflated:

- **Local-model evaluation** uses this repository's implementation of the
  published forward model. It supports controlled ablations and timing
  comparisons.
- **Organizer-hosted evaluation** renders through the challenge service. The
  camera-ready manuscript reports descriptive results for all 120 public scans.
- **Official or hidden-test results** can only be issued by the organizers. None
  are claimed in this repository.

The 120 public scans were also used while developing the fixed configuration.
Consequently, intervals and paired tests on that set measure within-set
consistency, not unseen-subject generalization. The data comprise 40
three-frame acquisition series rather than 120 independent subjects; inferential
analyses therefore group frames by series.

## Method-use assessment

Within the published technical task, the method performs the requested per-scan
inverse construction:

1. The supplied B-scan is the input to a fixed, deterministic algorithm.
2. RAP chooses a scanner-supported complex field; phase-pair encoding converts
   its coefficients to nonnegative scatterer energies.
3. The output is a distinct physical-parameter list for each input and is
   evaluated only after passing through the scanner.
4. Generation uses no external training data, pretrained generative weights,
   hidden-test access, or per-image hyperparameter selection.
5. Evaluation metrics are not invoked inside the generator.

High image similarity is therefore the expected result of inverting the
published scanner model, rather than a submission-format or metric-only
shortcut. Organizer-hosted rendering checks transfer beyond the local model.
This assessment does not pre-empt any organizer interpretation of the challenge
rules or intent.

## Scientific validity boundary

The strongest defensible claim is **high-fidelity, scanner-compatible synthesis
of the displayed B-scan**. The following claims are not supported:

- **Unique phase recovery.** The input is a quantized, log-compressed magnitude
  image and does not contain the acquired interferometric phase.
- **Recovery of tissue microstructure.** The phase-pair phantom is a
  mathematical encoding of scanner coefficients, not a biologically identified
  scatterer distribution. Many phantoms can produce nearly identical displays.
- **Independent physical validation by OAC, SC, and RSC.** Those views are
  deterministically derived from the same structural B-scan and are not
  independently calibrated optical-property ground truth.
- **Instrument validation.** The organizer service is a virtual scanner, not a
  physical OCT instrument.
- **Hidden-set generalization.** The reported local and hosted experiments use
  the public development set; hidden-test accuracy remains unknown.

The method intentionally uses small reflection amplitudes so that the virtual
scanner operates near its linear regime, and it represents one complex
coefficient with two sub-resolution file rows. These choices are valid under
the numerical phantom contract but should not be presented as a tissue-realistic
particle count or spatial arrangement.

## Reproducibility and repository hygiene

- [DATA.md](DATA.md) records the dataset version, license, archive checksums, and
  pairing rules.
- [METHODS.md](METHODS.md) documents the algorithm, local/hosted distinction,
  feature-map modes, and experiment history.
- [`experiments/`](../experiments/) contains the compact main-paper and
  supplementary numerical evidence.
- Raw data, scanner binaries, API credentials, generated runs, manuscript
  working files, and submission archives are intentionally excluded from Git.
