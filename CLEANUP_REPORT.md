# SynthOCT-Bibimbap — Repository Cleanup Report

**Date:** 2026-07-12
**Repository:** `https://github.com/MasalaKimchi/SynthOCT-Bibimbap.git`
**Purpose:** Prepare a concise, reproducible repository for final SynthOCT 2026
(MICCAI / SASHIMI) submission.

---

## 1. Summary

| Metric | Before | After |
|---|---:|---:|
| Working tree (with local data) | 2.7 GB | 2.7 GB* |
| `.git` directory (local) | 107 MB | 2.6 MB |
| **Fresh clone from origin** | **~107 MB+** | **3.0 MB** |
| Tracked files | 37 | 64 |
| Tracked content size | ~30 MB (inc. 15 MB `phantom.txt`) | 1.3 MB |
| Largest blob in history | 36 MB (`18095266.zip`) | 0.20 MB |
| Blobs > 1 MB in history | 5 | 0 |
| Branches (local / origin) | 2 / 1 | 1 / 1 |
| Commits on `main` | 27 | 28 (+1 cleanup commit) |

*The working tree stays 2.7 GB **on the author's disk** because the raw dataset
and generated outputs are intentionally **kept locally** — they are now excluded
from git rather than deleted, so the author loses nothing. What shrank is what
GitHub stores and what a collaborator downloads: **a fresh clone is 3.0 MB.**

**Final state:** `origin/main` @ `5cd92df` — the only branch, clean linear
history, no data blobs. Verified via the GitHub API and a test clone.

---

## 2. What the repository should — and should not — contain

**Principle applied.** A submission repository should contain everything needed
to *understand, install, and reproduce* the method, and nothing that can be
*regenerated, re-downloaded, or is purely local scratch*.

| Category | Decision | Rationale |
|---|---|---|
| Source package (`src/synthoct/`) | **Track** | The method itself. |
| Tests (`tests/`) | **Track** | Reproducibility / correctness evidence. |
| One-off tools (`tools/`) | **Track** | Audit + batch-run scripts referenced by docs. |
| Consolidated docs (`docs/`) | **Track** | Method, data, and compliance references. |
| Paper source (`paper/miccai2026/*.tex`, `.bib`, figures, `make_figures.py`) | **Track** | Manuscript is regenerable from these. |
| Packaging (`pyproject.toml`, `requirements.txt`, `environment.yml`) | **Track** | Install / environment definition. |
| One lightweight example run | **Track** | Lets a reviewer see a concrete result without running anything. |
| Raw dataset (`DATASET/`, 79 MB) | **Ignore** | Public Zenodo download; documented in `docs/DATA.md`. |
| Generated outputs (2.3 GB) | **Ignore** | Reproducible from the CLI. |
| Submission ZIPs (2 × ~200 MB) | **Ignore** | Build artifacts. |
| 15 MB `phantom.txt` | **Ignore** | Regenerated in seconds by the CLI. |
| Scanner binary (`Part2_Scanner.exe`) | **Ignore** | External, Windows-only, organizer-provided. |
| Paper `build/` (5.6 MB) | **Ignore** | LaTeX/figure output; regenerated. |
| `tmp/`, `output/`, caches, `.DS_Store`, `.egg-info` | **Ignore** | Scratch and tooling cruft. |

---

## 3. Files added to tracking (were untracked, but necessary)

These core modules and tests existed on disk but had **never been committed** —
a genuine risk of shipping a broken repo. All now tracked:

- `src/synthoct/submission.py` (600 lines) — deterministic ZIP packaging + verification
- `src/synthoct/benchmark.py` (211 lines) — batch evaluation (baseline Orchestrator path)
- `src/synthoct/evidence.py` (105 lines) — evidence-manifest builder / verifier
- `src/synthoct/provenance.py` (102 lines) — provenance capture
- `tests/test_submission.py`, `tests/test_feature_maps.py` — test coverage for the above
- `tools/audit_scientific_maps_public120.py`, `tools/run_hosted_api_public120.py`
- `paper/miccai2026/` — full manuscript source (previously entirely untracked)

---

## 4. Documentation consolidation

Eight scattered markdown files were consolidated into **three themed documents**
plus a rewritten README. Source content was preserved verbatim where it carried
identifiers, checksums, equations, or metric values.

| New file | Consolidated from |
|---|---|
| `docs/METHODS.md` | `phase_pair_method.md` + `baseline_mapping.md` + `feature_map_audit.md` + `experiments.md` |
| `docs/DATA.md` | `data_provenance.md` + `LABEL_CORRECTION.md` |
| `docs/COMPLIANCE.md` | `review/CODE_STRUCTURE_COMPLIANCE.md` + `review/METHOD_LEGITIMACY_ASSESSMENT.md` |
| `README.md` (rewritten) | Concise entry point: overview, results, layout, install, quickstart, full-submission reproduction, links to themed docs |

`paper/miccai2026/SUBMISSION_CHECKLIST.md` was **kept in place** (paper-specific,
co-located with the manuscript it describes).

**Removed** (untracked; content fully preserved in the consolidated docs, in the
saved artifacts, and in the backup bundle): the 5 old `docs/*.md` files and the
`review/` folder. Moved to system Trash (recoverable), not hard-deleted.

---

## 5. History rewrite (destructive — approved)

The `.git` directory was 107 MB because large blobs had been committed then
later removed from the tree — but they persist in history forever unless
rewritten.

**Blobs purged from all history** (`git filter-repo --strip-blobs-bigger-than 1M`):

- `18095266.zip` — 36 MB (dataset archive; added in `e76a359`, removed in `bc15126`)
- `outputs/holographic_inverse_contract300k/phantom.txt` — 15 MB
- Two further `phantom.txt` versions (~15 MB each) that were held alive only by
  stale `refs/codex/turn-diffs/checkpoints/*` refs — those 8 stray refs were
  deleted first.

**Result:** all 28 commits rewritten; `main` went `67850a1` → `5cd92df`;
0 blobs > 1 MB remain; local `.git` 107 MB → 2.6 MB; fresh clone → 3.0 MB.

**Publish:** force-pushed to `origin/main` (`67850a1...5cd92df` forced update).
The merged local branch `agent/retain-holographic-inversion` was deleted (it
never existed on origin).

> **Note on GitHub's reported size.** Immediately after a force-push, the GitHub
> API may still report the *old* size (here ~42 MB) because unreachable objects
> are only physically dropped by GitHub's periodic server-side GC. This does not
> affect clones — a fresh clone already pulls only the 3.0 MB of reachable
> history (verified).

---

## 6. Safety backup

A full mirror of the **pre-rewrite** state was bundled before any destructive
operation:

- `synthoct-backup-pre-rewrite.bundle` (88 MB) — contains original `main`
  (`67850a1`), the `agent/...` branch, and `origin/main`.

To recover the old history if ever needed:

```bash
git clone synthoct-backup-pre-rewrite.bundle recovered-repo
# or, into an existing repo:
git bundle unbundle synthoct-backup-pre-rewrite.bundle
```

The bundle is saved both in the repo root and as a session artifact.

---

## 7. Final repository structure (tracked files)

```
SynthOCT-Bibimbap/
├── README.md                     # concise entry point
├── .gitignore                    # excludes data/outputs/binaries/build/caches
├── pyproject.toml  requirements.txt  environment.yml
├── docs/
│   ├── METHODS.md                # method + baseline mapping + feature maps + experiments
│   ├── DATA.md                   # Zenodo provenance + checksums + label correction
│   └── COMPLIANCE.md             # code-structure compliance + method legitimacy
├── src/synthoct/
│   ├── __init__.py  cli.py  phantom.py  holographic_inverse.py
│   ├── benchmark.py  submission.py  evidence.py  provenance.py
│   ├── scanners/  (api.py  config.py  reference.py)
│   ├── features/  (extraction.py)
│   └── evaluation/  (maps.py  metrics.py)
├── tests/                        # 4 test modules
├── tools/                        # 2 audit / batch-run scripts
├── paper/miccai2026/             # LaTeX source, figures, make_figures.py, checklist
└── outputs/holographic_inverse_contract300k/
                                  # ONE lightweight example (maps, configs, CSVs;
                                  # phantom.txt excluded — regenerate via CLI)
```

**64 tracked files, 1.3 MB tracked content, 3.0 MB fresh clone.**

---

## 8. Verification performed

- ✅ All 24 tracked Python modules compile (`py_compile`, exit 0).
- ✅ `.gitignore` behavior confirmed with untracked probe files (data/outputs correctly ignored; example run's small files kept).
- ✅ 0 blobs > 1 MB in rewritten history.
- ✅ Force-push landed: GitHub API shows only `main` @ `5cd92df`, author `JustinNKim`.
- ✅ Fresh clone from origin = 3.0 MB, 64 files, 0 big blobs.
- ✅ Backup bundle contains pre-rewrite `main` @ `67850a1`.
