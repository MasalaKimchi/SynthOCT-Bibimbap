# Repository cleanup report

**Final cleanup:** 2026-07-13
**Purpose:** Keep the SynthOCT 2026 submission repository concise,
reproducible, and understandable to a new user.

## Final state

| Item | Result |
|---|---:|
| Local working copy before cleanup | 2.7 GB |
| Local working copy after cleanup | 282 MB |
| Canonical downloaded dataset retained locally | 79 MB |
| Final validated submission ZIP retained locally | 199 MB |
| Tracked generated runs | One lightweight example |
| Empty non-Git directories | 0 |

The downloaded dataset and final upload artifacts are intentionally ignored by
Git. A fresh clone contains the code, tests, documentation, and one small
inspectable example run. The manuscript draft remains local until review is
complete.

## Retained local artifacts

- `DATASET/` — the canonical Zenodo dataset used for reproduction.
- `outputs/preliminary_v3_200_balanced60_validated.zip` — the only submission
  archive to upload.
- `outputs/preliminary_v3_200_balanced60_validated.manifest.json` — binds the
  archive and all 60 members to their source references.
- `outputs/holographic_inverse_contract300k/` — the tracked lightweight example;
  its regenerable 15 MB `phantom.txt` is excluded.

The final archive passed `verify-submission` against `DATASET/DATASET_PNG`:

- status: `provenance_validated`
- members: 60
- rows per member: 300,000
- archive SHA-256:
  `c6fbcd0612442dfb44cb53f177b0b5d9835c8a7142df48711186128aaf021414`

## Removed

- The superseded `preliminary_v3_200_balanced60.zip`.
- The 1.8 GB extracted all-120 phantom batch used to build the archive.
- Historical v1/v3 benchmark directories and hosted API payloads after their
  conclusions were recorded in `docs/` and the local review workspace.
- The duplicate dataset extraction, upstream repository checkout, and other
  scratch content under `tmp/`.
- Stale paper PDFs/build trees under `output/` and `paper/**/build/`.
- Python bytecode, test/lint caches, package metadata, `.DS_Store` files, and
  empty directories.
- The local pre-history-rewrite bundle and downloaded challenge-brief copy;
  neither was part of the tracked repository.

## Output policy

Generated runs belong under `outputs/<run-name>/` and remain ignored by Git.
Keep an archive/manifest pair only while it is an active submission artifact.
Do not retain extracted archive copies, superseded ZIPs, full phantom batches,
or hosted payloads after their recorded conclusions are reproducible.

See `outputs/README.md` for the user-facing directory conventions. The
manuscript and its submission checklist are intentionally kept outside Git
until review is complete.

## Earlier repository-history cleanup

On 2026-07-12, large dataset/output blobs were removed from Git history, core
source files were added to tracking, and documentation was consolidated. The
manuscript was removed from tracking on 2026-07-13 pending extensive review.
That earlier operation changed what GitHub and fresh clones store; this
2026-07-13 pass additionally removed redundant ignored files from the author's
local working copy.
