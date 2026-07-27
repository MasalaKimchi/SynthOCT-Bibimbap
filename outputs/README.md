# Generated outputs

This directory is the single home for generated runs. Use a descriptive,
disposable subdirectory such as `outputs/candidate/` or
`outputs/hidden_test/`; do not write generated files into the repository root.

## What is versioned

`holographic_inverse_contract300k/` is the one lightweight example run kept in
Git. It shows the expected diagnostics, render, maps, metrics, configuration,
and provenance layout. Its 15 MB `phantom.txt` is omitted because the CLI can
regenerate it.

Everything else under `outputs/` is ignored by Git. A clean clone therefore
contains no exploratory runs, API payloads, or submission archives.

## Local final-submission files

The author's final working copy may contain exactly this ignored pair:

- `preliminary_v3_200_balanced60_validated.zip`
- `preliminary_v3_200_balanced60_validated.manifest.json`

The manifest binds all 60 archive members to their reference scans and records
the archive checksum. Verify the pair immediately before upload:

```bash
PYTHONPATH=src python -m synthoct.cli verify-submission \
  --archive outputs/preliminary_v3_200_balanced60_validated.zip \
  --manifest outputs/preliminary_v3_200_balanced60_validated.manifest.json \
  --reference-root DATASET/DATASET_PNG \
  --expected-count 60
```

Do not keep superseded ZIPs, extracted copies of an archive, full phantom batch
directories, hosted API payloads, or benchmark scratch after their conclusions
have been recorded in `docs/` or the local review workspace.
