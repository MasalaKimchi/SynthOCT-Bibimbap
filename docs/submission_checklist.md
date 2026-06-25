# Submission Checklist

## Prepared Artifacts

- Code package: `synthoct_bibimbap_code_submission.zip`
- Phantom package: `synthoct_h56_phantoms.zip`; older `synthoct_h41_phantoms.zip` packages are stale.
- Manifest: `submission_manifest.csv`

## Smoke Package

Use a small limit to verify upload formatting:

```bash
synthoct prepare-submission \
  --zip 18095266.zip \
  --out outputs/submission_ready_smoke \
  --limit 3 \
  --scatterers-count 300000
```

## Full Local Phantom Package

This generates one H56 phantom for every PNG B-scan in the Zenodo archive. It is expected to be large.

```bash
synthoct prepare-submission \
  --zip 18095266.zip \
  --out outputs/submission_ready_full \
  --scatterers-count 300000
```

## Before Upload

1. Confirm the portal expects one phantom per input B-scan and accepts `.zip` with `.txt` phantoms.
2. If the portal provides hidden input scans, run `synthoct baseline final` or adapt `prepare-submission` to that input directory.
3. Prefer validating H56, H41, and H11 with the official Windows `Part2_Scanner.exe` before final upload.
4. Upload the phantom package for preliminary validation.
5. Upload or link the code package for final code/model submission.

## Current Method

Primary submission method: `H56_h41_anti_anatomy`.

Previous incumbent: `H41_final_optimized`.

Conservative backup method: `H11_low_depth_comp`, which has better OAC/depth-profile agreement on the surrogate benchmark but lower competition proxy.
