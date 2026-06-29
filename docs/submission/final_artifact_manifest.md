# Final Artifact Manifest

The exact checksum manifest for the current local package is generated inside the package directory:

```text
outputs/submission_ready_learned_prior_sparse_p140_t32_full_corrected/artifact_manifest.md
```

That output-side file is written after both submission zips are finalized, so it can safely contain stable SHA-256 hashes for:

- `synthoct_bibimbap_code_submission.zip`;
- `synthoct_learned-prior-sparse-p140-t32_phantoms.zip`;
- `submission_manifest.csv`;
- `submission_validation.csv`;
- `submission_readiness_report.json`, when evidence-backed packaging is used.

Do not store exact zip hashes in this repository doc: the code zip includes repository docs, so embedding a code-zip checksum here would make the checksum stale whenever this file is packaged.

The current local full package should still report:

```text
method = learned-prior-sparse-p140-t32
manifest_rows = 120
validation_rows = 120
row_count_failures = 0
readiness_status = ready
hidden_holdout_final_score = false
```

## Code-Zip Smoke Test

After creating the package, verify the source zip independently from a temporary extraction directory:

```bash
PYTHONPATH=/tmp/synthoct_codezip_verify/extracted/src \
python -m synthoct.cli baseline learned-prior \
  --input /tmp/synthoct_codezip_verify/data/reference.png \
  --artifact /tmp/synthoct_codezip_verify/extracted/artifacts/goal_h_candidates_prior.npz \
  --out /tmp/synthoct_codezip_verify/out/generated_phantom.txt \
  --scatterers-count 4096
```

The smoke output should be a finite four-column scatterer table with the requested row count, `X` within `[-1536, 1536]`, `Z` within `[0, 1536]`, and `Energy` within `[0, 100]`.

This remains local fair-evidence readiness, not official hidden-holdout victory.
