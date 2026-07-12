# MICCAI/SASHIMI submission checklist

The compiled PDF is intentionally anonymized for double-blind review and uses
the unmodified Springer LNCS class distributed with the current proceedings
author package.

Before uploading:

- Replace `Anonymous Authors` and `Anonymous Institution` only for the
  camera-ready version, unless the target portal explicitly requests author
  metadata in the manuscript.
- Confirm the final author order, affiliations, ORCIDs, corresponding author,
  funding acknowledgements, and disclosure statement.
- Confirm the portal paper ID and add it only where the target instructions
  request it.
- Recheck the target event limit: SASHIMI 2026 currently specifies eight
  content pages plus up to two reference pages and the MICCAI 2026 template.
- Keep the current conservative claims: the 0.994079 composite is a local
  organizer-compatible public-development estimate, not an official or hidden
  score; the method reconstructs a scanner-equivalent phantom, not unique
  tissue microstructure.
- If a hidden-test result becomes available, add it as a separate result and
  preserve the public-development label on the current tables.
- Add a clean anonymous code/data URL if available before the deadline.
- Run `python make_figures.py`, compile with Tectonic, and visually inspect all
  pages after any textual or figure revision.

Preliminary phantom upload:

- Submit only `outputs/preliminary_v3_200_balanced60_validated.zip`, after
  checking the signed-in portal's exact root/naming convention.
- Do not submit the older `outputs/preliminary_v3_200_balanced60.zip`; it uses a
  different 60-case selection and is retained only for provenance.
- Verify the final artifact with its manifest and reference root:

  ```bash
  synthoct verify-submission \
    --archive outputs/preliminary_v3_200_balanced60_validated.zip \
    --manifest outputs/preliminary_v3_200_balanced60_validated.manifest.json \
    --reference-root tmp/dataset/extracted/DATASET_PNG \
    --expected-count 60
  ```
