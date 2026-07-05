# ML/DL Expert Review

This compact review summarizes the ML/DL conclusion. See [current_findings.md](current_findings.md) for current evidence and [next_approach.md](next_approach.md) for the execution plan.

## Verdict

The high-ceiling direction is not full neural density replacement. It is topology-preserving residual control around `learned-prior-sparse-p140-t32`.

## Evidence

- Standalone neural prior reduced Structural LPIPS on a small true-scanner split, but official score and OAC/SC/RSC maps regressed.
- Energy-only neural blends lowered Structural LPIPS slightly while preserving density topology, but broader grouped summaries still favored `p140-t32`.
- Learned surrogate inverse optimization produced strong local calibration but failed under hosted true-scanner rendering.
- Anchored surrogate residuals reduced extreme failures but still did not beat the base.

## Recommendation

Build ML/DL components as constrained residual predictors:

- predict low-frequency geometry and flow controls;
- predict attenuation-normalized energy controls;
- predict speckle texture statistics;
- estimate uncertainty for map-safety gating;
- train only from true-scanner teacher rows or clearly labeled surrogate rows with calibration.

## Guardrails

- Do not promote from preview or surrogate metrics.
- Do not replace the scatterer topology without grouped true-scanner proof.
- Keep `surrogate_scanner_is_true_scanner=false`.
- Require real LPIPS and Struct/OAC/SC/RSC metrics before promotion.
