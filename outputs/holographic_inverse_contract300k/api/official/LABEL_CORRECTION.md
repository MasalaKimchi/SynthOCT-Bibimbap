# Historical label correction

This directory predates the 2026-07-10 evidence terminology audit. Its
`official_score` column is a locally calculated, single-reference formula value
from a hosted scanner image; it is **not** an organizer-issued leaderboard or
hidden-test score. It also used a secondary grayscale copy.

The corrected organizer-compatible evaluation of the raw hosted PNG is in
`../organizer_compatible_estimate.csv`. Current code and documentation use the
name `competition_formula_estimate` for locally calculated values.
