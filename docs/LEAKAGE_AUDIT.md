# Data Leakage Safeguards

Implementation requires explicit leakage safeguards.

## Audit Checklist
- Target dataset experience is isolated.
- Target test data is hidden.
- Target validation data is isolated.
- Utility predictor training excludes target dataset.
- Preprocessing and normalization do not leak target statistics.
- Similarity calculation relies only on training data.
- Cached candidate outcomes do not leak to validation/test.
- Feature selection is strictly on training sets.
