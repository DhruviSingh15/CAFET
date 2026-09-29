# CAFET: Context-Aware Feature Experience Transfer

## Project Objective
An end-to-end research prototype to investigate whether historical feature-engineering experience from previously processed datasets can be transferred to a new unseen dataset using dataset context.

## Research Questions
Can context-aware cross-dataset feature experience transfer guide automated feature engineering toward useful feature transformations while reducing the number of expensive candidate evaluations required to reach near-exhaustive predictive performance?

## Phase Status
- **Phase 1 Complete**: Research documentation established in `docs/`.
- **Phase 2 Complete**: Clean AutoFE Baseline implemented and validated.
- **Phase 3 Complete**: Dataset Intelligence / Profiler implemented and validated.
- **Phase 4 Complete**: Experience Repository (SQLite) implemented and populated.
- **Phase 5 Complete**: Dataset Similarity Engine implemented and validated.
- **Phase 6 Complete**: Candidate-Level Representation (61-dim vector) implemented and validated.
- **Phase 7 NEEDS FIX**: Utility Predictor implemented; weak Spearman signal but P@K=0 for top candidates. Root cause: 3-dataset LODO pool too small; history features dominate. Fix required before Phase 8.
- **Phase 7.1 NEEDS FIX**: Similarity-weighted experience correction implemented. Model D (SimOnly) improves Spearman on 2/3 targets and reduces history dominance (44–67% vs 81–94%). P@10=0 persists on 2/3 targets. Fundamental bottleneck: 3-dataset pool too small for reliable candidate-level generalisation.
- **Phase 7.2 Complete**: Real-World Dataset Pool Expansion. 7 new datasets added (total 10). LODO feasibility confirmed (9 historical datasets per target). Ready for Phase 7.1 re-run.
- **Phase 7.3 FAIL**: Expanded-Pool Utility Prediction. Tested similarity-weighted predictions across the 10-dataset pool. Performance remained extremely poor (e.g. median Spearman near zero, P@10=0 on most datasets). Demonstrated that macro-level dataset context (meta-features) + similarity weighting does *not* provide generalizable candidate-level predictive signal.
- **Phase 7.4 FAIL**: Candidate Micro-Context Validation. Extracted detailed statistical properties (54 dims) of the source features and transformed output. While micro-context reduced the model's under-dispersion, prediction signal remained no better than random guessing across the 10 datasets.

## Current Roadblock / Next Steps
The core mechanism of CAFET (cross-dataset candidate-level utility prediction) has failed two critical validation phases:
1. **Macro-level failure (Phase 7.3):** Dataset-level summary statistics (e.g., % missing, % categorical) are insufficient to map historical transformation utility to a new dataset.
2. **Micro-level failure (Phase 7.4):** Local statistical properties of the specific source features (e.g., source standard deviations, pairwise correlations, output skewness) also failed to yield generalizable predictions for candidate ranking.

**Conclusion so far:** A transformation's utility (e.g., `DIVIDE(featureA, featureB)`) relies deeply on the joint distributions and relationship with the target variable *inside the specific dataset*. Attempting to predict this utility on an unseen dataset—without executing the candidate and observing the target—performs no better than random guessing.

**Action Required on Return:**
Before proceeding to candidate ranking or budgeted search (Phase 8), the hypothesis for utility prediction must be fundamentally reconsidered or pivoted. The current purely unsupervised representations (macro and micro context) do not carry enough signal to predict target-dependent utility.
