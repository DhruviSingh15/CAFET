# Phase 5: Dataset Similarity Engine Validation Report

## 1. Objective
Phase 5 implements the Dataset Similarity Engine, enabling a deterministic, mathematically grounded measure of contextual similarity between any two datasets. It relies entirely on the raw meta-features captured by the Phase 3 Profiler.

## 2. Similarity Architecture
Implemented in `backend/core/similarity/similarity_engine.py`, the `SimilarityEngine` handles computing similarities from 1D context vectors. A helper `Normalizer` standardizes meta-features to $\mu=0, \sigma=1$ dynamically to prevent features with vast absolute scales (e.g., `n_rows`) from dominating distances.

## 3. Context Vector
The engine utilizes the precise 14-element ordered `numeric_context_vector` defined in Phase 3:
1. `n_rows`
2. `n_cols`
3. `n_numeric`
4. `n_categorical`
5. `n_boolean`
6. `n_datetime`
7. `missing_ratio`
8. `cols_with_missing`
9. `mean_missingness_per_feature`
10. `max_feature_missingness`
11. `mean_cardinality`
12. `max_cardinality`
13. `avg_abs_correlation`
14. `max_abs_correlation`

## 4. Normalization
Normalization is critical for multi-scale meta-features. The `Normalizer` employs `nanmean` and `nanstd` to calculate $Z$-scores safely.
**LODO Protocol Constraint:** For rigorous research evaluation (LODO), the normalizer is fitted *exclusively* on the historical candidate contexts. The target context is mathematically excluded during `.fit()` to prevent data leakage.

## 5. Similarity Metric
The baseline metric relies on standard normalized Euclidean distance ($d$), smoothly transformed into a bounded similarity metric via:
$$ \text{similarity} = \frac{1}{1 + d} $$
This guarantees $0 < \text{similarity} \le 1$.

## 6. Similarity Matrix
A full diagnostic similarity matrix (where normalization is fitted on all 3 datasets globally):

|               | Breast Cancer | Wine      | Diabetes  |
| ------------- | ------------: | --------: | --------: |
| Breast Cancer |      1.000000 |  0.169435 |  0.192456 |
| Wine          |      0.169435 |  1.000000 |  0.352834 |
| Diabetes      |      0.192456 |  0.352834 |  1.000000 |

## 7. LODO Retrieval
LODO retrieval (where normalization is strictly fitted *only* on the reference pool, excluding the target) produces distinct rankings due to shifting reference scales:

**Target: breast_cancer**
1. wine — similarity 0.0451
2. diabetes — similarity 0.0440

**Target: wine**
1. diabetes — similarity 0.1904
2. breast_cancer — similarity 0.1172

**Target: diabetes**
1. wine — similarity 0.3901
2. breast_cancer — similarity 0.1978

## 8. Sanity Checks
- **Self similarity:** Euclidean distance of 0.0 transforms identically to 1.0.
- **Symmetry:** Matrix validates `sim(A,B) == sim(B,A)`.
- **Deterministic ranking:** Assured by tuple-sorting on `(-similarity, dataset_id)`.
- **Range checks:** Bounded safely in `(0, 1]`.
- **NaN checks:** Imputed pre-normalization and protected via zero-variance catching (`np.nan_to_num`).

## 9. Leakage Audit
The similarity engine strictly takes context vectors as inputs. It natively cannot access Phase 2 / Phase 4 candidate scores, validation performances, or transformations. Target outcomes remain wholly decoupled from dataset relations.

## 10. Target Isolation
During LODO, the API intentionally discards `target_dataset_id` from the reference dictionary *prior* to normalizer fitting and distance evaluation, guaranteeing pure historical boundary conditions.

## 11. Tests
Unit tests in `tests/test_phase5.py` report 6/6 passes covering determinism, symmetry, LODO exclusion, normalization variance boundaries, and mathematical identity.

## 12. Limitations
- Three datasets constitute a micro-pool. Normalization means/variances fluctuate wildly when computing LODO over a pool size of merely 2 datasets, leading to lower absolute similarities (e.g. `0.0451`) in LODO vs the global matrix. A larger repository is practically required for stable standardizations.
- Euclidean distance handles all 14 dimensions evenly. Unimportant structural features (e.g. `n_boolean` when all sets are exclusively numeric) could flatten distance resolution.

## 13. Scientific Interpretation
This experiment demonstrates that a purely meta-feature-driven mechanism can deterministically relate and rank datasets. It does **NOT** yet prove that a higher similarity score guarantees that optimal feature-engineering candidates are transferable; that hypothesis is strictly reserved for Phase 6 utility prediction modeling.

## 14. Phase Verdict
PASS
