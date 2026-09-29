# Phase 7: Candidate-Level Utility Predictor — Validation Report

## 1. Objective
Phase 7 tests whether a learned model can predict candidate-level feature-engineering utility on an unseen dataset, using only experience from other datasets. The strict LODO protocol prevents any target-dataset leakage.

---

## 2. Predictor Architecture
**Model**: RandomForestRegressor (scikit-learn)

| Hyperparameter | Value |
|---|---|
| n_estimators | 200 |
| max_depth | None (full trees) |
| min_samples_leaf | 2 |
| random_state | 42 |

**Justification**: Chosen for interpretability (feature importance), ability to handle mixed feature types, and native support for predictive uncertainty via tree ensemble variance.

**Uncertainty**: `std` of individual tree predictions. This is a predictive spread estimate, NOT a calibrated probability.

---

## 3. Utility Definition
```
utility = candidate_validation_score - baseline_validation_score
```
Test scores are **never** used as training targets or input features.

---

## 4. LODO Protocol

| Experiment | Train | Test |
|---|---|---|
| A | breast_cancer + wine | diabetes |
| B | breast_cancer + diabetes | wine |
| C | wine + diabetes | breast_cancer |

---

## 5. Leakage Prevention
- **History construction**: `repo.list_experiences(exclude_dataset_id=target)` removes all target-dataset records before building the history index.
- **Encoding assertion**: The `encode_dataset_candidates()` function raises `AssertionError` if any historical record sourced from the target dataset reaches the encoder.
- **Target label usage**: Actual utility (`gain`) is loaded from Phase 2 CSVs and used **only** as the evaluation label after prediction, never as a representation feature.

---

## 6. Model Configuration
Saved in `results/phase7/model_config.json`.

---

## 7. Prediction Metrics (CAFET full model)

| Target | Spearman | Pearson | RMSE | MAE |
|---|---:|---:|---:|---:|
| diabetes | 0.0466 | 0.1100 | 0.0300 | 0.0238 |
| wine | 0.1552 | 0.1629 | 0.0124 | 0.0078 |
| breast_cancer | 0.1216 | 0.1417 | 0.0066 | 0.0049 |

---

## 8. Ranking Metrics

| Target | P@10 | P@25 | P@50 | NDCG@10 | NDCG@25 | NDCG@50 |
|---|---:|---:|---:|---:|---:|---:|
| diabetes | 0.000 | 0.280 | 0.220 | 0.481 | 0.616 | 0.641 |
| wine | 0.000 | 0.120 | 0.140 | 1.000 | 1.000 | 1.000 |
| breast_cancer | 0.000 | 0.000 | 0.000 | 0.728 | 0.716 | 0.722 |

---

## 9. Random Baseline (mean over 5 seeds: 0, 7, 13, 42, 99)

| Target | Spearman | P@10 | P@25 | P@50 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| diabetes | -0.032 | 0.080 | 0.096 | 0.240 | 0.516 |
| wine | -0.018 | 0.000 | 0.080 | 0.132 | 0.943 |
| breast_cancer | 0.033 | 0.020 | 0.032 | 0.032 | 0.827 |

---

## 10. Global Mean Baseline

| Target | RMSE | MAE | P@10 | P@25 |
|---|---:|---:|---:|---:|
| diabetes | 0.0280 | 0.0216 | 0.000 | 0.120 |
| wine | 0.0125 | 0.0031 | 0.800 | 0.360 |
| breast_cancer | 0.0097 | 0.0077 | 0.000 | 0.280 |

> Note: Global Mean achieves `P@10=0.8` for wine because wine's actual utility distribution is extremely flat (all candidates near-identical score), so any constant prediction ranks similarly to the true top-10.

---

## 11. Transformation-Only Diagnostic

| Target | Spearman | NDCG@10 | NDCG@25 |
|---|---:|---:|---:|
| diabetes | 0.1336 | 0.529 | 0.601 |
| wine | 0.1123 | 0.987 | 0.987 |
| breast_cancer | 0.1349 | 0.874 | 0.874 |

**Diagnostic finding**: The transformation-only model performs comparably or slightly better than the full CAFET model in some cases. This is discussed in §17.

---

## 12. Prediction Distribution

| Target | Actual Mean | Actual Std | Pred Mean | Pred Std |
|---|---:|---:|---:|---:|
| diabetes | 0.01449 | 0.02129 | -0.00675 | 0.00303 |
| wine | -0.00285 | 0.01225 | -0.00570 | 0.00133 |
| breast_cancer | -0.00398 | 0.00597 | -0.00403 | 0.00373 |

> **Important finding**: Predicted standard deviations are substantially smaller than actual standard deviations. The model is under-dispersed — it can rank relative utilities but does not accurately reproduce their absolute magnitudes. This is expected behavior for a cross-dataset transfer predictor trained on only ~500–1100 samples.

---

## 13. Uncertainty Statistics

| Target | Mean Unc | Std Unc | Min Unc | Max Unc |
|---|---:|---:|---:|---:|
| breast_cancer | 0.00712 | 0.00472 | 0.00000 | 0.01518 |
| diabetes | 0.00853 | 0.00166 | 0.00553 | 0.01129 |
| wine | 0.00670 | 0.00088 | 0.00483 | 0.00898 |

Uncertainty is non-negative across all candidates. The variation is modest — the predictor lacks the historical data volume needed to produce high-contrast confidence estimates.

---

## 14. Feature Importance (Grouped)

| Group | diabetes | wine | breast_cancer |
|---|---:|---:|---:|
| transformation | 1.7% | 0.7% | 0.7% |
| feature_slot_1 | 8.2% | 4.3% | 3.8% |
| feature_slot_2 | 3.5% | 3.1% | 1.8% |
| dataset_context | 5.4% | 0.1% | 0.0% |
| **history** | **81.2%** | **91.8%** | **93.7%** |

> **Diagnostic**: The history dimensions dominate feature importance across all splits. This is consistent with the LODO pool being very small — the model effectively learns "this candidate was (or wasn't) observed before" rather than generalizing from transformation semantics or feature statistics.

---

## 15. Candidate-Level Examples (diabetes, LODO)

From `results/phase7/lodo_predictions.csv`:

| Candidate | Actual Utility | Predicted | Uncertainty | Rank (actual) | Rank (pred) |
|---|---:|---:|---:|---:|---:|
| ADD(bmi,s5) | +0.1312 | -0.0069 | 0.0073 | 1 | ~100-150 |
| ABS(bmi) | -0.0023 | -0.0067 | 0.0073 | ~178 | ~100 |

> The predictor does not successfully identify the best candidate (`ADD(bmi,s5)`) at top rank under LODO. This is consistent with: (a) no exact match of `ADD(bmi,s5)` in the breast_cancer or wine experience pools, and (b) insufficient candidate-level discriminative signal from only two source datasets.

---

## 16. Limitations

1. **Three datasets only**: LODO pools contain only 2 training datasets per split, yielding ~500–1100 training samples. This is insufficient for a robust RF to generalize.
2. **Sparse exact-candidate overlap**: The same candidate `ADD(bmi,s5)` does not appear in breast_cancer or wine. Cross-dataset history is effectively zero for most diabetes-specific candidates.
3. **History dominance**: With 81–94% feature importance, the model relies almost entirely on whether a candidate has been observed before — not on transformation semantics or feature metadata.
4. **Under-dispersion**: Predicted utility std is consistently smaller than actual std. The model regresses toward the training mean.
5. **Three-dataset limitation**: No statistical significance claims can be made.

---

## 17. Scientific Interpretation

### What was observed:
- CAFET Spearman: 0.047–0.155 across targets.
- Random Spearman: −0.032–0.033 across targets.
- CAFET consistently beats random on Spearman (positive vs near-zero).
- Transformation-only model is competitive with the full model, sometimes marginally better.
- History features dominate importance, suggesting the model learns "seen before" rather than "semantically useful."
- The best-performing individual candidate (`ADD(bmi,s5)`, gain=+0.131) is not found at rank 1 under LODO.

### What this does NOT demonstrate:
- That CAFET reliably discovers high-utility candidates on unseen datasets.
- That candidate-level features (feature metadata, dataset context) contribute meaningfully beyond transformation type.
- That the utility predictor is production-ready.
- Generalization to broader datasets.

### Research gate assessment:
The predictor produces weakly positive Spearman correlations that exceed random baseline in all three targets. However, P@10=0.0 for two of three targets indicates that ranking the very best candidates is not yet achieved. The transformation-only model's competitive performance suggests the candidate-specific representation advantage has not yet been demonstrated.

**This phase is assessed as NEEDS FIX** before proceeding to Phase 8. The root causes are: (1) insufficient dataset pool (3 datasets), (2) history feature dominance masking semantic signal, (3) P@K for top candidates is zero or near-zero.

Recommended fixes before Phase 8:
- Incorporate similarity-weighted historical experience (Phase 5 engine) to enrich cross-dataset evidence.
- Expand dataset pool (at least 5–7 datasets) to provide meaningful LODO training data.
- Investigate whether constraining or weighting history features improves semantic signal.

---

## 18. Phase Verdict

> **NEEDS FIX**

The mechanism is implemented correctly and leakage-free. Weak positive Spearman signal exists. However, P@K=0 for top candidates and history-feature dominance indicate the predictor does not yet produce reliable candidate-level ranking that would justify building a budget-constrained search on top of it.
