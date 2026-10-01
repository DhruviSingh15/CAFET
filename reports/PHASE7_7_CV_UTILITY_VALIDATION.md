# PHASE 7.7 — Cross-Validated Utility Estimation Under Independent Evaluation

## 1. Research question
Does replacing a single validation-split utility estimate with cross-validated utility reduce search overfitting and improve the independent generalization of CAFET's adaptive feature search?

## 2. Hypothesis
Estimating candidate utility via 5-fold cross validation on the search/training split will yield a more robust and generalized evaluation signal than a single 50/50 inner split, reducing the gap between search utility and independent validation utility.

## 3. Protocol
The pipeline applies a strict three-way outer split and a 5-fold inner CV:
- The target dataset is outer-split into 3 partitions (62.5% Search/Training, 18.75% Independent, 18.75% Final Test).
- Adaptive feature search uses only the Search/Training split.
- During search, each evaluated candidate undergoes a localized 5-fold CV.
- Preprocessing and baseline model fitting happen strictly within each fold.
- CV Utility is defined as the mean difference between candidate fold-score and baseline fold-score.

## 4. Dataset pool
10 diverse regression and classification datasets: breast_cancer, wine, diabetes, california_housing, iris, titanic, credit_g, blood_transfusion, vehicle, spambase. (California housing is downsampled to a 5000-row subset deterministically).

## 5. Outer split protocol
5 random seeds (42, 7, 13, 99, 123) are used to randomly outer-split the datasets.

## 6. CV protocol
A 5-fold Cross-Validation is run on the Search/Training partition.
- `StratifiedKFold(n_splits=5)` for classification tasks.
- `KFold(n_splits=5)` for regression tasks.
- The random seed is fixed to 42 for inner fold generation.

## 7. Leakage controls
- Candidate Generation does not use test labels.
- Preprocessing is fitted locally to the 4/5 training data in each CV fold; the remaining 1/5 validation fold is only transformed. 
- Independent and Final Test partitions are explicitly held out and untouched by the search algorithm.
- Hyperparameters are unmodified.

## 8. Adaptive strategy
Uses a Random Forest surrogate to predict candidate CV utility from contextual embeddings, optimizing acquisition via Upper Confidence Bound (UCB). 

## 9. Random baseline
A baseline search strategy that randomly acquires candidates without utilizing predicted CV utilities.

## 10. Budget definitions
Candidate evaluation budgets are restricted to 5%, 10%, and 20% of the total generated candidate universe.

## 11. Aggregate Metrics by Budget

### Budget: 5.0%
* **Strategy: Random**
  * **Search Utility:** Mean=0.0089, Median=0.0056, Std=0.0089
  * **Independent Gain:** Mean=0.0039, Median=0.0000, Std=0.0123
  * **Final Test Gain:** Mean=-0.0016, Median=0.0000, Std=0.0135
  * **Search-to-Indep Gap:** Mean=0.0050, Median=0.0028, Std=0.0129
* **Strategy: Adaptive**
  * **Search Utility:** Mean=0.0082, Median=0.0066, Std=0.0082
  * **Independent Gain:** Mean=-0.0005, Median=0.0000, Std=0.0165
  * **Final Test Gain:** Mean=0.0032, Median=0.0000, Std=0.0091
  * **Search-to-Indep Gap:** Mean=0.0087, Median=0.0060, Std=0.0178

### Budget: 10.0%
* **Strategy: Random**
  * **Search Utility:** Mean=0.0114, Median=0.0087, Std=0.0098
  * **Independent Gain:** Mean=0.0050, Median=0.0000, Std=0.0127
  * **Final Test Gain:** Mean=0.0003, Median=0.0000, Std=0.0137
  * **Search-to-Indep Gap:** Mean=0.0064, Median=0.0053, Std=0.0133
* **Strategy: Adaptive**
  * **Search Utility:** Mean=0.0114, Median=0.0080, Std=0.0115
  * **Independent Gain:** Mean=0.0003, Median=0.0000, Std=0.0201
  * **Final Test Gain:** Mean=0.0021, Median=0.0000, Std=0.0122
  * **Search-to-Indep Gap:** Mean=0.0111, Median=0.0063, Std=0.0216

### Budget: 20.0%
* **Strategy: Random**
  * **Search Utility:** Mean=0.0127, Median=0.0101, Std=0.0101
  * **Independent Gain:** Mean=0.0028, Median=0.0000, Std=0.0178
  * **Final Test Gain:** Mean=0.0012, Median=0.0000, Std=0.0139
  * **Search-to-Indep Gap:** Mean=0.0099, Median=0.0056, Std=0.0209
* **Strategy: Adaptive**
  * **Search Utility:** Mean=0.0135, Median=0.0099, Std=0.0115
  * **Independent Gain:** Mean=-0.0002, Median=0.0000, Std=0.0219
  * **Final Test Gain:** Mean=0.0050, Median=0.0000, Std=0.0138
  * **Search-to-Indep Gap:** Mean=0.0137, Median=0.0085, Std=0.0238

## 12. Dataset-level Statistical Tests (Independent Gain)
* **Budget: 5.0% | N=10 datasets**
  * Adaptive Mean: -0.0005, Random Mean: 0.0039
  * Mean Diff: -0.0044, Median Diff: -0.0025
  * Wins: 2, Losses: 8, Ties: 0
  * Wilcoxon p-value: 0.0273 (Random significantly better)
* **Budget: 10.0% | N=10 datasets**
  * Adaptive Mean: 0.0003, Random Mean: 0.0050
  * Mean Diff: -0.0047, Median Diff: -0.0033
  * Wins: 2, Losses: 8, Ties: 0
  * Wilcoxon p-value: 0.1309
* **Budget: 20.0% | N=10 datasets**
  * Adaptive Mean: -0.0002, Random Mean: 0.0028
  * Mean Diff: -0.0030, Median Diff: -0.0039
  * Wins: 4, Losses: 6, Ties: 0
  * Wilcoxon p-value: 0.4316

## 13. Candidate Stability across 5 outer splits
(Lower unique count is better, minimum 1)

* **Budget 5.0%**
  * Random: Mean Unique Exact = 4.90, Transformations = 3.30
  * Adaptive: Mean Unique Exact = 5.00, Transformations = 3.70
* **Budget 10.0%**
  * Random: Mean Unique Exact = 4.70, Transformations = 3.10
  * Adaptive: Mean Unique Exact = 4.80, Transformations = 3.30
* **Budget 20.0%**
  * Random: Mean Unique Exact = 4.70, Transformations = 3.10
  * Adaptive: Mean Unique Exact = 4.60, Transformations = 2.60

## 14. Search vs Independent Correlation
* Budget 5.0% - Adaptive: Spearman=0.0691 (p=0.6337)
* Budget 10.0% - Adaptive: Spearman=0.0356 (p=0.8059)
* Budget 20.0% - Adaptive: Spearman=0.1286 (p=0.3736)

## 15. Most Important Comparison (Phase 7.7 vs 7.6)
Did cross-validated utility:
1. **Reduce the search-to-independent gap?** No. The gap remained substantial (~0.010 on average for Adaptive).
2. **Improve independent performance?** No. Adaptive Independent performance degraded (mean ~0.0000 across budgets) compared to Phase 7.6.
3. **Improve final-test performance?** Marginally, but unreliably. Final test gain was positive but independent gain was not.
4. **Improve candidate stability?** No. Adaptive Search picked roughly 5 distinct exact candidates across 5 seeds on average, meaning almost zero stability.
5. **Improve relationship between search utility and independent utility?** No. Spearman correlation is virtually zero (0.03 - 0.12) and not statistically significant.
6. **Produce stronger evidence for Adaptive Search over Random Search?** No. In fact, Random Search outperformed Adaptive Search on independent validation across all budgets (significantly so at 5% budget).

## 16. Limitations
The full cross-validation design evaluated 5 Baseline Models (and 5 feature transformations) for every candidate evaluated. The computational cost was roughly 5x higher than Phase 7.6, taking approximately 50 minutes for the full evaluation.

## 17. Scientific interpretation
**NO MEANINGFUL IMPROVEMENT — FREEZE THIS RESEARCH BRANCH**

The evidence strongly concludes that attempting to predict candidate utility using purely structural representation context vector without knowing the target interaction does not generalize reliably, regardless of whether utility is estimated via a single validation split (Phase 7.6) or 5-fold cross-validation (Phase 7.7). 

Adaptive search consistently underperforms or fails to separate itself meaningfully from Random Search, suffering from severe candidate instability and zero correlation between the surrogate utility model and actual out-of-sample independent gain. Replacing the utility target evaluation will require a fundamental pivoting of the candidate representation approach (e.g. observing true target utility during search, target-aware embeddings, or direct gradient-based meta-learning) rather than further modifying the validation protocol.
