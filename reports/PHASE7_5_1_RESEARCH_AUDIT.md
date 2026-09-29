# PHASE 7.5.1 — ADAPTIVE ACQUISITION RESEARCH VALIDATION AUDIT

## 1. Objective
This phase acts as a statistical audit of Phase 7.5 to verify whether the initial findings of Adaptive CAFET's superiority are mathematically and scientifically justified when treating the 10 datasets as the primary paired statistical unit, eliminating pseudo-replication of stochastic seeds.

## 2. Primary Statistical Unit
The dataset ($N=10$) is strictly used as the statistical unit. Means over the 5 random seeds are computed per-dataset before paired testing to prevent significance inflation.

## 3. Required Statistical Comparisons (Dataset-Level)
Wilcoxon signed-rank paired tests on seed-mean best utilities (Adaptive vs Random):
- **1% Budget**: Mean diff: 0.00046, Median diff: 0.00000, Std diff: 0.00125. (Adaptive > Random: 3, Adaptive < Random: 0, Ties: 7). p-value: 0.250
- **2% Budget**: Mean diff: -0.00041, Median diff: 0.00000, Std diff: 0.00067. (Adaptive > Random: 0, Adaptive < Random: 3, Ties: 3). p-value: 0.250
- **5% Budget**: Mean diff: -0.00008, Median diff: 0.00000, Std diff: 0.00347. (Adaptive > Random: 3, Adaptive < Random: 2, Ties: 3). p-value: 0.812
- **10% Budget**: Mean diff: 0.00185, Median diff: 0.00042, Std diff: 0.00368. (Adaptive > Random: 5, Adaptive < Random: 1, Ties: 2). p-value: 0.218
- **20% Budget**: Mean diff: 0.00002, Median diff: 0.00000, Std diff: 0.00580. (Adaptive > Random: 4, Adaptive < Random: 2, Ties: 4). p-value: 0.562

Conclusion: **No statistical significance is established** at any budget when properly paired at the dataset level. 

## 4. Historical Prior Comparisons (Budget 10%)
- **Adaptive vs Transformation-Frequency Prior**: 
  - Mean diff: 0.00526, p-value: 0.039 (Statistically significant improvement over the static prior).
- **Adaptive vs Similarity Prior**:
  - Mean diff: 0.00811, p-value: 0.027 (Statistically significant improvement over the similarity prior).
  
Historical priors continue to demonstrably hinder search efficiency compared to adaptive learning.

## 5. Complete Per-Dataset Results (10% Budget)
| Dataset | Strategy | Mean Evals | Mean Regret | Mean Val Gain | Median Val Gain | Std Val Gain |
|---|---|---|---|---|---|---|
| breast_cancer | Adaptive | 88 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| breast_cancer | Random | 88 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| california_housing | Adaptive | 14 | 0.02771 | 0.00964 | 0.00356 | 0.01552 |
| california_housing | Random | 14 | 0.02590 | 0.01146 | 0.00356 | 0.01473 |
| credit_g | Adaptive | 100 | 0.00266 | 0.02400 | 0.02666 | 0.00365 |
| credit_g | Random | 100 | 0.00800 | 0.01866 | 0.02000 | 0.00557 |
| diabetes | Adaptive | 22 | 0.07208 | 0.05911 | 0.05855 | 0.01244 |
| diabetes | Random | 22 | 0.07233 | 0.05885 | 0.05290 | 0.01039 |
| spambase | Adaptive | 98 | 0.00376 | 0.00782 | 0.00724 | 0.00417 |
| spambase | Random | 98 | 0.00434 | 0.00724 | 0.00724 | 0.00324 |
| titanic | Adaptive | 907 | 0.00406 | 0.00609 | 0.00507 | 0.00227 |
| titanic | Random | 907 | 0.00507 | 0.00507 | 0.00507 | 0.00000 |
| vehicle | Adaptive | 68 | 0.00472 | 0.03464 | 0.03937 | 0.00704 |
| vehicle | Random | 68 | 0.01417 | 0.02519 | 0.02362 | 0.00658 |
| wine | Adaptive | 36 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| wine | Random | 36 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
*(Note: iris and blood_transfusion truncated for brevity, but evaluated exactly the same)*

**Counts (Adaptive vs Random at 10% Budget)**
- Adaptive > Random: 5
- Adaptive < Random: 1
- Adaptive = Random: 2

## 6. Learning Curves
| Budget | Strategy | Mean Utility | Median Utility | Mean Regret | Median Regret |
|---|---|---|---|---|---|
| 0.01 | Adaptive | 0.0133 | 0.0080 | 0.0210 | 0.0113 |
| 0.01 | Random | 0.0128 | 0.0072 | 0.0215 | 0.0133 |
| 0.02 | Adaptive | 0.0068 | 0.0045 | 0.0078 | 0.0063 |
| 0.02 | Random | 0.0072 | 0.0050 | 0.0074 | 0.0058 |
| 0.05 | Adaptive | 0.0131 | 0.0059 | 0.0188 | 0.0068 |
| 0.05 | Random | 0.0132 | 0.0079 | 0.0188 | 0.0066 |
| 0.10 | Adaptive | 0.0176 | 0.0087 | 0.0143 | 0.0039 |
| 0.10 | Random | 0.0158 | 0.0093 | 0.0162 | 0.0065 |
| 0.20 | Adaptive | 0.0236 | 0.0188 | 0.0107 | 0.0011 |
| 0.20 | Random | 0.0236 | 0.0186 | 0.0107 | 0.0036 |

## 7. Regret Audit
- Verified: `regret(B) = oracle_best_utility - best_observed_utility(B)`. The oracle best utility is only calculated retrospectively over the validation data. It never enters the acquisition step.

## 8 & 9. Final Test Aggregate Analysis (Budget 10%)
| Strategy | Mean Test Gain | Median Test Gain | Std Test Gain |
|---|---|---|---|
| Adaptive | 0.00289 | 0.00086 | 0.00450 |
| Random | 0.00062 | 0.00000 | 0.00683 |
| Similarity-Prior | 0.00140 | 0.00000 | 0.00388 |
| Transform-Freq | -0.00155 | -0.00066 | 0.00451 |

**Adaptive Test Gain Counts:** Positive: 5 | Negative: 0 | Zero: 3

## 10. Validation-to-Test Consistency
The ratio `test_gain / validation_gain` varies drastically across datasets (e.g. 0.038 on diabetes, 0.672 on california_housing). 
Crucially, there are **78 instances** (across all strategies/seeds at 10% budget) where `validation_gain > 0` but `test_gain <= 0`. This points to a fundamental instability in evaluating candidate utility solely on the small validation set.

## 11. Search Diversity Audit
On diabetes (budget 10%), Adaptive evaluates a mean of 2.8 unique transformations across 4.0 candidates, matching Random. Similarity-Prior and Transform-Freq collapse diversity entirely, evaluating only 1.0 unique transformation. The Random Forest UCB preserves exploration structurally.

## 12. Initialization Audit
The script evaluated exactly 5 random candidates before surrogate activation. For extremely small candidate spaces (e.g. iris with 34 candidates, blood_transfusion with 35), the 1% and 2% budgets correspond to < 5 candidates. Therefore, learning *cannot* be established at the 1% or 2% budgets since they are functionally just random initialization. 

## 13, 14, 15, 16. Structural Audits
- **Adaptive Acquisition**: Passed. No future or oracle utility was visible to the model.
- **Random Baseline**: Passed. Sampled without replacement matching budget steps perfectly.
- **Historical Priors**: Passed. LODO strictly enforced.
- **Candidate Universe**: Passed. 10 datasets retained their Phase 2 sizes (titanic=7672, spambase=868, etc.).

## 17. Corrected Scientific Conclusions
**Outcome B**: Adaptive CAFET shows a promising empirical improvement (mean differences are positive at 10% budget, final test gains are strictly non-negative), but the 10-dataset benchmark **does not establish statistical significance** when strictly paired at the dataset level (p = 0.218 vs Random).
The validation-to-test consistency is low, meaning the search can easily overfit to the validation slice. 

## 18. Required Final Verdict
**NEEDS FIX**
