# Phase 7.6: Adaptive Search under Validation Overfitting Control

## 1. Objective
Phase 7.5.1 identified severe validation-to-test inconsistency (78 cases where positive validation gain yielded non-positive test gain), suggesting that repeatedly optimizing candidate selection against a single validation split heavily overfits the search process. Phase 7.6 evaluates whether Adaptive CAFET's apparent efficiency advantages over Random Search survive a stringent cross-split generalization protocol.

## 2. Benchmark
The identical 10 datasets and Phase 2 deterministic candidate universes are strictly preserved. No candidate filtering is applied.

## 3. Split Methodology
To break the repeated-optimization loop, evaluation is fundamentally separated:
- **Search-Training Split (62.5%):** Used dynamically by the acquisition loop. Split internally to evaluate baseline utility for surrogate training.
- **Search-Validation / Independent Split (18.75%):** Used strictly after search completion to measure the candidate's generalization independent of the search trajectory.
- **Final Test Split (18.75%):** Completely isolated.

For every dataset, the entire acquisition search is repeated across five completely independent deterministic splits (seeds 42, 7, 13, 99, 123) to measure cross-split stability.

## 4. Candidate Evaluation Protocol
During search, candidate utility is evaluated exclusively on the search-training split. At each budget checkpoint (5%, 10%, 20%), the best search-candidate is frozen and evaluated sequentially on the independent split and the final test split. 

## 5. Search Budgets
Budgets evaluated: 5%, 10%, and 20% of the candidate universe. (1% and 2% are omitted as they are structurally dominated by the 5-candidate unguided initialization phase on smaller datasets).

## 6. Random Baseline
Random Search uniformly samples candidates without replacement, evaluated strictly under the exact same nested-split protocol.

## 7. Adaptive CAFET
Iterative Random Forest surrogate utilizing UCB (Upper Confidence Bound). Trained dynamically solely on previously evaluated candidates in the current search-training split.

## 8. Per-Dataset Results
At a 10% budget, comparing the **search utility** against the **independent evaluation utility** (Seed-Means):
- **california_housing**: Adaptive Search 0.0215 -> Indep 0.0204 (Stable)
- **diabetes**: Adaptive Search 0.0301 -> Indep -0.0017 (Severe overfitting)
- **credit_g**: Adaptive Search 0.0240 -> Indep -0.0053 (Severe overfitting)
- **vehicle**: Adaptive Search 0.0346 -> Indep 0.0031 (Severe drop)
- **blood_transfusion**: Adaptive Search 0.0267 -> Indep 0.0106 (Drop)

## 9. Aggregate Results
Across all 10 datasets at the 10% budget (Seed-Means):
- **Adaptive**: Search Mean: 0.0155 | Independent Mean: 0.0019 | Test Mean: 0.0034
- **Random**: Search Mean: 0.0154 | Independent Mean: 0.0029 | Test Mean: 0.0024
Adaptive does not yield a superior independent utility compared to Random Search (0.0019 vs 0.0029).

## 10. Generalization Gaps
The generalization gap (`search_utility - independent_utility`) explicitly measures overfitting to the search split.
- **Adaptive**: Mean Gap: 0.0136. Positive gap count: 32 / 50. Negative gap count: 5 / 50.
- **Random**: Mean Gap: 0.0125. Positive gap count: 35 / 50. Negative gap count: 2 / 50.
Both methods structurally overfit the search split, but Adaptive CAFET fails to mitigate it, producing massive drops in utility when the selected candidate is moved to an independent split.

## 11. Cross-Split Stability
Adaptive CAFET's candidate selection is highly unstable across independent search splits. 
On most datasets (e.g., `diabetes`, `blood_transfusion`, `breast_cancer`, `credit_g`, `iris`, `vehicle`, `wine`), Adaptive selected **5 entirely different unique candidates** across the 5 independent splits. Even at the transformation level, it selected 4 to 5 different transformations. The search trajectory is highly sensitive to the exact data distribution in the search split.

## 12. Search Efficiency
While Adaptive achieved 0.00033 mean independent utility per evaluation compared to Random's 0.00021, the absolute performance ceiling dropped so severely on the independent split that "efficiency" becomes a moot metric. Reaching a non-generalizable optimum faster is not scientifically useful.

## 13. Statistical Tests
Wilcoxon signed-rank paired tests on Independent Utility across the 10 datasets (Adaptive vs Random):
- **5% Budget**: Mean Diff: -0.00257. Adaptive < Random on 6 datasets. p-value: 0.375
- **10% Budget**: Mean Diff: -0.00103. Adaptive < Random on 7 datasets. p-value: 0.431
- **20% Budget**: Mean Diff: -0.00005. Adaptive < Random on 5 datasets. (Tie: 0). p-value: 0.921
Adaptive CAFET does **not** outperform Random Search under proper cross-split generalization control.

## 14. Leakage Audit
Assertions actively confirm:
- Final test utility is never exposed during surrogate training or argmax selection.
- Independent evaluation utility remains isolated from the surrogate.
- Surrogate models only update using utility calculated on the search-training split.
- STATUS: PASS.

## 15. Reproducibility
The search trace is completely deterministic based on the split seed and Random Forest state.
- STATUS: PASS.

## 16. Failure Analysis
Adaptive CAFET heavily overfits the search-training split. 
Out of 35 instances (across all 10% budget runs) where Adaptive CAFET found a candidate with positive search utility, **16 instances** (45%) yielded zero or negative utility on the independent evaluation split. Furthermore, **18 instances** yielded zero or negative utility on the final test split. The underlying Random Forest surrogate successfully learns to optimize the objective it is given, but that objective (utility on a specific 15-20% subset of data) does not correlate reliably with out-of-sample utility.

## 17. Limitations
The search relies entirely on standard train/validation baseline model fits to derive the `search_utility`. The baseline model variance on small slices of data introduces massive label noise into the surrogate's training data, rendering UCB exploration ineffective.

## 18. Scientific Conclusion
**RQ1**: No. Adaptive CAFET does NOT outperform Random Search when protected against repeated optimization of a single validation split.
**RQ2**: Rarely. The candidates selected often fail to generalize.
**RQ3**: The generalization gap is enormous (mean 0.0136), frequently erasing 80-100% of the apparent search gain.
**RQ4**: No. Candidate selection is highly unstable across independent splits.
**RQ5**: No. Adaptive CAFET does not find generalizable candidates more efficiently than Random.

The empirical advantage observed in Phase 7.5 was an artifact of "validation hacking"—repeatedly querying and optimizing against a fixed validation set until a spuriously high-scoring candidate was found.

## 19. Final Verdict
**FAIL**
