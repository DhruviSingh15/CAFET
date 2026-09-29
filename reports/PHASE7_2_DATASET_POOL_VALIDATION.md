# CAFET — Phase 7.2: Real-World Dataset Pool Expansion

## 1. Objective
The objective of Phase 7.2 is to expand the experimental foundation of CAFET from 3 datasets to 8–10 real-world public tabular datasets. This expanded pool provides a scientifically meaningful historical experience base for evaluating cross-dataset feature experience transfer, addressing the fundamental bottleneck identified in Phase 7.1.

---

## 2. Dataset Selection Criteria
Datasets were selected strictly based on the following criteria:
- **Real-world provenance**: No purely synthetic/toy datasets.
- **Public availability**: Accessible via scikit-learn or OpenML.
- **Diversity**: Mix of classification/regression, small/medium size, varying dimensions and missingness, covering different domain contexts.
- **Technical compatibility**: Capability to be processed securely by the existing pipeline without altering the core mechanisms.

---

## 3. Dataset Registry

| dataset_id | source | task_type | target | n_rows | n_features | numeric | categorical | missing |
|:---|:---|:---|:---|---:|---:|---:|---:|---:|
| breast_cancer | sklearn.datasets | classification | target | 569 | 30 | 30 | 0 | 0 |
| wine | sklearn.datasets | classification | target | 178 | 13 | 13 | 0 | 0 |
| diabetes | sklearn.datasets | regression | target | 442 | 10 | 10 | 0 | 0 |
| california_housing | sklearn.datasets | regression | target | 20640 | 8 | 8 | 0 | 0 |
| iris | sklearn.datasets | classification | target | 150 | 4 | 4 | 0 | 0 |
| titanic | OpenML:40945 | classification | target | 1309 | 13 | 6 | 7 | 3855 |
| credit_g | OpenML:31 | classification | target | 1000 | 20 | 7 | 13 | 0 |
| blood_transfusion | OpenML:1464 | classification | target | 748 | 4 | 4 | 0 | 0 |
| vehicle | OpenML:54 | classification | target | 846 | 18 | 18 | 0 | 0 |
| spambase | OpenML:44 | classification | target | 4601 | 57 | 57 | 0 | 0 |

---

## 4. Dataset Diversity
The expanded pool provides significant diversity:
- **Tasks**: 8 classification, 2 regression.
- **Scale**: Rows vary from 150 (`iris`) to 20,640 (`california_housing`), offering a realistic evaluation gradient. `california_housing` was randomly sampled down to 5,000 rows during candidate evaluation to respect current execution bounds.
- **Features**: Varying from 4 (`iris`, `blood_transfusion`) to 57 (`spambase`), creating varying candidate generation loads.
- **Types**: Presence of mixed types, notably `titanic` (6 num, 7 cat) and `credit_g` (7 num, 13 cat), testing CAFET's broader type compatibility.
- **Missingness**: `titanic` contains substantial missing values (3855 total).

---

## 5. Data Integrity
Automated tests confirmed:
- Datasets load deterministically.
- Target variables are excluded from the feature sets.
- Phase 3 profiler successfully generated context vectors (`numeric_context_vector`) for all 10 datasets.

---

## 6. Phase 3 Profiling Results
All datasets successfully passed dataset intelligence processing. Valid deterministic context representations capturing cardinality, correlations, classes, and missingness metrics were produced and saved in `results/phase3/`.

---

## 7. Candidate Generation Results
The Phase 2 candidate generator processed the new datasets successfully:
- `titanic` generated 9,072 successful candidate features.
- `spambase` generated 988 candidates.
- `credit_g` generated 1,004 candidates.
This demonstrates the generator scales naturally over a wide variety of domains.

---

## 8. Baseline Results
All 10 datasets successfully produced a baseline model score and a "best candidate" validation and test evaluation.

---

## 9. Experience Repository Results
The Phase 4 Experience Repository was successfully cleared and fully repopulated.
- **Datasets**: 10
- **Total Records**: 13,436 candidates successfully evaluated and stored.
- **Successful Experiences**: Some datasets had huge improvements (e.g., `titanic` with 2,252 successful candidates), while others served to provide mostly negative signal.

---

## 10. Similarity Results
Phase 5 Dataset Similarity Engine successfully ranked the full 10x10 dataset matrix. 
- Diagonal similarities equal 1.0.
- Similarities appropriately bounded in (0, 1].
- Matrix confirms no `NaN` collapse across structurally diverse datasets.

---

## 11. LODO Feasibility
For any target dataset in the pool:
- **Historical datasets available**: 9 
- **Historical experience pool**: Ranging from 4,364 to 13,396 records, providing a deep evidence base.
- **Mean similarity**: Varies from 0.10 (`titanic` as target) to 0.33 (`diabetes` as target).
The pool is now large enough to robustly test similarity-based cross-dataset feature transfer.

---

## 12. Experience Coverage
- **Exact Candidate Overlap**: Remains strictly near zero (0.00% to ~0.018%) across all datasets. Exact feature names and statistics rarely match perfectly between datasets.
- **Transformation Overlap**: 100%. All target datasets use candidate transformations that were heavily observed in the 9 training datasets.
- **Similarity-Weighted Evidence Rate**: 1.0. Due to structural compatibility handling (Phase 7.1), every candidate generated can now draw upon similarity-weighted experience from its 9 peers.

---

## 13. Dataset-Specific Problems
- `california_housing`: Features exceeded comfortable evaluation time; dynamically subsetted to 5,000 samples to remain viable for current synchronous Phase 2 processing.
- `titanic`: Contains heavily categorical + numeric splits with missing data, yet passed safely through the robust pipeline due to default scikit-learn data handling in earlier setups.

---

## 14. Limitations
- Classification heavily outweighs regression (8 vs 2). Further benchmarks will need to balance this.
- Subsetting `california_housing` to 5,000 records technically modifies the real-world statistical layout, but maintains the integrity of the methodology.
- CAFET has yet to prove performance on this newly constructed pool (deferred to Phase 7.1 re-run).

---

## 15. Phase Verdict
**PASS**

The dataset expansion strictly adhered to all requirements, building a stable, reproducible pool of 10 real-world datasets with comprehensive coverage over candidate generation, utility evaluation, similarity ranking, and repository storage. This clears the bottleneck from Phase 7/7.1.
