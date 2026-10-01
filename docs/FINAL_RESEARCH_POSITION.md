# FINAL RESEARCH POSITION

## A. Problem

Automated feature engineering can generate many candidate transformations, making exhaustive evaluation computationally expensive. CAFET investigates whether dataset intelligence, stored experience, and budget-aware search can organize this process.

## B. System

The complete end-to-end pipeline operates as follows:

Dataset 
→ Dataset Profile 
→ Candidate Generation 
→ Search Budget 
→ Candidate Evaluation 
→ Best Candidate 
→ Experience Repository 
→ Dashboard

Separate from the core pipeline, experimental research branches were designed to test whether experience and intelligent utility prediction could optimize Candidate Evaluation under strict budgets.

## C. What CAFET Demonstrates

CAFET currently demonstrates the following implemented capabilities:
1. Automated feature candidate generation.
2. Candidate evaluation under an explicit search budget.
3. Dataset profiling / dataset intelligence.
4. Persistent experience storage.
5. Candidate and transformation representation.
6. Dataset similarity computation.
7. End-to-end feature-search execution.
8. Separation between search/validation evaluation and final held-out test evaluation.
9. A working interactive dashboard.
10. A reproducible experimental infrastructure across multiple real-world datasets.

## D. What the Experiments Tested

CAFET systematically tested several hypotheses over progressive experimental phases:
* **Phase 2 baseline**: Establishing random budget-constrained search.
* **Dataset intelligence**: Representing datasets via numerical profiles.
* **Experience repository**: Tracking historical feature utility.
* **Dataset similarity**: Weighting history based on dataset profile distances.
* **Candidate representation**: Encoding transformations and source features.
* **Utility prediction**: Predicting expected feature utility from contextual embeddings.
* **Similarity-weighted experience**: Modulating predictions with historical dataset-similarity.
* **Expanded dataset pool**: Scaling from 3 to 10 diverse datasets.
* **Micro-context representation**: Using fine-grained feature interaction statistics.
* **Adaptive search**: Acquiring candidates via Upper Confidence Bound (UCB).
* **Independent evaluation**: Strict isolation of test partitions to prevent target leakage.
* **Cross-validation**: 5-fold CV to stabilize utility estimation during search.

## E. Main Scientific Finding

The evaluated cross-dataset candidate-utility prediction and adaptive-search mechanisms did not demonstrate reliable out-of-sample improvement over simple baselines under independent evaluation.

*(Note: The working framework and the investigated research hypothesis are separate. The core AutoFE framework functions reliably, but the advanced utility prediction hypothesis yielded a negative result).*

## F. Limitations

* Limited number of benchmark datasets (10).
* Dataset distribution is classification-heavy.
* California Housing uses a deterministic 5,000-row subset for performance.
* Candidate space is constrained.
* Some representations rely on handcrafted statistics.
* Cross-dataset candidate overlap is sparse.
* Adaptive-search experiments showed substantial search-to-independent generalization gaps.
* Candidate selection can be sensitive to dataset/split characteristics.
* No claim of universal cross-dataset transfer.
* No claim of proven continual-learning improvement.

## G. Future Research

* target-aware representations
* richer dataset/task embeddings
* meta-learning
* larger and more diverse dataset pools
* stronger candidate-family modeling
* repeated independent datasets/splits
* alternative acquisition functions
* more robust continual-learning evaluation
