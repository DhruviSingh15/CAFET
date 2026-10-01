# NOVELTY AND CLAIM BOUNDARY

This document carefully delineates the scientific and technical claims associated with CAFET based directly on empirical evidence.

## Safe claims

These claims are directly supported by implementation and experiments:
* CAFET implements a reproducible, modular Automated Feature Engineering (AutoFE) pipeline.
* CAFET strictly enforces candidate evaluation budgets.
* CAFET securely logs dataset intelligence and feature engineering results into an Experience Repository.
* CAFET successfully enforces strict boundaries between search/validation partitions and final held-out test partitions to prevent target leakage.
* CAFET provides a functional interactive dashboard to orchestrate end-to-end AutoFE tasks.

## Research claims

These claims were investigated but require qualification:
* **Experience-aware Candidate Representation:** CAFET investigated embedding candidate features and dataset profiles to predict utility, though out-of-sample generalization was unreliable.
* **Cross-dataset Utility Prediction:** CAFET evaluated similarity-weighted historical experience for candidate scoring; empirical results show this approach did not yield consistent improvements without target-awareness.
* **Adaptive Search:** CAFET tested using Random Forest surrogates and UCB acquisition to guide search, finding it underperformed random search baselines during strict independent evaluation.

## Unsupported claims

We explicitly do NOT make the following claims:
* "first automated feature engineering system"
* "proven continual learning"
* "proven cross-dataset transfer"
* "guaranteed adaptive-search improvement"
* "patentable invention"
* "state-of-the-art"

Note: Patentability and prior-art novelty require a separate prior-art/patent search and cannot be concluded from the current experiments alone.
