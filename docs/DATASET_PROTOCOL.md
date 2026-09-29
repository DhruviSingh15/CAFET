# Dataset and Experiment Protocol

## 7. Experiment Flow
Every run executes: Dataset Load -> Context Profiling -> Retrieve Experience -> Candidate Generation -> Predict Utility & Risk -> Rank -> Truncate to Budget -> Evaluate -> Train Downstream Model -> Measure Performance -> Update Experience Repository.

## 8. Dataset LODO (Leave-One-Dataset-Out) Protocol
- Primary Benchmark: 10–15 real-world public tabular datasets (OpenML, UCI).
- For a target dataset D, training experience comes from all other datasets EXCEPT D.
- Before evaluating D, it must not contribute any feature-engineering experience or candidate outcomes. Test data must not influence feature selection or similarity calculations.

## 9. Baseline Methods
- **METHOD A (Random / Uninformed Search)**: Generates candidate universe and randomly selects candidates under the fixed evaluation budget.
- **METHOD B (Global Experience)**: Uses historical transformation performance without considering dataset-specific similarity context.

## 10. CAFET Method
- **METHOD C (Context-Aware Transfer)**: Uses dataset similarity, historical transformation performance, task compatibility, and feature compatibility.
- **METHOD D (Full CAFET)**: Employs context-aware transfer alongside candidate-level utility prediction, uncertainty/risk-aware ranking, budget-constrained search, and continual experience updating.

## 11. Exhaustive Reference
- **METHOD E (Exhaustive Reference)**: Evaluates the complete candidate space where computationally feasible. Establishes the approximate performance ceiling. (Regret = Exhaustive Score - Budgeted Score).

## 13. Testing Strategy
- **Unit Tests**: Test dataset profiler, feature profiler, similarity metrics, experience storage/retrieval, candidate generation, utility prediction, ranking, budget enforcement, and leakage safeguards.
- **Integration Tests**: Full data flow from dataset profile through ranking to evaluation and experience update.
- **Validation**: Rank correlation checks (Spearman, Precision@K, NDCG@K) before expensive full benchmark.

## 14. Leakage Prevention
Strict isolation safeguards:
- Target dataset isolated from training.
- Target test/validation data fully hidden from utility predictor training.
- Preprocessing/normalization fit only on training sets.
- Candidate caching ensures no leakage across folds.

## 15. Reproducibility Strategy
- Strict use of fixed random seeds (minimum 5 seeds for final benchmark).
- Track experiment IDs, dataset, seed, method, budget, baseline/final scores, search time, and experience state.
- Scripts provided for environment setup, dataset acquisition, experiment execution, and dashboard generation.
