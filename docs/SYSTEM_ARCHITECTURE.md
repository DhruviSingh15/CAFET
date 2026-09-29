# System Architecture

## 4. System Components
The CAFET architecture is modular, divided into Core Engine, Services, Database, API, and Dashboard.
- **Dataset Profiler**: Extracts meta-features from tabular data.
- **Experience Repository**: Stores context-aware transformation successes/failures.
- **Similarity Engine**: Computes dataset distance and identifies useful peers.
- **Candidate Generator**: Generates transformation candidates based on features.
- **Utility Predictor**: ML-based scorer evaluating utility and uncertainty for candidates.
- **Ranker/Budget Enforcer**: Prioritizes candidates via exploitation/exploration under a fixed budget.

## 5. Data Flow
1. **REAL DATASET** -> Dataset Profiler -> **DATASET CONTEXT**
2. **DATASET CONTEXT** -> Similarity Engine -> **EXPERIENCE RETRIEVAL**
3. **EXPERIENCE RETRIEVAL** -> Candidate Generator -> **CANDIDATE GENERATION**
4. **CANDIDATE GENERATION** -> Utility Predictor -> **CANDIDATE-LEVEL UTILITY PREDICTION** (with UNCERTAINTY + RISK)
5. **PREDICTION** -> Ranker -> **CANDIDATE RANKING**
6. **RANKING** -> Budget Enforcer -> **FIXED BUDGET**
7. **FIXED BUDGET** -> Evaluator -> **EXPENSIVE EVALUATION** -> FEATURE SELECTION -> DOWNSTREAM MODEL
8. **EVALUATION RESULTS** -> Database -> **EXPERIENCE UPDATE**

## 6. Database Entities
- **Dataset**: `id`, `name`, `meta_features` (JSON)
- **Feature**: `id`, `dataset_id`, `name`, `type`, `statistics`
- **Transformation**: `id`, `name`, `logic`
- **Candidate**: `id`, `dataset_id`, `transformation_id`, `source_feature_ids`
- **Experience**: `id`, `source_dataset_id`, `candidate_id`, `utility_score`, `risk`, `computational_cost`, `is_successful`, `timestamp`
- **ExperimentRun**: `id`, `dataset_id`, `method`, `budget`, `seed`, `performance_score`

## 12. Dashboard Architecture
- **Tech Stack**: Frontend web framework interacting with backend FastAPI.
- **Pages/Views**:
  - Overview & Dataset Intelligence
  - Experience Repository & Dataset Similarity
  - Candidate Intelligence (Table showing candidate, similarity, predicted utility, uncertainty, priority, actual gain)
  - Budget-Constrained Search (Budget, candidates available vs. evaluated, regret, gap closure)
  - Experiment Comparison, Negative Transfer, Continual Learning
