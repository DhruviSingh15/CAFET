from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from backend.core.dataset_intelligence.profiler import DatasetProfiler
from backend.core.candidate_generation.generator import CandidateGenerator
from backend.core.candidate_evaluation.evaluator import CandidateEvaluator
from backend.core.preprocessing.preprocessor import BaselinePreprocessor
from backend.models.baseline_model.model import BaselineModel

# 1. Load Data
d = load_diabetes(as_frame=True)
df = d.frame
task = 'regression'
target_col = 'target'
X = df.drop(columns=[target_col])
y = df[target_col]

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
prep = BaselinePreprocessor()
X_train_prep = prep.fit_transform(X_train)
X_val_prep = prep.transform(X_val)
X_test_prep = prep.transform(X_test)

# 2. Generator
generator = CandidateGenerator()
candidates = generator.generate(X_train_prep.columns.tolist())
total_cands = len(candidates)

# 3. Budget
budget_pct = 5
allowed_evals = max(1, int((budget_pct / 100.0) * total_cands))

# 4. Search
baseline_model = BaselineModel(task=task, random_state=42)
baseline_model.fit(X_train_prep, y_train)
baseline_val_score = baseline_model.evaluate(X_val_prep, y_val)

evaluator = CandidateEvaluator()
rng = np.random.default_rng(42)
search_indices = rng.choice(range(total_cands), size=allowed_evals, replace=False)

best_candidate = None
best_candidate_score = -float('inf')

for idx in search_indices:
    cand = candidates[idx]
    try:
        train_feat = evaluator.apply_transform(X_train_prep, cand)
        val_feat = evaluator.apply_transform(X_val_prep, cand)
        if train_feat.isna().any() or np.isinf(train_feat).any():
            continue
            
        X_train_c = X_train_prep.copy()
        X_val_c = X_val_prep.copy()
        X_train_c[cand['id']] = train_feat
        X_val_c[cand['id']] = val_feat
        
        model = BaselineModel(task=task, random_state=42)
        model.fit(X_train_c, y_train)
        score = model.evaluate(X_val_c, y_val)
        
        if score > best_candidate_score:
            best_candidate_score = score
            best_candidate = cand
    except:
        pass

# Final evaluation
baseline_test_score = baseline_model.evaluate(X_test_prep, y_test)
train_feat = evaluator.apply_transform(X_train_prep, best_candidate)
test_feat = evaluator.apply_transform(X_test_prep, best_candidate)
X_train_c = X_train_prep.copy()
X_test_c = X_test_prep.copy()
X_train_c[best_candidate['id']] = train_feat
X_test_c[best_candidate['id']] = test_feat

final_model = BaselineModel(task=task, random_state=42)
final_model.fit(X_train_c, y_train)
best_candidate_test_score = final_model.evaluate(X_test_c, y_test)

print(f"Dataset: diabetes")
print(f"Candidate Count: {total_cands}")
print(f"Budget: {budget_pct}%")
print(f"Evaluated: {allowed_evals}")
print(f"Baseline Search Score: {baseline_val_score}")
print(f"Best Candidate: {best_candidate['transform']}({best_candidate['features']})")
print(f"Candidate Search Score: {best_candidate_score}")
print(f"Search Utility: {best_candidate_score - baseline_val_score}")
print(f"Baseline Test Score: {baseline_test_score}")
print(f"Candidate Test Score: {best_candidate_test_score}")
print(f"Final Test Gain: {best_candidate_test_score - baseline_test_score}")

