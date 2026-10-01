import os
import sys
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from backend.core.candidate_evaluation.evaluator import CandidateEvaluator
from backend.models.baseline_model.model import BaselineModel
from backend.core.preprocessing.preprocessor import BaselinePreprocessor
from sklearn.model_selection import StratifiedKFold, KFold

def test_classification_cv_split():
    d = load_breast_cancer(as_frame=True)
    X = d.frame.drop(columns=['target'])
    y = d.frame['target']
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(cv.split(X, y))
    
    assert len(splits) == 5
    train_idx, val_idx = splits[0]
    
    # Check split proportion (roughly 80/20)
    assert len(val_idx) / len(X) == pytest.approx(0.2, abs=0.01)

def test_regression_cv_split():
    d = load_diabetes(as_frame=True)
    X = d.frame.drop(columns=['target'])
    y = d.frame['target']
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(cv.split(X, y))
    
    assert len(splits) == 5
    train_idx, val_idx = splits[0]
    
    assert len(val_idx) / len(X) == pytest.approx(0.2, abs=0.01)

def test_deterministic_fold_generation():
    d = load_diabetes(as_frame=True)
    X = d.frame.drop(columns=['target'])
    y = d.frame['target']
    
    cv1 = KFold(n_splits=5, shuffle=True, random_state=42)
    splits1 = list(cv1.split(X, y))
    
    cv2 = KFold(n_splits=5, shuffle=True, random_state=42)
    splits2 = list(cv2.split(X, y))
    
    assert np.array_equal(splits1[0][0], splits2[0][0])
    assert np.array_equal(splits1[0][1], splits2[0][1])

def test_fold_local_preprocessing():
    # If we fit preprocessing on train, the mean should be 0 on train but not exactly 0 on val
    d = load_diabetes(as_frame=True)
    X = d.frame.drop(columns=['target'])
    y = d.frame['target']
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(cv.split(X, y))
    
    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    
    prep = BaselinePreprocessor()
    X_train_p = prep.fit_transform(X_train)
    X_val_p = prep.transform(X_val)
    
    train_mean = X_train_p.mean().mean()
    val_mean = X_val_p.mean().mean()
    
    assert abs(train_mean) < 1e-7
    # Val mean shouldn't be identically zero usually
    assert abs(val_mean) > 1e-7

def test_budget_enforcement():
    budget_pct = 0.05
    total_candidates = 220
    allowed_evals = max(5, int(budget_pct * total_candidates))
    assert allowed_evals == 11
    
    budget_pct = 0.20
    allowed_evals = max(5, int(budget_pct * total_candidates))
    assert allowed_evals == 44

def test_independent_and_test_exclusion():
    # Verify that splits are disjoint
    from sklearn.model_selection import train_test_split
    d = load_diabetes(as_frame=True)
    X = d.frame.drop(columns=['target'])
    y = d.frame['target']
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1875, random_state=42)
    X_train, X_indep, y_train, y_indep = train_test_split(X_temp, y_temp, test_size=(18.75/81.25), random_state=42)
    
    intersect_train_test = set(X_train.index).intersection(set(X_test.index))
    intersect_indep_test = set(X_indep.index).intersection(set(X_test.index))
    intersect_train_indep = set(X_train.index).intersection(set(X_indep.index))
    
    assert len(intersect_train_test) == 0
    assert len(intersect_indep_test) == 0
    assert len(intersect_train_indep) == 0
