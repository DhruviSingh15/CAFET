import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from sklearn.datasets import load_breast_cancer
from backend.core.preprocessing.preprocessor import BaselinePreprocessor
from backend.core.candidate_generation.generator import CandidateGenerator
from backend.services.experiment_runner.runner import ExperimentRunner

def test_preprocessing_leakage():
    X = pd.DataFrame({'a': [1, np.nan, 3], 'b': ['cat', 'dog', 'cat']})
    prep = BaselinePreprocessor()
    X_train = prep.fit_transform(X)
    assert not X_train.isna().any().any()
    
    X_test = pd.DataFrame({'a': [np.nan], 'b': ['bird']})
    X_test_prep = prep.transform(X_test)
    assert not X_test_prep.isna().any().any()

def test_candidates():
    gen = CandidateGenerator()
    cands = gen.generate(['f1', 'f2'])
    ids = [c['id'] for c in cands]
    assert len(ids) == len(set(ids)) 
    assert any('ADD(f1,f2)' in i or 'ADD(f2,f1)' in i for i in ids)

def test_reproducibility():
    d = load_breast_cancer(as_frame=True)
    df = d.frame.sample(50, random_state=42)
    
    r1 = ExperimentRunner("test", df, "target", task="classification", seed=42).run()
    r2 = ExperimentRunner("test", df, "target", task="classification", seed=42).run()
    
    assert r1['summary']['baseline_val_score'] == r2['summary']['baseline_val_score']
    assert r1['summary']['best_candidate'] == r2['summary']['best_candidate']

def test_integration():
    d = load_breast_cancer(as_frame=True)
    df = d.frame.sample(50, random_state=42)
    r = ExperimentRunner("test", df, "target", task="classification", seed=42).run()
    assert r['summary']['successful_evaluations'] > 0
