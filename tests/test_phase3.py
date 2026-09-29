import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from backend.core.dataset_intelligence.profiler import DatasetProfiler

def get_dummy_data():
    np.random.seed(42)
    df = pd.DataFrame({
        'num1': np.random.randn(100),
        'num2': [1.0, 2.0, np.nan] + list(np.random.randn(97)),
        'cat1': ['A', 'B', 'A', 'C'] * 25,
        'bool1': [True, False] * 50,
        'target': [0, 1] * 50
    })
    return df

def test_profiler_determinism():
    df = get_dummy_data()
    profiler = DatasetProfiler('test', 'classification')
    p1 = profiler.profile(df, 'target')
    p2 = profiler.profile(df, 'target')
    assert p1 == p2

def test_profiler_serialization(tmp_path):
    df = get_dummy_data()
    profiler = DatasetProfiler('test', 'classification')
    p1 = profiler.profile(df, 'target')
    
    filepath = tmp_path / "test_profile.json"
    profiler.serialize(p1, str(filepath))
    p2 = profiler.deserialize(str(filepath))
    
    assert p1 == p2

def test_feature_types():
    df = get_dummy_data()
    profiler = DatasetProfiler('test', 'classification')
    p = profiler.profile(df, 'target')
    
    feat_profs = p['raw_context']['feature_profiles']
    assert feat_profs['num1']['feature_type'] == 'numerical'
    assert feat_profs['cat1']['feature_type'] == 'categorical'
    assert feat_profs['bool1']['feature_type'] == 'boolean'
    
    struct = p['raw_context']['structural']
    assert struct['n_numeric'] == 2
    assert struct['n_categorical'] == 1
    assert struct['n_boolean'] == 1
    assert struct['n_cols'] == 4

def test_leakage_safety():
    df = get_dummy_data()
    profiler = DatasetProfiler('test', 'classification')
    p = profiler.profile(df, 'target')
    
    # Target shouldn't be in feature profiles
    assert 'target' not in p['raw_context']['feature_profiles']
    
def test_regression_target():
    df = get_dummy_data()
    df['target'] = np.random.randn(100)
    profiler = DatasetProfiler('test', 'regression')
    p = profiler.profile(df, 'target')
    assert 'target_mean' in p['raw_context']['target']

def test_context_vector_ordering():
    df = get_dummy_data()
    profiler = DatasetProfiler('test', 'classification')
    p = profiler.profile(df, 'target')
    
    # Check fixed length and matching keys
    assert len(p['numeric_context_vector']) == len(profiler.vector_keys)
    assert p['numeric_context_vector'][0] == p['raw_context']['structural']['n_rows']
