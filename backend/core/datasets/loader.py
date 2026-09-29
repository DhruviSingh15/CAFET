import os
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer, load_wine, load_diabetes, 
    fetch_california_housing, load_iris, fetch_openml
)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))

def _fetch_openml_cached(name, data_id, target_col):
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = os.path.join(DATA_DIR, f"{name}.csv")
    
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
    else:
        # Fetch from openml
        data = fetch_openml(data_id=data_id, as_frame=True, parser='auto')
        df = data.frame
        # Rename target column to 'target' for consistency if it's not already
        original_target = data.target_names[0] if hasattr(data, 'target_names') and data.target_names else data.target.name
        if original_target != 'target':
            if 'target' in df.columns:
                df = df.drop(columns=['target'])
            df = df.rename(columns={original_target: 'target'})
        df.to_csv(cache_path, index=False)
        
    return df, 'target'

def _load_sklearn_cached(name, loader_func):
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = os.path.join(DATA_DIR, f"{name}.csv")
    
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
    else:
        data = loader_func(as_frame=True)
        df = data.frame
        df.to_csv(cache_path, index=False)
        
    return df, 'target'

def get_dataset(name):
    """
    Returns (DataFrame, target_column_name, task_type, original_source_info)
    """
    if name == 'breast_cancer':
        df, tgt = _load_sklearn_cached(name, load_breast_cancer)
        return df, tgt, 'classification', 'sklearn.datasets.load_breast_cancer'
    elif name == 'wine':
        df, tgt = _load_sklearn_cached(name, load_wine)
        return df, tgt, 'classification', 'sklearn.datasets.load_wine'
    elif name == 'diabetes':
        df, tgt = _load_sklearn_cached(name, load_diabetes)
        return df, tgt, 'regression', 'sklearn.datasets.load_diabetes'
    elif name == 'california_housing':
        # the frame returned by fetch_california_housing has 'MedHouseVal' as target
        df, tgt = _load_sklearn_cached(name, fetch_california_housing)
        if 'target' not in df.columns and 'MedHouseVal' in df.columns:
            df = df.rename(columns={'MedHouseVal': 'target'})
        return df, 'target', 'regression', 'sklearn.datasets.fetch_california_housing'
    elif name == 'iris':
        df, tgt = _load_sklearn_cached(name, load_iris)
        return df, tgt, 'classification', 'sklearn.datasets.load_iris'
    elif name == 'titanic':
        df, tgt = _fetch_openml_cached(name, 40945, 'survived')
        return df, tgt, 'classification', 'OpenML:40945 (Titanic)'
    elif name == 'credit_g':
        df, tgt = _fetch_openml_cached(name, 31, 'class')
        return df, tgt, 'classification', 'OpenML:31 (Credit-g)'
    elif name == 'blood_transfusion':
        df, tgt = _fetch_openml_cached(name, 1464, 'Class')
        return df, tgt, 'classification', 'OpenML:1464 (Blood Transfusion)'
    elif name == 'vehicle':
        df, tgt = _fetch_openml_cached(name, 54, 'Class')
        return df, tgt, 'classification', 'OpenML:54 (Vehicle)'
    elif name == 'spambase':
        df, tgt = _fetch_openml_cached(name, 44, 'class')
        return df, tgt, 'classification', 'OpenML:44 (Spambase)'
    else:
        raise ValueError(f"Unknown dataset: {name}")

DATASETS_LIST = [
    'breast_cancer', 'wine', 'diabetes', 'california_housing', 'iris',
    'titanic', 'credit_g', 'blood_transfusion', 'vehicle', 'spambase'
]
