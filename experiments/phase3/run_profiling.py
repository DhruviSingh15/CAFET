import os
import sys
import json
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from sklearn.datasets import load_breast_cancer, load_wine, load_diabetes
from backend.core.dataset_intelligence.profiler import DatasetProfiler

def get_dataset(name):
    if name == 'breast_cancer':
        d = load_breast_cancer(as_frame=True)
        task = 'classification'
    elif name == 'wine':
        d = load_wine(as_frame=True)
        task = 'classification'
    elif name == 'diabetes':
        d = load_diabetes(as_frame=True)
        task = 'regression'
    
    df = pd.concat([d.data, d.target], axis=1)
    target_col = d.target.name
    return df, target_col, task

def main():
    datasets = ['breast_cancer', 'wine', 'diabetes']
    os.makedirs('results/phase3', exist_ok=True)
    
    schema = {
        "raw_context": "Hierarchical dictionary of metadata",
        "numeric_context_vector": "Deterministic 1D array",
        "vector_keys": "Explicit feature ordering list"
    }
    with open('results/phase3/profile_schema.json', 'w') as f:
        json.dump(schema, f, indent=4)
    
    for name in datasets:
        df, target_col, task = get_dataset(name)
        profiler = DatasetProfiler(dataset_name=name, task_type=task)
        profile = profiler.profile(df, target_col)
        
        profiler.serialize(profile, f"results/phase3/{name}_profile.json")
        print(f"Profiled {name} successfully.")
        
if __name__ == "__main__":
    main()
