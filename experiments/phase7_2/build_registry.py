import os
import sys
import pandas as pd
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from backend.core.datasets.loader import get_dataset, DATASETS_LIST

def build_registry():
    registry_path = "data/dataset_registry.csv"
    os.makedirs("data", exist_ok=True)
    os.makedirs("results/phase7_2", exist_ok=True)
    
    registry = []
    
    for name in DATASETS_LIST:
        print(f"Loading {name}...")
        df, tgt, task, source = get_dataset(name)
        
        # Determine feature types
        features = df.drop(columns=[tgt])
        num_cols = features.select_dtypes(include=['number']).columns.tolist()
        cat_cols = features.select_dtypes(exclude=['number']).columns.tolist()
        missing = df.isnull().sum().sum()
        
        registry.append({
            "dataset_id": name,
            "name": name,
            "source": source,
            "task_type": task,
            "target": tgt,
            "n_rows": len(df),
            "n_features": len(features.columns),
            "numeric_features": len(num_cols),
            "categorical_features": len(cat_cols),
            "missing_values": missing,
        })
        
    registry_df = pd.DataFrame(registry)
    registry_df.to_csv(registry_path, index=False)
    registry_df.to_csv("results/phase7_2/dataset_registry.csv", index=False)
    
    print("Registry built successfully!")
    print(registry_df.to_string(index=False))

if __name__ == "__main__":
    build_registry()
