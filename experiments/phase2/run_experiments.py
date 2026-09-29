import os
import sys
import pandas as pd
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sklearn.datasets import load_breast_cancer, load_wine, load_diabetes
from backend.services.experiment_runner.runner import ExperimentRunner

def get_dataset(name):
    if name == 'breast_cancer':
        d = load_breast_cancer(as_frame=True)
        df = d.frame
        task = 'classification'
    elif name == 'wine':
        d = load_wine(as_frame=True)
        df = d.frame
        task = 'classification'
    elif name == 'diabetes':
        d = load_diabetes(as_frame=True)
        df = d.frame
        task = 'regression'
    return df, 'target', task

def run_phase2():
    datasets = ['breast_cancer', 'wine', 'diabetes']
    os.makedirs('results/phase2', exist_ok=True)
    all_summaries = []
    
    for name in datasets:
        print(f"Running baseline on dataset: {name}")
        df, target_col, task = get_dataset(name)
        
        runner = ExperimentRunner(name, df, target_col, task=task, seed=42)
        results = runner.run()
        
        cand_df = pd.DataFrame(results['candidates'])
        cand_df.to_csv(f"results/phase2/{name}_candidates.csv", index=False)
        
        with open(f"results/phase2/{name}_baseline.json", "w") as f:
            json.dump(results['baseline'], f, indent=4)
            
        all_summaries.append(results['summary'])
        
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv("results/phase2/summary.csv", index=False)
    
    with open("results/phase2/summary.json", "w") as f:
        json.dump(all_summaries, f, indent=4)
        
if __name__ == "__main__":
    run_phase2()
