import json
import pandas as pd

with open('results/phase2/summary.json', 'r') as f:
    summaries = json.load(f)

for ds in summaries:
    print(f"==============================")
    print(f"DATASET: {ds['dataset']}")
    print(f"Candidates generated: {ds['candidate_count']}")
    print(f"Successfully evaluated: {ds['successful_evaluations']}")
    print(f"Failed evaluations: {ds['failed_evaluations']}")
    
    df = pd.read_csv(f"results/phase2/{ds['dataset']}_candidates.csv")
    print("\nTransformation types:")
    print(df['transformation'].value_counts())
    
    # Calculate gains directly from JSON
    val_gain = ds['best_candidate_val_score'] - ds['baseline_val_score']
    test_gain = ds['best_candidate_test_score'] - ds['baseline_test_score']
    print(f"\nValidation gain: {val_gain:.5f}")
    print(f"Test gain: {test_gain:.5f}")
    
    if val_gain > 0 and test_gain <= 0:
        print("WARNING: Validation gain > 0 but Test gain <= 0 (Overfitting Validation!)")
