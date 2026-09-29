import sys
import os
import json
import uuid
import datetime
import pandas as pd
import ast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from backend.core.experience.repository import ExperienceRepository

def main():
    os.makedirs('results/phase4', exist_ok=True)
    repo = ExperienceRepository("results/phase4/experience.db")
    repo.clear()
    
    datasets = ['breast_cancer', 'wine', 'diabetes']
    
    with open('results/phase2/summary.json', 'r') as f:
        p2_summaries = json.load(f)
        
    counts = {}
    for ds_name in datasets:
        with open(f'results/phase3/{ds_name}_profile.json', 'r') as f:
            context = json.load(f)['raw_context']
            
        summary = next(s for s in p2_summaries if s['dataset'] == ds_name)
        baseline_score = summary['baseline_val_score']
        
        df = pd.read_csv(f'results/phase2/{ds_name}_candidates.csv')
        df = df[df['status'] == 'success']
        df = df.sort_values(by='validation_score', ascending=False).reset_index(drop=True)
        
        dataset_count = 0
        for idx, row in df.iterrows():
            cand_id = row['candidate_id']
            val_score = row['validation_score']
            utility = val_score - baseline_score
            is_successful = val_score > baseline_score
            
            try:
                src_feats = ast.literal_eval(row['source_features'])
            except:
                src_feats = [row['source_features']]
                
            exp = {
                'experience_id': str(uuid.uuid4()),
                'source_dataset_id': ds_name,
                'source_dataset_name': ds_name,
                'dataset_context': context,
                'candidate_id': cand_id,
                'transformation': row['transformation'],
                'source_features': src_feats,
                'baseline_score': baseline_score,
                'candidate_score': val_score,
                'utility_score': utility,
                'gain': utility,
                'evaluation_time': row['evaluation_time'],
                'computational_cost': row['evaluation_time'],
                'rank': idx + 1,
                'is_successful': is_successful,
                'random_seed': 42,
                'timestamp': datetime.datetime.now().isoformat(),
                'schema_version': "1.0"
            }
            repo.add_experience(exp)
            dataset_count += 1
            
        counts[ds_name] = dataset_count
        print(f"Imported {dataset_count} experiences for {ds_name}.")
        
    print(f"Total imported: {repo.count()}")
    
    # Example experiences to check
    with open('results/phase4/import_stats.json', 'w') as f:
        json.dump(counts, f, indent=4)
        
    exps = repo.list_experiences(source_dataset_id='diabetes', is_successful=True)
    if exps:
        with open('results/phase4/example_successful.json', 'w') as f:
            json.dump(exps[0], f, indent=4)
            
    exps_unsucc = repo.list_experiences(source_dataset_id='diabetes', is_successful=False)
    if exps_unsucc:
        with open('results/phase4/example_unsuccessful.json', 'w') as f:
            json.dump(exps_unsucc[0], f, indent=4)

if __name__ == '__main__':
    main()
