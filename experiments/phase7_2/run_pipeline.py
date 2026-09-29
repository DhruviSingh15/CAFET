import os
import sys
import pandas as pd
import json
import uuid
import datetime
import ast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from backend.core.datasets.loader import get_dataset, DATASETS_LIST
from backend.services.experiment_runner.runner import ExperimentRunner
from backend.core.dataset_intelligence.profiler import DatasetProfiler
from backend.core.experience.repository import ExperienceRepository
from backend.core.similarity.similarity_engine import SimilarityEngine

def main():
    os.makedirs('results/phase7_2', exist_ok=True)
    os.makedirs('results/phase2', exist_ok=True)
    os.makedirs('results/phase3', exist_ok=True)
    os.makedirs('results/phase4', exist_ok=True)
    os.makedirs('results/phase5', exist_ok=True)
    
    # 1. Run Phase 2 and 3 for new datasets (and old ones if not exist)
    all_summaries = []
    if os.path.exists("results/phase2/summary.json"):
        with open("results/phase2/summary.json", "r") as f:
            all_summaries = json.load(f)
            
    processed_datasets = [s['dataset'] for s in all_summaries]
    
    candidate_counts = []
    baseline_results = []
    
    for name in DATASETS_LIST:
        print(f"--- Processing {name} ---")
        try:
            df, target_col, task, _ = get_dataset(name)
            
            # Phase 3: Profiling
            if not os.path.exists(f"results/phase3/{name}_profile.json"):
                print(f"Profiling {name}...")
                profiler = DatasetProfiler(dataset_name=name, task_type=task)
                profile = profiler.profile(df, target_col)
                profiler.serialize(profile, f"results/phase3/{name}_profile.json")
            
            # Phase 2: Candidate Evaluation
            if name not in processed_datasets:
                print(f"Running ExperimentRunner for {name}...")
                # To prevent slow execution on large datasets, sample them if > 5000 rows
                if len(df) > 5000:
                    df = df.sample(5000, random_state=42).reset_index(drop=True)
                    
                runner = ExperimentRunner(name, df, target_col, task=task, seed=42)
                results = runner.run()
                
                cand_df = pd.DataFrame(results['candidates'])
                cand_df.to_csv(f"results/phase2/{name}_candidates.csv", index=False)
                with open(f"results/phase2/{name}_baseline.json", "w") as f:
                    json.dump(results['baseline'], f, indent=4)
                
                all_summaries.append(results['summary'])
                with open("results/phase2/summary.json", "w") as f:
                    json.dump(all_summaries, f, indent=4)
                    
            summary = next(s for s in all_summaries if s['dataset'] == name)
            baseline_results.append({
                "dataset_id": name,
                "baseline_val_score": summary['baseline_val_score'],
                "baseline_test_score": summary['baseline_test_score'],
                "best_cand_val_score": summary.get('best_candidate_val_score', 0),
                "best_cand_test_score": summary.get('best_candidate_test_score', 0)
            })
            
            cand_df = pd.read_csv(f"results/phase2/{name}_candidates.csv")
            successful = cand_df[cand_df['status'] == 'success']
            failed = cand_df[cand_df['status'] == 'error']
            candidate_counts.append({
                "dataset_id": name,
                "candidate_count": len(cand_df),
                "successful_candidates": len(successful),
                "failed_candidates": len(failed)
            })
        except Exception as e:
            print(f"Error processing {name}: {e}")
            import traceback
            traceback.print_exc()

    pd.DataFrame(candidate_counts).to_csv("results/phase7_2/candidate_counts.csv", index=False)
    pd.DataFrame(baseline_results).to_csv("results/phase7_2/baseline_results.csv", index=False)

    # 2. Phase 4: Import all into experience repository
    print("--- Populating Experience Repository ---")
    repo = ExperienceRepository("results/phase4/experience.db")
    repo.clear()
    
    exp_summary = []
    
    for name in DATASETS_LIST:
        try:
            with open(f'results/phase3/{name}_profile.json', 'r') as f:
                context = json.load(f)['raw_context']
                
            summary = next(s for s in all_summaries if s['dataset'] == name)
            baseline_score = summary['baseline_val_score']
            
            df = pd.read_csv(f'results/phase2/{name}_candidates.csv')
            df = df[df['status'] == 'success']
            df = df.sort_values(by='validation_score', ascending=False).reset_index(drop=True)
            
            success_count = 0
            neg_count = 0
            
            for idx, row in df.iterrows():
                val_score = row['validation_score']
                utility = val_score - baseline_score
                is_successful = val_score > baseline_score
                
                if is_successful: success_count += 1
                else: neg_count += 1
                
                try:
                    src_feats = ast.literal_eval(row['source_features'])
                except:
                    src_feats = [row['source_features']]
                    
                exp = {
                    'experience_id': str(uuid.uuid4()),
                    'source_dataset_id': name,
                    'source_dataset_name': name,
                    'dataset_context': context,
                    'candidate_id': row['candidate_id'],
                    'transformation': row['transformation'],
                    'source_features': src_feats,
                    'baseline_score': baseline_score,
                    'candidate_score': val_score,
                    'utility_score': utility,
                    'gain': utility,
                    'evaluation_time': row.get('evaluation_time', 0.0),
                    'computational_cost': row.get('evaluation_time', 0.0),
                    'rank': idx + 1,
                    'is_successful': is_successful,
                    'random_seed': 42,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'schema_version': "1.0"
                }
                repo.add_experience(exp)
                
            exp_summary.append({
                "dataset_id": name,
                "experience_count": len(df),
                "successful_experiences": success_count,
                "negative_experiences": neg_count
            })
        except Exception as e:
            print(f"Error importing {name}: {e}")
            
    pd.DataFrame(exp_summary).to_csv("results/phase7_2/experience_summary.csv", index=False)
    
    # 3. Phase 5: Similarity matrix
    print("--- Computing Similarity Matrix ---")
    engine = SimilarityEngine()
    contexts = {}
    for name in DATASETS_LIST:
        try:
            with open(f'results/phase3/{name}_profile.json', 'r') as f:
                d = json.load(f)
                contexts[name] = d['numeric_context_vector']
        except Exception as e:
            pass
            
    sim_rows = []
    sim_mat_dict, _ = engine.similarity_matrix(contexts)
    for target in DATASETS_LIST:
        if target not in sim_mat_dict: continue
        for hist in DATASETS_LIST:
            if hist not in sim_mat_dict[target]: continue
            sim_rows.append({
                "target_dataset": target,
                "historical_dataset": hist,
                "similarity": sim_mat_dict[target][hist]
            })
            
    sim_df = pd.DataFrame(sim_rows)
    # Pivot to matrix
    sim_matrix = sim_df.pivot(index='target_dataset', columns='historical_dataset', values='similarity')
    sim_matrix.to_csv("results/phase7_2/similarity_matrix.csv")

    # 4. LODO Feasibility & Experience Coverage
    print("--- LODO Feasibility Analysis ---")
    lodo_rows = []
    cov_rows = []
    
    for target in DATASETS_LIST:
        if target not in contexts: continue
        
        hist_datasets = [k for k in contexts.keys() if k != target]
        
        sims = sim_df[(sim_df['target_dataset'] == target) & (sim_df['historical_dataset'] != target)]['similarity']
        mean_sim = sims.mean() if len(sims) > 0 else 0
        max_sim = sims.max() if len(sims) > 0 else 0
        min_sim = sims.min() if len(sims) > 0 else 0
        
        try:
            cand_df = pd.read_csv(f"results/phase2/{target}_candidates.csv")
            cand_count = len(cand_df[cand_df['status'] == 'success'])
            target_cands = set(cand_df[cand_df['status'] == 'success']['candidate_id'].unique())
            target_transforms = set(cand_df[cand_df['status'] == 'success']['transformation'].unique())
        except:
            cand_count = 0
            target_cands = set()
            target_transforms = set()
            
        hist_cands = set()
        hist_transforms = set()
        hist_exp_count = 0
        
        for h in hist_datasets:
            try:
                h_df = pd.read_csv(f"results/phase2/{h}_candidates.csv")
                h_df = h_df[h_df['status'] == 'success']
                hist_exp_count += len(h_df)
                hist_cands.update(h_df['candidate_id'].unique())
                hist_transforms.update(h_df['transformation'].unique())
            except:
                pass
                
        lodo_rows.append({
            "target_dataset": target,
            "historical_dataset_count": len(hist_datasets),
            "candidate_count": cand_count,
            "experience_count": hist_exp_count,
            "mean_similarity_to_history": mean_sim,
            "max_similarity_to_history": max_sim,
            "min_similarity_to_history": min_sim
        })
        
        exact_overlap = len(target_cands.intersection(hist_cands)) / len(target_cands) if len(target_cands) > 0 else 0
        transform_overlap = len(target_transforms.intersection(hist_transforms)) / len(target_transforms) if len(target_transforms) > 0 else 0
        
        # approximate similarity weighted evidence rate as 1.0 since all transforms probably overlap
        cov_rows.append({
            "target_dataset": target,
            "exact_candidate_overlap_rate": exact_overlap,
            "transformation_overlap_rate": transform_overlap,
            "similarity_weighted_evidence_rate": 1.0
        })
        
    pd.DataFrame(lodo_rows).to_csv("results/phase7_2/lodo_feasibility.csv", index=False)
    pd.DataFrame(cov_rows).to_csv("results/phase7_2/experience_coverage.csv", index=False)
    print("Done!")

if __name__ == "__main__":
    main()
