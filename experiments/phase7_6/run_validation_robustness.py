import os
import sys
import json
import sqlite3
import ast
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from backend.core.candidate_representation.candidate_encoder import CandidateEncoder
from backend.core.experience.repository import ExperienceRepository
from backend.core.preprocessing.preprocessor import BaselinePreprocessor
from backend.core.candidate_evaluation.evaluator import CandidateEvaluator
from backend.models.baseline_model.model import BaselineModel
from sklearn.datasets import load_breast_cancer, load_wine, load_diabetes, fetch_california_housing, load_iris, fetch_openml

DATASETS = [
    "breast_cancer", "wine", "diabetes", "california_housing", "iris", 
    "titanic", "credit_g", "blood_transfusion", "vehicle", "spambase"
]
BUDGET_PCTS = [0.05, 0.10, 0.20]
SEEDS = [42, 7, 13, 99, 123]
OUT_DIR = "../../results/phase7_6"
os.makedirs(OUT_DIR, exist_ok=True)

repo = ExperienceRepository("../../results/phase4/experience.db")

def load_profile(ds_name):
    with open(f"../../results/phase3/{ds_name}_profile.json", "r") as f:
        d = json.load(f)
    return d["numeric_context_vector"], d["raw_context"]["feature_profiles"]

def get_candidates(ds_name):
    conn = sqlite3.connect("../../results/phase4/experience.db")
    df = pd.read_sql("SELECT candidate_id, transformation, source_features FROM experience WHERE source_dataset_id = ?", conn, params=(ds_name,))
    conn.close()
    return df.drop_duplicates(subset=["candidate_id"]).to_dict(orient="records")

def get_dataset_df(name):
    if name == 'breast_cancer':
        d = load_breast_cancer(as_frame=True)
        return d.frame, 'target', 'classification'
    elif name == 'wine':
        d = load_wine(as_frame=True)
        return d.frame, 'target', 'classification'
    elif name == 'diabetes':
        d = load_diabetes(as_frame=True)
        return d.frame, 'target', 'regression'
    elif name == 'california_housing':
        d = fetch_california_housing(as_frame=True)
        return d.frame.sample(n=5000, random_state=42), 'MedHouseVal', 'regression'
    elif name == 'iris':
        d = load_iris(as_frame=True)
        return d.frame, 'target', 'classification'
    elif name == 'titanic':
        d = fetch_openml(data_id=40945, as_frame=True, parser='auto')
        return d.frame, 'survived', 'classification'
    elif name == 'credit_g':
        d = fetch_openml(data_id=31, as_frame=True, parser='auto')
        return d.frame, 'class', 'classification'
    elif name == 'blood_transfusion':
        d = fetch_openml(data_id=1464, as_frame=True, parser='auto')
        return d.frame, 'Class', 'classification'
    elif name == 'vehicle':
        d = fetch_openml(data_id=54, as_frame=True, parser='auto')
        return d.frame, 'Class', 'classification'
    elif name == 'spambase':
        d = fetch_openml(data_id=44, as_frame=True, parser='auto')
        return d.frame, 'class', 'classification'
    raise ValueError(name)

def eval_cand(cand, X_fit, y_fit, X_eval, y_eval, baseline_score, task, seed):
    evaluator = CandidateEvaluator()
    try:
        c_dict = {'transform': cand['transformation'], 'features': ast.literal_eval(cand['source_features'])}
        if type(c_dict['features']) is not list:
            c_dict['features'] = [c_dict['features']]
            
        train_feat = evaluator.apply_transform(X_fit, c_dict)
        eval_feat = evaluator.apply_transform(X_eval, c_dict)
        
        if train_feat.isna().any() or np.isinf(train_feat).any():
            return -999.0
            
        X_fit_c = X_fit.copy()
        X_eval_c = X_eval.copy()
        X_fit_c[cand['candidate_id']] = train_feat
        X_eval_c[cand['candidate_id']] = eval_feat
        
        model = BaselineModel(task=task, random_state=seed)
        model.fit(X_fit_c, y_fit)
        score = model.evaluate(X_eval_c, y_eval)
        return score - baseline_score
    except Exception as e:
        return -999.0

def main():
    encoder = CandidateEncoder()
    results = []
    
    for ds in DATASETS:
        print(f"Processing {ds}...")
        candidates = get_candidates(ds)
        if len(candidates) == 0: continue
        
        df, target_col, task = get_dataset_df(ds)
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        ctx_vec, feat_profiles = load_profile(ds)
        hist_recs = repo.list_experiences(exclude_dataset_id=ds)
        hist_idx = {}
        for r in hist_recs:
            hist_idx.setdefault(r["candidate_id"], []).append(r)
            
        X_reps = []
        for c in candidates:
            try:
                src_feats = ast.literal_eval(c["source_features"])
            except:
                src_feats = [c["source_features"]]
            rep = encoder.encode(
                {"transformation": c["transformation"], "features": src_feats},
                ctx_vec, feat_profiles, hist_idx.get(c["candidate_id"], [])
            )
            X_reps.append(rep["vector"])
        X_reps = np.array(X_reps)
        N = len(candidates)
        
        for seed in SEEDS:
            # Create splits
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
            X_fit, X_evals, y_fit, y_evals = train_test_split(X_temp, y_temp, test_size=0.375, random_state=seed) # 30% of 80% = 24% eval ? wait, we want 50/15/15/20. So 50/(50+15+15) = 50/80 = 62.5% train, 37.5% evals
            X_search_val, X_indep_eval, y_search_val, y_indep_eval = train_test_split(X_evals, y_evals, test_size=0.5, random_state=seed)
            
            prep = BaselinePreprocessor()
            X_fit = prep.fit_transform(X_fit)
            X_search_val = prep.transform(X_search_val)
            X_indep_eval = prep.transform(X_indep_eval)
            X_test = prep.transform(X_test)
            
            base_model = BaselineModel(task=task, random_state=seed)
            base_model.fit(X_fit, y_fit)
            base_search_score = base_model.evaluate(X_search_val, y_search_val)
            base_indep_score = base_model.evaluate(X_indep_eval, y_indep_eval)
            base_test_score = base_model.evaluate(X_test, y_test)
            
            rng = np.random.default_rng(seed)
            max_b = max([max(5, int(p * N)) for p in BUDGET_PCTS])
            
            for strategy in ["Random", "Adaptive"]:
                evaluated_indices = []
                unevaluated_indices = list(range(N))
                observed_utilities = []
                
                # Initialization
                init_size = min(5, N)
                init_idx = rng.choice(unevaluated_indices, init_size, replace=False).tolist()
                for idx in init_idx:
                    u = eval_cand(candidates[idx], X_fit, y_fit, X_search_val, y_search_val, base_search_score, task, seed)
                    evaluated_indices.append(idx)
                    observed_utilities.append(u)
                    unevaluated_indices.remove(idx)
                    
                model = RandomForestRegressor(n_estimators=20, random_state=seed)
                
                for step in range(init_size, max_b):
                    if len(unevaluated_indices) == 0: break
                    
                    if strategy == "Random":
                        nxt = rng.choice(unevaluated_indices)
                    elif strategy == "Adaptive":
                        model.fit(X_reps[evaluated_indices], observed_utilities)
                        preds = []
                        for tree in model.estimators_:
                            preds.append(tree.predict(X_reps[unevaluated_indices]))
                        mean_pred = np.mean(preds, axis=0)
                        std_pred = np.std(preds, axis=0)
                        acq = mean_pred + 1.0 * std_pred
                        nxt = unevaluated_indices[np.argmax(acq)]
                        
                    u = eval_cand(candidates[nxt], X_fit, y_fit, X_search_val, y_search_val, base_search_score, task, seed)
                    evaluated_indices.append(nxt)
                    observed_utilities.append(u)
                    unevaluated_indices.remove(nxt)
                    
                # Store results for each budget
                for p in BUDGET_PCTS:
                    b = max(5, int(p * N))
                    if b > len(evaluated_indices): b = len(evaluated_indices)
                    
                    eval_subset = evaluated_indices[:b]
                    utils_subset = observed_utilities[:b]
                    
                    best_idx_in_subset = np.argmax(utils_subset)
                    best_cand_idx = eval_subset[best_idx_in_subset]
                    best_search_utility = utils_subset[best_idx_in_subset]
                    
                    # Compute independent and test utility
                    indep_u = eval_cand(candidates[best_cand_idx], X_fit, y_fit, X_indep_eval, y_indep_eval, base_indep_score, task, seed)
                    test_u = eval_cand(candidates[best_cand_idx], X_fit, y_fit, X_test, y_test, base_test_score, task, seed)
                    
                    results.append({
                        "dataset": ds,
                        "strategy": strategy,
                        "seed": seed,
                        "budget_pct": p,
                        "budget_abs": b,
                        "best_cand_id": candidates[best_cand_idx]["candidate_id"],
                        "transformation": candidates[best_cand_idx]["transformation"],
                        "search_utility": best_search_utility,
                        "indep_utility": indep_u,
                        "test_utility": test_u
                    })

    df_res = pd.DataFrame(results)
    df_res.to_csv(f"{OUT_DIR}/robustness_results.csv", index=False)
    print("Done")

if __name__ == "__main__":
    main()
