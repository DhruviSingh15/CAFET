import os
import sys
import json
import sqlite3
import ast
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

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
OUT_DIR = "../../results/phase7_7"
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
        return d.frame.sample(n=5000, random_state=42).reset_index(drop=True), 'MedHouseVal', 'regression'
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

def eval_cand_outer(cand, X_fit, y_fit, X_eval, y_eval, baseline_score, task, seed):
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

def eval_cand_cv(cand, X_train, y_train, task, seed, base_scores):
    if task == 'classification':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
    evaluator = CandidateEvaluator()
    try:
        c_dict = {'transform': cand['transformation'], 'features': ast.literal_eval(cand['source_features'])}
        if type(c_dict['features']) is not list:
            c_dict['features'] = [c_dict['features']]
    except:
        c_dict = {'transform': cand['transformation'], 'features': [cand['source_features']]}

    fold_utilities = []
    
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)

    X_train_reset = X_train.reset_index(drop=True)
    y_train_reset = y_train.reset_index(drop=True)

    fold_idx = 0
    for train_idx, val_idx in cv.split(X_train_reset, y_train_reset):
        X_fold_train = X_train_reset.iloc[train_idx]
        y_fold_train = y_train_reset.iloc[train_idx]
        X_fold_val = X_train_reset.iloc[val_idx]
        y_fold_val = y_train_reset.iloc[val_idx]
        
        prep = BaselinePreprocessor()
        try:
            X_fold_train_p = prep.fit_transform(X_fold_train)
            X_fold_val_p = prep.transform(X_fold_val)
        except Exception:
            fold_utilities.append(-999.0)
            continue
            
        base_score = base_scores[fold_idx] if fold_idx < len(base_scores) else -999.0
            
        try:
            train_feat = evaluator.apply_transform(X_fold_train_p, c_dict)
            val_feat = evaluator.apply_transform(X_fold_val_p, c_dict)
            if train_feat.isna().any() or np.isinf(train_feat).any():
                fold_utilities.append(-999.0)
                continue
                
            X_fold_train_c = X_fold_train_p.copy()
            X_fold_val_c = X_fold_val_p.copy()
            X_fold_train_c[cand['candidate_id']] = train_feat
            X_fold_val_c[cand['candidate_id']] = val_feat
            
            c_model = BaselineModel(task=task, random_state=seed)
            c_model.fit(X_fold_train_c, y_fold_train)
            c_score = c_model.evaluate(X_fold_val_c, y_fold_val)
            
            fold_utilities.append(c_score - base_score)
        except:
            fold_utilities.append(-999.0)
            
        fold_idx += 1
            
    if all(u == -999.0 for u in fold_utilities):
        return -999.0, [-999.0]*5
        
    valid_utils = [u for u in fold_utilities if u != -999.0]
    return np.mean(valid_utils), fold_utilities

def main():
    encoder = CandidateEncoder()
    results = []
    
    # Full experiment
    datasets_to_run = DATASETS
    seeds_to_run = SEEDS

    for ds in datasets_to_run:
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
        
        for seed in seeds_to_run:
            # 62.5% Search/Training, 18.75% Independent, 18.75% Final Test
            # X_temp is 81.25%, X_test is 18.75%
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1875, random_state=seed)
            # We want 18.75% out of original for Independent. That is 18.75 / 81.25 = 0.230769 of X_temp
            # So X_train = 62.5 / 81.25 = 0.76923
            X_train, X_indep, y_train, y_indep = train_test_split(X_temp, y_temp, test_size=(18.75/81.25), random_state=seed)
            
            prep = BaselinePreprocessor()
            X_train_prep = prep.fit_transform(X_train)
            X_indep_prep = prep.transform(X_indep)
            X_test_prep = prep.transform(X_test)
            
            base_outer_model = BaselineModel(task=task, random_state=seed)
            base_outer_model.fit(X_train_prep, y_train)
            base_indep_score = base_outer_model.evaluate(X_indep_prep, y_indep)
            base_test_score = base_outer_model.evaluate(X_test_prep, y_test)
            
            rng = np.random.default_rng(seed)
            max_b = max([max(5, int(p * N)) for p in BUDGET_PCTS])
            
            # Precompute Baseline CV scores
            if task == 'classification':
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            else:
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                
            base_cv_scores = []
            X_train_reset = X_train.reset_index(drop=True)
            y_train_reset = y_train.reset_index(drop=True)
            
            for train_idx, val_idx in cv.split(X_train_reset, y_train_reset):
                try:
                    X_fold_train = X_train_reset.iloc[train_idx]
                    y_fold_train = y_train_reset.iloc[train_idx]
                    X_fold_val = X_train_reset.iloc[val_idx]
                    y_fold_val = y_train_reset.iloc[val_idx]
                    prep = BaselinePreprocessor()
                    X_fold_train_p = prep.fit_transform(X_fold_train)
                    X_fold_val_p = prep.transform(X_fold_val)
                    b_model = BaselineModel(task=task, random_state=seed)
                    b_model.fit(X_fold_train_p, y_fold_train)
                    base_cv_scores.append(b_model.evaluate(X_fold_val_p, y_fold_val))
                except:
                    base_cv_scores.append(-999.0)
            
            cv_cache = {}
            
            for strategy in ["Random", "Adaptive"]:
                evaluated_indices = []
                unevaluated_indices = list(range(N))
                observed_utilities = []
                fold_utilities_log = []
                
                init_size = min(5, N)
                init_idx = rng.choice(unevaluated_indices, init_size, replace=False).tolist()
                for idx in init_idx:
                    if idx not in cv_cache:
                        mean_u, folds_u = eval_cand_cv(candidates[idx], X_train, y_train, task, seed, base_cv_scores)
                        cv_cache[idx] = (mean_u, folds_u)
                    mean_u, folds_u = cv_cache[idx]
                    
                    evaluated_indices.append(idx)
                    observed_utilities.append(mean_u)
                    fold_utilities_log.append(folds_u)
                    unevaluated_indices.remove(idx)
                    
                model = RandomForestRegressor(n_estimators=20, random_state=seed)
                
                for step in range(init_size, max_b):
                    if len(unevaluated_indices) == 0: break
                    
                    if strategy == "Random":
                        nxt = rng.choice(unevaluated_indices)
                    elif strategy == "Adaptive":
                        valid_mask = np.array(observed_utilities) > -900
                        if valid_mask.sum() > 0:
                            model.fit(X_reps[np.array(evaluated_indices)[valid_mask]], np.array(observed_utilities)[valid_mask])
                            preds = []
                            for tree in model.estimators_:
                                preds.append(tree.predict(X_reps[unevaluated_indices]))
                            mean_pred = np.mean(preds, axis=0)
                            std_pred = np.std(preds, axis=0)
                            acq = mean_pred + 1.0 * std_pred
                            nxt = unevaluated_indices[np.argmax(acq)]
                        else:
                            nxt = rng.choice(unevaluated_indices)
                        
                    if nxt not in cv_cache:
                        mean_u, folds_u = eval_cand_cv(candidates[nxt], X_train, y_train, task, seed, base_cv_scores)
                        cv_cache[nxt] = (mean_u, folds_u)
                    mean_u, folds_u = cv_cache[nxt]
                    
                    evaluated_indices.append(nxt)
                    observed_utilities.append(mean_u)
                    fold_utilities_log.append(folds_u)
                    unevaluated_indices.remove(nxt)
                    
                for p in BUDGET_PCTS:
                    b = max(5, int(p * N))
                    if b > len(evaluated_indices): b = len(evaluated_indices)
                    
                    eval_subset = evaluated_indices[:b]
                    utils_subset = observed_utilities[:b]
                    folds_subset = fold_utilities_log[:b]
                    
                    best_idx_in_subset = np.argmax(utils_subset)
                    best_cand_idx = eval_subset[best_idx_in_subset]
                    best_search_utility = utils_subset[best_idx_in_subset]
                    best_fold_utils = folds_subset[best_idx_in_subset]
                    
                    if best_search_utility <= -900:
                        indep_u = 0.0
                        test_u = 0.0
                    else:
                        indep_u = eval_cand_outer(candidates[best_cand_idx], X_train_prep, y_train, X_indep_prep, y_indep, base_indep_score, task, seed)
                        test_u = eval_cand_outer(candidates[best_cand_idx], X_train_prep, y_train, X_test_prep, y_test, base_test_score, task, seed)
                    
                    valid_folds = [u for u in best_fold_utils if u > -900]
                    cv_std = np.std(valid_folds) if len(valid_folds)>0 else 0.0
                    cv_se = cv_std / np.sqrt(len(valid_folds)) if len(valid_folds)>0 else 0.0
                    
                    results.append({
                        "dataset": ds,
                        "strategy": strategy,
                        "seed": seed,
                        "budget_pct": p,
                        "budget_abs": b,
                        "best_cand_id": candidates[best_cand_idx]["candidate_id"],
                        "transformation": candidates[best_cand_idx]["transformation"],
                        "search_utility_cv": best_search_utility,
                        "cv_std": cv_std,
                        "cv_se": cv_se,
                        "indep_utility": indep_u,
                        "test_utility": test_u
                    })

            # Save intermediate results after each seed
            df_res = pd.DataFrame(results)
            df_res.to_csv(f"{OUT_DIR}/cv_robustness_results.csv", index=False)
            
    print("Done")

if __name__ == "__main__":
    main()
