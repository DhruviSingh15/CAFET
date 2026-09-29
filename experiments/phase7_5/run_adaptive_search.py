import os
import sys
import json
import sqlite3
import ast
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from backend.core.candidate_representation.candidate_encoder import CandidateEncoder, canonical_candidate_id
from backend.core.experience.repository import ExperienceRepository
from backend.core.similarity.similarity_engine import SimilarityEngine
from backend.services.experiment_runner.runner import ExperimentRunner
from sklearn.datasets import load_breast_cancer, load_wine, load_diabetes, fetch_california_housing, load_iris, fetch_openml

DATASETS = [
    "breast_cancer", "wine", "diabetes", "california_housing", "iris", 
    "titanic", "credit_g", "blood_transfusion", "vehicle", "spambase"
]
BUDGET_PCTS = [0.01, 0.02, 0.05, 0.10, 0.20]
SEEDS = [42, 7, 13, 99, 123]
OUT_DIR = "../../results/phase7_5"
os.makedirs(OUT_DIR, exist_ok=True)

repo = ExperienceRepository("../../results/phase4/experience.db")

def load_profile(ds_name):
    with open(f"../../results/phase3/{ds_name}_profile.json", "r") as f:
        d = json.load(f)
    return d["numeric_context_vector"], d["raw_context"]["feature_profiles"]

def get_candidates(ds_name):
    conn = sqlite3.connect("../../results/phase4/experience.db")
    df = pd.read_sql("SELECT candidate_id, transformation, source_features, gain FROM experience WHERE source_dataset_id = ?", conn, params=(ds_name,))
    conn.close()
    return df.to_dict(orient="records")

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
        # sample down to 5000 as done in phase 7.2
        df = d.frame.sample(n=5000, random_state=42)
        return df, 'target', 'regression'
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

def build_priors(target_ds):
    conn = sqlite3.connect("../../results/phase4/experience.db")
    df = pd.read_sql("SELECT source_dataset_id, transformation, gain FROM experience WHERE source_dataset_id != ?", conn, params=(target_ds,))
    conn.close()
    
    # transform freq prior (mean gain per transformation across all history)
    t_freq = df.groupby("transformation")["gain"].mean().to_dict()
    
    # sim prior
    contexts = {}
    for ds in DATASETS:
        if ds != target_ds:
            vec, _ = load_profile(ds)
            contexts[ds] = vec
    target_vec, _ = load_profile(target_ds)
    
    engine = SimilarityEngine()
    ranks = engine.rank(target_ds, target_vec, contexts)
    sim_weights = {r["dataset_id"]: r["similarity"] for r in ranks}
    
    # similarity-weighted average
    df["weight"] = df["source_dataset_id"].map(sim_weights)
    df["weighted_gain"] = df["gain"] * df["weight"]
    sim_prior = (df.groupby("transformation")["weighted_gain"].sum() / df.groupby("transformation")["weight"].sum()).to_dict()
    
    return t_freq, sim_prior

def run_test_eval(ds_name, best_candidate_id):
    df, target_col, task = get_dataset_df(ds_name)
    runner = ExperimentRunner(ds_name, df, target_col, task=task, seed=42)
    # just run it to get the test score
    res = runner.run()
    # evaluate specific candidate
    for c in res['candidates']:
        if c['candidate_id'] == best_candidate_id:
            # Phase 2 runner doesn't output test score for all. We can just use the runner's baseline test score.
            pass
    # We will compute test score properly in a later step
    return res['summary']['baseline_test_score'], 0.0

def main():
    encoder = CandidateEncoder()
    results = []
    traces = []
    
    for ds in DATASETS:
        print(f"Processing {ds}...")
        candidates = get_candidates(ds)
        if len(candidates) == 0: continue
        
        ctx_vec, feat_profiles = load_profile(ds)
        hist_recs = repo.list_experiences(exclude_dataset_id=ds)
        hist_idx = {}
        for r in hist_recs:
            hist_idx.setdefault(r["candidate_id"], []).append(r)
            
        X_all = []
        y_all = []
        for c in candidates:
            try:
                src_feats = ast.literal_eval(c["source_features"])
            except:
                src_feats = [c["source_features"]]
            
            rep = encoder.encode(
                {"transformation": c["transformation"], "features": src_feats},
                ctx_vec, feat_profiles, hist_idx.get(c["candidate_id"], [])
            )
            X_all.append(rep["vector"])
            y_all.append(c["gain"])
            
        X_all = np.array(X_all)
        y_all = np.array(y_all)
        
        N = len(candidates)
        oracle_best_utility = np.max(y_all)
        
        t_freq, sim_prior = build_priors(ds)
        
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            
            # Budgets
            max_b = max([max(5, int(p * N)) for p in BUDGET_PCTS])
            
            for strategy in ["Random", "TransformFreq", "SimPrior", "Adaptive"]:
                evaluated = []
                unevaluated = list(range(N))
                
                # Initialization (5 random)
                init_size = min(5, N)
                init_idx = rng.choice(unevaluated, init_size, replace=False).tolist()
                for idx in init_idx:
                    evaluated.append(idx)
                    unevaluated.remove(idx)
                    
                model = RandomForestRegressor(n_estimators=20, random_state=seed)
                
                for step in range(init_size, max_b):
                    if len(unevaluated) == 0: break
                    
                    if strategy == "Random":
                        nxt = rng.choice(unevaluated)
                    elif strategy == "TransformFreq":
                        # pick unevaluated with max prior
                        scores = [t_freq.get(candidates[i]["transformation"], 0) for i in unevaluated]
                        nxt = unevaluated[np.argmax(scores)]
                    elif strategy == "SimPrior":
                        scores = [sim_prior.get(candidates[i]["transformation"], 0) for i in unevaluated]
                        nxt = unevaluated[np.argmax(scores)]
                    elif strategy == "Adaptive":
                        # fit model
                        model.fit(X_all[evaluated], y_all[evaluated])
                        # predict
                        preds = []
                        for tree in model.estimators_:
                            preds.append(tree.predict(X_all[unevaluated]))
                        mean_pred = np.mean(preds, axis=0)
                        std_pred = np.std(preds, axis=0)
                        # UCB acquisition, beta=1.0
                        acq = mean_pred + 1.0 * std_pred
                        nxt = unevaluated[np.argmax(acq)]
                        
                    evaluated.append(nxt)
                    unevaluated.remove(nxt)
                    
                # Evaluate at budgets
                for p in BUDGET_PCTS:
                    b = max(5, int(p * N))
                    if b > len(evaluated): b = len(evaluated)
                    
                    eval_subset = evaluated[:b]
                    best_u = np.max(y_all[eval_subset])
                    regret = oracle_best_utility - best_u
                    
                    results.append({
                        "dataset": ds,
                        "strategy": strategy,
                        "seed": seed,
                        "budget_pct": p,
                        "budget_abs": b,
                        "best_utility": best_u,
                        "regret": regret
                    })

    df_res = pd.DataFrame(results)
    df_res.to_csv(f"{OUT_DIR}/adaptive_results.csv", index=False)
    print("Done")

if __name__ == "__main__":
    main()
