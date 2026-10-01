import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import ast
import sqlite3
from sklearn.datasets import load_breast_cancer, load_wine, load_diabetes
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from backend.core.dataset_intelligence.profiler import DatasetProfiler
from backend.core.candidate_generation.generator import CandidateGenerator
from backend.core.candidate_evaluation.evaluator import CandidateEvaluator
from backend.core.preprocessing.preprocessor import BaselinePreprocessor
from backend.models.baseline_model.model import BaselineModel
from backend.core.experience.repository import ExperienceRepository

st.set_page_config(page_title="CAFET", layout="wide")

# ==========================================
# 5. DASHBOARD STRUCTURE
# ==========================================
st.title("CAFET")
st.subheader("Automated Feature Engineering Framework")
st.write("Automatically generate, evaluate, and select useful engineered features under a computational budget.")
st.markdown("**Pipeline:** Dataset ➔ Profile ➔ Generate ➔ Search ➔ Select ➔ Test ➔ Learn")
st.markdown("---")

# ==========================================
# 6. DATASET SECTION
# ==========================================
st.header("1. Dataset")
dataset_choice = st.selectbox(
    "Select Dataset",
    ["diabetes", "breast_cancer", "wine"],
    help="Select a built-in dataset to run the CAFET pipeline."
)

budget_choice = st.selectbox(
    "Evaluation Budget (%)",
    [1, 5, 10, 20, 100],
    index=1,
    help="Percentage of generated candidates to actually evaluate."
)

if st.button(f"Run CAFET on {dataset_choice}"):
    
    # 1. Load Data
    loading_text = st.empty()
    loading_text.success(f"Dataset loaded: {dataset_choice}")
    if dataset_choice == 'diabetes':
        d = load_diabetes(as_frame=True)
        task = 'regression'
        target_col = 'target'
    elif dataset_choice == 'breast_cancer':
        d = load_breast_cancer(as_frame=True)
        task = 'classification'
        target_col = 'target'
    elif dataset_choice == 'wine':
        d = load_wine(as_frame=True)
        task = 'classification'
        target_col = 'target'

    df = d.frame
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split into Train, Val, Test (60%, 20%, 20%)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    # Preprocess
    prep = BaselinePreprocessor()
    X_train_prep = prep.fit_transform(X_train)
    X_val_prep = prep.transform(X_val)
    X_test_prep = prep.transform(X_test)

    # ==========================================
    # 7. DATASET INTELLIGENCE VIEW
    # ==========================================
    st.header("2. Understand: Dataset Intelligence")
    st.write("CAFET profiles the dataset before feature search to characterize the data and determine the available feature-engineering context.")
    profiler = DatasetProfiler(dataset_choice, task_type=task)
    profile = profiler.profile(df, target_col)
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", profile['raw_context']['structural']['n_rows'])
    c2.metric("Original Features", profile['raw_context']['structural']['n_cols'])
    c3.metric("Numeric Features", profile['raw_context']['structural']['n_numeric'])
    c4.metric("Categorical Features", profile['raw_context']['structural']['n_categorical'])
    c5.metric("Task", profile['raw_context']['structural']['task_type'])

    with st.expander("Feature Profiles"):
        feat_df = pd.DataFrame(profile['raw_context']['feature_profiles']).T
        st.dataframe(feat_df)
        
    st.markdown("---")
    
    # ==========================================
    # 8. FEATURE GENERATION VIEW
    # ==========================================
    st.header("3. Generate: Automated Feature Generation")
    st.write("CAFET generates candidate engineered features by applying supported transformations to the original features.")
    generator = CandidateGenerator()
    candidates = generator.generate(X_train_prep.columns.tolist())
    
    transforms = set([c['transform'] for c in candidates])
    
    st.text(f"Original features      {len(X_train_prep.columns)}\n"
            f"Transformations          {len(transforms)}\n"
            f"Generated candidates   {len(candidates)}")
    
    st.write(f"Showing 10 of {len(candidates)} generated candidates.")
    with st.expander("Candidate List (Top 10)"):
        cand_df = pd.DataFrame(candidates[:10])
        st.dataframe(cand_df)
        
    st.markdown("---")

    # ==========================================
    # 9. SEARCH / BUDGET CONTROL & 10. EVALUATION
    # ==========================================
    st.header("4. Search: Budgeted Feature Search")
    st.write("CAFET evaluates only a limited portion of the generated candidate space according to the selected evaluation budget.")
    
    # Evaluate Baseline
    baseline_model = BaselineModel(task=task, random_state=42)
    baseline_model.fit(X_train_prep, y_train)
    baseline_val_score = baseline_model.evaluate(X_val_prep, y_val)
    
    # Setup search
    total_cands = len(candidates)
    allowed_evals = max(1, int((budget_choice / 100.0) * total_cands))
    
    st.text(f"Generated candidates: {total_cands}\n"
            f"Evaluation budget: {budget_choice}%\n"
            f"Allowed evaluations: {allowed_evals}\n"
            f"Evaluated: {allowed_evals}\n"
            f"Not evaluated: {total_cands - allowed_evals}")
    
    evaluator = CandidateEvaluator()
    rng = np.random.default_rng(42)
    search_indices = rng.choice(range(total_cands), size=allowed_evals, replace=False)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    best_candidate = None
    best_candidate_score = -float('inf')
    evals_completed = 0
    
    for i, idx in enumerate(search_indices):
        cand = candidates[idx]
        try:
            train_feat = evaluator.apply_transform(X_train_prep, cand)
            test_feat = evaluator.apply_transform(X_val_prep, cand)
            
            if train_feat.isna().any() or np.isinf(train_feat).any():
                raise ValueError("NaN or Inf generated")
                
            X_train_c = X_train_prep.copy()
            X_val_c = X_val_prep.copy()
            X_train_c[cand['id']] = train_feat
            X_val_c[cand['id']] = test_feat
            
            model = BaselineModel(task=task, random_state=42)
            model.fit(X_train_c, y_train)
            score = model.evaluate(X_val_c, y_val)
            
            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate = cand
                
            evals_completed += 1
            
        except Exception as e:
            # 15. ERROR HANDLING (survive bad candidates)
            st.toast(f"Candidate {cand['id']} failed: {str(e)}")
            evals_completed += 1
            
        progress_bar.progress((i + 1) / allowed_evals)
        status_text.text(f"Evaluating {i+1}/{allowed_evals} ... Best Val Score: {max(best_candidate_score, baseline_val_score):.4f}")
        
    status_text.text(f"Search complete\n\n{evals_completed} / {allowed_evals} candidates evaluated\n{budget_choice}% of the generated candidate space evaluated")
    
    st.markdown("---")

    # ==========================================
    # 11. BEST FEATURE RESULT & 12. BASELINE VS CAFET
    # ==========================================
    if best_candidate is None:
        st.warning("No valid candidate found.")
    else:
        st.header("5. Select: Best Engineered Feature")
        
        st.markdown(f"# `{best_candidate['id']}`")
        st.write(f"**Transformation:** {best_candidate['transform']}")
        st.write(f"**Source feature(s):** {', '.join(best_candidate['features'])}")
        st.write(f"**Candidate ID:** {best_candidate['id']}")
        
        st.write("CAFET selected this engineered feature because it achieved the highest search/validation score among the candidates evaluated under the selected budget. The selected feature was evaluated once on the held-out test set.")
        
        st.subheader("Search / Validation")
        st.write("These scores were used during candidate selection.")
        
        val_utility = best_candidate_score - baseline_val_score
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline Search/Validation Score", f"{baseline_val_score:.4f}")
        c2.metric("Candidate Search/Validation Score", f"{best_candidate_score:.4f}")
        c3.metric("Search Utility / Gain", f"{val_utility:+.4f}")
        
        st.markdown("---")
        
        st.header("6. Test: Final Held-out Test")
        st.write("This test set was not used to select the engineered feature.")
        
        # Final Test Evaluation
        baseline_test_score = baseline_model.evaluate(X_test_prep, y_test)
        
        train_feat = evaluator.apply_transform(X_train_prep, best_candidate)
        test_feat = evaluator.apply_transform(X_test_prep, best_candidate)
        X_train_c = X_train_prep.copy()
        X_test_c = X_test_prep.copy()
        X_train_c[best_candidate['id']] = train_feat
        X_test_c[best_candidate['id']] = test_feat
        
        final_model = BaselineModel(task=task, random_state=42)
        final_model.fit(X_train_c, y_train)
        best_candidate_test_score = final_model.evaluate(X_test_c, y_test)
        
        test_utility = best_candidate_test_score - baseline_test_score
        
        comp_data = {
            "Metric": ["Features", "Search/Validation Score", "Final Test Score"],
            "Baseline": [f"{len(X_train_prep.columns)}", f"{baseline_val_score:.4f}", f"{baseline_test_score:.4f}"],
            "CAFET": [f"{len(X_train_prep.columns) + 1}", f"{best_candidate_score:.4f}", f"{best_candidate_test_score:.4f}"]
        }
        st.table(pd.DataFrame(comp_data).set_index("Metric"))
        
        st.markdown(f"### Final Test Gain\n**{test_utility:+.4f}**")
        
        st.markdown("---")
        st.subheader("CAFET Run Summary")
        st.text(f"Dataset: {dataset_choice}\nGenerated candidates: {total_cands}\nEvaluation budget: {budget_choice}%\nCandidates evaluated: {evals_completed}\nBest engineered feature: {best_candidate['id']}\nSearch utility: {val_utility:+.4f}\nFinal test gain: {test_utility:+.4f}")

    st.markdown("---")

    # ==========================================
    # 13. EXPERIENCE SECTION
    # ==========================================
    st.header("7. Learn: Historical Experience")
    st.write("These are previously recorded CAFET experience records. They are displayed for transparency and are not used to select the current candidate in this demo.")
    db_path = "results/phase4/experience.db"
    if os.path.exists(db_path):
        try:
            repo = ExperienceRepository(db_path)
            # Find experience for this dataset
            # We must map the current dataset choice to the one in DB
            conn = sqlite3.connect(db_path)
            exp_df = pd.read_sql("SELECT candidate_id, transformation, source_features, gain, is_successful FROM experience WHERE source_dataset_id = ? ORDER BY gain DESC LIMIT 10", conn, params=(dataset_choice,))
            conn.close()
            
            if len(exp_df) > 0:
                st.write(f"Showing top historical experience records for **{dataset_choice}**:")
                st.dataframe(exp_df)
            else:
                st.info("No prior experience available for this dataset.")
        except Exception as e:
            st.info(f"No prior experience available. (Error: {str(e)})")
    else:
        st.info("No prior experience available (DB not found).")

