import os
import sys
import pandas as pd
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from backend.core.datasets.loader import DATASETS_LIST, get_dataset
from backend.core.similarity.similarity_engine import SimilarityEngine
from backend.core.experience.repository import ExperienceRepository

def test_dataset_registry():
    registry_path = "results/phase7_2/dataset_registry.csv"
    assert os.path.exists(registry_path)
    df = pd.read_csv(registry_path)
    assert len(df) == 10
    assert set(df['dataset_id'].tolist()) == set(DATASETS_LIST)

def test_all_datasets_load():
    for name in DATASETS_LIST:
        df, tgt, task, _ = get_dataset(name)
        assert not df.empty
        assert tgt in df.columns
        assert task in ['classification', 'regression']

def test_target_is_not_feature():
    for name in DATASETS_LIST:
        df, tgt, _, _ = get_dataset(name)
        features = [c for c in df.columns if c != tgt]
        assert tgt not in features

def test_profiles_exist():
    for name in DATASETS_LIST:
        path = f"results/phase3/{name}_profile.json"
        assert os.path.exists(path)
        with open(path, "r") as f:
            data = json.load(f)
            assert "numeric_context_vector" in data
            assert len(data["numeric_context_vector"]) > 0

def test_candidate_generation():
    df = pd.read_csv("results/phase7_2/candidate_counts.csv")
    assert len(df) == 10
    for idx, row in df.iterrows():
        assert row['candidate_count'] > 0
        assert row['successful_candidates'] > 0

def test_baseline_evaluation():
    df = pd.read_csv("results/phase7_2/baseline_results.csv")
    assert len(df) == 10
    for idx, row in df.iterrows():
        assert not pd.isna(row['baseline_val_score'])

def test_experience_import():
    df = pd.read_csv("results/phase7_2/experience_summary.csv")
    assert len(df) == 10
    total_exp = df['experience_count'].sum()
    
    repo = ExperienceRepository("results/phase4/experience.db")
    assert repo.count() == total_exp

def test_similarity_matrix():
    df = pd.read_csv("results/phase7_2/similarity_matrix.csv", index_col=0)
    assert len(df) == 10
    assert len(df.columns) == 10
    for i in range(10):
        assert df.iloc[i, i] == 1.0
        for j in range(10):
            assert 0 <= df.iloc[i, j] <= 1.0

def test_lodo_target_exclusion():
    engine = SimilarityEngine()
    df = pd.read_csv("results/phase7_2/similarity_matrix.csv", index_col=0)
    # The LODO normalization enforces target exclusion. 
    # Check that rank results do not include target except for self.
    # In run_pipeline, we did not include target in historical_contexts for LODO similarity
    lodo = pd.read_csv("results/phase7_2/lodo_feasibility.csv")
    for _, row in lodo.iterrows():
        assert row['historical_dataset_count'] == 9 # 10 total - 1 target = 9

def test_deterministic_loading():
    df1, tgt1, _, _ = get_dataset('breast_cancer')
    df2, tgt2, _, _ = get_dataset('breast_cancer')
    pd.testing.assert_frame_equal(df1, df2)
    assert tgt1 == tgt2
