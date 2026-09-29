import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from backend.core.similarity.similarity_engine import SimilarityEngine, Normalizer

def test_normalization():
    norm = Normalizer()
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    norm.fit(data)
    transformed = norm.transform(data)
    
    assert np.allclose(np.mean(transformed, axis=0), 0.0)
    assert np.allclose(np.std(transformed, axis=0), 1.0)
    
def test_normalization_zero_variance():
    norm = Normalizer()
    data = np.array([[1.0, 2.0], [1.0, 4.0]])
    norm.fit(data)
    transformed = norm.transform(data)
    assert np.allclose(transformed[:, 0], [0.0, 0.0]) # std was 1.0 because of zero var guard

def test_distance_and_similarity():
    engine = SimilarityEngine()
    vec_a = np.array([0.0, 0.0])
    vec_b = np.array([3.0, 4.0])
    
    dist = engine.compute_distance(vec_a, vec_b)
    assert dist == 5.0
    
    sim = engine.compute_similarity(vec_a, vec_b)
    assert np.isclose(sim, 1.0 / (1.0 + 5.0))
    
    # Self similarity
    assert engine.compute_distance(vec_a, vec_a) == 0.0
    assert engine.compute_similarity(vec_a, vec_a) == 1.0
    
    # Symmetry
    assert engine.compute_similarity(vec_a, vec_b) == engine.compute_similarity(vec_b, vec_a)

def test_ranking_determinism():
    engine = SimilarityEngine()
    curr = [1.0, 1.0]
    # Provide multiple identical contexts
    hist = {
        'A': [2.0, 2.0],
        'B': [2.0, 2.0],
        'C': [2.0, 2.0]
    }
    
    ranks = engine.rank('target', curr, hist)
    assert [r['dataset_id'] for r in ranks] == ['A', 'B', 'C']

def test_lodo_exclusion():
    engine = SimilarityEngine()
    curr = [1.0, 1.0]
    hist = {
        'target': [1.0, 1.0],
        'A': [2.0, 2.0],
    }
    
    ranks = engine.rank('target', curr, hist)
    assert len(ranks) == 1
    assert ranks[0]['dataset_id'] == 'A'
    
def test_similarity_matrix():
    engine = SimilarityEngine()
    contexts = {
        'ds1': [1.0, 0.0],
        'ds2': [1.0, 0.0],
        'ds3': [0.0, 1.0]
    }
    
    mat, _ = engine.similarity_matrix(contexts)
    # Self sim = 1.0
    assert np.isclose(mat['ds1']['ds1'], 1.0)
    # Symmetry
    assert np.isclose(mat['ds1']['ds3'], mat['ds3']['ds1'])
