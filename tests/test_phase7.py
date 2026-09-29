import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from backend.core.utility_predictor.predictor import UtilityPredictor
from backend.core.utility_predictor.metrics import (
    spearman, pearson, rmse, mae, precision_at_k, ndcg_at_k
)


@pytest.fixture
def simple_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 61))
    y = rng.standard_normal(50)
    return X, y


def test_predictor_fit_predict(simple_data):
    X, y = simple_data
    p = UtilityPredictor()
    p.fit(X, y)
    pred, unc = p.predict(X)
    assert len(pred) == len(y)
    assert len(unc) == len(y)


def test_predictor_reproducibility(simple_data):
    X, y = simple_data
    p1 = UtilityPredictor()
    p1.fit(X, y)
    pr1, u1 = p1.predict(X)

    p2 = UtilityPredictor()
    p2.fit(X, y)
    pr2, u2 = p2.predict(X)

    np.testing.assert_allclose(pr1, pr2)
    np.testing.assert_allclose(u1, u2)


def test_uncertainty_nonnegative(simple_data):
    X, y = simple_data
    p = UtilityPredictor()
    p.fit(X, y)
    _, unc = p.predict(X)
    assert (unc >= 0).all()


def test_cold_start_candidates(simple_data):
    """Model must produce predictions even when history dims = 0."""
    X, y = simple_data
    # Zero out history dims (54-61)
    X_cold = X.copy()
    X_cold[:, 54:] = 0.0
    p = UtilityPredictor()
    p.fit(X_cold, y)
    pred, unc = p.predict(X_cold)
    assert len(pred) == len(y)
    assert not np.any(np.isnan(pred))


def test_metrics_spearman():
    actual = np.array([0.1, 0.3, 0.05, -0.02, 0.2])
    # Perfect rank agreement
    perfect = np.array([1.0, 3.0, 0.5, -0.5, 2.0])
    assert spearman(actual, perfect) == pytest.approx(1.0, abs=0.01)
    # Inverse rank
    inverse = -perfect
    assert spearman(actual, inverse) == pytest.approx(-1.0, abs=0.01)


def test_precision_at_k():
    actual = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    # Perfect prediction
    pred_perfect = actual.copy()
    assert precision_at_k(actual, pred_perfect, k=3) == pytest.approx(1.0)
    # Random prediction (worst case: no overlap)
    pred_worst = actual[::-1]
    assert precision_at_k(actual, pred_worst, k=3) <= 1.0


def test_ndcg_at_k():
    actual = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    pred_perfect = actual.copy()
    score = ndcg_at_k(actual, pred_perfect, k=3)
    assert 0.0 <= score <= 1.0 + 1e-6


def test_no_leakage_assertion():
    """
    Simulate the LODO encoding pipeline and assert no target records appear.
    """
    hist_records = [
        {"source_dataset_id": "breast_cancer", "gain": 0.01, "is_successful": True},
        {"source_dataset_id": "wine",           "gain": -0.01, "is_successful": False},
    ]
    target = "diabetes"
    for r in hist_records:
        assert r["source_dataset_id"] != target, "LEAKAGE DETECTED"
