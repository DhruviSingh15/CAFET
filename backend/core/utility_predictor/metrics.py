"""
Evaluation metrics for Phase 7 utility predictor validation.

All metrics are computed at the candidate level.
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import ndcg_score


def spearman(actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) < 2:
        return float("nan")
    rho, _ = spearmanr(actual, predicted)
    return float(rho)


def pearson(actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) < 2:
        return float("nan")
    r, _ = pearsonr(actual, predicted)
    return float(r)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def precision_at_k(actual: np.ndarray, predicted: np.ndarray, k: int) -> float:
    """
    P@K = |{top-K actual} ∩ {top-K predicted}| / K
    'Top-K actual' = indices with highest actual utility.
    """
    k = min(k, len(actual))
    if k == 0:
        return float("nan")
    top_k_actual = set(np.argsort(actual)[::-1][:k])
    top_k_pred = set(np.argsort(predicted)[::-1][:k])
    return float(len(top_k_actual & top_k_pred) / k)


def ndcg_at_k(actual: np.ndarray, predicted: np.ndarray, k: int) -> float:
    """
    NDCG@K using sklearn's ndcg_score.
    Relevance = actual utility shifted to non-negative (min-shift).
    """
    k = min(k, len(actual))
    if k == 0:
        return float("nan")
    # Shift to non-negative for NDCG (requires non-negative relevance)
    shift = min(0.0, float(actual.min()))
    relevance = (actual - shift).reshape(1, -1)
    scores = predicted.reshape(1, -1)
    return float(ndcg_score(relevance, scores, k=k))


def all_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "spearman": spearman(actual, predicted),
        "pearson": pearson(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mae": mae(actual, predicted),
        "precision_at_10": precision_at_k(actual, predicted, 10),
        "precision_at_25": precision_at_k(actual, predicted, 25),
        "precision_at_50": precision_at_k(actual, predicted, 50),
        "ndcg_at_10": ndcg_at_k(actual, predicted, 10),
        "ndcg_at_25": ndcg_at_k(actual, predicted, 25),
        "ndcg_at_50": ndcg_at_k(actual, predicted, 50),
    }
