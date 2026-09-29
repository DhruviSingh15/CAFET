"""
Similarity-Weighted Experience Builder — Phase 7.1

For a target dataset D and a candidate C:
  1. Compute similarity(D, Hi) for every historical dataset Hi (Phase 5 engine, LODO-safe).
  2. Retrieve all experience records from H = {H1..Hn} (D excluded).
  3. For each record from Hi with transformation T and arity A:
     - dataset_weight      = similarity(D, Hi)            ∈ (0, 1]
     - candidate_compat    = compatibility(C, record)      ∈ {1.0, 0.5, 0.0}
     - combined_weight     = dataset_weight * candidate_compat
  4. Aggregate weighted_gain statistics.

Candidate compatibility rule (deterministic, interpretable):
  - same transformation AND same arity  → 1.0 (strong compatibility)
  - different transformation, same arity → 0.5 (partial compatibility: same structural complexity)
  - different arity                      → 0.0 (incompatible structural form)

This ensures similarity-weighted experience is candidate-relevant, not just dataset-level averaging.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from backend.core.similarity.similarity_engine import SimilarityEngine, Normalizer
from backend.core.candidate_representation.schema import TRANSFORMATION_PROPERTIES

SIM_HISTORY_DIM = 7  # 7 similarity-weighted features


def _candidate_compat(transform_c: str, arity_c: int, record: Dict[str, Any]) -> float:
    """
    Returns compatibility weight between a candidate (transform, arity) and a historical record.
    """
    t_r = record.get("transformation", "")
    feats_r = record.get("source_features", [])
    arity_r = len(feats_r) if isinstance(feats_r, list) else 1

    if t_r == transform_c and arity_r == arity_c:
        return 1.0  # exact transformation + same arity
    elif arity_r == arity_c:
        return 0.5  # different transform, same structural complexity
    else:
        return 0.0  # incompatible arity (unary vs binary)


def build_similarity_weighted_features(
    target_context_vector: List[float],
    target_dataset_id: str,
    candidate_transform: str,
    candidate_arity: int,
    historical_contexts: Dict[str, List[float]],
    historical_records: List[Dict[str, Any]],  # already excludes target dataset
) -> List[float]:
    """
    Returns a 7-element similarity-weighted experience vector.

    Fields:
        [0] sim_weighted_mean_gain
        [1] sim_weighted_median_gain   (weighted median approximated via sorted list)
        [2] sim_weighted_success_rate
        [3] sim_weighted_best_gain
        [4] sim_weighted_gain_std
        [5] sim_weighted_observation_count   (effective count)
        [6] similarity_weight_sum
    """
    if not historical_records or not historical_contexts:
        return [0.0] * SIM_HISTORY_DIM

    # Compute LODO-safe dataset similarities
    engine = SimilarityEngine()
    sim_results = engine.rank(
        current_dataset_id=target_dataset_id,
        current_vector=target_context_vector,
        historical_contexts=historical_contexts,
    )
    sim_map = {r["dataset_id"]: r["similarity"] for r in sim_results}

    # Collect weighted observations
    weighted_gains = []
    weighted_successes = []
    weight_sum = 0.0

    for rec in historical_records:
        ds_id = rec.get("source_dataset_id", "")
        ds_sim = sim_map.get(ds_id, 0.0)
        compat = _candidate_compat(candidate_transform, candidate_arity, rec)
        combined_w = ds_sim * compat

        if combined_w <= 0.0:
            continue

        gain = rec.get("gain")
        succ = rec.get("is_successful")
        if gain is None:
            continue

        weighted_gains.append((float(gain), combined_w))
        if succ is not None:
            weighted_successes.append((1.0 if succ else 0.0, combined_w))
        weight_sum += combined_w

    if not weighted_gains:
        return [0.0] * SIM_HISTORY_DIM

    gains = np.array([g for g, _ in weighted_gains])
    weights = np.array([w for _, w in weighted_gains])

    # Weighted mean gain
    w_mean = float(np.average(gains, weights=weights))

    # Weighted median (approximated via sorted weighted cumsum)
    sort_idx = np.argsort(gains)
    sorted_g = gains[sort_idx]
    sorted_w = weights[sort_idx]
    cumw = np.cumsum(sorted_w)
    median_idx = np.searchsorted(cumw, cumw[-1] / 2)
    w_median = float(sorted_g[min(median_idx, len(sorted_g) - 1)])

    # Weighted success rate
    if weighted_successes:
        s_vals = np.array([v for v, _ in weighted_successes])
        s_ws = np.array([w for _, w in weighted_successes])
        w_success = float(np.average(s_vals, weights=s_ws))
    else:
        w_success = 0.0

    w_best = float(np.max(gains))
    w_std = float(np.sqrt(np.average((gains - w_mean) ** 2, weights=weights)))
    eff_count = float(len(weighted_gains))

    return [w_mean, w_median, w_success, w_best, w_std, eff_count, float(weight_sum)]
