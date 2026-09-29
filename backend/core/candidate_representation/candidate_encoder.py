"""
Candidate Encoder — Phase 6 core module.

Produces a deterministic fixed-length representation for an individual candidate.

REPRESENTATION STRUCTURE (flat vector):
    [transformation_encoding (12)]
    + [feature_slot_1 (14)]
    + [feature_slot_2 (14)]        ← zeros if unary
    + [dataset_context (14)]       ← Phase 3 numeric_context_vector
    + [history (7)]                ← aggregated Phase 4 experience
    ─────────────────────────────
    TOTAL: 12 + 14 + 14 + 14 + 7 = 61 dimensions

CANDIDATE IDENTITY (string ID):
    Commutative binary ops → features sorted alphabetically.
    Non-commutative binary ops → features in original order.
    Unary ops → single feature name.

Example IDs:
    ADD(age,bmi)     ← sorted: commutative
    SUB(bmi,age)     ← NOT sorted: non-commutative
    LOG(age)
"""

from typing import Optional, List, Dict, Any

from .schema import TRANSFORMATION_PROPERTIES, MAX_ARITY
from .transformation_encoder import encode_transformation, TRANSFORMATION_VECTOR_DIM
from .feature_encoder import encode_feature, encode_empty_feature_slot, FEATURE_SLOT_DIM

DATASET_CONTEXT_DIM = 14   # Phase 3 numeric_context_vector length
HISTORY_DIM = 7             # 7 historical feature fields
TOTAL_DIM = TRANSFORMATION_VECTOR_DIM + (MAX_ARITY * FEATURE_SLOT_DIM) + DATASET_CONTEXT_DIM + HISTORY_DIM
# = 12 + 28 + 14 + 7 = 61


def canonical_candidate_id(transform: str, source_features: List[str]) -> str:
    """
    Produces a deterministic, canonical candidate ID string.

    For commutative binary ops (ADD, MUL): source features sorted alphabetically.
    For non-commutative ops (SUB, DIV) and unary ops: features in original order.
    """
    props = TRANSFORMATION_PROPERTIES.get(transform, {})
    is_commutative = props.get("commutative", False)

    if is_commutative and len(source_features) == 2:
        ordered = sorted(source_features)
    else:
        ordered = list(source_features)

    feature_str = ",".join(ordered)
    return f"{transform}({feature_str})"


def _encode_history(historical_records: List[Dict[str, Any]]) -> list:
    """
    Aggregates Phase 4 experience records for a single candidate.
    Accepts pre-filtered records (target dataset already excluded by caller).

    Returns a 7-element list:
        [has_history, count, mean_gain, median_gain, success_rate, best_gain, gain_std]
    """
    import numpy as np

    if not historical_records:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    gains = [r["gain"] for r in historical_records if r.get("gain") is not None]
    successes = [r["is_successful"] for r in historical_records if r.get("is_successful") is not None]

    count = float(len(historical_records))
    mean_gain = float(np.mean(gains)) if gains else 0.0
    median_gain = float(np.median(gains)) if gains else 0.0
    success_rate = float(np.mean([1.0 if s else 0.0 for s in successes])) if successes else 0.0
    best_gain = float(np.max(gains)) if gains else 0.0
    gain_std = float(np.std(gains)) if len(gains) > 1 else 0.0
    has_history = 1.0

    return [has_history, count, mean_gain, median_gain, success_rate, best_gain, gain_std]


class CandidateEncoder:
    """
    Encodes a single candidate into a deterministic fixed-length representation.

    Parameters
    ----------
    candidate : dict
        Must contain: 'transformation', 'features' (list of source feature names).
    dataset_context_vector : list
        Phase 3 numeric_context_vector (14 floats, fixed ordering).
    feature_profiles : dict
        Phase 3 feature_profiles keyed by feature name.
    historical_records : list, optional
        Phase 4 experience records for this candidate.
        MUST already exclude the target dataset (LODO enforced by caller).
    """

    def encode(
        self,
        candidate: Dict[str, Any],
        dataset_context_vector: List[float],
        feature_profiles: Dict[str, Any],
        historical_records: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:

        transform = candidate["transformation"]
        source_features = candidate["features"]

        # Canonical candidate ID
        cand_id = canonical_candidate_id(transform, source_features)

        # Transformation encoding (12 dims)
        trans_vec = encode_transformation(transform)

        # Feature slot encodings (14 dims each × 2 slots = 28)
        feat_vecs = []
        for i in range(MAX_ARITY):
            if i < len(source_features):
                feat_vecs.extend(encode_feature(source_features[i], feature_profiles))
            else:
                feat_vecs.extend(encode_empty_feature_slot())

        # Dataset context (14 dims)
        ctx_vec = list(dataset_context_vector)[:DATASET_CONTEXT_DIM]
        # Pad if shorter (should not happen with valid Phase 3 output)
        ctx_vec += [0.0] * (DATASET_CONTEXT_DIM - len(ctx_vec))

        # Historical encoding (7 dims)
        hist_vec = _encode_history(historical_records or [])

        # Final flat vector
        vector = trans_vec + feat_vecs + ctx_vec + hist_vec

        return {
            "candidate_id": cand_id,
            "transformation": transform,
            "source_features": source_features,
            "arity": len(source_features),
            "has_history": int(hist_vec[0]),
            "history_count": int(hist_vec[1]),
            "vector": vector,
            "vector_dim": len(vector),
            "schema_version": "1.0",
        }
