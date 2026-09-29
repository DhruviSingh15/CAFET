"""
Feature encoder.

Produces a deterministic fixed-length vector for a single source feature.

Vector = [one_hot_feature_type(5)] + [8 numerical stats]
       = 5 + 8 = 13 dimensions per feature slot.

For unary operations, the second slot is filled with zeros and a special
marker `is_present = 0.0` is set.

Each slot:
  [type_onehot(5)] + [missing_ratio, unique_ratio, mean, std, min, max, median, skewness]
  = 13 dimensions.

Plus `is_present` flag = 1 dimension.

Final slot dim = 13 + 1 = 14.
"""

from .schema import FEATURE_TYPES, FEATURE_NUMERIC_STATS

FEATURE_SLOT_DIM = len(FEATURE_TYPES) + len(FEATURE_NUMERIC_STATS) + 1  # 5 + 8 + 1 = 14


def encode_feature(feature_name: str, feature_profiles: dict) -> list:
    """
    Encodes a single feature using its profile from Phase 3.
    Returns a 14-element float vector.
    is_present = 1.0.
    """
    prof = feature_profiles.get(feature_name, {})
    ftype = prof.get("feature_type", "unknown")

    # One-hot type encoding
    type_onehot = [1.0 if t == ftype else 0.0 for t in FEATURE_TYPES]

    # Numerical stats — None/NaN → 0.0 (no invented values for categoricals)
    stats = []
    for stat in FEATURE_NUMERIC_STATS:
        val = prof.get(stat, None)
        if val is None or (isinstance(val, float) and (val != val)):  # NaN check
            stats.append(0.0)
        else:
            stats.append(float(val))

    is_present = [1.0]
    return is_present + type_onehot + stats  # 1 + 5 + 8 = 14


def encode_empty_feature_slot() -> list:
    """Pad slot for unary candidates (no second feature). is_present = 0."""
    is_present = [0.0]
    type_zeros = [0.0] * len(FEATURE_TYPES)
    stat_zeros = [0.0] * len(FEATURE_NUMERIC_STATS)
    return is_present + type_zeros + stat_zeros  # 1 + 5 + 8 = 14
