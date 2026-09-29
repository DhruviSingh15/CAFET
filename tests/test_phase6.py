import pytest
import sys
import os
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from backend.core.candidate_representation.candidate_encoder import (
    CandidateEncoder, canonical_candidate_id, TOTAL_DIM,
)
from backend.core.candidate_representation.transformation_encoder import encode_transformation
from backend.core.candidate_representation.feature_encoder import encode_feature, encode_empty_feature_slot, FEATURE_SLOT_DIM


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def context_vector():
    return [float(i) for i in range(14)]


@pytest.fixture
def feature_profiles():
    return {
        "age":  {"feature_type": "numerical", "missing_ratio": 0.0, "unique_ratio": 0.3, "mean": 0.1, "std": 0.05, "min": -0.1, "max": 0.1, "median": 0.0, "skewness": 0.2},
        "bmi":  {"feature_type": "numerical", "missing_ratio": 0.0, "unique_ratio": 0.4, "mean": 0.0, "std": 0.05, "min": -0.09, "max": 0.17, "median": -0.007, "skewness": 0.6},
        "sex":  {"feature_type": "numerical", "missing_ratio": 0.0, "unique_ratio": 0.01, "mean": 0.0, "std": 0.05, "min": -0.04, "max": 0.05, "median": -0.04, "skewness": 0.1},
        "cat1": {"feature_type": "categorical", "missing_ratio": 0.0, "unique_ratio": 0.01, "cardinality": 3},
    }


@pytest.fixture
def encoder():
    return CandidateEncoder()


# ─── Candidate identity tests ──────────────────────────────────────────────────

def test_canonical_id_commutative():
    """ADD is commutative — sorted features."""
    assert canonical_candidate_id("ADD", ["bmi", "age"]) == "ADD(age,bmi)"
    assert canonical_candidate_id("ADD", ["age", "bmi"]) == "ADD(age,bmi)"
    assert canonical_candidate_id("MUL", ["bmi", "age"]) == "MUL(age,bmi)"


def test_canonical_id_non_commutative():
    """SUB and DIV are non-commutative — order preserved."""
    assert canonical_candidate_id("SUB", ["bmi", "age"]) == "SUB(bmi,age)"
    assert canonical_candidate_id("SUB", ["age", "bmi"]) == "SUB(age,bmi)"
    # These must differ
    assert canonical_candidate_id("SUB", ["a", "b"]) != canonical_candidate_id("SUB", ["b", "a"])


def test_canonical_id_unary():
    assert canonical_candidate_id("LOG", ["age"]) == "LOG(age)"
    assert canonical_candidate_id("SQRT", ["bmi"]) == "SQRT(bmi)"


def test_collision_prevention():
    """Distinct candidate pairs must produce distinct canonical IDs."""
    ids = {
        canonical_candidate_id("ADD", ["a", "b"]),
        canonical_candidate_id("ADD", ["a", "c"]),
        canonical_candidate_id("ADD", ["b", "c"]),
        canonical_candidate_id("SUB", ["a", "b"]),
        canonical_candidate_id("SUB", ["b", "a"]),
    }
    assert len(ids) == 5  # All five are distinct


# ─── Transformation encoding tests ────────────────────────────────────────────

def test_transformation_encoding_distinct():
    """Every transformation must produce a different one-hot vector."""
    transforms = ["ADD", "SUB", "MUL", "DIV", "LOG", "SQRT", "ABS", "SQUARE"]
    vecs = [tuple(encode_transformation(t)) for t in transforms]
    assert len(set(vecs)) == len(transforms)


def test_transformation_encoding_dim():
    vec = encode_transformation("ADD")
    assert len(vec) == 12  # 8 one-hot + 4 properties


def test_transformation_commutative_flag():
    add_vec = encode_transformation("ADD")
    sub_vec = encode_transformation("SUB")
    # commutative flag is at index 9 (8 one-hot + arity + commutative)
    assert add_vec[9] == 1.0  # commutative
    assert sub_vec[9] == 0.0  # not commutative


# ─── Feature encoding tests ────────────────────────────────────────────────────

def test_feature_slot_dim(feature_profiles):
    vec = encode_feature("age", feature_profiles)
    assert len(vec) == FEATURE_SLOT_DIM  # 14


def test_empty_slot_dim():
    vec = encode_empty_feature_slot()
    assert len(vec) == FEATURE_SLOT_DIM
    assert vec[0] == 0.0  # is_present = 0


def test_numerical_feature_type_encoding(feature_profiles):
    vec = encode_feature("age", feature_profiles)
    assert vec[0] == 1.0  # is_present
    assert vec[1] == 1.0  # numerical is first in FEATURE_TYPES


def test_categorical_feature_no_numeric_stats(feature_profiles):
    """
    Categorical features must NOT have fabricated purely-numerical stats
    (mean, std, min, max, median, skewness). However missing_ratio and
    unique_ratio are universal meta-statistics valid for all feature types.
    FEATURE_NUMERIC_STATS order: missing_ratio(0), unique_ratio(1), mean(2),
    std(3), min(4), max(5), median(6), skewness(7).
    Indices 6-13 in full vec: skip is_present(1)+type(5)+missing_ratio(1)+unique_ratio(1)=8.
    """
    vec = encode_feature("cat1", feature_profiles)
    # Purely numerical stats start at index 8 (is_present + 5-type-oh + 2 universal stats)
    numerical_only_stats = vec[8:]
    assert all(s == 0.0 for s in numerical_only_stats)


# ─── Full candidate encoding tests ────────────────────────────────────────────

def test_total_dim(encoder, context_vector, feature_profiles):
    cand = {"transformation": "ADD", "features": ["age", "bmi"]}
    rep = encoder.encode(cand, context_vector, feature_profiles)
    assert rep["vector_dim"] == TOTAL_DIM == 61


def test_determinism(encoder, context_vector, feature_profiles):
    cand = {"transformation": "LOG", "features": ["age"]}
    r1 = encoder.encode(cand, context_vector, feature_profiles)
    r2 = encoder.encode(cand, context_vector, feature_profiles)
    assert r1["vector"] == r2["vector"]
    assert r1["candidate_id"] == r2["candidate_id"]


def test_cold_start(encoder, context_vector, feature_profiles):
    """Candidate with no historical records must still produce a valid vector."""
    cand = {"transformation": "SQRT", "features": ["bmi"]}
    rep = encoder.encode(cand, context_vector, feature_profiles, historical_records=[])
    assert rep["has_history"] == 0
    assert rep["history_count"] == 0
    assert rep["vector_dim"] == TOTAL_DIM
    # No NaNs
    assert all(v == v for v in rep["vector"])  # NaN != NaN


def test_with_history(encoder, context_vector, feature_profiles):
    cand = {"transformation": "ADD", "features": ["age", "bmi"]}
    history = [
        {"gain": 0.1, "is_successful": True},
        {"gain": -0.05, "is_successful": False},
        {"gain": 0.2, "is_successful": True},
    ]
    rep = encoder.encode(cand, context_vector, feature_profiles, historical_records=history)
    assert rep["has_history"] == 1
    assert rep["history_count"] == 3


def test_leakage_prevention(encoder, context_vector, feature_profiles):
    """
    Historical records filtered for target dataset BEFORE encoding is the caller's
    responsibility. This test verifies that if only non-target records are passed,
    those are what appear in the representation.
    """
    cand = {"transformation": "LOG", "features": ["age"]}
    # Simulate: only wine records passed (diabetes excluded externally)
    wine_records = [{"gain": 0.05, "is_successful": True}]
    rep = encoder.encode(cand, context_vector, feature_profiles, historical_records=wine_records)
    assert rep["history_count"] == 1


def test_distinct_candidates_produce_distinct_vectors(encoder, context_vector, feature_profiles):
    """ADD(age,bmi) and ADD(age,sex) must produce different vectors."""
    c1 = {"transformation": "ADD", "features": ["age", "bmi"]}
    c2 = {"transformation": "ADD", "features": ["age", "sex"]}
    r1 = encoder.encode(c1, context_vector, feature_profiles)
    r2 = encoder.encode(c2, context_vector, feature_profiles)
    assert r1["candidate_id"] != r2["candidate_id"]
    assert r1["vector"] != r2["vector"]
