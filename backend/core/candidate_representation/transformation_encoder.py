"""
Transformation encoder.

Produces a deterministic fixed-length vector for a given transformation name.
Encoding = [one_hot(8)] + [arity, commutative, can_produce_nan, can_produce_inf]
         = 8 + 4 = 12 dimensions

Commutative: 1.0 if True, 0.0 if False, 0.5 if N/A (unary has no concept of commutativity).
"""

from .schema import TRANSFORMATIONS, TRANSFORMATION_PROPERTIES

TRANSFORMATION_VECTOR_DIM = len(TRANSFORMATIONS) + 4  # 8 one-hot + 4 properties


def encode_transformation(transform_name: str) -> list:
    """Returns a deterministic 12-element float vector for the given transformation."""
    # One-hot
    onehot = [1.0 if t == transform_name else 0.0 for t in TRANSFORMATIONS]

    props = TRANSFORMATION_PROPERTIES.get(transform_name, {
        "arity": 0, "commutative": None, "can_produce_nan": False, "can_produce_inf": False
    })

    arity = float(props.get("arity", 0))
    comm_raw = props.get("commutative", None)
    commutative = 1.0 if comm_raw is True else (0.0 if comm_raw is False else 0.5)
    can_nan = 1.0 if props.get("can_produce_nan", False) else 0.0
    can_inf = 1.0 if props.get("can_produce_inf", False) else 0.0

    return onehot + [arity, commutative, can_nan, can_inf]
