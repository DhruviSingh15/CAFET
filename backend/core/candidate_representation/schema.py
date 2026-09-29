"""
Transformation schema: static properties for each supported transformation.
These are NOT learned — they are fixed definitions derived from mathematical semantics.
"""

# Ordered canonical list — this ordering defines the one-hot encoding dimensions.
TRANSFORMATIONS = ["ADD", "SUB", "MUL", "DIV", "LOG", "SQRT", "ABS", "SQUARE"]

TRANSFORMATION_PROPERTIES = {
    "ADD":    {"arity": 2, "commutative": True,  "requires_numeric": True,  "can_produce_nan": False, "can_produce_inf": False},
    "SUB":    {"arity": 2, "commutative": False, "requires_numeric": True,  "can_produce_nan": False, "can_produce_inf": False},
    "MUL":    {"arity": 2, "commutative": True,  "requires_numeric": True,  "can_produce_nan": False, "can_produce_inf": False},
    "DIV":    {"arity": 2, "commutative": False, "requires_numeric": True,  "can_produce_nan": False, "can_produce_inf": True},
    "LOG":    {"arity": 1, "commutative": None,  "requires_numeric": True,  "can_produce_nan": True,  "can_produce_inf": False},
    "SQRT":   {"arity": 1, "commutative": None,  "requires_numeric": True,  "can_produce_nan": True,  "can_produce_inf": False},
    "ABS":    {"arity": 1, "commutative": None,  "requires_numeric": True,  "can_produce_nan": False, "can_produce_inf": False},
    "SQUARE": {"arity": 1, "commutative": None,  "requires_numeric": True,  "can_produce_nan": False, "can_produce_inf": False},
}

# Ordered canonical list for feature types — defines one-hot encoding dimensions.
FEATURE_TYPES = ["numerical", "categorical", "boolean", "datetime", "unknown"]

# Ordered list of numerical feature statistics used in feature encoding.
FEATURE_NUMERIC_STATS = ["missing_ratio", "unique_ratio", "mean", "std", "min", "max", "median", "skewness"]

# Number of source features supported in the fixed-length representation.
# Unary: 1 feature, Binary: 2 features. We always pad to MAX_ARITY slots.
MAX_ARITY = 2
