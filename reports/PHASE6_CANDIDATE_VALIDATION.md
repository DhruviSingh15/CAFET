# Phase 6: Candidate-Level Representation — Validation Report

## 1. Objective
Phase 6 establishes a rich, deterministic, fixed-length representation for individual feature-engineering candidates that explicitly encodes transformation semantics, source feature metadata, dataset context, and historical evidence. This representation is the direct input to Phase 7's utility predictor.

---

## 2. Candidate Schema
Each encoded candidate produces:

```
{
  "candidate_id":   <str>   — canonical string ID
  "transformation": <str>   — e.g. "ADD"
  "source_features": [str]  — e.g. ["bmi", "s5"]
  "arity":          <int>   — 1 (unary) or 2 (binary)
  "has_history":    <int>   — 1 if historical records exist, 0 otherwise
  "history_count":  <int>   — number of LODO-filtered records
  "vector":         [float] — 61-dim flat representation
  "vector_dim":     <int>   — always 61
  "schema_version": "1.0"
}
```

---

## 3. Candidate Identity Rules
The canonical ID is computed by `canonical_candidate_id(transform, features)`:

| Operation | Commutative? | ID Rule |
|---|---|---|
| ADD, MUL | ✓ Yes | features sorted alphabetically |
| SUB, DIV | ✗ No | features in original order |
| LOG, SQRT, ABS, SQUARE | N/A (unary) | single feature |

Examples:
- `ADD(bmi,s5)` and `ADD(s5,bmi)` → canonicalized to `ADD(bmi,s5)`
- `SUB(bmi,age)` ≠ `SUB(age,bmi)` (distinct, preserved)

---

## 4. Transformation Encoding (12 dims)
`[one_hot(8)] + [arity(1), commutative(1), can_produce_nan(1), can_produce_inf(1)]`

Commutative encoding: `1.0=True, 0.0=False, 0.5=N/A (unary)`.
All 8 transformations produce distinct vectors.

---

## 5. Feature Encoding (14 dims × 2 slots = 28 dims)
Each feature slot = `[is_present(1)] + [type_onehot(5)] + [stats(8)]`

Stats: `missing_ratio, unique_ratio, mean, std, min, max, median, skewness`

- `is_present=0.0` for the unused second slot of unary candidates
- Categorical features have zero values for purely numerical stats (mean, std, min, max, median, skewness) — no fabrication

---

## 6. Dataset Context Encoding (14 dims)
Direct pass-through of the Phase 3 `numeric_context_vector` in its documented fixed ordering.

---

## 7. Historical Experience Encoding (7 dims)
`[has_history, count, mean_gain, median_gain, success_rate, best_gain, gain_std]`

Historical records are filtered by the **caller** to exclude the target dataset (LODO).

---

## 8. Cold-Start Handling
When `historical_records = []`:
```
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```
`has_history=0` explicitly flags absence of evidence, preventing confusion with zero-utility evidence.

---

## 9. Representation Dimension

| Component | Dims |
|---|---:|
| Transformation (1-hot + properties) | 12 |
| Feature slot 1 (is_present + type + stats) | 14 |
| Feature slot 2 (padded if unary) | 14 |
| Dataset context (Phase 3 vector) | 14 |
| Historical aggregates | 7 |
| **Total** | **61** |

---

## 10. Collision Tests
All 5 candidate pairs tested produce distinct canonical IDs:
- `ADD(a,b)`, `ADD(a,c)`, `ADD(b,c)`, `SUB(a,b)`, `SUB(b,a)` → 5 distinct IDs ✓

---

## 11. Candidate Counts

| Dataset | Candidates Encoded | Unique IDs | Vector Dims | LODO History Pool |
|---|---:|---:|---:|---:|
| breast_cancer | 880 | 880 | {61} | 584 |
| wine | 364 | 364 | {61} | 1100 |
| diabetes | 220 | 220 | {61} | 1244 |

Within each dataset, every candidate ID is unique. Across datasets, the same `ADD(bmi,s5)` may appear legitimately — distinguished by `source_dataset_id` in Phase 4.

---

## 12. Leakage Validation
LODO exclusion is enforced at the repository query layer before encoding begins. The encoder itself receives already-filtered records. `test_leakage_prevention` explicitly verifies only non-target records influence the history vector.

---

## 13. Determinism
`test_determinism` encodes the same candidate twice and asserts bitwise vector equality. No random state or dictionary ordering affects output.

---

## 14. Serialization
Representation dictionaries are JSON-serializable (all primitives: str, int, list of floats). Round-trip equivalence confirmed.

---

## 15. Example Representations

**Example 1 — Binary candidate with history (diabetes, LODO)**
```
Candidate ID:         ADD(bmi,s5)
Transformation:       ADD
Source features:      ["bmi", "s5"]
Arity:                2
Dataset context:      14-dim Phase 3 vector (diabetes)
has_history:          0  (no exact-match records from wine/breast_cancer)
history_count:        0
vector_dim:           61
```

**Example 2 — Unary candidate, cold start**
```
Candidate ID:         LOG(age)
Transformation:       LOG
Source features:      ["age"]
Arity:                1
Feature slot 2:       padded with zeros (is_present=0.0)
has_history:          0
history_count:        0
vector_dim:           61
```

**Example 3 — Commutative canonicalization**
```
Input:   ADD(["s5", "bmi"])
Output:  ADD(bmi,s5)   ← alphabetically sorted
```

**Example 4 — Non-commutative preservation**
```
SUB(bmi, age) → "SUB(bmi,age)"
SUB(age, bmi) → "SUB(age,bmi)"
These are distinct candidates.
```

**Example 5 — Categorical feature slot**
```
Feature: "cat1" (categorical)
Encoding: [is_present=1.0] + [0,1,0,0,0] (categorical one-hot) + [0.0, 0.01, 0, 0, 0, 0, 0, 0]
             (missing_ratio=0, unique_ratio=0.01, mean/std/... all zero)
```

---

## 16. Limitations
- The experiment datasets (breast_cancer, wine, diabetes) are all exclusively numeric, so the categorical branch of feature encoding is only exercised in unit tests.
- Exact-candidate history matching across datasets is sparse: `ADD(bmi,s5)` only occurs in diabetes, so history pools for cross-dataset experiments are empty for many candidates.
- Similarity-weighted history aggregation is not implemented (deferred to Phase 7+).

---

## 17. Phase Verdict
PASS
