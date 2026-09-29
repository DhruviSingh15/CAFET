# Phase 4: Experience Repository Validation Report

## 1. Objective
Phase 4 implements the persistent memory layer for CAFET: the Experience Repository. This module stores exact observations of candidate evaluation success/failure on past datasets alongside the context of those datasets. It validates the capability to precisely query historical experience at the candidate level without introducing artificial leakage or predictive logic.

## 2. Architecture
The architecture comprises `backend/core/experience/repository.py`, leveraging a native `sqlite3` database for persistence. A single table `experience` is employed, enforcing unique constraints on the combination of `(source_dataset_id, candidate_id)` to ensure deduplicated storage while enabling indexing.

## 3. Schema
The `experience` table defines:
- `experience_id` (TEXT, PK)
- `source_dataset_id` (TEXT)
- `source_dataset_name` (TEXT)
- `dataset_context` (TEXT, serialized JSON object)
- `candidate_id` (TEXT, exact signature like `ADD(bmi,s5)`)
- `transformation` (TEXT)
- `source_features` (TEXT, serialized JSON list)
- `baseline_score` (REAL)
- `candidate_score` (REAL)
- `utility_score` (REAL)
- `gain` (REAL)
- `evaluation_time` (REAL)
- `computational_cost` (REAL)
- `rank` (INTEGER)
- `is_successful` (BOOLEAN)
- `random_seed` (INTEGER)
- `timestamp` (TEXT)
- `schema_version` (TEXT)

## 4. Data Import
Data was imported strictly from Phase 2 CSV candidate logs (`results/phase2/*_candidates.csv`) and Phase 3 dataset profiles (`results/phase3/*_profile.json`). A Python script `experiments/phase4/import_phase2.py` merges these inputs synchronously and iterates row-by-row to seed the database without executing models anew. 

## 5. Experience Counts
| Dataset       | Phase 2 Candidates | Experiences Stored |
| ------------- | -----------------: | -----------------: |
| breast_cancer |                880 |                880 |
| wine          |                364 |                364 |
| diabetes      |                220 |                220 |
| Total         |               1464 |               1464 |

## 6. Candidate-Level Integrity
The repository separates candidates based on strict string ID representations (`ADD(bmi,s5)` vs `ADD(age,sex)`). Tests in `test_phase4.py::test_candidate_identity` enforce that candidates with the same transformation but different source features remain distinct entities in `sqlite3`.

## 7. Context Preservation
The Phase 3 dataset context dictionary is converted to a string and safely stored in the `dataset_context` JSON field. The `.get_experience()` and `.list_experiences()` methods automatically deserialize it upon retrieval without data degradation, ensuring historical context is perfectly coupled with historical outcome.

## 8. Utility Integrity
Utility is exclusively defined as `utility_score = candidate_validation_score - baseline_validation_score`. Test dataset scores are entirely omitted from the schema.

## 9. Dataset Isolation
The API supports an `exclude_dataset_id` argument. `test_phase4.py::test_filtering` explicitly demonstrates that historical experience from a given target dataset (e.g., `diabetes`) can be surgically filtered out to simulate unseen LODO (Leave-One-Dataset-Out) evaluation.

## 10. Persistence
SQLite ensures atomic storage. Custom serializers transparently wrap Python `<->` JSON translations for arrays/dicts. `test_insert_and_read` asserts read/write symmetry.

## 11. Tests
A complete suite passed in `tests/test_phase4.py`:
- `test_insert_and_read`
- `test_duplicate_handling`
- `test_filtering`
- `test_candidate_identity`

## 12. Limitations
- Single-table monolithic architecture.
- JSON blobs for `dataset_context` cannot be easily queried at the individual meta-feature level natively via SQL without extraction.

## 13. Phase Verdict
PASS
