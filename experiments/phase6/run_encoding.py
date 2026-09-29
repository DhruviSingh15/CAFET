"""
Phase 6 validation experiment.

For each dataset:
  1. Load Phase 2 candidates.
  2. Load Phase 3 dataset context and feature profiles.
  3. Load Phase 4 historical experience (excluding target dataset).
  4. Encode all candidates.
  5. Verify uniqueness and dimensions.
  6. Save sample representations.
"""

import os
import sys
import json
import ast
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from backend.core.candidate_representation.candidate_encoder import (
    CandidateEncoder, canonical_candidate_id, TOTAL_DIM,
)
from backend.core.experience.repository import ExperienceRepository


DATASETS = ["breast_cancer", "wine", "diabetes"]
REPO_PATH = "results/phase4/experience.db"


def load_phase3_profile(dataset_name):
    with open(f"results/phase3/{dataset_name}_profile.json", "r") as f:
        data = json.load(f)
    return data["numeric_context_vector"], data["raw_context"]["feature_profiles"]


def main():
    os.makedirs("results/phase6", exist_ok=True)
    repo = ExperienceRepository(REPO_PATH)
    encoder = CandidateEncoder()
    
    all_stats = {}
    sample_representations = []

    for target_ds in DATASETS:
        ctx_vec, feat_profiles = load_phase3_profile(target_ds)

        # Load Phase 2 candidates for this dataset
        cand_df = pd.read_csv(f"results/phase2/{target_ds}_candidates.csv")
        cand_df = cand_df[cand_df["status"] == "success"]

        # Load historical experiences EXCLUDING the target dataset (LODO)
        historical = repo.list_experiences(exclude_dataset_id=target_ds)

        # Build index: (candidate_id) → list of records from other datasets
        hist_index = {}
        for rec in historical:
            cid = rec["candidate_id"]
            if cid not in hist_index:
                hist_index[cid] = []
            hist_index[cid].append(rec)

        # Encode all candidates
        encoded_ids = []
        vectors = []

        for _, row in cand_df.iterrows():
            transform = row["transformation"]
            try:
                src_feats = ast.literal_eval(row["source_features"])
            except Exception:
                src_feats = [row["source_features"]]

            cand = {"transformation": transform, "features": src_feats}
            cid = canonical_candidate_id(transform, src_feats)
            hist_recs = hist_index.get(cid, [])

            rep = encoder.encode(cand, ctx_vec, feat_profiles, hist_recs)
            encoded_ids.append(rep["candidate_id"])
            vectors.append(rep["vector"])

            # Collect sample representations (first 5 per dataset)
            if len(sample_representations) < 5 * len(DATASETS):
                sample_representations.append({
                    "dataset": target_ds,
                    "candidate_id": rep["candidate_id"],
                    "transformation": transform,
                    "source_features": src_feats,
                    "has_history": rep["has_history"],
                    "history_count": rep["history_count"],
                    "vector_dim": rep["vector_dim"],
                })

        n_total = len(encoded_ids)
        n_unique = len(set(encoded_ids))
        all_dim = set(len(v) for v in vectors)

        all_stats[target_ds] = {
            "total_candidates": n_total,
            "unique_candidate_ids": n_unique,
            "vector_dims": list(all_dim),
            "historical_pool_size": len(historical),
        }

        print(f"[{target_ds}]  total={n_total}  unique={n_unique}  dim={all_dim}  hist_pool={len(historical)}")

    # Save stats and samples
    with open("results/phase6/encoding_stats.json", "w") as f:
        json.dump(all_stats, f, indent=4)
    with open("results/phase6/sample_representations.json", "w") as f:
        json.dump(sample_representations, f, indent=4)

    print(f"\nExpected TOTAL_DIM = {TOTAL_DIM}")
    print("Done. Outputs saved to results/phase6/")


if __name__ == "__main__":
    main()
