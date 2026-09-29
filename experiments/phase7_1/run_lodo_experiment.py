"""
Phase 7.1 LODO Experiment — Similarity-Weighted Experience Correction.

Four ablation models:
  A: Transformation-only (12 dims)           — Phase 7 baseline
  B: Original Phase 7 (61 dims)              — Phase 7 baseline
  C: Original 61 dims + sim-weighted 7 dims  — Phase 7.1 full model
  D: Candidate/context 54 dims (no exact-hist) + sim-weighted 7 dims = 61 dims

Same candidates, same seeds, same LODO splits as Phase 7.
"""

import os
import sys
import json
import ast
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from backend.core.candidate_representation.candidate_encoder import (
    CandidateEncoder, canonical_candidate_id, TOTAL_DIM,
)
from backend.core.candidate_representation.transformation_encoder import encode_transformation
from backend.core.utility_predictor.predictor import UtilityPredictor, RF_CONFIG
from backend.core.utility_predictor.metrics import all_metrics, precision_at_k, ndcg_at_k
from backend.core.utility_predictor.similarity_experience import (
    build_similarity_weighted_features, SIM_HISTORY_DIM,
)
from backend.core.experience.repository import ExperienceRepository

DATASETS = ["breast_cancer", "wine", "diabetes"]
REPO_PATH = "results/phase4/experience.db"
OUT_DIR = "results/phase7_1"

LODO_SPLITS = [
    {"target": "diabetes",      "train": ["breast_cancer", "wine"]},
    {"target": "wine",          "train": ["breast_cancer", "diabetes"]},
    {"target": "breast_cancer", "train": ["wine", "diabetes"]},
]


def load_profile(ds_name):
    with open(f"results/phase3/{ds_name}_profile.json") as f:
        d = json.load(f)
    return d["numeric_context_vector"], d["raw_context"]["feature_profiles"]


def load_candidates(ds_name):
    df = pd.read_csv(f"results/phase2/{ds_name}_candidates.csv")
    return df[df["status"] == "success"].copy()


def build_hist_index(repo, exclude_dataset):
    recs = repo.list_experiences(exclude_dataset_id=exclude_dataset)
    idx = {}
    for r in recs:
        cid = r["candidate_id"]
        idx.setdefault(cid, []).append(r)
    return idx, recs  # idx by candidate_id, and full flat list


def encode_dataset(ds_name, encoder, hist_index, all_hist_recs,
                   target_ds, train_contexts):
    ctx_vec, feat_profiles = load_profile(ds_name)
    cand_df = load_candidates(ds_name)
    rows = []

    for _, row in cand_df.iterrows():
        transform = row["transformation"]
        try:
            src_feats = ast.literal_eval(row["source_features"])
        except Exception:
            src_feats = [str(row["source_features"])]

        cid = canonical_candidate_id(transform, src_feats)
        arity = len(src_feats)

        # LODO exact history
        hist_recs = hist_index.get(cid, [])
        # Leakage assertion
        for r in hist_recs:
            assert r["source_dataset_id"] != target_ds, f"LEAKAGE in {cid}"

        rep = encoder.encode(
            {"transformation": transform, "features": src_feats},
            ctx_vec, feat_profiles, hist_recs
        )

        # Similarity-weighted experience (uses ALL non-target records)
        sim_feats = build_similarity_weighted_features(
            target_context_vector=ctx_vec,
            target_dataset_id=target_ds,
            candidate_transform=transform,
            candidate_arity=arity,
            historical_contexts=train_contexts,
            historical_records=all_hist_recs,
        )

        # Leakage: sim_feats must not use target records
        for r in all_hist_recs:
            assert r["source_dataset_id"] != target_ds

        rows.append({
            "dataset": ds_name,
            "candidate_id": cid,
            "transform": transform,
            "arity": arity,
            "vector_61": rep["vector"],                 # Model B
            "vector_tonly": encode_transformation(transform),  # Model A
            "sim_feats": sim_feats,                     # 7-dim extension
            "actual_utility": float(row["gain"]),
            "has_history": rep["has_history"],
            "sim_weight_sum": sim_feats[6],
        })
    return rows


def random_preds(n, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.05, 0.05, n)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    repo = ExperienceRepository(REPO_PATH)
    encoder = CandidateEncoder()

    all_pred_rows = []
    all_metric_rows = []
    all_importance_rows = []
    all_coldstart_rows = []
    all_sim_summary_rows = []

    for split in LODO_SPLITS:
        target = split["target"]
        train_datasets = split["train"]
        print(f"\n{'='*60}")
        print(f"Target: {target}  Train: {train_datasets}")

        # Load train context vectors for similarity engine
        train_contexts = {ds: load_profile(ds)[0] for ds in train_datasets}

        # History excluding target
        hist_index, all_hist_recs = build_hist_index(repo, exclude_dataset=target)

        # Encode training and target candidates
        train_rows = []
        for ds in train_datasets:
            train_rows.extend(encode_dataset(
                ds, encoder, hist_index, all_hist_recs, target, train_contexts
            ))

        target_rows = encode_dataset(
            target, encoder, hist_index, all_hist_recs, target, train_contexts
        )

        # Build numpy arrays for 4 models
        def stack(rows, key):
            return np.array([r[key] for r in rows], dtype=float)

        # Model A: transformation-only (12)
        Xtr_A = stack(train_rows, "vector_tonly")
        Xte_A = stack(target_rows, "vector_tonly")

        # Model B: original Phase 7 61-dim
        Xtr_B = stack(train_rows, "vector_61")
        Xte_B = stack(target_rows, "vector_61")

        # Model C: 61 + 7 sim-weighted = 68
        Xtr_C = np.hstack([Xtr_B, stack(train_rows, "sim_feats")])
        Xte_C = np.hstack([Xte_B, stack(target_rows, "sim_feats")])

        # Model D: 54 (no exact-hist) + 7 sim-weighted = 61
        # exact history occupies dims 54:61 in the 61-dim vector
        Xtr_D = np.hstack([Xtr_B[:, :54], stack(train_rows, "sim_feats")])
        Xte_D = np.hstack([Xte_B[:, :54], stack(target_rows, "sim_feats")])

        y_tr = np.array([r["actual_utility"] for r in train_rows])
        y_te = np.array([r["actual_utility"] for r in target_rows])

        models = {
            "A_TransformOnly": (Xtr_A, Xte_A),
            "B_Phase7":        (Xtr_B, Xte_B),
            "C_Phase7_1_Full": (Xtr_C, Xte_C),
            "D_SimOnly":       (Xtr_D, Xte_D),
        }

        model_preds = {}
        model_uncs = {}

        for mname, (Xtr, Xte) in models.items():
            pred_model = UtilityPredictor()
            pred_model.fit(Xtr, y_tr)
            y_pred, y_unc = pred_model.predict(Xte)
            model_preds[mname] = y_pred
            model_uncs[mname] = y_unc

            m = all_metrics(y_te, y_pred)
            m["target"] = target
            m["model"] = mname
            m["pred_std"] = float(y_pred.std())
            m["actual_std"] = float(y_te.std())
            m["std_ratio"] = float(y_pred.std() / (y_te.std() + 1e-12))
            all_metric_rows.append(m)

            # Feature importance (grouped)
            imp = pred_model.feature_importances()
            n = len(imp)
            if mname == "A_TransformOnly":
                groups = {"transformation": list(range(n))}
            elif mname == "B_Phase7":
                groups = {
                    "transformation": list(range(0, 12)),
                    "feature_slot_1": list(range(12, 26)),
                    "feature_slot_2": list(range(26, 40)),
                    "dataset_context": list(range(40, 54)),
                    "exact_history": list(range(54, 61)),
                }
            elif mname == "C_Phase7_1_Full":
                groups = {
                    "transformation": list(range(0, 12)),
                    "feature_slot_1": list(range(12, 26)),
                    "feature_slot_2": list(range(26, 40)),
                    "dataset_context": list(range(40, 54)),
                    "exact_history": list(range(54, 61)),
                    "sim_history": list(range(61, 68)),
                }
            else:  # D
                groups = {
                    "transformation": list(range(0, 12)),
                    "feature_slot_1": list(range(12, 26)),
                    "feature_slot_2": list(range(26, 40)),
                    "dataset_context": list(range(40, 54)),
                    "sim_history": list(range(54, 61)),
                }

            for grp, idxs in groups.items():
                valid_idxs = [i for i in idxs if i < len(imp)]
                all_importance_rows.append({
                    "target": target, "model": mname, "group": grp,
                    "importance": float(imp[valid_idxs].sum()) if valid_idxs else 0.0,
                })

        # Random baseline (5 seeds)
        rand_metrics_accum = {}
        for seed in [0, 7, 13, 42, 99]:
            y_rand = random_preds(len(y_te), seed)
            rm = all_metrics(y_te, y_rand)
            for k, v in rm.items():
                rand_metrics_accum.setdefault(k, []).append(v)
        rand_mean = {k: float(np.nanmean(v)) for k, v in rand_metrics_accum.items()}
        rand_mean["target"] = target
        rand_mean["model"] = "Random"
        rand_mean["pred_std"] = float(np.nan)
        rand_mean["actual_std"] = float(y_te.std())
        rand_mean["std_ratio"] = float(np.nan)
        all_metric_rows.append(rand_mean)

        # Global mean baseline
        gm = float(y_tr.mean())
        gm_preds = np.full(len(y_te), gm)
        gm_metrics = all_metrics(y_te, gm_preds)
        gm_metrics["target"] = target
        gm_metrics["model"] = "GlobalMean"
        gm_metrics["pred_std"] = 0.0
        gm_metrics["actual_std"] = float(y_te.std())
        gm_metrics["std_ratio"] = 0.0
        all_metric_rows.append(gm_metrics)

        # Cold-start analysis (candidates with has_history=0)
        cold_mask = np.array([r["has_history"] == 0 for r in target_rows])
        cold_n = int(cold_mask.sum())
        if cold_n >= 2:
            y_cold = y_te[cold_mask]
            for mname, y_pred in model_preds.items():
                y_pred_cold = y_pred[cold_mask]
                from backend.core.utility_predictor.metrics import spearman
                all_coldstart_rows.append({
                    "target": target, "model": mname,
                    "cold_count": cold_n,
                    "cold_spearman": spearman(y_cold, y_pred_cold),
                    "cold_p10": precision_at_k(y_cold, y_pred_cold, 10),
                    "cold_ndcg10": ndcg_at_k(y_cold, y_pred_cold, 10),
                })
        else:
            all_coldstart_rows.append({
                "target": target, "model": "all",
                "cold_count": cold_n,
                "cold_spearman": None, "cold_p10": None, "cold_ndcg10": None
            })

        # Similarity experience summary
        sim_ws = [r["sim_weight_sum"] for r in target_rows]
        all_sim_summary_rows.append({
            "target": target,
            "mean_sim_weight": float(np.mean(sim_ws)),
            "median_sim_weight": float(np.median(sim_ws)),
            "min_sim_weight": float(np.min(sim_ws)),
            "max_sim_weight": float(np.max(sim_ws)),
            "pct_nonzero_sim": float(np.mean([w > 0 for w in sim_ws])),
        })

        # Per-candidate predictions (Model C = Phase 7.1)
        y_pred_C = model_preds["C_Phase7_1_Full"]
        y_unc_C = model_uncs["C_Phase7_1_Full"]
        for i, tr in enumerate(target_rows):
            all_pred_rows.append({
                "target_dataset": target,
                "candidate_id": tr["candidate_id"],
                "actual_utility": float(y_te[i]),
                "predicted_utility": float(y_pred_C[i]),
                "uncertainty": float(y_unc_C[i]),
                "has_exact_history": tr["has_history"],
                "similarity_weight_sum": tr["sim_weight_sum"],
                "rank_actual": int(np.argsort(y_te)[::-1].tolist().index(i) + 1),
                "rank_predicted": int(np.argsort(y_pred_C)[::-1].tolist().index(i) + 1),
            })

        # Console summary
        for mname in ["A_TransformOnly", "B_Phase7", "C_Phase7_1_Full", "D_SimOnly"]:
            ym = [r for r in all_metric_rows if r["target"] == target and r["model"] == mname][-1]
            print(f"  [{mname}]  Spearman={ym['spearman']:.4f}  P@10={ym['precision_at_10']:.3f}  NDCG@10={ym['ndcg_at_10']:.4f}  pred_std={ym['pred_std']:.5f}")
        print(f"  [Random]  Spearman={rand_mean['spearman']:.4f}")

    # Write outputs
    pd.DataFrame(all_pred_rows).to_csv(f"{OUT_DIR}/lodo_predictions.csv", index=False)
    pd.DataFrame(all_metric_rows).to_csv(f"{OUT_DIR}/lodo_metrics.csv", index=False)
    pd.DataFrame(all_importance_rows).to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)
    pd.DataFrame(all_coldstart_rows).to_csv(f"{OUT_DIR}/cold_start_metrics.csv", index=False)
    pd.DataFrame(all_sim_summary_rows).to_csv(f"{OUT_DIR}/similarity_experience_summary.csv", index=False)

    unc_df = pd.DataFrame(all_pred_rows)
    unc_df.groupby("target_dataset")["uncertainty"].agg(
        ["mean", "std", "min", "max"]
    ).reset_index().to_csv(f"{OUT_DIR}/uncertainty_summary.csv", index=False)

    with open(f"{OUT_DIR}/model_config.json", "w") as f:
        json.dump(RF_CONFIG, f, indent=4)

    print(f"\nOutputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
