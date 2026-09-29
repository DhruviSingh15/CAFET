"""
Phase 7: LODO Utility Prediction Experiment.

Three experiments, each following strict LODO:
  A: Train breast_cancer + wine    → Test diabetes
  B: Train breast_cancer + diabetes → Test wine
  C: Train wine + diabetes          → Test breast_cancer

For every candidate, the Phase 6 representation is built with LODO-filtered history.
The predictor is trained on observed validation gain (NOT test gain).
"""

import os
import sys
import json
import ast
import csv
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from backend.core.candidate_representation.candidate_encoder import (
    CandidateEncoder, canonical_candidate_id, TOTAL_DIM,
)
from backend.core.candidate_representation.transformation_encoder import (
    encode_transformation, TRANSFORMATION_VECTOR_DIM,
)
from backend.core.utility_predictor.predictor import UtilityPredictor, RF_CONFIG
from backend.core.utility_predictor.metrics import all_metrics
from backend.core.experience.repository import ExperienceRepository

DATASETS = ["breast_cancer", "wine", "diabetes"]
REPO_PATH = "results/phase4/experience.db"
OUT_DIR = "results/phase7"

LODO_SPLITS = [
    {"target": "diabetes",     "train": ["breast_cancer", "wine"]},
    {"target": "wine",         "train": ["breast_cancer", "diabetes"]},
    {"target": "breast_cancer","train": ["wine", "diabetes"]},
]


def load_profile(ds_name):
    with open(f"results/phase3/{ds_name}_profile.json", "r") as f:
        d = json.load(f)
    return d["numeric_context_vector"], d["raw_context"]["feature_profiles"]


def load_candidates(ds_name):
    df = pd.read_csv(f"results/phase2/{ds_name}_candidates.csv")
    return df[df["status"] == "success"].copy()


def build_hist_index(repo: ExperienceRepository, exclude_dataset: str):
    """Retrieve all experiences except the target dataset."""
    recs = repo.list_experiences(exclude_dataset_id=exclude_dataset)
    idx = {}
    for r in recs:
        cid = r["candidate_id"]
        idx.setdefault(cid, []).append(r)
    return idx


def encode_dataset_candidates(
    ds_name: str,
    encoder: CandidateEncoder,
    hist_index: dict,
    leakage_assert_dataset: str = None,
):
    ctx_vec, feat_profiles = load_profile(ds_name)
    cand_df = load_candidates(ds_name)

    rows = []
    for _, row in cand_df.iterrows():
        transform = row["transformation"]
        try:
            src_feats = ast.literal_eval(row["source_features"])
        except Exception:
            src_feats = [row["source_features"]]

        cid = canonical_candidate_id(transform, src_feats)

        # Leakage assertion: history must NOT contain leakage_assert_dataset records
        hist_recs = hist_index.get(cid, [])
        if leakage_assert_dataset:
            for r in hist_recs:
                assert r["source_dataset_id"] != leakage_assert_dataset, (
                    f"LEAKAGE: {leakage_assert_dataset} record found in history of {cid}"
                )

        rep = encoder.encode(
            {"transformation": transform, "features": src_feats},
            ctx_vec,
            feat_profiles,
            hist_recs,
        )
        rows.append({
            "dataset": ds_name,
            "candidate_id": cid,
            "vector": rep["vector"],
            "actual_utility": float(row["gain"]),
            "has_history": rep["has_history"],
        })
    return rows


def transformation_only_vector(transform: str) -> list:
    """Baseline: encode only transformation type (12 dims), ignore everything else."""
    return encode_transformation(transform)


def random_predictions(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.05, 0.05, n)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    repo = ExperienceRepository(REPO_PATH)
    encoder = CandidateEncoder()

    all_predictions = []
    all_metrics_rows = []
    feature_importance_rows = []

    for split in LODO_SPLITS:
        target = split["target"]
        train_datasets = split["train"]
        print(f"\n{'='*60}")
        print(f"LODO: target={target}  train={train_datasets}")

        # Build history excluding target
        hist_index = build_hist_index(repo, exclude_dataset=target)

        # ── Encode training candidates ──────────────────────────────────────
        train_rows = []
        for ds in train_datasets:
            train_rows.extend(
                encode_dataset_candidates(ds, encoder, hist_index,
                                          leakage_assert_dataset=target)
            )

        # ── Encode target candidates ────────────────────────────────────────
        target_rows = encode_dataset_candidates(
            target, encoder, hist_index, leakage_assert_dataset=target
        )

        X_train = np.array([r["vector"] for r in train_rows])
        y_train = np.array([r["actual_utility"] for r in train_rows])
        X_test  = np.array([r["vector"]  for r in target_rows])
        y_test  = np.array([r["actual_utility"] for r in target_rows])

        # ── Full model (CAFET) ──────────────────────────────────────────────
        predictor = UtilityPredictor()
        predictor.fit(X_train, y_train)
        y_pred, y_unc = predictor.predict(X_test)

        cafet_metrics = all_metrics(y_test, y_pred)

        # ── Transformation-only baseline ────────────────────────────────────
        def get_transform(row):
            parts = row["candidate_id"].split("(")
            return parts[0]

        X_train_tonly = np.array([
            transformation_only_vector(get_transform(r)) for r in train_rows
        ])
        X_test_tonly = np.array([
            transformation_only_vector(get_transform(r)) for r in target_rows
        ])
        pred_tonly = UtilityPredictor()
        pred_tonly.fit(X_train_tonly, y_train)
        y_pred_tonly, _ = pred_tonly.predict(X_test_tonly)
        tonly_metrics = all_metrics(y_test, y_pred_tonly)

        # ── Global mean baseline ────────────────────────────────────────────
        global_mean = float(y_train.mean())
        y_pred_global = np.full(len(y_test), global_mean)
        global_metrics = all_metrics(y_test, y_pred_global)

        # ── Random baseline (5 seeds) ───────────────────────────────────────
        random_seeds = [0, 7, 13, 42, 99]
        random_all = {k: [] for k in cafet_metrics}
        for seed in random_seeds:
            y_rand = random_predictions(len(y_test), seed)
            rm = all_metrics(y_test, y_rand)
            for k in rm:
                random_all[k].append(rm[k])
        random_mean_metrics = {k: float(np.nanmean(v)) for k, v in random_all.items()}

        # ── Feature importance ──────────────────────────────────────────────
        imp = predictor.feature_importances()
        # Grouped importance
        groups = {
            "transformation": list(range(0, 12)),
            "feature_slot_1": list(range(12, 26)),
            "feature_slot_2": list(range(26, 40)),
            "dataset_context": list(range(40, 54)),
            "history": list(range(54, 61)),
        }
        for grp, idxs in groups.items():
            feature_importance_rows.append({
                "target": target,
                "group": grp,
                "importance": float(imp[idxs].sum()),
            })

        # ── Print summary ───────────────────────────────────────────────────
        print(f"  CAFET:  Spearman={cafet_metrics['spearman']:.4f}  "
              f"P@10={cafet_metrics['precision_at_10']:.3f}  "
              f"NDCG@10={cafet_metrics['ndcg_at_10']:.4f}")
        print(f"  T-only: Spearman={tonly_metrics['spearman']:.4f}  "
              f"P@10={tonly_metrics['precision_at_10']:.3f}  "
              f"NDCG@10={tonly_metrics['ndcg_at_10']:.4f}")
        print(f"  Global: Spearman={global_metrics['spearman']:.4f}  "
              f"P@10={global_metrics['precision_at_10']:.3f}")
        print(f"  Random: Spearman={random_mean_metrics['spearman']:.4f}  "
              f"P@10={random_mean_metrics['precision_at_10']:.3f}")
        print(f"  Pred dist: mean={y_pred.mean():.5f}  std={y_pred.std():.5f}  "
              f"actual mean={y_test.mean():.5f}  actual std={y_test.std():.5f}")

        # ── Save metrics ────────────────────────────────────────────────────
        for model, mdict in [
            ("CAFET", cafet_metrics),
            ("TransformOnly", tonly_metrics),
            ("GlobalMean", global_metrics),
            ("Random", random_mean_metrics),
        ]:
            row = {"target": target, "model": model}
            row.update(mdict)
            all_metrics_rows.append(row)

        # ── Save per-candidate predictions ──────────────────────────────────
        for i, tr in enumerate(target_rows):
            all_predictions.append({
                "target_dataset": target,
                "candidate_id": tr["candidate_id"],
                "actual_utility": float(y_test[i]),
                "predicted_utility": float(y_pred[i]),
                "uncertainty": float(y_unc[i]),
                "has_history": tr["has_history"],
                "rank_actual": int(np.argsort(y_test)[::-1].tolist().index(i) + 1),
                "rank_predicted": int(np.argsort(y_pred)[::-1].tolist().index(i) + 1),
            })

    # ── Write outputs ────────────────────────────────────────────────────────
    pd.DataFrame(all_predictions).to_csv(f"{OUT_DIR}/lodo_predictions.csv", index=False)
    pd.DataFrame(all_metrics_rows).to_csv(f"{OUT_DIR}/lodo_metrics.csv", index=False)
    pd.DataFrame(feature_importance_rows).to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)

    # Uncertainty summary
    pred_df = pd.DataFrame(all_predictions)
    unc_summary = pred_df.groupby("target_dataset")["uncertainty"].agg(
        ["mean", "std", "min", "max"]
    ).reset_index()
    unc_summary.to_csv(f"{OUT_DIR}/uncertainty_summary.csv", index=False)

    with open(f"{OUT_DIR}/model_config.json", "w") as f:
        json.dump(RF_CONFIG, f, indent=4)

    print(f"\nOutputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
