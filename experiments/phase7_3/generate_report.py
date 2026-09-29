import pandas as pd
import numpy as np

OUT_DIR = "results/phase7_3"
REPORT_PATH = "reports/PHASE7_3_EXPANDED_UTILITY_VALIDATION.md"

metrics = pd.read_csv(f"{OUT_DIR}/lodo_metrics.csv")
comp = pd.read_csv(f"{OUT_DIR}/model_comparison.csv")
cold = pd.read_csv(f"{OUT_DIR}/cold_start_metrics.csv")
sim = pd.read_csv(f"{OUT_DIR}/similarity_experience_summary.csv")
var = pd.read_csv(f"{OUT_DIR}/prediction_variance.csv")
unc = pd.read_csv(f"{OUT_DIR}/uncertainty_summary.csv")
imp = pd.read_csv(f"{OUT_DIR}/feature_importance.csv")
stat = pd.read_csv(f"{OUT_DIR}/statistical_comparison.csv")
preds = pd.read_csv(f"{OUT_DIR}/lodo_predictions.csv")

def format_report():
    report = []
    report.append("# CAFET — Phase 7.3: Expanded-Pool Utility Prediction\n")
    report.append("## 1. Objective\nTest whether increasing from 3 to 10 real-world datasets improves candidate-level utility prediction using similarity-weighted experience.\n")
    
    report.append("## 2. Dataset Pool\n10 datasets (validated in Phase 7.2): breast_cancer, wine, diabetes, california_housing, iris, titanic, credit_g, blood_transfusion, vehicle, spambase.\n")
    
    report.append("## 3. Experimental Protocol\n10 LODO (Leave-One-Dataset-Out) experiments.\n")
    
    report.append("## 4. LODO Design\nEach target dataset was held out. The remaining 9 were used for training.\n")
    
    report.append("## 5. Leakage Controls\nTarget dataset was excluded from historical records before any similarities or weighted features were computed.\n")
    
    report.append("## 6. Models\n- Model A: Transformation Only\n- Model B: Original Phase 7\n- Model C: Phase 7.1 Full\n- Model D: Phase 7.1 SimOnly\n- Random\n- GlobalMean\n")
    
    report.append("## 7. Per-Dataset Results\n")
    per_ds = metrics[metrics["model"].isin(["D_SimOnly", "A_TransformOnly", "Random"])].pivot(index="target", columns="model", values=["spearman", "precision_at_10", "ndcg_at_10"]).reset_index()
    per_ds.columns = ["_".join(c).strip('_') for c in per_ds.columns]
    report.append(per_ds.to_markdown(index=False) + "\n")
    
    report.append("## 8. Aggregate Results\n")
    agg = metrics.groupby("model")[["spearman", "precision_at_10", "precision_at_25", "ndcg_at_10", "ndcg_at_25", "rmse"]].agg(["mean", "median", "std", "min", "max"])
    # Format aggregate
    report.append(agg.to_markdown() + "\n")
    
    report.append("## 9. Random Baseline\nRandom mean Spearman is ~0.0 across all datasets.\n")
    report.append("## 10. Transformation-Only Baseline\nSee Model A. Generally performs poorly but sometimes beats SimOnly.\n")
    report.append("## 11. Original Phase 7 Comparison\nSee Model B.\n")
    report.append("## 12. Phase 7.1 Comparison\nSee Model C and D.\n")
    
    report.append("## 13. Cold-Start Analysis\n")
    report.append(cold.to_markdown(index=False) + "\n")
    
    report.append("## 14. Similarity Evidence Coverage\n")
    report.append(sim.to_markdown(index=False) + "\n")
    
    report.append("## 15. Prediction Variance\n")
    report.append(var.groupby("model")[["actual_std", "pred_std", "std_ratio"]].mean().reset_index().to_markdown(index=False) + "\n")
    
    report.append("## 16. Uncertainty\n")
    report.append(unc.groupby("model")[["mean", "median", "std", "min", "max"]].mean().reset_index().to_markdown(index=False) + "\n")
    
    report.append("## 17. Feature Importance\n")
    imp_mean = imp.groupby(["model", "group"])["importance"].mean().reset_index()
    report.append(imp_mean.to_markdown(index=False) + "\n")
    
    report.append("## 18. Statistical Comparison\n")
    report.append(stat.to_markdown(index=False) + "\n")
    
    report.append("## 19. Failure Cases\n")
    report.append("The similarity-weighted model (D_SimOnly) exhibits negative Spearman correlations on half the datasets (iris, titanic, credit_g, blood_transfusion, vehicle, spambase). Precision@10 is 0.0 on 6 out of 10 datasets. This demonstrates a fundamental lack of generalized predictive signal.\n")
    
    report.append("## 20. Limitations\n")
    report.append("- Transformation overlap is 100%, but actual utility of identical transformations on different datasets varies wildly based on data distribution, meaning dataset meta-features are insufficient to contextualize the transformation.\n")
    
    report.append("## 21. Scientific Interpretation\n")
    report.append("Observed result: Increasing the historical dataset pool from 3 to 10 did not yield a positive candidate-level predictive signal. P@10 remains zero frequently, Spearman is commonly negative, and variance remains highly under-dispersed.\nInterpretation: CAFET's core mechanism—using high-level dataset meta-features to transfer feature utility—fails. The mapping from dataset macro-statistics to specific candidate utility is either too complex for 10 datasets to capture, or fundamentally non-existent. A transformation like `ADD(f1, f2)` depends heavily on the joint distribution of `f1` and `f2` and the target, which macro-level dataset context vectors (e.g. % missing, % categorical) cannot express.\n")
    
    report.append("## 22. Phase Verdict\n")
    report.append("FAIL\n")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report))
        
    print(f"Report written to {REPORT_PATH}")
    
    # Print the specific output required for the prompt
    print("\n\n--- FOR PROMPT ---")
    print("Phase: 7.3\nStatus: FAIL\n")
    print("Dataset count: 10\nHistorical datasets per target: 9\n")
    print("California Housing sampling:\n- original rows: 20640\n- experimental rows: 5000\n- method: df.sample(5000, random_state=42).reset_index(drop=True)\n- seed: 42\n")
    
if __name__ == "__main__":
    format_report()
