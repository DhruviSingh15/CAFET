import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

df = pd.read_csv("../../results/phase7_5/adaptive_results.csv")

print("=== AGGREGATE MEAN UTILITY ===")
agg = df.groupby(["budget_pct", "strategy"])["best_utility"].mean().unstack()
print(agg)

print("\n=== AGGREGATE MEAN REGRET ===")
agg_regret = df.groupby(["budget_pct", "strategy"])["regret"].mean().unstack()
print(agg_regret)

print("\n=== PER-DATASET BUDGET 0.10 ===")
ds_agg = df[df["budget_pct"] == 0.10].groupby(["dataset", "strategy"])["best_utility"].mean().unstack()
print(ds_agg)

print("\n=== STATISTICAL COMPARISON (Budget 0.10) ===")
d10 = df[df["budget_pct"] == 0.10]
def pval(s1, s2):
    try:
        x = d10[d10["strategy"] == s1].groupby(["dataset", "seed"])["best_utility"].mean()
        y = d10[d10["strategy"] == s2].groupby(["dataset", "seed"])["best_utility"].mean()
        # Pair by (dataset, seed)
        merged = pd.merge(x, y, on=["dataset", "seed"])
        stat, p = wilcoxon(merged["best_utility_x"], merged["best_utility_y"])
        return p
    except:
        return 1.0

print("Adaptive vs Random:", pval("Adaptive", "Random"))
print("Adaptive vs TransformFreq:", pval("Adaptive", "TransformFreq"))
print("Adaptive vs SimPrior:", pval("Adaptive", "SimPrior"))

