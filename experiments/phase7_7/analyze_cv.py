import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, spearmanr

def main():
    df = pd.read_csv("../../results/phase7_7/cv_robustness_results.csv")
    
    # 1. Aggregate Metrics by Budget
    print("=== Aggregate Metrics by Budget ===")
    budgets = df["budget_pct"].unique()
    for b in sorted(budgets):
        print(f"\nBudget: {b*100}%")
        b_df = df[df["budget_pct"] == b]
        for strat in ["Random", "Adaptive"]:
            s_df = b_df[b_df["strategy"] == strat]
            print(f"  Strategy: {strat}")
            print(f"    Search Utility:    Mean={s_df['search_utility_cv'].mean():.4f}, Median={s_df['search_utility_cv'].median():.4f}, Std={s_df['search_utility_cv'].std():.4f}")
            print(f"    Independent Gain:  Mean={s_df['indep_utility'].mean():.4f}, Median={s_df['indep_utility'].median():.4f}, Std={s_df['indep_utility'].std():.4f}")
            print(f"    Final Test Gain:   Mean={s_df['test_utility'].mean():.4f}, Median={s_df['test_utility'].median():.4f}, Std={s_df['test_utility'].std():.4f}")
            
            gap = s_df['search_utility_cv'] - s_df['indep_utility']
            print(f"    Search-to-Indep Gap: Mean={gap.mean():.4f}, Median={gap.median():.4f}, Std={gap.std():.4f}")
            
    # 2. Dataset-level Statistical Test (Adaptive vs Random Independent Gain)
    print("\n=== Dataset-level Statistical Tests (Independent Gain) ===")
    for b in sorted(budgets):
        b_df = df[df["budget_pct"] == b]
        
        # Aggregate across 5 seeds per dataset
        agg_df = b_df.groupby(["dataset", "strategy"])["indep_utility"].mean().reset_index()
        
        adapt = agg_df[agg_df["strategy"] == "Adaptive"].set_index("dataset")["indep_utility"]
        randm = agg_df[agg_df["strategy"] == "Random"].set_index("dataset")["indep_utility"]
        
        # Align them
        merged = pd.concat([adapt, randm], axis=1, keys=["Adaptive", "Random"]).dropna()
        
        if len(merged) < 5:
            print(f"Budget {b*100}%: Not enough data for Wilcoxon.")
            continue
            
        stat, pval = wilcoxon(merged["Adaptive"], merged["Random"])
        diff = merged["Adaptive"] - merged["Random"]
        
        wins = sum(diff > 0)
        losses = sum(diff < 0)
        ties = sum(diff == 0)
        
        print(f"Budget: {b*100}% | N={len(merged)} datasets")
        print(f"  Adaptive Mean: {merged['Adaptive'].mean():.4f}, Random Mean: {merged['Random'].mean():.4f}")
        print(f"  Mean Diff: {diff.mean():.4f}, Median Diff: {diff.median():.4f}")
        print(f"  Wins: {wins}, Losses: {losses}, Ties: {ties}")
        print(f"  Wilcoxon p-value: {pval:.4f}")
        
    # 3. Candidate Stability
    print("\n=== Candidate Stability across 5 outer splits ===")
    for b in sorted(budgets):
        print(f"\nBudget {b*100}%")
        b_df = df[df["budget_pct"] == b]
        for strat in ["Random", "Adaptive"]:
            s_df = b_df[b_df["strategy"] == strat]
            # Exact candidate match
            exact_counts = s_df.groupby("dataset")["best_cand_id"].nunique()
            # Transformation match
            trans_counts = s_df.groupby("dataset")["transformation"].nunique()
            
            # Lower unique count = higher stability (1 means exact same across all seeds)
            print(f"  {strat}: Mean Unique Exact Candidates/Dataset = {exact_counts.mean():.2f} (lower is better, min 1)")
            print(f"  {strat}: Mean Unique Transformations/Dataset  = {trans_counts.mean():.2f}")
            
    # 4. Correlation Analysis
    print("\n=== Search vs Independent Correlation ===")
    for b in sorted(budgets):
        b_df = df[df["budget_pct"] == b]
        # Calculate Spearman correlation across all selections for Adaptive
        adapt_df = b_df[b_df["strategy"] == "Adaptive"]
        
        if len(adapt_df) > 1:
            spear, p_spear = spearmanr(adapt_df["search_utility_cv"], adapt_df["indep_utility"])
            print(f"Budget {b*100}% - Adaptive: Spearman={spear:.4f} (p={p_spear:.4f})")

if __name__ == "__main__":
    main()
