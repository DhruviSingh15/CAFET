import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

traces = pd.read_csv('../../results/phase7_5/adaptive_search_traces.csv')
results = pd.read_csv('../../results/phase7_5/adaptive_search_results.csv')

def write_md(f, text):
    f.write(text + "\n")

with open('audit_output.md', 'w') as f:
    write_md(f, "## 3. & 4. Statistical Comparisons\n")
    
    # Calculate seed means
    seed_means = results.groupby(['dataset', 'strategy', 'budget_pct'])['best_val_gain'].mean().reset_index()
    
    for b in [0.01, 0.02, 0.05, 0.1, 0.2]:
        write_md(f, f"### Budget: {b*100}%")
        b_data = seed_means[seed_means['budget_pct'] == b]
        
        for comp in ["Random", "TransformFreq", "SimPrior"]:
            ad = b_data[b_data['strategy'] == 'Adaptive'].set_index('dataset')['best_val_gain']
            ot = b_data[b_data['strategy'] == comp].set_index('dataset')['best_val_gain']
            merged = pd.DataFrame({'ad': ad, 'ot': ot}).dropna()
            if len(merged) == 0: continue
            
            diff = merged['ad'] - merged['ot']
            mean_diff = diff.mean()
            median_diff = diff.median()
            std_diff = diff.std()
            n_greater = (diff > 0).sum()
            n_less = (diff < 0).sum()
            n_tie = (diff == 0).sum()
            
            # wilcoxon requires non-zero differences
            non_zero_diffs = diff[diff != 0]
            if len(non_zero_diffs) > 0:
                try:
                    stat, pval = wilcoxon(merged['ad'], merged['ot'], zero_method='wilcox')
                except Exception as e:
                    pval = float('nan')
            else:
                pval = 1.0
                
            write_md(f, f"**Adaptive vs {comp}**")
            write_md(f, f"- Mean diff: {mean_diff:.5f}")
            write_md(f, f"- Median diff: {median_diff:.5f}")
            write_md(f, f"- Std diff: {std_diff:.5f}")
            write_md(f, f"- Adaptive > {comp}: {n_greater}")
            write_md(f, f"- Adaptive < {comp}: {n_less}")
            write_md(f, f"- Ties: {n_tie}")
            write_md(f, f"- Wilcoxon p-value: {pval:.5e}\n")

    write_md(f, "## 5. Complete Per-Dataset Results (Budget 0.10 Example)\n")
    d10 = results[results['budget_pct'] == 0.10]
    agg10 = d10.groupby(['dataset', 'strategy']).agg(
        evals=('budget_count', 'mean'),
        regret_mean=('regret', 'mean'),
        val_mean=('best_val_gain', 'mean'),
        val_median=('best_val_gain', 'median'),
        val_std=('best_val_gain', 'std')
    ).reset_index()
    write_md(f, agg10.to_string(index=False) + "\n")

    write_md(f, "## 6. Learning Curves (Aggregated by Seed-Means first)\n")
    agg_learning = seed_means.groupby(['budget_pct', 'strategy'])['best_val_gain'].agg(['mean', 'median', 'std']).reset_index()
    write_md(f, agg_learning.to_string(index=False) + "\n")
    
    seed_regret = results.groupby(['dataset', 'strategy', 'budget_pct'])['regret'].mean().reset_index()
    agg_regret = seed_regret.groupby(['budget_pct', 'strategy'])['regret'].agg(['mean', 'median', 'std']).reset_index()
    write_md(f, "Regret:\n" + agg_regret.to_string(index=False) + "\n")

    write_md(f, "## 9. Final-Test Aggregate Analysis (Budget 0.10)\n")
    test_seed_means = results[results['budget_pct'] == 0.10].groupby(['dataset', 'strategy'])['test_gain'].mean().reset_index()
    test_agg = test_seed_means.groupby('strategy')['test_gain'].agg(['mean', 'median', 'std']).reset_index()
    write_md(f, test_agg.to_string(index=False) + "\n")
    
    write_md(f, "Positive/Negative/Zero counts for Adaptive (Budget 0.10):")
    ad_test = test_seed_means[test_seed_means['strategy'] == 'Adaptive']['test_gain']
    write_md(f, f"- > 0: {(ad_test > 0).sum()}")
    write_md(f, f"- < 0: {(ad_test < 0).sum()}")
    write_md(f, f"- == 0: {(ad_test == 0).sum()}\n")
    
    write_md(f, "## 10. Validation-to-Test Consistency (Adaptive, Budget 0.10, Seed Mean)\n")
    val_seed_means = results[results['budget_pct'] == 0.10].groupby(['dataset', 'strategy'])['best_val_gain'].mean().reset_index()
    merge_vt = pd.merge(test_seed_means, val_seed_means, on=['dataset', 'strategy'])
    merge_vt_ad = merge_vt[merge_vt['strategy'] == 'Adaptive'].copy()
    merge_vt_ad['ratio'] = np.where(merge_vt_ad['best_val_gain'] != 0, merge_vt_ad['test_gain'] / merge_vt_ad['best_val_gain'], float('nan'))
    write_md(f, merge_vt_ad[['dataset', 'test_gain', 'best_val_gain', 'ratio']].to_string(index=False) + "\n")
    
    write_md(f, "Cases where val > 0 but test <= 0 (across all strategies/seeds at 10% budget):")
    bad_gen = results[(results['budget_pct'] == 0.1) & (results['best_val_gain'] > 0) & (results['test_gain'] <= 0)]
    write_md(f, f"{len(bad_gen)} instances found.\n")

    write_md(f, "## 11. Search Diversity Audit (Budget 0.10)\n")
    t10 = traces[traces['step'] <= traces.groupby(['dataset', 'strategy', 'seed'])['step'].transform('max') * 0.10] # approx
    div = t10.groupby(['dataset', 'strategy', 'seed']).agg(
        unique_transforms=('transformation', 'nunique'),
        unique_cands=('candidate_id', 'nunique'),
        total_steps=('step', 'count')
    ).reset_index()
    
    div_agg = div.groupby(['dataset', 'strategy']).agg(
        mean_unique_transforms=('unique_transforms', 'mean'),
        mean_unique_cands=('unique_cands', 'mean')
    ).reset_index()
    write_md(f, div_agg[div_agg['dataset'] == 'diabetes'].to_string(index=False) + "\n (Showing Diabetes as example)\n")
    
    write_md(f, "## 16. Candidate Universe Audit\n")
    cand_counts = traces.groupby('dataset')['candidate_id'].nunique().reset_index()
    write_md(f, cand_counts.to_string(index=False) + "\n")
