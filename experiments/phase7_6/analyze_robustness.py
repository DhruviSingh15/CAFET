import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# Load results
df = pd.read_csv('../../results/phase7_6/robustness_results.csv')

def write_md(f, text):
    f.write(text + "\n")

with open('robustness_analysis.md', 'w') as f:
    
    # 9. Primary Metrics (Budget 0.10 for example)
    write_md(f, "## 9. Primary Metrics (Budget 0.10)\n")
    d10 = df[df['budget_pct'] == 0.1]
    agg_ds = d10.groupby(['dataset', 'strategy']).agg(
        search_mean=('search_utility', 'mean'),
        indep_mean=('indep_utility', 'mean'),
        test_mean=('test_utility', 'mean')
    ).reset_index()
    
    write_md(f, "Per-dataset Aggregates (Seed-Mean):")
    write_md(f, agg_ds.to_string(index=False) + "\n")
    
    overall_agg = agg_ds.groupby('strategy').agg(
        search_mean=('search_mean', 'mean'),
        search_median=('search_mean', 'median'),
        search_std=('search_mean', 'std'),
        indep_mean=('indep_mean', 'mean'),
        test_mean=('test_mean', 'mean')
    ).reset_index()
    write_md(f, "Overall Aggregates:")
    write_md(f, overall_agg.to_string(index=False) + "\n")
    
    # 10. Generalization Gap
    write_md(f, "## 10. Generalization Gap (Budget 0.10)\n")
    d10 = d10.copy()
    d10['gap'] = d10['search_utility'] - d10['indep_utility']
    
    gap_agg = d10.groupby('strategy').agg(
        mean_gap=('gap', 'mean'),
        median_gap=('gap', 'median'),
        std_gap=('gap', 'std'),
        pos_gap=('gap', lambda x: (x > 0).sum()),
        neg_gap=('gap', lambda x: (x < 0).sum())
    ).reset_index()
    write_md(f, gap_agg.to_string(index=False) + "\n")
    
    # 11. Cross-Split Stability (Adaptive, Budget 0.10)
    write_md(f, "## 11. Cross-Split Stability (Adaptive, Budget 0.10)\n")
    ad10 = d10[d10['strategy'] == 'Adaptive']
    stab = ad10.groupby('dataset').agg(
        unique_cands=('best_cand_id', 'nunique'),
        unique_transforms=('transformation', 'nunique')
    ).reset_index()
    write_md(f, stab.to_string(index=False) + "\n")
    
    # 12. Test / Independent Evaluation Consistency (Adaptive, Budget 0.10)
    write_md(f, "## 12. Consistency (Adaptive, Budget 0.10)\n")
    s_pos_i_pos = len(ad10[(ad10['search_utility'] > 0) & (ad10['indep_utility'] > 0)])
    s_pos_i_nonpos = len(ad10[(ad10['search_utility'] > 0) & (ad10['indep_utility'] <= 0)])
    i_pos_t_pos = len(ad10[(ad10['indep_utility'] > 0) & (ad10['test_utility'] > 0)])
    s_pos_t_nonpos = len(ad10[(ad10['search_utility'] > 0) & (ad10['test_utility'] <= 0)])
    
    write_md(f, f"- search > 0 -> indep > 0: {s_pos_i_pos}")
    write_md(f, f"- search > 0 -> indep <= 0: {s_pos_i_nonpos}")
    write_md(f, f"- indep > 0 -> test > 0: {i_pos_t_pos}")
    write_md(f, f"- search > 0 -> test <= 0: {s_pos_t_nonpos}\n")
    
    # 13. Statistical Testing (Dataset level, paired)
    write_md(f, "## 13. Statistical Testing\n")
    for b in [0.05, 0.10, 0.20]:
        write_md(f, f"### Budget {b}")
        b_data = df[df['budget_pct'] == b]
        ds_mean = b_data.groupby(['dataset', 'strategy'])['indep_utility'].mean().reset_index()
        
        ad = ds_mean[ds_mean['strategy'] == 'Adaptive'].set_index('dataset')['indep_utility']
        rd = ds_mean[ds_mean['strategy'] == 'Random'].set_index('dataset')['indep_utility']
        
        merged = pd.DataFrame({'ad': ad, 'rd': rd}).dropna()
        diff = merged['ad'] - merged['rd']
        
        try:
            stat, p = wilcoxon(merged['ad'], merged['rd'], zero_method='wilcox')
        except:
            p = 1.0
            
        write_md(f, f"Mean Diff: {diff.mean():.5f}, Median Diff: {diff.median():.5f}")
        write_md(f, f"Adaptive > Random: {(diff > 0).sum()}, < Random: {(diff < 0).sum()}, Ties: {(diff == 0).sum()}")
        write_md(f, f"Wilcoxon p-value: {p:.5e}\n")
        
    # 14. Search Efficiency (Budget 0.10)
    write_md(f, "## 14. Search Efficiency (Budget 0.10)\n")
    d10['eff'] = d10['indep_utility'] / d10['budget_abs']
    eff_agg = d10.groupby('strategy')['eff'].mean()
    write_md(f, f"Mean Indep Utility per evaluation:\n{eff_agg.to_string()}\n")
