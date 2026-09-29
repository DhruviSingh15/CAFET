## 3. & 4. Statistical Comparisons

### Budget: 1.0%
**Adaptive vs Random**
- Mean diff: 0.00046
- Median diff: 0.00000
- Std diff: 0.00125
- Adaptive > Random: 3
- Adaptive < Random: 0
- Ties: 7
- Wilcoxon p-value: 2.50000e-01

### Budget: 2.0%
**Adaptive vs Random**
- Mean diff: -0.00041
- Median diff: -0.00000
- Std diff: 0.00067
- Adaptive > Random: 0
- Adaptive < Random: 3
- Ties: 3
- Wilcoxon p-value: 2.50000e-01

### Budget: 5.0%
**Adaptive vs Random**
- Mean diff: -0.00008
- Median diff: 0.00000
- Std diff: 0.00347
- Adaptive > Random: 3
- Adaptive < Random: 2
- Ties: 3
- Wilcoxon p-value: 8.12500e-01

### Budget: 10.0%
**Adaptive vs Random**
- Mean diff: 0.00185
- Median diff: 0.00042
- Std diff: 0.00368
- Adaptive > Random: 5
- Adaptive < Random: 1
- Ties: 2
- Wilcoxon p-value: 2.18750e-01

### Budget: 20.0%
**Adaptive vs Random**
- Mean diff: 0.00002
- Median diff: 0.00000
- Std diff: 0.00580
- Adaptive > Random: 4
- Adaptive < Random: 2
- Ties: 4
- Wilcoxon p-value: 5.62500e-01

## 5. Complete Per-Dataset Results (Budget 0.10 Example)

           dataset                 strategy  evals  regret_mean  val_mean  val_median  val_std
     breast_cancer                 Adaptive   88.0     0.000000  0.000000    0.000000 0.000000
     breast_cancer                   Random   88.0     0.000000  0.000000    0.000000 0.000000
     breast_cancer         Similarity-Prior   88.0     0.000000  0.000000    0.000000 0.000000
     breast_cancer Transformation-Frequency   88.0     0.000000  0.000000    0.000000 0.000000
california_housing                 Adaptive   14.0     0.027719  0.009647    0.003563 0.015529
california_housing                   Random   14.0     0.025906  0.011460    0.003563 0.014734
california_housing         Similarity-Prior   14.0     0.027619  0.009748    0.009748 0.000000
california_housing Transformation-Frequency   14.0     0.032111  0.005256    0.003563 0.004030
          credit_g                 Adaptive  100.0     0.002667  0.024000    0.026667 0.003651
          credit_g                   Random  100.0     0.008000  0.018667    0.020000 0.005578
          credit_g         Similarity-Prior  100.0     0.020000  0.006667    0.006667 0.000000
          credit_g Transformation-Frequency  100.0     0.018667  0.008000    0.006667 0.002981
          diabetes                 Adaptive   22.0     0.072080  0.059117    0.058557 0.012449
          diabetes                   Random   22.0     0.072339  0.058859    0.052905 0.010390
          diabetes         Similarity-Prior   22.0     0.109466  0.021731    0.021731 0.000000
          diabetes Transformation-Frequency   22.0     0.083656  0.047541    0.052905 0.012959
          spambase                 Adaptive   98.0     0.003768  0.007826    0.007246 0.004175
          spambase                   Random   98.0     0.004348  0.007246    0.007246 0.003241
          spambase         Similarity-Prior   98.0     0.002029  0.009565    0.010145 0.000794
          spambase Transformation-Frequency   98.0     0.002899  0.008696    0.008696 0.000000
           titanic                 Adaptive  907.0     0.004061  0.006091    0.005076 0.002270
           titanic                   Random  907.0     0.005076  0.005076    0.005076 0.000000
           titanic         Similarity-Prior  907.0     0.005076  0.005076    0.005076 0.000000
           titanic Transformation-Frequency  907.0     0.004061  0.006091    0.005076 0.002270
           vehicle                 Adaptive   68.0     0.004724  0.034646    0.039370 0.007043
           vehicle                   Random   68.0     0.014173  0.025197    0.023622 0.006588
           vehicle         Similarity-Prior   68.0     0.015748  0.023622    0.023622 0.000000
           vehicle Transformation-Frequency   68.0     0.015748  0.023622    0.023622 0.000000
              wine                 Adaptive   36.0     0.000000  0.000000    0.000000 0.000000
              wine                   Random   36.0     0.000000  0.000000    0.000000 0.000000
              wine         Similarity-Prior   36.0     0.000000  0.000000    0.000000 0.000000
              wine Transformation-Frequency   36.0     0.000000  0.000000    0.000000 0.000000

## 6. Learning Curves (Aggregated by Seed-Means first)

 budget_pct                 strategy     mean   median      std
       0.01                 Adaptive 0.013327 0.008050 0.016073
       0.01                   Random 0.012865 0.007205 0.016101
       0.01         Similarity-Prior 0.008785 0.004538 0.013356
       0.01 Transformation-Frequency 0.007092 0.004538 0.008748
       0.02                 Adaptive 0.006814 0.004567 0.007317
       0.02                   Random 0.007221 0.005002 0.007659
       0.02         Similarity-Prior 0.005596 0.005582 0.005785
       0.02 Transformation-Frequency 0.005596 0.005582 0.005785
       0.05                 Adaptive 0.013145 0.005944 0.016622
       0.05                   Random 0.013221 0.007912 0.015193
       0.05         Similarity-Prior 0.008136 0.005871 0.009066
       0.05 Transformation-Frequency 0.008121 0.005871 0.009175
       0.10                 Adaptive 0.017666 0.008736 0.020584
       0.10                   Random 0.015813 0.009353 0.019482
       0.10         Similarity-Prior 0.009551 0.008116 0.008922
       0.10 Transformation-Frequency 0.012401 0.007046 0.016001
       0.20                 Adaptive 0.023660 0.018832 0.021634
       0.20                   Random 0.023641 0.018646 0.024822
       0.20         Similarity-Prior 0.016153 0.010671 0.019037
       0.20 Transformation-Frequency 0.014562 0.009526 0.016780

Regret:
 budget_pct                 strategy     mean   median      std
       0.01                 Adaptive 0.021081 0.011304 0.027847
       0.01                   Random 0.021543 0.013304 0.027845
       0.01         Similarity-Prior 0.025623 0.014087 0.035889
       0.01 Transformation-Frequency 0.027316 0.020183 0.034867
       0.02                 Adaptive 0.007817 0.006306 0.008806
       0.02                   Random 0.007409 0.005871 0.008254
       0.02         Similarity-Prior 0.009034 0.005292 0.010241
       0.02 Transformation-Frequency 0.009034 0.005292 0.010241
       0.05                 Adaptive 0.018899 0.006899 0.028581
       0.05                   Random 0.018823 0.006609 0.028597
       0.05         Similarity-Prior 0.023907 0.011199 0.036794
       0.05 Transformation-Frequency 0.023922 0.012538 0.035728
       0.10                 Adaptive 0.014377 0.003915 0.024987
       0.10                   Random 0.016230 0.006538 0.024220
       0.10         Similarity-Prior 0.022492 0.010412 0.036599
       0.10 Transformation-Frequency 0.019643 0.009904 0.028193
       0.20                 Adaptive 0.010747 0.001174 0.020544
       0.20                   Random 0.010766 0.003625 0.015812
       0.20         Similarity-Prior 0.018254 0.009205 0.025156
       0.20 Transformation-Frequency 0.019845 0.014541 0.024522

## 9. Final-Test Aggregate Analysis (Budget 0.10)

                strategy      mean    median      std
                Adaptive  0.002890  0.000868 0.004509
                  Random  0.000623  0.000000 0.006836
        Similarity-Prior  0.001406  0.000000 0.003882
Transformation-Frequency -0.001559 -0.000667 0.004519

Positive/Negative/Zero counts for Adaptive (Budget 0.10):
- > 0: 5
- < 0: 0
- == 0: 3

## 10. Validation-to-Test Consistency (Adaptive, Budget 0.10, Seed Mean)

           dataset    test_gain  best_val_gain        ratio
     breast_cancer 0.000000e+00       0.000000          NaN
california_housing 6.492348e-03       0.009647 6.729981e-01
          credit_g 4.024558e-17       0.024000 1.676899e-15
          diabetes 2.293822e-03       0.059117 3.880112e-02
          spambase 1.736614e-03       0.007826 2.219006e-01
           titanic 0.000000e+00       0.006091 0.000000e+00
           vehicle 1.259843e-02       0.034646 3.636364e-01
              wine 0.000000e+00       0.000000          NaN

Cases where val > 0 but test <= 0 (across all strategies/seeds at 10% budget):
78 instances found.

## 11. Search Diversity Audit (Budget 0.10)

 dataset                 strategy  mean_unique_transforms  mean_unique_cands
diabetes                 Adaptive                     2.8                4.0
diabetes                   Random                     2.8                4.0
diabetes         Similarity-Prior                     1.0                4.0
diabetes Transformation-Frequency                     1.0                4.0
 (Showing Diabetes as example)

## 16. Candidate Universe Audit

           dataset  candidate_id
 blood_transfusion            35
     breast_cancer           803
california_housing           132
          credit_g           923
          diabetes           208
              iris            34
          spambase           868
           titanic          7672
           vehicle           628
              wine           320

