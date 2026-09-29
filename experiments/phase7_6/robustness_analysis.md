## 9. Primary Metrics (Budget 0.10)

Per-dataset Aggregates (Seed-Mean):
           dataset strategy  search_mean  indep_mean     test_mean
 blood_transfusion Adaptive     0.026786    0.010619 -1.333333e-03
 blood_transfusion   Random     0.032143    0.014159  1.066667e-02
     breast_cancer Adaptive     0.007059    0.000000 -3.508772e-03
     breast_cancer   Random     0.007059    0.002326  1.977585e-17
california_housing Adaptive     0.021511    0.020475  2.167831e-02
california_housing   Random     0.009942    0.004043  1.222382e-02
          credit_g Adaptive     0.024000   -0.005333 -4.000000e-03
          credit_g   Random     0.021333   -0.002667 -3.000000e-03
          diabetes Adaptive     0.030157   -0.001770 -5.380271e-03
          diabetes   Random     0.033521    0.016145  5.472653e-03
              iris Adaptive     0.000000    0.000000  1.333333e-02
              iris   Random     0.009091   -0.008696  0.000000e+00
          spambase Adaptive     0.005507    0.001739  4.343105e-03
          spambase   Random     0.005507    0.002899  5.646037e-03
           titanic Adaptive     0.006122   -0.002030  3.816794e-03
           titanic   Random     0.002041    0.000000  0.000000e+00
           vehicle Adaptive     0.034646    0.003150  5.882353e-03
           vehicle   Random     0.026772    0.001575 -1.176471e-03
              wine Adaptive     0.000000   -0.007407  0.000000e+00
              wine   Random     0.007407    0.000000 -5.555556e-03

Overall Aggregates:
strategy  search_mean  search_median  search_std  indep_mean  test_mean
Adaptive     0.015579       0.014285    0.013151    0.001944   0.003483
  Random     0.015482       0.009516    0.011797    0.002978   0.002428

## 10. Generalization Gap (Budget 0.10)

strategy  mean_gap  median_gap  std_gap  pos_gap  neg_gap
Adaptive  0.013635    0.008929 0.017929       32        5
  Random  0.012503    0.007141 0.014767       35        2

## 11. Cross-Split Stability (Adaptive, Budget 0.10)

           dataset  unique_cands  unique_transforms
 blood_transfusion             5                  4
     breast_cancer             5                  5
california_housing             3                  2
          credit_g             5                  3
          diabetes             5                  4
              iris             5                  4
          spambase             4                  4
           titanic             4                  4
           vehicle             5                  2
              wine             5                  2

## 12. Consistency (Adaptive, Budget 0.10)

- search > 0 -> indep > 0: 19
- search > 0 -> indep <= 0: 16
- indep > 0 -> test > 0: 12
- search > 0 -> test <= 0: 18

## 13. Statistical Testing

### Budget 0.05
Mean Diff: -0.00257, Median Diff: -0.00180
Adaptive > Random: 4, < Random: 6, Ties: 0
Wilcoxon p-value: 3.75000e-01

### Budget 0.1
Mean Diff: -0.00103, Median Diff: -0.00218
Adaptive > Random: 3, < Random: 7, Ties: 0
Wilcoxon p-value: 4.31641e-01

### Budget 0.2
Mean Diff: -0.00005, Median Diff: -0.00050
Adaptive > Random: 5, < Random: 5, Ties: 0
Wilcoxon p-value: 9.21875e-01

## 14. Search Efficiency (Budget 0.10)

Mean Indep Utility per evaluation:
strategy
Adaptive    0.000331
Random      0.000217

