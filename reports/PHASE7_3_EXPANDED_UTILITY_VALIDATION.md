# CAFET — Phase 7.3: Expanded-Pool Utility Prediction

## 1. Objective
Test whether increasing from 3 to 10 real-world datasets improves candidate-level utility prediction using similarity-weighted experience.

## 2. Dataset Pool
10 datasets (validated in Phase 7.2): breast_cancer, wine, diabetes, california_housing, iris, titanic, credit_g, blood_transfusion, vehicle, spambase.

## 3. Experimental Protocol
10 LODO (Leave-One-Dataset-Out) experiments.

## 4. LODO Design
Each target dataset was held out. The remaining 9 were used for training.

## 5. Leakage Controls
Target dataset was excluded from historical records before any similarities or weighted features were computed.

## 6. Models
- Model A: Transformation Only
- Model B: Original Phase 7
- Model C: Phase 7.1 Full
- Model D: Phase 7.1 SimOnly
- Random
- GlobalMean

## 7. Per-Dataset Results

| target             |   spearman_A_TransformOnly |   spearman_D_SimOnly |   spearman_Random |   precision_at_10_A_TransformOnly |   precision_at_10_D_SimOnly |   precision_at_10_Random |   ndcg_at_10_A_TransformOnly |   ndcg_at_10_D_SimOnly |   ndcg_at_10_Random |
|:-------------------|---------------------------:|---------------------:|------------------:|----------------------------------:|----------------------------:|-------------------------:|-----------------------------:|-----------------------:|--------------------:|
| blood_transfusion  |                -0.185859   |           -0.152279  |       -0.0151442  |                               0.2 |                         0.2 |                     0.22 |                     0.254037 |               0.311091 |            0.204769 |
| breast_cancer      |                 0.0865695  |            0.0944999 |        0.0325528  |                               0   |                         0   |                     0.02 |                     0.886842 |               0.898381 |            0.827345 |
| california_housing |                 0.0494341  |            0.164918  |       -0.04338    |                               0.1 |                         0.1 |                     0.1  |                     0.393247 |               0.37039  |            0.370245 |
| credit_g           |                -0.407383   |           -0.0906224 |       -0.00737895 |                               0   |                         0   |                     0    |                     0.383039 |               0.413033 |            0.392968 |
| diabetes           |                -0.111934   |            0.049354  |       -0.0318061  |                               0   |                         0   |                     0.08 |                     0.433674 |               0.493277 |            0.516381 |
| iris               |                 0.00599953 |           -0.0770195 |        0.0538928  |                               0.2 |                         0.2 |                     0.3  |                     0.502228 |               0.526309 |            0.624568 |
| spambase           |                 0.0912405  |           -0.0208823 |        0.0183987  |                               0   |                         0   |                     0.04 |                     0.4247   |               0.460584 |            0.445974 |
| titanic            |                -0.126735   |           -0.130547  |       -0.0020653  |                               0.1 |                         0   |                     0    |                     0.993783 |               0.990365 |            0.991544 |
| vehicle            |                 0.0640279  |           -0.0424839 |       -0.0129566  |                               0   |                         0   |                     0    |                     0.443293 |               0.554853 |            0.589073 |
| wine               |                 0.131371   |            0.11457   |       -0.0182955  |                               0.1 |                         0   |                     0    |                     0.923077 |               1        |            0.942731 |

## 8. Aggregate Results

| model           |   ('spearman', 'mean') |   ('spearman', 'median') |   ('spearman', 'std') |   ('spearman', 'min') |   ('spearman', 'max') |   ('precision_at_10', 'mean') |   ('precision_at_10', 'median') |   ('precision_at_10', 'std') |   ('precision_at_10', 'min') |   ('precision_at_10', 'max') |   ('precision_at_25', 'mean') |   ('precision_at_25', 'median') |   ('precision_at_25', 'std') |   ('precision_at_25', 'min') |   ('precision_at_25', 'max') |   ('ndcg_at_10', 'mean') |   ('ndcg_at_10', 'median') |   ('ndcg_at_10', 'std') |   ('ndcg_at_10', 'min') |   ('ndcg_at_10', 'max') |   ('ndcg_at_25', 'mean') |   ('ndcg_at_25', 'median') |   ('ndcg_at_25', 'std') |   ('ndcg_at_25', 'min') |   ('ndcg_at_25', 'max') |   ('rmse', 'mean') |   ('rmse', 'median') |   ('rmse', 'std') |   ('rmse', 'min') |   ('rmse', 'max') |
|:----------------|-----------------------:|-------------------------:|----------------------:|----------------------:|----------------------:|------------------------------:|--------------------------------:|-----------------------------:|-----------------------------:|-----------------------------:|------------------------------:|--------------------------------:|-----------------------------:|-----------------------------:|-----------------------------:|-------------------------:|---------------------------:|------------------------:|------------------------:|------------------------:|-------------------------:|---------------------------:|------------------------:|------------------------:|------------------------:|-------------------:|---------------------:|------------------:|------------------:|------------------:|
| A_TransformOnly |            -0.0403267  |                0.0277168 |             0.167494  |             -0.407383 |             0.131371  |                         0.07  |                            0.05 |                    0.0823273 |                            0 |                          0.2 |                        0.156  |                           0.06  |                     0.211198 |                            0 |                        0.56  |                 0.563792 |                   0.438483 |                0.264708 |               0.254037  |                0.993783 |                 0.630039 |                   0.504296 |                0.228179 |                0.40577  |                0.994594 |          0.0131685 |            0.0109501 |        0.00876279 |        0.00248997 |         0.0290889 |
| B_Phase7        |             0.00707812 |                0.0292136 |             0.100831  |             -0.154372 |             0.152162  |                         0.09  |                            0.05 |                    0.0994429 |                            0 |                          0.2 |                        0.172  |                           0.08  |                     0.229434 |                            0 |                        0.6   |                 0.60557  |                   0.521865 |                0.268508 |               0.281357  |                1        |                 0.662709 |                   0.571362 |                0.222398 |                0.394461 |                1        |          0.0137414 |            0.0121104 |        0.00865829 |        0.00541479 |         0.029326  |
| C_Phase7_1_Full |            -0.0270258  |               -0.0281321 |             0.117563  |             -0.174269 |             0.166716  |                         0.07  |                            0.05 |                    0.0948683 |                            0 |                          0.3 |                        0.192  |                           0.06  |                     0.273122 |                            0 |                        0.72  |                 0.590229 |                   0.602767 |                0.263243 |               0.0491432 |                0.991364 |                 0.675416 |                   0.697621 |                0.205121 |                0.374807 |                0.993431 |          0.0139062 |            0.0117447 |        0.00866349 |        0.00410254 |         0.0293484 |
| D_SimOnly       |            -0.00904918 |               -0.0316831 |             0.109285  |             -0.152279 |             0.164918  |                         0.05  |                            0    |                    0.0849837 |                            0 |                          0.2 |                        0.172  |                           0.06  |                     0.242249 |                            0 |                        0.64  |                 0.601828 |                   0.509793 |                0.260517 |               0.311091  |                1        |                 0.668276 |                   0.578609 |                0.211628 |                0.41597  |                1        |          0.0137621 |            0.0127586 |        0.00903411 |        0.00168828 |         0.0290701 |
| GlobalMean      |           nan          |              nan         |           nan         |            nan        |           nan         |                         0.16  |                            0    |                    0.259058  |                            0 |                          0.8 |                        0.252  |                           0.14  |                     0.261992 |                            0 |                        0.76  |                 0.589797 |                   0.523587 |                0.249775 |               0.283228  |                0.991005 |                 0.651361 |                   0.584401 |                0.208259 |                0.452842 |                0.991813 |          0.0128008 |            0.0108558 |        0.00894496 |        0.00174024 |         0.0290575 |
| Random          |            -0.00261824 |               -0.0101678 |             0.0296614 |             -0.04338  |             0.0538928 |                         0.076 |                            0.03 |                    0.104902  |                            0 |                          0.3 |                        0.1656 |                           0.056 |                     0.239081 |                            0 |                        0.616 |                 0.59056  |                   0.552727 |                0.259243 |               0.204769  |                0.991544 |                 0.649307 |                   0.596106 |                0.213734 |                0.405456 |                0.992272 |          0.0322867 |            0.0307216 |        0.00415458 |        0.0286422  |         0.0404361 |

## 9. Random Baseline
Random mean Spearman is ~0.0 across all datasets.

## 10. Transformation-Only Baseline
See Model A. Generally performs poorly but sometimes beats SimOnly.

## 11. Original Phase 7 Comparison
See Model B.

## 12. Phase 7.1 Comparison
See Model C and D.

## 13. Cold-Start Analysis

| target             | model           |   cold_count |   cold_spearman |   cold_p10 |   cold_p25 |   cold_ndcg10 |
|:-------------------|:----------------|-------------:|----------------:|-----------:|-----------:|--------------:|
| breast_cancer      | A_TransformOnly |          880 |      0.0865695  |        0   |       0    |     0.886842  |
| breast_cancer      | B_Phase7        |          880 |      0.152162   |        0   |       0    |     0.933474  |
| breast_cancer      | C_Phase7_1_Full |          880 |      0.0342568  |        0   |       0    |     0.724127  |
| breast_cancer      | D_SimOnly       |          880 |      0.0944999  |        0   |       0    |     0.898381  |
| wine               | A_TransformOnly |          364 |      0.131371   |        0.1 |       0.16 |     0.923077  |
| wine               | B_Phase7        |          364 |      0.0586148  |        0.2 |       0.2  |     1         |
| wine               | C_Phase7_1_Full |          364 |      0.00371835 |        0   |       0    |     0.861138  |
| wine               | D_SimOnly       |          364 |      0.11457    |        0   |       0    |     1         |
| diabetes           | A_TransformOnly |          216 |     -0.0915781  |        0   |       0.08 |     0.44124   |
| diabetes           | B_Phase7        |          216 |      0.103806   |        0   |       0.12 |     0.51749   |
| diabetes           | C_Phase7_1_Full |          216 |     -0.0450844  |        0   |       0    |     0.444748  |
| diabetes           | D_SimOnly       |          216 |      0.0729441  |        0   |       0.12 |     0.515574  |
| california_housing | A_TransformOnly |          144 |      0.0494341  |        0.1 |       0.08 |     0.393247  |
| california_housing | B_Phase7        |          144 |      0.0452301  |        0.2 |       0.2  |     0.488217  |
| california_housing | C_Phase7_1_Full |          144 |      0.166716   |        0.3 |       0.28 |     0.695743  |
| california_housing | D_SimOnly       |          144 |      0.164918   |        0.1 |       0.28 |     0.37039   |
| iris               | A_TransformOnly |           40 |      0.00599953 |        0.2 |       0.56 |     0.502228  |
| iris               | B_Phase7        |           40 |     -0.154372   |        0.2 |       0.56 |     0.555514  |
| iris               | C_Phase7_1_Full |           40 |      0.0885658  |        0.1 |       0.72 |     0.557283  |
| iris               | D_SimOnly       |           40 |     -0.0770195  |        0.2 |       0.64 |     0.526309  |
| titanic            | A_TransformOnly |         9068 |     -0.126288   |        0   |       0.04 |     0.993783  |
| titanic            | B_Phase7        |         9068 |     -0.0122433  |        0   |       0    |     0.991563  |
| titanic            | C_Phase7_1_Full |         9068 |     -0.0727591  |        0   |       0    |     0.994094  |
| titanic            | D_SimOnly       |         9068 |     -0.131398   |        0   |       0    |     0.990365  |
| credit_g           | A_TransformOnly |         1000 |     -0.40758    |        0   |       0    |     0.383432  |
| credit_g           | B_Phase7        |         1000 |     -0.0681216  |        0   |       0    |     0.366809  |
| credit_g           | C_Phase7_1_Full |         1000 |     -0.162632   |        0   |       0    |     0.399361  |
| credit_g           | D_SimOnly       |         1000 |     -0.0877043  |        0   |       0    |     0.417724  |
| blood_transfusion  | A_TransformOnly |           40 |     -0.185859   |        0.2 |       0.52 |     0.254037  |
| blood_transfusion  | B_Phase7        |           40 |      0.10712    |        0.2 |       0.6  |     0.281357  |
| blood_transfusion  | C_Phase7_1_Full |           40 |     -0.174269   |        0.1 |       0.64 |     0.0491432 |
| blood_transfusion  | D_SimOnly       |           40 |     -0.152279   |        0.2 |       0.56 |     0.311091  |
| vehicle            | A_TransformOnly |          684 |      0.0640279  |        0   |       0.04 |     0.443293  |
| vehicle            | B_Phase7        |          684 |     -0.13827    |        0.1 |       0.04 |     0.56417   |
| vehicle            | C_Phase7_1_Full |          684 |     -0.160072   |        0.1 |       0.08 |     0.61392   |
| vehicle            | D_SimOnly       |          684 |     -0.0424839  |        0   |       0.04 |     0.554853  |
| spambase           | A_TransformOnly |          988 |      0.0912405  |        0   |       0    |     0.4247    |
| spambase           | B_Phase7        |          988 |      0.0131971  |        0   |       0    |     0.432848  |
| spambase           | C_Phase7_1_Full |          988 |      0.0647976  |        0.1 |       0.16 |     0.591614  |
| spambase           | D_SimOnly       |          988 |     -0.0208823  |        0   |       0.12 |     0.460584  |

## 14. Similarity Evidence Coverage

| target             |   mean_sim_weight |   median_sim_weight |   min_sim_weight |   max_sim_weight |   pct_nonzero_sim |   nonzero_count |
|:-------------------|------------------:|--------------------:|-----------------:|-----------------:|------------------:|----------------:|
| breast_cancer      |         571.051   |          559.272    |       559.272    |         645.651  |                 1 |             880 |
| wine               |         708.239   |          709.212    |       702.407    |         709.212  |                 1 |             364 |
| diabetes           |         749.427   |          759.824    |       702.642    |         759.824  |                 1 |             220 |
| california_housing |         194.835   |          159.778    |       159.778    |         317.535  |                 1 |             144 |
| iris               |         592.263   |          566.03     |       566.03     |         631.614  |                 1 |              40 |
| titanic            |           1.26494 |            0.989754 |         0.989754 |           4.2746 |                 1 |            9072 |
| credit_g           |         350.417   |          268.852    |       268.852    |         604.473  |                 1 |            1004 |
| blood_transfusion  |         662.641   |          662.522    |       662.522    |         662.82   |                 1 |              40 |
| vehicle            |         643.077   |          639.225    |       639.225    |         675.812  |                 1 |             684 |
| spambase           |         292.844   |          248.322    |       248.322    |         441.25   |                 1 |             988 |

## 15. Prediction Variance

| model           |   actual_std |     pred_std |   std_ratio |
|:----------------|-------------:|-------------:|------------:|
| A_TransformOnly |    0.0117022 |   0.00175978 |    0.321912 |
| B_Phase7        |    0.0117022 |   0.00162231 |    0.350371 |
| C_Phase7_1_Full |    0.0117022 |   0.00193341 |    0.339816 |
| D_SimOnly       |    0.0117022 |   0.00310552 |    0.405957 |
| GlobalMean      |    0.0117022 |   0          |    0        |
| Random          |    0.0117022 | nan          |  nan        |

## 16. Uncertainty

| model           |        mean |      median |         std |        min |        max |
|:----------------|------------:|------------:|------------:|-----------:|-----------:|
| A_TransformOnly | 0.000473393 | 0.000274235 | 0.000501064 | 6.9252e-05 | 0.00146626 |
| B_Phase7        | 0.0101036   | 0.0103129   | 0.00169302  | 0.00556583 | 0.0136887  |
| C_Phase7_1_Full | 0.0109186   | 0.0110201   | 0.00176983  | 0.00588911 | 0.0144799  |
| D_SimOnly       | 0.0119825   | 0.0117589   | 0.00303243  | 0.00584902 | 0.0201109  |

## 17. Feature Importance

| model           | group           |   importance |
|:----------------|:----------------|-------------:|
| A_TransformOnly | transformation  |   1          |
| B_Phase7        | dataset_context |   0.00257623 |
| B_Phase7        | exact_history   |   0.973828   |
| B_Phase7        | feature_slot_1  |   0.0120281  |
| B_Phase7        | feature_slot_2  |   0.00862121 |
| B_Phase7        | transformation  |   0.00294657 |
| C_Phase7_1_Full | dataset_context |   0.00139563 |
| C_Phase7_1_Full | exact_history   |   0.973964   |
| C_Phase7_1_Full | feature_slot_1  |   0.0113258  |
| C_Phase7_1_Full | feature_slot_2  |   0.0076608  |
| C_Phase7_1_Full | sim_history     |   0.00449477 |
| C_Phase7_1_Full | transformation  |   0.00115903 |
| D_SimOnly       | dataset_context |   0.00978417 |
| D_SimOnly       | feature_slot_1  |   0.0878608  |
| D_SimOnly       | feature_slot_2  |   0.253556   |
| D_SimOnly       | sim_history     |   0.365793   |
| D_SimOnly       | transformation  |   0.283007   |

## 18. Statistical Comparison

| comparison                   | metric          |   statistic |   p_value |   n_samples |
|:-----------------------------|:----------------|------------:|----------:|------------:|
| D_SimOnly vs Random          | spearman        |          25 | 0.845703  |          10 |
| D_SimOnly vs Random          | ndcg_at_10      |          22 | 0.625     |          10 |
| D_SimOnly vs Random          | precision_at_10 |           0 | 0.0625    |          10 |
| D_SimOnly vs A_TransformOnly | spearman        |          22 | 0.625     |          10 |
| D_SimOnly vs A_TransformOnly | ndcg_at_10      |           4 | 0.0136719 |          10 |
| D_SimOnly vs A_TransformOnly | precision_at_10 |           0 | 0.5       |          10 |
| D_SimOnly vs B_Phase7        | spearman        |          26 | 0.921875  |          10 |
| D_SimOnly vs B_Phase7        | ndcg_at_10      |          22 | 1         |          10 |
| D_SimOnly vs B_Phase7        | precision_at_10 |           0 | 0.25      |          10 |

## 19. Failure Cases

The similarity-weighted model (D_SimOnly) exhibits negative Spearman correlations on half the datasets (iris, titanic, credit_g, blood_transfusion, vehicle, spambase). Precision@10 is 0.0 on 6 out of 10 datasets. This demonstrates a fundamental lack of generalized predictive signal.

## 20. Limitations

- Transformation overlap is 100%, but actual utility of identical transformations on different datasets varies wildly based on data distribution, meaning dataset meta-features are insufficient to contextualize the transformation.

## 21. Scientific Interpretation

Observed result: Increasing the historical dataset pool from 3 to 10 did not yield a positive candidate-level predictive signal. P@10 remains zero frequently, Spearman is commonly negative, and variance remains highly under-dispersed.
Interpretation: CAFET's core mechanism—using high-level dataset meta-features to transfer feature utility—fails. The mapping from dataset macro-statistics to specific candidate utility is either too complex for 10 datasets to capture, or fundamentally non-existent. A transformation like `ADD(f1, f2)` depends heavily on the joint distribution of `f1` and `f2` and the target, which macro-level dataset context vectors (e.g. % missing, % categorical) cannot express.

## 22. Phase Verdict

FAIL
