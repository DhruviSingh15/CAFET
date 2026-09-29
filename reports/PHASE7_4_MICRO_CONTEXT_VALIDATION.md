# CAFET — Phase 7.4: Candidate Micro-Context Validation

## 1. Objective
Determine whether representing candidate utility via source-feature statistical properties (micro-context) improves candidate-level predictions, overcoming the failures of macro-level dataset similarity.

## 2. Research hypothesis
Candidate utility is driven by the specific, localised statistical relationships of the transformed features rather than global dataset constraints. Providing this micro-context will improve predictor generalisation across datasets.

## 3. Dataset Benchmark
10 LODO experiments identical to Phase 7.3 (breast_cancer, wine, diabetes, california_housing, iris, titanic, credit_g, blood_transfusion, vehicle, spambase).

## 4. Representation definition
Micro-context uses 54 dimensions: 15 per source feature, 12 pairwise interactions, 12 output feature stats. Encoded with transformation type (12 dims) and optional similarity history.

## 5. Leakage protocol
Strict 10-way LODO. Micro-context computes only over target-blind `X` characteristics. No validation/test utility is leaked into representations.

## 6. Model definitions
- A_TransformOnly
- B_Phase7
- C_Phase7_1_Full
- D_SimOnly
- E_MicroCtxOnly
- F_MicroCtxSim
- Random
- GlobalMean

## 7. Per-dataset results

| target             |   spearman_D_SimOnly |   spearman_E_MicroCtxOnly |   spearman_F_MicroCtxSim |   spearman_Random |   precision_at_10_D_SimOnly |   precision_at_10_E_MicroCtxOnly |   precision_at_10_F_MicroCtxSim |   precision_at_10_Random |   ndcg_at_10_D_SimOnly |   ndcg_at_10_E_MicroCtxOnly |   ndcg_at_10_F_MicroCtxSim |   ndcg_at_10_Random |
|:-------------------|---------------------:|--------------------------:|-------------------------:|------------------:|----------------------------:|---------------------------------:|--------------------------------:|-------------------------:|-----------------------:|----------------------------:|---------------------------:|--------------------:|
| blood_transfusion  |           -0.152279  |                 0.0604515 |                0.0260422 |       -0.0151442  |                         0.2 |                              0.5 |                             0.4 |                     0.22 |               0.311091 |                    0.619772 |                   0.46662  |            0.204769 |
| breast_cancer      |            0.0944999 |                 0.118055  |                0.123889  |        0.0325528  |                         0   |                              0   |                             0   |                     0.02 |               0.898381 |                    0.84084  |                   0.828152 |            0.827345 |
| california_housing |            0.164918  |                 0.149204  |                0.0846958 |       -0.04338    |                         0.1 |                              0.1 |                             0.1 |                     0.1  |               0.37039  |                    0.328857 |                   0.35237  |            0.370245 |
| credit_g           |           -0.0906224 |                -0.148895  |               -0.170326  |       -0.00737895 |                         0   |                              0   |                             0   |                     0    |               0.413033 |                    0.456948 |                   0.4411   |            0.392968 |
| diabetes           |            0.049354  |                -0.0184833 |               -0.01939   |       -0.0318061  |                         0   |                              0.1 |                             0   |                     0.08 |               0.493277 |                    0.513728 |                   0.49229  |            0.516381 |
| iris               |           -0.0770195 |                 0.111981  |               -0.114381  |        0.0538928  |                         0.2 |                              0.2 |                             0.2 |                     0.3  |               0.526309 |                    0.656123 |                   0.584125 |            0.624568 |
| spambase           |           -0.0208823 |                 0.0589873 |                0.0372968 |        0.0183987  |                         0   |                              0   |                             0.1 |                     0.04 |               0.460584 |                    0.480055 |                   0.556923 |            0.445974 |
| titanic            |           -0.130547  |                 0.168569  |               -0.031131  |       -0.0020653  |                         0   |                              0   |                             0   |                     0    |               0.990365 |                    0.990508 |                   0.938704 |            0.991544 |
| vehicle            |           -0.0424839 |                 0.0309696 |                0.0244438 |       -0.0129566  |                         0   |                              0   |                             0   |                     0    |               0.554853 |                    0.512454 |                   0.554446 |            0.589073 |
| wine               |            0.11457   |                 0.0448529 |                0.141452  |       -0.0182955  |                         0   |                              0   |                             0   |                     0    |               1        |                    1        |                   1        |            0.942731 |

## 8. Aggregate results

| model           |   ('spearman', 'mean') |   ('spearman', 'median') |   ('spearman', 'std') |   ('spearman', 'min') |   ('spearman', 'max') |   ('precision_at_10', 'mean') |   ('precision_at_10', 'median') |   ('precision_at_10', 'std') |   ('precision_at_10', 'min') |   ('precision_at_10', 'max') |   ('precision_at_25', 'mean') |   ('precision_at_25', 'median') |   ('precision_at_25', 'std') |   ('precision_at_25', 'min') |   ('precision_at_25', 'max') |   ('ndcg_at_10', 'mean') |   ('ndcg_at_10', 'median') |   ('ndcg_at_10', 'std') |   ('ndcg_at_10', 'min') |   ('ndcg_at_10', 'max') |   ('ndcg_at_25', 'mean') |   ('ndcg_at_25', 'median') |   ('ndcg_at_25', 'std') |   ('ndcg_at_25', 'min') |   ('ndcg_at_25', 'max') |   ('rmse', 'mean') |   ('rmse', 'median') |   ('rmse', 'std') |   ('rmse', 'min') |   ('rmse', 'max') |
|:----------------|-----------------------:|-------------------------:|----------------------:|----------------------:|----------------------:|------------------------------:|--------------------------------:|-----------------------------:|-----------------------------:|-----------------------------:|------------------------------:|--------------------------------:|-----------------------------:|-----------------------------:|-----------------------------:|-------------------------:|---------------------------:|------------------------:|------------------------:|------------------------:|-------------------------:|---------------------------:|------------------------:|------------------------:|------------------------:|-------------------:|---------------------:|------------------:|------------------:|------------------:|
| A_TransformOnly |            -0.0403267  |                0.0277168 |             0.167494  |             -0.407383 |             0.131371  |                         0.07  |                            0.05 |                    0.0823273 |                            0 |                          0.2 |                        0.156  |                           0.06  |                     0.211198 |                            0 |                        0.56  |                 0.563792 |                   0.438483 |                0.264708 |               0.254037  |                0.993783 |                 0.630039 |                   0.504296 |                0.228179 |                0.40577  |                0.994594 |          0.0131685 |            0.0109501 |        0.00876279 |        0.00248997 |         0.0290889 |
| B_Phase7        |             0.00707812 |                0.0292136 |             0.100831  |             -0.154372 |             0.152162  |                         0.09  |                            0.05 |                    0.0994429 |                            0 |                          0.2 |                        0.172  |                           0.08  |                     0.229434 |                            0 |                        0.6   |                 0.60557  |                   0.521865 |                0.268508 |               0.281357  |                1        |                 0.662709 |                   0.571362 |                0.222398 |                0.394461 |                1        |          0.0137414 |            0.0121104 |        0.00865829 |        0.00541479 |         0.029326  |
| C_Phase7_1_Full |            -0.0270258  |               -0.0281321 |             0.117563  |             -0.174269 |             0.166716  |                         0.07  |                            0.05 |                    0.0948683 |                            0 |                          0.3 |                        0.192  |                           0.06  |                     0.273122 |                            0 |                        0.72  |                 0.590229 |                   0.602767 |                0.263243 |               0.0491432 |                0.991364 |                 0.675416 |                   0.697621 |                0.205121 |                0.374807 |                0.993431 |          0.0139062 |            0.0117447 |        0.00866349 |        0.00410254 |         0.0293484 |
| D_SimOnly       |            -0.00904918 |               -0.0316831 |             0.109285  |             -0.152279 |             0.164918  |                         0.05  |                            0    |                    0.0849837 |                            0 |                          0.2 |                        0.172  |                           0.06  |                     0.242249 |                            0 |                        0.64  |                 0.601828 |                   0.509793 |                0.260517 |               0.311091  |                1        |                 0.668276 |                   0.578609 |                0.211628 |                0.41597  |                1        |          0.0137621 |            0.0127586 |        0.00903411 |        0.00168828 |         0.0290701 |
| E_MicroCtxOnly  |             0.0575693  |                0.0597194 |             0.0922725 |             -0.148895 |             0.168569  |                         0.09  |                            0    |                    0.159513  |                            0 |                          0.5 |                        0.184  |                           0.08  |                     0.237917 |                            0 |                        0.6   |                 0.639928 |                   0.56675  |                0.231319 |               0.328857  |                1        |                 0.69095  |                   0.60626  |                0.197863 |                0.453169 |                1        |          0.01433   |            0.0115894 |        0.00805053 |        0.00581646 |         0.0291313 |
| F_MicroCtxSim   |             0.0102592  |                0.025243  |             0.0986486 |             -0.170326 |             0.141452  |                         0.08  |                            0    |                    0.131656  |                            0 |                          0.4 |                        0.184  |                           0.08  |                     0.234151 |                            0 |                        0.64  |                 0.621473 |                   0.555685 |                0.221687 |               0.35237   |                1        |                 0.684391 |                   0.622058 |                0.190177 |                0.443705 |                1        |          0.0145533 |            0.0124525 |        0.008535   |        0.00617368 |         0.0295791 |
| GlobalMean      |           nan          |              nan         |           nan         |            nan        |           nan         |                         0.16  |                            0    |                    0.259058  |                            0 |                          0.8 |                        0.252  |                           0.14  |                     0.261992 |                            0 |                        0.76  |                 0.589797 |                   0.523587 |                0.249775 |               0.283228  |                0.991005 |                 0.651361 |                   0.584401 |                0.208259 |                0.452842 |                0.991813 |          0.0128008 |            0.0108558 |        0.00894496 |        0.00174024 |         0.0290575 |
| Random          |            -0.00261824 |               -0.0101678 |             0.0296614 |             -0.04338  |             0.0538928 |                         0.076 |                            0.03 |                    0.104902  |                            0 |                          0.3 |                        0.1656 |                           0.056 |                     0.239081 |                            0 |                        0.616 |                 0.59056  |                   0.552727 |                0.259243 |               0.204769  |                0.991544 |                 0.649307 |                   0.596106 |                0.213734 |                0.405456 |                0.992272 |          0.0322867 |            0.0307216 |        0.00415458 |        0.0286422  |         0.0404361 |

## 9. Statistical comparisons

| comparison                       | metric          |   statistic |   p_value |   n_samples |
|:---------------------------------|:----------------|------------:|----------:|------------:|
| F_MicroCtxSim vs Random          | spearman        |        22   | 0.625     |          10 |
| F_MicroCtxSim vs Random          | ndcg_at_10      |        21   | 0.556641  |          10 |
| F_MicroCtxSim vs Random          | precision_at_10 |         7   | 1         |          10 |
| F_MicroCtxSim vs A_TransformOnly | spearman        |        17   | 0.322266  |          10 |
| F_MicroCtxSim vs A_TransformOnly | ndcg_at_10      |         8   | 0.0488281 |          10 |
| F_MicroCtxSim vs A_TransformOnly | precision_at_10 |         4   | 1         |          10 |
| F_MicroCtxSim vs B_Phase7        | spearman        |        27   | 1         |          10 |
| F_MicroCtxSim vs B_Phase7        | ndcg_at_10      |        19   | 0.734375  |          10 |
| F_MicroCtxSim vs B_Phase7        | precision_at_10 |         6.5 | 1         |          10 |
| F_MicroCtxSim vs C_Phase7_1_Full | spearman        |        18   | 0.375     |          10 |
| F_MicroCtxSim vs C_Phase7_1_Full | ndcg_at_10      |        20   | 0.492188  |          10 |
| F_MicroCtxSim vs C_Phase7_1_Full | precision_at_10 |         4.5 | 1         |          10 |
| F_MicroCtxSim vs D_SimOnly       | spearman        |        24   | 0.769531  |          10 |
| F_MicroCtxSim vs D_SimOnly       | ndcg_at_10      |        18   | 0.652344  |          10 |
| F_MicroCtxSim vs D_SimOnly       | precision_at_10 |         0   | 0.5       |          10 |
| F_MicroCtxSim vs E_MicroCtxOnly  | spearman        |        10   | 0.0839844 |          10 |
| F_MicroCtxSim vs E_MicroCtxOnly  | ndcg_at_10      |        17   | 0.570312  |          10 |
| F_MicroCtxSim vs E_MicroCtxOnly  | precision_at_10 |         2.5 | 1         |          10 |

## 10. Cold-start results

| target             | model           |   cold_count |   cold_spearman |   cold_p10 |   cold_p25 |   cold_ndcg10 |
|:-------------------|:----------------|-------------:|----------------:|-----------:|-----------:|--------------:|
| breast_cancer      | A_TransformOnly |          880 |      0.0865695  |        0   |       0    |     0.886842  |
| breast_cancer      | B_Phase7        |          880 |      0.152162   |        0   |       0    |     0.933474  |
| breast_cancer      | C_Phase7_1_Full |          880 |      0.0342568  |        0   |       0    |     0.724127  |
| breast_cancer      | D_SimOnly       |          880 |      0.0944999  |        0   |       0    |     0.898381  |
| breast_cancer      | E_MicroCtxOnly  |          880 |      0.118055   |        0   |       0    |     0.84084   |
| breast_cancer      | F_MicroCtxSim   |          880 |      0.123889   |        0   |       0    |     0.828152  |
| wine               | A_TransformOnly |          364 |      0.131371   |        0.1 |       0.16 |     0.923077  |
| wine               | B_Phase7        |          364 |      0.0586148  |        0.2 |       0.2  |     1         |
| wine               | C_Phase7_1_Full |          364 |      0.00371835 |        0   |       0    |     0.861138  |
| wine               | D_SimOnly       |          364 |      0.11457    |        0   |       0    |     1         |
| wine               | E_MicroCtxOnly  |          364 |      0.0448529  |        0   |       0.24 |     1         |
| wine               | F_MicroCtxSim   |          364 |      0.141452   |        0   |       0    |     1         |
| diabetes           | A_TransformOnly |          216 |     -0.0915781  |        0   |       0.08 |     0.44124   |
| diabetes           | B_Phase7        |          216 |      0.103806   |        0   |       0.12 |     0.51749   |
| diabetes           | C_Phase7_1_Full |          216 |     -0.0450844  |        0   |       0    |     0.444748  |
| diabetes           | D_SimOnly       |          216 |      0.0729441  |        0   |       0.12 |     0.515574  |
| diabetes           | E_MicroCtxOnly  |          216 |      0.00761205 |        0.1 |       0.08 |     0.533731  |
| diabetes           | F_MicroCtxSim   |          216 |     -0.00100859 |        0   |       0.04 |     0.49229   |
| california_housing | A_TransformOnly |          144 |      0.0494341  |        0.1 |       0.08 |     0.393247  |
| california_housing | B_Phase7        |          144 |      0.0452301  |        0.2 |       0.2  |     0.488217  |
| california_housing | C_Phase7_1_Full |          144 |      0.166716   |        0.3 |       0.28 |     0.695743  |
| california_housing | D_SimOnly       |          144 |      0.164918   |        0.1 |       0.28 |     0.37039   |
| california_housing | E_MicroCtxOnly  |          144 |      0.149204   |        0.1 |       0.24 |     0.328857  |
| california_housing | F_MicroCtxSim   |          144 |      0.0846958  |        0.1 |       0.28 |     0.35237   |
| iris               | A_TransformOnly |           40 |      0.00599953 |        0.2 |       0.56 |     0.502228  |
| iris               | B_Phase7        |           40 |     -0.154372   |        0.2 |       0.56 |     0.555514  |
| iris               | C_Phase7_1_Full |           40 |      0.0885658  |        0.1 |       0.72 |     0.557283  |
| iris               | D_SimOnly       |           40 |     -0.0770195  |        0.2 |       0.64 |     0.526309  |
| iris               | E_MicroCtxOnly  |           40 |      0.111981   |        0.2 |       0.6  |     0.656123  |
| iris               | F_MicroCtxSim   |           40 |     -0.114381   |        0.2 |       0.52 |     0.584125  |
| titanic            | A_TransformOnly |         9068 |     -0.126288   |        0   |       0.04 |     0.993783  |
| titanic            | B_Phase7        |         9068 |     -0.0122433  |        0   |       0    |     0.991563  |
| titanic            | C_Phase7_1_Full |         9068 |     -0.0727591  |        0   |       0    |     0.994094  |
| titanic            | D_SimOnly       |         9068 |     -0.131398   |        0   |       0    |     0.990365  |
| titanic            | E_MicroCtxOnly  |         9068 |      0.170128   |        0   |       0    |     0.993084  |
| titanic            | F_MicroCtxSim   |         9068 |     -0.031826   |        0   |       0    |     0.938704  |
| credit_g           | A_TransformOnly |         1000 |     -0.40758    |        0   |       0    |     0.383432  |
| credit_g           | B_Phase7        |         1000 |     -0.0681216  |        0   |       0    |     0.366809  |
| credit_g           | C_Phase7_1_Full |         1000 |     -0.162632   |        0   |       0    |     0.399361  |
| credit_g           | D_SimOnly       |         1000 |     -0.0877043  |        0   |       0    |     0.417724  |
| credit_g           | E_MicroCtxOnly  |         1000 |     -0.14965    |        0   |       0    |     0.456948  |
| credit_g           | F_MicroCtxSim   |         1000 |     -0.168092   |        0   |       0    |     0.442718  |
| blood_transfusion  | A_TransformOnly |           40 |     -0.185859   |        0.2 |       0.52 |     0.254037  |
| blood_transfusion  | B_Phase7        |           40 |      0.10712    |        0.2 |       0.6  |     0.281357  |
| blood_transfusion  | C_Phase7_1_Full |           40 |     -0.174269   |        0.1 |       0.64 |     0.0491432 |
| blood_transfusion  | D_SimOnly       |           40 |     -0.152279   |        0.2 |       0.56 |     0.311091  |
| blood_transfusion  | E_MicroCtxOnly  |           40 |      0.0604515  |        0.5 |       0.6  |     0.619772  |
| blood_transfusion  | F_MicroCtxSim   |           40 |      0.0260422  |        0.4 |       0.64 |     0.46662   |
| vehicle            | A_TransformOnly |          684 |      0.0640279  |        0   |       0.04 |     0.443293  |
| vehicle            | B_Phase7        |          684 |     -0.13827    |        0.1 |       0.04 |     0.56417   |
| vehicle            | C_Phase7_1_Full |          684 |     -0.160072   |        0.1 |       0.08 |     0.61392   |
| vehicle            | D_SimOnly       |          684 |     -0.0424839  |        0   |       0.04 |     0.554853  |
| vehicle            | E_MicroCtxOnly  |          684 |      0.0309696  |        0   |       0.08 |     0.512454  |
| vehicle            | F_MicroCtxSim   |          684 |      0.0244438  |        0   |       0.12 |     0.554446  |
| spambase           | A_TransformOnly |          988 |      0.0912405  |        0   |       0    |     0.4247    |
| spambase           | B_Phase7        |          988 |      0.0131971  |        0   |       0    |     0.432848  |
| spambase           | C_Phase7_1_Full |          988 |      0.0647976  |        0.1 |       0.16 |     0.591614  |
| spambase           | D_SimOnly       |          988 |     -0.0208823  |        0   |       0.12 |     0.460584  |
| spambase           | E_MicroCtxOnly  |          988 |      0.0589873  |        0   |       0    |     0.480055  |
| spambase           | F_MicroCtxSim   |          988 |      0.0372968  |        0.1 |       0.24 |     0.556923  |

## 11. Feature importance

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
| E_MicroCtxOnly  | output_stats    |   0.444066   |
| E_MicroCtxOnly  | pairwise_stats  |   0.0428366  |
| E_MicroCtxOnly  | source1_stats   |   0.0655161  |
| E_MicroCtxOnly  | source2_stats   |   0.22075    |
| E_MicroCtxOnly  | transformation  |   0.226832   |
| F_MicroCtxSim   | dataset_context |   0.00826957 |
| F_MicroCtxSim   | output_stats    |   0.311719   |
| F_MicroCtxSim   | pairwise_stats  |   0.0434919  |
| F_MicroCtxSim   | sim_history     |   0.225307   |
| F_MicroCtxSim   | source1_stats   |   0.0412459  |
| F_MicroCtxSim   | source2_stats   |   0.211235   |
| F_MicroCtxSim   | transformation  |   0.158732   |

## 12. Prediction variance

| model           |   actual_std |     pred_std |   std_ratio |
|:----------------|-------------:|-------------:|------------:|
| A_TransformOnly |    0.0117022 |   0.00175978 |    0.321912 |
| B_Phase7        |    0.0117022 |   0.00162231 |    0.350371 |
| C_Phase7_1_Full |    0.0117022 |   0.00193341 |    0.339816 |
| D_SimOnly       |    0.0117022 |   0.00310552 |    0.405957 |
| E_MicroCtxOnly  |    0.0117022 |   0.00467645 |    1.10959  |
| F_MicroCtxSim   |    0.0117022 |   0.00422873 |    0.812243 |
| GlobalMean      |    0.0117022 |   0          |    0        |
| Random          |    0.0117022 | nan          |  nan        |

## 13. Reproducibility
Verified. Second run matched predictions exactly within Random Forest stochasticity limits (controlled by `random_state=42`).

## 14. Failure analysis
Micro-context representation failed to produce a reliable, generalizable signal across datasets.

## 15. Limitations
Requires access to the raw input features during encoding. Relies entirely on unsupervised summary statistics.

## 16. Final verdict
FAIL
