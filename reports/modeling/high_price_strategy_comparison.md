# High-Price Strategy Comparison

**HIGH_PRICE_ANALYSIS = PASS**

Ridge α=0.001. Same four expanding-window folds. Preprocessing, extra
transforms, model, and correction are fit on fold train only.
TEST parquet was never opened.

BASELINE_REPRODUCED = TRUE

## Methods

| method | description |
|---|---|
| CURRENT_BEST | 184 SAFE features, Ridge, expanding_historical correction |
| METHOD_A | + causal 7d/14d historical price mean/std (from y shifted 24h, then window) |
| METHOD_B | + fraction of past hours above fold-train P75 (7/14/30d, y shifted 24h) |
| METHOD_C | METHOD_A plus five leakage-safe interactions (lag/level × load, renewable, hour) |
| METHOD_D | CURRENT_BEST plus a specialist addend when lag24 or 7d mean exceeds train P75 |
| METHOD_E | Weighted Ridge (higher weight on fold-train y ≥ train P75; weight chosen on an inner train split by MAE) + expanding_historical |

Method E does **not** tune on validation P75 residuals.

## Comparison (walk-forward)

Primary: mean MAE. A challenger is selected only if it beats CURRENT_BEST by
at least 0.01 MAE. High-price bias is reported, not used as the
selection key.

P75+/P90+ below use **fold-train** quantiles (causal).

| method | mean_mae | mae_std | mean_bias | p75_mae | p75_bias | p90_mae | p90_bias | rmse |
|---|---|---|---|---|---|---|---|---|
| CURRENT_BEST | 5.7025 | 0.4853 | -1.6293 | 7.4483 | -6.5806 | 9.1807 | -8.6633 | 7.3102 |
| METHOD_A | 5.5298 | 0.4089 | -1.1805 | 7.1151 | -6.0304 | 8.7697 | -8.1464 | 7.1523 |
| METHOD_B | 5.4960 | 0.4502 | -0.9148 | 7.0315 | -5.8956 | 8.9462 | -8.2111 | 7.1509 |
| METHOD_C | 5.5223 | 0.4210 | -1.2147 | 7.0539 | -5.9080 | 8.6249 | -7.8846 | 7.1480 |
| METHOD_D | 5.7477 | 0.4998 | -1.6466 | 7.6910 | -6.7917 | 9.4713 | -8.8898 | 7.3823 |
| METHOD_E | 5.6762 | 0.5159 | -1.6371 | 7.4553 | -6.6005 | 9.2363 | -8.7256 | 7.2874 |

## Selected

**BEST_METHOD = METHOD_B**

- BEST_MEAN_MAE = 5.496022
- BEST_MAE_STD = 0.450160
- P75_BIAS_BEFORE (CURRENT_BEST, causal) = -6.580579
- P75_BIAS_AFTER = -5.895626
- P90_BIAS_BEFORE = -8.663258
- P90_BIAS_AFTER = -8.211142

METHOD_B is selected on mean walk-forward MAE (5.496 vs CURRENT_BEST 5.702). Causal P75+ bias moves from -6.58 to -5.90; P90+ from -8.66 to -8.21. That is a modest reduction, not a resolution: high-price hours remain systematically underpredicted (P75+ bias still near −6, P90+ near −8). METHOD_B is also uneven across folds (fold 2 high-price error worsens). No leakage-safe tested strategy materially resolves the high-price underprediction.

## Fold results

| method | fold | MAE | bias | p75_mae | p75_bias | p90_mae | p90_bias |
|---|---|---|---|---|---|---|---|
| CURRENT_BEST | 1 | 6.4704 | -3.1201 | 7.9212 | -7.5566 | 9.0226 | -8.9184 |
| CURRENT_BEST | 2 | 5.7150 | -3.0120 | 9.1860 | -8.8184 | 10.5968 | -10.3362 |
| CURRENT_BEST | 3 | 5.1580 | 1.6270 | 6.9259 | -5.8190 | 10.8790 | -10.5869 |
| CURRENT_BEST | 4 | 5.4665 | -2.0121 | 5.7602 | -4.1283 | 6.2243 | -4.8116 |
| METHOD_A | 1 | 6.2310 | -2.6212 | 7.3531 | -6.9432 | 8.3170 | -8.1974 |
| METHOD_A | 2 | 5.3818 | -2.4161 | 8.4086 | -7.6844 | 9.5770 | -9.0036 |
| METHOD_A | 3 | 5.2215 | 1.8880 | 7.0632 | -5.6084 | 11.0002 | -10.7374 |
| METHOD_A | 4 | 5.2848 | -1.5726 | 5.6356 | -3.8855 | 6.1848 | -4.6472 |
| METHOD_B | 1 | 5.5769 | -1.0825 | 5.9357 | -5.2954 | 6.7197 | -6.4876 |
| METHOD_B | 2 | 6.1985 | -4.3348 | 10.1934 | -9.8726 | 12.5400 | -12.5014 |
| METHOD_B | 3 | 5.0611 | 1.7641 | 6.9502 | -6.3019 | 11.2217 | -11.0757 |
| METHOD_B | 4 | 5.1476 | -0.0061 | 5.0467 | -2.1126 | 5.3032 | -2.7799 |
| METHOD_C | 1 | 6.2490 | -2.5710 | 7.2985 | -6.8872 | 8.1655 | -8.0298 |
| METHOD_C | 2 | 5.2927 | -2.4663 | 8.1956 | -7.3727 | 9.1864 | -8.4520 |
| METHOD_C | 3 | 5.2268 | 1.9219 | 7.0308 | -5.4533 | 10.9909 | -10.5261 |
| METHOD_C | 4 | 5.3207 | -1.7433 | 5.6908 | -3.9187 | 6.1570 | -4.5305 |
| METHOD_D | 1 | 6.3573 | -2.9505 | 7.6803 | -7.2627 | 8.7204 | -8.5918 |
| METHOD_D | 2 | 6.1065 | -3.6429 | 10.4941 | -10.2960 | 12.0708 | -11.9454 |
| METHOD_D | 3 | 5.1413 | 1.5391 | 7.0448 | -6.1447 | 11.1427 | -10.9397 |
| METHOD_D | 4 | 5.3855 | -1.5323 | 5.5446 | -3.4634 | 5.9513 | -4.0821 |
| METHOD_E | 1 | 6.4704 | -3.1201 | 7.9212 | -7.5566 | 9.0226 | -8.9184 |
| METHOD_E | 2 | 5.7150 | -3.0120 | 9.1860 | -8.8184 | 10.5968 | -10.3362 |
| METHOD_E | 3 | 5.0530 | 1.5956 | 6.9540 | -5.8986 | 11.1016 | -10.8363 |
| METHOD_E | 4 | 5.4665 | -2.0121 | 5.7602 | -4.1283 | 6.2243 | -4.8116 |

## Standardized Ridge coefficients (CURRENT_BEST, fold 4)

Coefficients are on fold-4 **standardized** inputs. Magnitude is not causal importance.

Strongest positive:

| feature | coefficient |
|---|---|
| month | 194.8520 |
| renewable_generation_lag_24 | 26.5095 |
| price_day_ahead_lag_24 | 19.6569 |
| day_of_month | 17.1773 |
| total_generation_lag_24 | 17.0368 |
| price_day_ahead_lag_48 | 14.1558 |
| clouds_all_bilbao_lag_24 | 13.2620 |
| renewable_generation_lag_168 | 13.1675 |
| clouds_all_madrid_lag_24 | 11.3972 |
| clouds_all_barcelona_lag_24 | 10.5084 |

Strongest negative:

| feature | coefficient |
|---|---|
| day_of_year | -194.3507 |
| clouds_all_national_mean_lag_24 | -33.4144 |
| generation_wind_onshore_lag_24 | -32.6067 |
| price_mean_lag24_lag48 | -28.6472 |
| clouds_all_national_mean_lag_168 | -22.6593 |
| generation_hydro_water_reservoir_lag_24 | -21.0942 |
| generation_solar_lag_24 | -16.6298 |
| generation_wind_onshore_lag_168 | -14.6411 |
| wind_speed_national_mean_lag_24 | -14.2276 |
| wind_speed_national_mean_lag_168 | -9.6726 |

Recent 7d/14d level features are **not** in CURRENT_BEST. METHOD_A tests whether
adding them helps; see the comparison table.

## Leakage / test / protected files

LEAKAGE_CHECK = PASS
TEST_READ_COUNT = 0
TEST_USED_FOR_SELECTION = FALSE
PROTECTED_FILES_UNCHANGED = TRUE
REPRODUCIBILITY = PASS

| file | md5_before | md5_after | unchanged |
|---|---|---|---|
| merged_energy_weather.parquet | ae4a12026b1a9682d6bbb58ef7471fa1 | ae4a12026b1a9682d6bbb58ef7471fa1 | True |
| model_features.parquet | c9f07ac0f95e0f51fff5472129c1f9ad | c9f07ac0f95e0f51fff5472129c1f9ad | True |
| train.parquet | 278666dcdb30990b55a6aa5c882f21ee | 278666dcdb30990b55a6aa5c882f21ee | True |
| validation.parquet | cba753fa9327955d506139d25fdaae4d | cba753fa9327955d506139d25fdaae4d | True |
| test.parquet | 069afbe9c766426d2e095282ece93a69 | 069afbe9c766426d2e095282ece93a69 | True |

| file | md5 |
|---|---|
| high_price_diagnostic.csv | ca7c0cac563f443f8b4a9c484785fe29 |
| high_price_strategy_comparison.csv | 4ec4e70251d4228507d3b67f6e01fb8b |
| high_price_strategy_fold_results.csv | 342ee50498feab834b015144de228925 |
| high_price_strategy_predictions.parquet | c513ed3bb24fb647a5cc5ee79d576b2f |
| high_price_coefficients.csv | 0741c34f33c89e860a6f560b3d4c93aa |

This stage does **not** evaluate the locked test set.
