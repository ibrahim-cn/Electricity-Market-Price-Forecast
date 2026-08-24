# Residual / Level-Shift Correction

**RESIDUAL_CORRECTION = PASS**

Ridge `alpha=0.001` is unchanged. Feature engineering, splits, fold timestamps, and
fold-train preprocessing are unchanged. TEST parquet was never opened.

`random_state = 42` is recorded for pipeline consistency. Ridge and the
corrections are deterministic closed-form NumPy.

## Baseline (no correction)

From the previous tuning stage, reproduced here as Method A:

- BEST_ALPHA = 0.001
- walk-forward mean MAE = 5.796612
- MAE std = 0.648924
- mean bias = -1.8966
- HIGH_PRICE_BIAS_BEFORE = -6.115697

## Methods

All correction coefficients use **fold-train predictions and residuals only**.
Reporting bias is `mean(pred − y)`. The additive correction is `mean(y − pred)` so
that a train underprediction raises the validation forecast.

| method | rule |
|---|---|
| no_correction | raw Ridge prediction |
| fold_train_bias | add fold-train `mean(y − pred)` (≈ 0 because Ridge has an intercept) |
| expanding_historical | add time-weighted train `mean(y − pred)`, weights `exp((t − t_max)/720h)`, causal |
| regime_aware | train-target P25/P50/P75 edges; add regime `mean(y − pred)` by **predicted** price bin |
| linear_calibration | `y = a + b * pred` OLS on fold train; apply `a + b * pred` on validation |

Quantile **edges for Method D** come from fold-train **y only**. Validation y is never
used to set thresholds or coefficients. Evaluation quantiles below use pooled
walk-forward **y_true** only after predictions exist, for reporting.

## Folds (same expanding window)

| fold | n_train | n_val | train_start | train_end | val_start | val_end | train_frac |
|---|---|---|---|---|---|---|---|
| 1 | 14902 | 2980 | 2014-12-31T23:00:00+00:00 | 2016-09-12T20:00:00+00:00 | 2016-09-12T21:00:00+00:00 | 2017-01-15T00:00:00+00:00 | 0.5000 |
| 2 | 17882 | 2980 | 2014-12-31T23:00:00+00:00 | 2017-01-15T00:00:00+00:00 | 2017-01-15T01:00:00+00:00 | 2017-05-19T04:00:00+00:00 | 0.6000 |
| 3 | 20862 | 2981 | 2014-12-31T23:00:00+00:00 | 2017-05-19T04:00:00+00:00 | 2017-05-19T05:00:00+00:00 | 2017-09-20T09:00:00+00:00 | 0.7000 |
| 4 | 23843 | 5961 | 2014-12-31T23:00:00+00:00 | 2017-09-20T09:00:00+00:00 | 2017-09-20T10:00:00+00:00 | 2018-05-26T18:00:00+00:00 | 0.8000 |

## Walk-forward comparison

Primary: mean MAE. Secondary: MAE std. Third: \|mean bias\|.
A correction is selected only if it beats no_correction by at least 0.01 MAE.

| method | mean_MAE | std_MAE | mean_RMSE | mean_sMAPE | mean_bias | abs_mean_bias | n_folds |
|---|---|---|---|---|---|---|---|
| no_correction | 5.7966 | 0.6489 | 7.4142 | 12.8376 | -1.8966 | 1.8966 | 4 |
| fold_train_bias | 5.7966 | 0.6489 | 7.4142 | 12.8376 | -1.8966 | 1.8966 | 4 |
| expanding_historical | 5.7025 | 0.4853 | 7.3102 | 12.5918 | -1.6293 | 1.6293 | 4 |
| regime_aware | 5.7793 | 0.6522 | 7.4091 | 12.7964 | -1.9358 | 1.9358 | 4 |
| linear_calibration | 5.7966 | 0.6489 | 7.4142 | 12.8377 | -1.8965 | 1.8965 | 4 |

## Selected method

**BEST_METHOD = expanding_historical**

- BEST_MEAN_MAE = 5.702475
- BEST_MAE_STD = 0.485263
- mean RMSE = 7.3102
- mean sMAPE = 12.5918
- mean bias = -1.6293

**expanding_historical** improved mean walk-forward MAE by 0.0941.

## Fold results

| method | fold | MAE | RMSE | sMAPE | bias | y_mean |
|---|---|---|---|---|---|---|
| expanding_historical | 1 | 6.4704 | 8.3041 | 15.8546 | -3.1201 | 49.0733 |
| expanding_historical | 2 | 5.7150 | 7.3395 | 11.7592 | -3.0120 | 51.9526 |
| expanding_historical | 3 | 5.1580 | 6.5002 | 9.9701 | 1.6270 | 50.4882 |
| expanding_historical | 4 | 5.4665 | 7.0971 | 12.7831 | -2.0121 | 52.8151 |
| fold_train_bias | 1 | 6.2823 | 8.1058 | 15.4300 | -2.5458 | 49.0733 |
| fold_train_bias | 2 | 6.5728 | 8.2673 | 13.7156 | -4.8531 | 51.9526 |
| fold_train_bias | 3 | 5.0080 | 6.3736 | 9.7592 | 1.0083 | 50.4882 |
| fold_train_bias | 4 | 5.3233 | 6.9102 | 12.4456 | -1.1959 | 52.8151 |
| linear_calibration | 1 | 6.2823 | 8.1057 | 15.4300 | -2.5457 | 49.0733 |
| linear_calibration | 2 | 6.5728 | 8.2672 | 13.7157 | -4.8530 | 51.9526 |
| linear_calibration | 3 | 5.0081 | 6.3737 | 9.7595 | 1.0084 | 50.4882 |
| linear_calibration | 4 | 5.3233 | 6.9102 | 12.4457 | -1.1958 | 52.8151 |
| no_correction | 1 | 6.2823 | 8.1058 | 15.4300 | -2.5458 | 49.0733 |
| no_correction | 2 | 6.5728 | 8.2673 | 13.7156 | -4.8531 | 51.9526 |
| no_correction | 3 | 5.0080 | 6.3736 | 9.7592 | 1.0083 | 50.4882 |
| no_correction | 4 | 5.3233 | 6.9102 | 12.4456 | -1.1959 | 52.8151 |
| regime_aware | 1 | 6.2506 | 8.0940 | 15.3644 | -2.5491 | 49.0733 |
| regime_aware | 2 | 6.5669 | 8.2485 | 13.7016 | -4.8734 | 51.9526 |
| regime_aware | 3 | 4.9672 | 6.3612 | 9.6649 | 0.9046 | 50.4882 |
| regime_aware | 4 | 5.3326 | 6.9326 | 12.4548 | -1.2251 | 52.8151 |

## Error by price quantile (pooled walk-forward y_true)

| method | quantile | MAE | bias | y_mean | n |
|---|---|---|---|---|---|
| expanding_historical | P25_below | 5.6193 | 1.1155 | 35.3267 | 3732 |
| expanding_historical | P25_P50 | 4.5354 | -0.5746 | 48.0285 | 3733 |
| expanding_historical | P50_P75 | 5.1945 | -1.0590 | 54.6286 | 3713 |
| expanding_historical | P75_above | 7.2732 | -6.3114 | 67.7841 | 3724 |
| fold_train_bias | P25_below | 5.7336 | 1.0191 | 35.3267 | 3732 |
| fold_train_bias | P25_P50 | 4.5957 | -0.7758 | 48.0285 | 3733 |
| fold_train_bias | P50_P75 | 5.1725 | -1.1592 | 54.6286 | 3713 |
| fold_train_bias | P75_above | 7.3068 | -6.1157 | 67.7841 | 3724 |
| linear_calibration | P25_below | 5.7337 | 1.0188 | 35.3267 | 3732 |
| linear_calibration | P25_P50 | 4.5959 | -0.7757 | 48.0285 | 3733 |
| linear_calibration | P50_P75 | 5.1726 | -1.1590 | 54.6286 | 3713 |
| linear_calibration | P75_above | 7.3065 | -6.1152 | 67.7841 | 3724 |
| no_correction | P25_below | 5.7336 | 1.0191 | 35.3267 | 3732 |
| no_correction | P25_P50 | 4.5957 | -0.7758 | 48.0285 | 3733 |
| no_correction | P50_P75 | 5.1725 | -1.1592 | 54.6286 | 3713 |
| no_correction | P75_above | 7.3068 | -6.1157 | 67.7841 | 3724 |
| regime_aware | P25_below | 5.6928 | 1.1050 | 35.3267 | 3732 |
| regime_aware | P25_P50 | 4.4567 | -0.8813 | 48.0285 | 3733 |
| regime_aware | P50_P75 | 5.2907 | -1.2996 | 54.6286 | 3713 |
| regime_aware | P75_above | 7.3212 | -6.1047 | 67.7841 | 3724 |

HIGH_PRICE_BIAS_BEFORE = -6.115697
HIGH_PRICE_BIAS_AFTER (expanding_historical) = -6.311367

No method reduced |P75+ bias| by at least 0.10 versus no_correction. The high-price underprediction is essentially unchanged by intercept-style or regime add-ons. expanding_historical improves overall MAE mainly on mid-price hours and fold 2; P75+ bias is slightly worse.

Did correction actually help? **expanding_historical** improved mean walk-forward MAE by 0.0941.

## Leakage audit

LEAKAGE_CHECK = PASS

- Preprocessing fit on fold train only (`assert_preproc_train_only`).
- Ridge fit on fold train only.
- Train residuals/calibration use fold-train `y` only.
- Validation `y` is used only to **score** after correction is frozen.
- `price actual` absent; 184 SAFE features unchanged.

## Test access audit

TEST_READ_COUNT = 0
TEST_USED_FOR_SELECTION = FALSE

`read_parquet_locked` raises if the path is `test.parquet`. No test metric was computed.

## Protected file hashes

PROTECTED_FILES_UNCHANGED = TRUE

| file | md5_before | md5_after | unchanged |
|---|---|---|---|
| merged_energy_weather.parquet | ae4a12026b1a9682d6bbb58ef7471fa1 | ae4a12026b1a9682d6bbb58ef7471fa1 | True |
| model_features.parquet | c9f07ac0f95e0f51fff5472129c1f9ad | c9f07ac0f95e0f51fff5472129c1f9ad | True |
| train.parquet | 278666dcdb30990b55a6aa5c882f21ee | 278666dcdb30990b55a6aa5c882f21ee | True |
| validation.parquet | cba753fa9327955d506139d25fdaae4d | cba753fa9327955d506139d25fdaae4d | True |
| test.parquet | 069afbe9c766426d2e095282ece93a69 | 069afbe9c766426d2e095282ece93a69 | True |

## Reproducibility

REPRODUCIBILITY = PASS

Pipeline runs twice in one process. Comparison / fold / prediction / quantile hashes must match.

| file | md5 |
|---|---|
| residual_correction_comparison.csv | b4f1af6a70e5d5c313bd46f218445b87 |
| residual_correction_fold_results.csv | 483f1e112e51e5a6b44f781fe8fde845 |
| residual_correction_predictions.parquet | ba4a41324d96087a9574015f53baa928 |
| residual_error_by_price_quantile.csv | 970dca9afac3e51edf7f0d7c390c8b2d |

## Final status

RESIDUAL_CORRECTION = PASS
TEST_READ_COUNT = 0
TEST_USED_FOR_SELECTION = FALSE
LEAKAGE_CHECK = PASS
PROTECTED_FILES_UNCHANGED = TRUE
REPRODUCIBILITY = PASS
BEST_METHOD = expanding_historical
BEST_MEAN_MAE = 5.702475
BEST_MAE_STD = 0.485263
HIGH_PRICE_BIAS_BEFORE = -6.115697
HIGH_PRICE_BIAS_AFTER = -6.311367

This stage does **not** evaluate the locked test set.
