# Ridge Alpha Tuning

**RIDGE_TUNING = PASS**

TEST parquet was never loaded. Alpha was selected on TRAIN+VALIDATION expanding-window folds only.
No locked-test metric was computed. Feature engineering was not changed.

`random_state = 42` is recorded for pipeline consistency. Ridge itself is a closed-form
NumPy solve and does not use a random number generator.

## 1. Alpha grid

[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

Grid was chosen before looking at any test metric. It densifies the previous walk-forward Ridge set
around the winning region (0.01) without using the locked test set.

## 2. Walk-forward methodology

Same expanding-window scheme as `src/walk_forward_validation.py`:

- Combine official TRAIN then VALIDATION, sort by `timestamp_utc` (UTC, hourly, no shuffle).
- Fold train fractions: [0.5, 0.6, 0.7, 0.8]; each validation block is the next chronological remainder up to the next cut (fold 4 uses the final 20%).
- For every fold and every alpha:
  1. Fit median impute + standard scale on **fold train only**.
  2. Transform fold validation with those statistics.
  3. Fit Ridge `(X'X + αI)w = X'y` (centered y intercept).
  4. Score MAE, RMSE, R², sMAPE, bias on that fold's validation block.
- Primary selection: mean walk-forward MAE.
- If mean MAE values are within 0.05: lower MAE std, then lower |mean bias|, then smaller alpha.

## 3. Fold timestamps

| fold | n_train | n_val | train_start | train_end | val_start | val_end | train_frac |
|---|---|---|---|---|---|---|---|
| 1 | 14902 | 2980 | 2014-12-31T23:00:00+00:00 | 2016-09-12T20:00:00+00:00 | 2016-09-12T21:00:00+00:00 | 2017-01-15T00:00:00+00:00 | 0.5000 |
| 2 | 17882 | 2980 | 2014-12-31T23:00:00+00:00 | 2017-01-15T00:00:00+00:00 | 2017-01-15T01:00:00+00:00 | 2017-05-19T04:00:00+00:00 | 0.6000 |
| 3 | 20862 | 2981 | 2014-12-31T23:00:00+00:00 | 2017-05-19T04:00:00+00:00 | 2017-05-19T05:00:00+00:00 | 2017-09-20T09:00:00+00:00 | 0.7000 |
| 4 | 23843 | 5961 | 2014-12-31T23:00:00+00:00 | 2017-09-20T09:00:00+00:00 | 2017-09-20T10:00:00+00:00 | 2018-05-26T18:00:00+00:00 | 0.8000 |

Each fold satisfies `max(train timestamp) < min(validation timestamp)`. Validation blocks do not overlap.

## 4–6. Alpha comparison (mean ± std over 4 folds)

| alpha | mean_MAE | std_MAE | mean_RMSE | std_RMSE | mean_R2 | mean_sMAPE | mean_bias | abs_mean_bias | n_folds |
|---|---|---|---|---|---|---|---|---|---|
| 0.0010 | 5.7966 | 0.6489 | 7.4142 | 0.7973 | 0.6221 | 12.8376 | -1.8966 | 1.8966 | 4.0000 |
| 0.0030 | 5.7982 | 0.6531 | 7.4158 | 0.8006 | 0.6221 | 12.8434 | -1.9094 | 1.9094 | 4.0000 |
| 0.0100 | 5.8043 | 0.6679 | 7.4219 | 0.8121 | 0.6219 | 12.8639 | -1.9507 | 1.9507 | 4.0000 |
| 0.0300 | 5.8219 | 0.7082 | 7.4398 | 0.8434 | 0.6212 | 12.9217 | -2.0464 | 2.0464 | 4.0000 |
| 0.1000 | 5.8713 | 0.8111 | 7.4902 | 0.9209 | 0.6185 | 13.0708 | -2.2445 | 2.2445 | 4.0000 |
| 0.3000 | 5.9332 | 0.9254 | 7.5573 | 1.0064 | 0.6141 | 13.2577 | -2.4576 | 2.4576 | 4.0000 |
| 1.0000 | 5.9835 | 1.0013 | 7.6160 | 1.0620 | 0.6097 | 13.4096 | -2.6201 | 2.6201 | 4.0000 |
| 3.0000 | 6.0069 | 1.0288 | 7.6447 | 1.0816 | 0.6073 | 13.4756 | -2.6896 | 2.6896 | 4.0000 |
| 10.0000 | 6.0204 | 1.0381 | 7.6619 | 1.0883 | 0.6057 | 13.5098 | -2.7195 | 2.7195 | 4.0000 |

## 7. Selected alpha

**BEST_ALPHA = 0.001**

- BEST_MEAN_MAE = 5.796612
- BEST_MAE_STD = 0.648924
- mean RMSE = 7.4142 (std 0.7973)
- mean R2 = 0.6221
- mean sMAPE = 12.8376
- mean bias = -1.8966

Selected-alpha fold detail:

| fold | val_start | val_end | y_mean | y_std | MAE | RMSE | R2 | sMAPE | bias |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2016-09-12T21:00:00+00:00 | 2017-01-15T00:00:00+00:00 | 49.0733 | 14.6807 | 6.2823 | 8.1058 | 0.6951 | 15.4300 | -2.5458 |
| 2 | 2017-01-15T01:00:00+00:00 | 2017-05-19T04:00:00+00:00 | 51.9526 | 14.8874 | 6.5728 | 8.2673 | 0.6916 | 13.7156 | -4.8531 |
| 3 | 2017-05-19T05:00:00+00:00 | 2017-09-20T09:00:00+00:00 | 50.4882 | 7.9766 | 5.0080 | 6.3736 | 0.3615 | 9.7592 | 1.0083 |
| 4 | 2017-09-20T10:00:00+00:00 | 2018-05-26T18:00:00+00:00 | 52.8151 | 13.5598 | 5.3233 | 6.9102 | 0.7403 | 12.4456 | -1.1959 |

## 8. High-price regime (selected alpha, pooled walk-forward validation blocks)

Pooled residual mean = -1.7563  
Pooled residual std = 7.1379

### Price quantiles (P25 / P50 / P75 of pooled y_true)

| quantile | MAE | bias | residual_mean | residual_std | y_mean | n |
|---|---|---|---|---|---|---|
| P25_below | 5.7336 | 1.0191 | 1.0191 | 7.3225 | 35.3267 | 3732 |
| P25_P50 | 4.5957 | -0.7758 | -0.7758 | 5.7220 | 48.0285 | 3733 |
| P50_P75 | 5.1725 | -1.1592 | -1.1592 | 6.4525 | 54.6286 | 3713 |
| P75_above | 7.3068 | -6.1157 | -6.1157 | 6.9115 | 67.7841 | 3724 |

HIGH_PRICE_BIAS (P75_above) = -6.1157  
P75_above MAE = 7.3068 vs P25_below MAE = 5.7336

The selected Ridge still systematically underpredicts the upper price quartile.

### Hour / weekday / month MAE

Worst hours: [0, 7, 1, 8, 19]  
Worst weekdays (Mon=0 … Sun=6): [6, 1, 5]  
Worst months: [1, 12, 11]

Hourly MAE range: 5.285–5.940.

Weekday table is in `reports/ridge_error_by_weekday.csv` (extra diagnostic; not used for alpha selection).

## 9. Test lock

TEST_USED_FOR_TUNING = FALSE

`read_parquet_locked` raises if the path is `data/processed/test.parquet`.
No test prediction, MAE, RMSE, R², sMAPE, or bias was computed.

## 10. Protected file hashes

LEAKAGE_CHECK = PASS  
PROTECTED_FILES_UNCHANGED = TRUE

| file | md5_before | md5_after | unchanged |
|---|---|---|---|
| merged_energy_weather.parquet | ae4a12026b1a9682d6bbb58ef7471fa1 | ae4a12026b1a9682d6bbb58ef7471fa1 | True |
| model_features.parquet | c9f07ac0f95e0f51fff5472129c1f9ad | c9f07ac0f95e0f51fff5472129c1f9ad | True |
| train.parquet | 278666dcdb30990b55a6aa5c882f21ee | 278666dcdb30990b55a6aa5c882f21ee | True |
| validation.parquet | cba753fa9327955d506139d25fdaae4d | cba753fa9327955d506139d25fdaae4d | True |
| test.parquet | 069afbe9c766426d2e095282ece93a69 | 069afbe9c766426d2e095282ece93a69 | True |

## 11. Reproducibility

REPRODUCIBILITY = PASS

The pipeline is executed twice in one process. Comparison / fold / prediction / error hashes must match.

| file | md5 |
|---|---|
| ridge_alpha_comparison.csv | 235aceebc272853f0c8a0bb146920057 |
| ridge_tuning_fold_results.csv | e8f47e3fe800a98596048f7ea78d2ad4 |
| ridge_walk_forward_validation_predictions.parquet | 2d4109b8348be68d5194e9e6a8500f35 |
| ridge_error_by_hour.csv | 2d7d954ff96935434fca6d245531f113 |
| ridge_error_by_month.csv | 8b845704172f40e0bfe0a52fe4a739ad |
| ridge_error_by_weekday.csv | 8e6d539f85f31f7061e0d18b82c8abb9 |
| ridge_error_by_price_quantile.csv | 47d92ea096f41f7c87e9b982bc77db3b |

## 12. Final status

RIDGE_TUNING = PASS
TEST_USED_FOR_TUNING = FALSE
LEAKAGE_CHECK = PASS
PROTECTED_FILES_UNCHANGED = TRUE
REPRODUCIBILITY = PASS
BEST_ALPHA = 0.001
BEST_MEAN_MAE = 5.796612
BEST_MAE_STD = 0.648924
HIGH_PRICE_BIAS = -6.115697

This stage does **not** evaluate a production/final model on the locked test set.
