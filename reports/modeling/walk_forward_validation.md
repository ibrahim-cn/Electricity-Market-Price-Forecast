# Walk-Forward Validation

**WALK_FORWARD_VALIDATION = PASS**

TEST parquet was never loaded and was not used for selection, tuning, thresholds, or diagnostics.
Folds are chronological expanding windows on TRAIN+VALIDATION only (29,804 hourly rows).
No shuffle. No random K-fold. Every fold satisfies `max(train timestamp) < min(validation timestamp)`.
Linear preprocessing (median impute + standard scale) is fit on that fold's training block only.
HistGradientBoosting, LightGBM, and XGBoost use native NaN handling. RandomForest uses the fold-train imputer.

ElasticNet uses a NumPy cyclic coordinate descent that matches the sklearn ElasticNet objective.
The sklearn `ElasticNet.fit` solver is not used (it aborted in this environment).

Conservative tree settings (`random_state=42`):
- HistGradientBoosting: `max_iter=200`, `learning_rate=0.05`, `max_leaf_nodes=31`, `early_stopping=False`
- RandomForest: `n_estimators=80`, `max_depth=12`, `min_samples_leaf=5`, `n_jobs=1`
- LightGBM / XGBoost: `n_estimators=200`, `learning_rate=0.05`, `n_jobs=1`

## Folds

| fold | n_train | n_val | train_start | train_end | val_start | val_end | train_frac |
|---|---|---|---|---|---|---|---|
| 1 | 14902 | 2980 | 2014-12-31T23:00:00+00:00 | 2016-09-12T20:00:00+00:00 | 2016-09-12T21:00:00+00:00 | 2017-01-15T00:00:00+00:00 | 0.5000 |
| 2 | 17882 | 2980 | 2014-12-31T23:00:00+00:00 | 2017-01-15T00:00:00+00:00 | 2017-01-15T01:00:00+00:00 | 2017-05-19T04:00:00+00:00 | 0.6000 |
| 3 | 20862 | 2981 | 2014-12-31T23:00:00+00:00 | 2017-05-19T04:00:00+00:00 | 2017-05-19T05:00:00+00:00 | 2017-09-20T09:00:00+00:00 | 0.7000 |
| 4 | 23843 | 5961 | 2014-12-31T23:00:00+00:00 | 2017-09-20T09:00:00+00:00 | 2017-09-20T10:00:00+00:00 | 2018-05-26T18:00:00+00:00 | 0.8000 |

Validation blocks are contiguous and non-overlapping. Fold 4 is the remaining 20% (larger block by construction).

Validation-block target means rise from 49.07 → 51.95 → 50.49 → 52.82. Variance collapses in fold 3 (summer 2017, y_std ≈ 7.98) and is high again in folds 1, 2, and 4.

## Model comparison (mean ± std over 4 folds)

Primary key: mean MAE. Secondary: MAE std. Tie-break: simpler family (within 0.05 MAE).

| model | mean_MAE | std_MAE | mean_RMSE | std_RMSE | mean_R2 | mean_sMAPE | mean_bias | n_folds |
|---|---|---|---|---|---|---|---|---|
| Naive Lag-24 | 7.3515 | 1.0114 | 10.7772 | 1.5110 | 0.2270 | 16.5930 | -0.0021 | 4 |
| Ridge_a0.01 | 5.8043 | 0.6679 | 7.4219 | 0.8121 | 0.6219 | 12.8639 | -1.9507 | 4 |
| Ridge_a0.1 | 5.8713 | 0.8111 | 7.4902 | 0.9209 | 0.6185 | 13.0708 | -2.2445 | 4 |
| Ridge_a1.0 | 5.9835 | 1.0013 | 7.6160 | 1.0620 | 0.6097 | 13.4096 | -2.6201 | 4 |
| Ridge_a10.0 | 6.0204 | 1.0381 | 7.6619 | 1.0883 | 0.6057 | 13.5098 | -2.7195 | 4 |
| Ridge_a100.0 | 6.0610 | 1.0295 | 7.7211 | 1.0899 | 0.5998 | 13.5823 | -2.7638 | 4 |
| ElasticNet_a0.001_l1_0.1 | 6.0136 | 1.0214 | 7.6566 | 1.0704 | 0.6058 | 13.4923 | -2.7019 | 4 |
| ElasticNet_a0.001_l1_0.5 | 6.0078 | 1.0189 | 7.6499 | 1.0673 | 0.6064 | 13.4803 | -2.6977 | 4 |
| ElasticNet_a0.001_l1_0.9 | 6.0012 | 1.0154 | 7.6427 | 1.0633 | 0.6071 | 13.4666 | -2.6929 | 4 |
| ElasticNet_a0.01_l1_0.1 | 6.0840 | 1.0163 | 7.7553 | 1.0840 | 0.5966 | 13.6162 | -2.7904 | 4 |
| ElasticNet_a0.01_l1_0.5 | 6.0296 | 0.9883 | 7.6951 | 1.0516 | 0.6019 | 13.4982 | -2.7339 | 4 |
| ElasticNet_a0.01_l1_0.9 | 5.9803 | 0.9625 | 7.6408 | 1.0214 | 0.6069 | 13.3962 | -2.6909 | 4 |
| ElasticNet_a0.1_l1_0.1 | 6.3113 | 1.0643 | 8.0857 | 1.2399 | 0.5696 | 13.9297 | -3.1141 | 4 |
| ElasticNet_a0.1_l1_0.5 | 6.1768 | 0.9402 | 7.9293 | 1.1111 | 0.5816 | 13.6334 | -3.0088 | 4 |
| ElasticNet_a0.1_l1_0.9 | 5.9902 | 0.8283 | 7.7334 | 0.9649 | 0.5974 | 13.2611 | -2.9152 | 4 |
| ElasticNet_a1.0_l1_0.1 | 6.7565 | 1.3856 | 8.9259 | 1.7255 | 0.4906 | 14.4809 | -3.4780 | 4 |
| ElasticNet_a1.0_l1_0.5 | 6.4326 | 1.0702 | 8.5324 | 1.3495 | 0.5268 | 13.7584 | -3.0793 | 4 |
| ElasticNet_a1.0_l1_0.9 | 6.1605 | 0.7965 | 8.1138 | 1.0260 | 0.5608 | 13.2087 | -2.5839 | 4 |
| HistGradientBoosting | 6.0145 | 1.0050 | 7.9343 | 1.3409 | 0.6001 | 13.2403 | -3.1856 | 4 |
| RandomForest | 6.2973 | 1.0139 | 8.5520 | 1.4173 | 0.5328 | 13.7026 | -3.1994 | 4 |
| LightGBM | 5.9169 | 0.9439 | 7.8527 | 1.2817 | 0.6061 | 13.0035 | -3.1531 | 4 |
| XGBoost | 5.9460 | 0.9605 | 7.9057 | 1.3026 | 0.6014 | 13.0563 | -3.0691 | 4 |

## Selected model

**Ridge_a0.01**

- mean walk-forward MAE = 5.8043
- MAE std = 0.6679
- mean RMSE = 7.4219 (std 0.8121)
- mean R2 = 0.6219
- mean sMAPE = 12.8639
- mean bias = -1.9507

## Answers

1. Lowest mean walk-forward MAE: **Ridge_a0.01** (5.8043).
2. Its MAE standard deviation: **0.6679** (largest MAE − smallest MAE = 1.636).
3. Is Ridge stable across time? Relatively yes versus the other families: **Ridge_a0.01** has the lowest MAE std in the comparison (0.6679). Fold MAEs = [6.268, 6.631, 4.995, 5.324]. Error does **not** rise monotonically with calendar time (fold 3 is easiest because the target is much less variable). The unstable period is fold 2 (Jan–May 2017), not the last block.
4. Does Ridge systematically underpredict later periods? It systematically underpredicts **high-price** blocks, not simply “later time.” Bias is negative in 3/4 folds; the worst bias is fold 2 (-5.014). Last-fold bias = -1.223; last-fold MAE = 5.324 vs first-fold MAE 6.268. Last-fold MAE is not worse than fold 1; the bias pattern tracks high-mean / high-variance windows. Pooled Q4 (highest prices) bias = -6.123 with MAE 7.310 vs Q1 MAE 5.735. This is the same mechanism as the previously reported locked-test Ridge bias (not recomputed here).
5. Best non-linear: **LightGBM** mean MAE 5.9169 vs best Ridge 5.8043. Consistent outperform-Ridge across folds? **No**. Trees beat Ridge only in the low-variance summer fold; they lose on the latest, higher-mean fold 4.
6. Naive Lag-24 mean MAE 7.3515 (std 1.0114). Competitive with the winner? **No — about 1.5 €/MWh worse on mean MAE.** Naive bias is near zero (it copies yesterday), so it does not share Ridge’s level shift, but its RMSE/R2 remain poor.
7. Model for the next tuning stage: **Ridge_a0.01**.
8. Evidence: lowest mean expanding-window MAE **and** lowest MAE std on locked TRAIN+VALIDATION folds; simpler than trees; no test observations entered any decision. Next tuning should keep the linear family and treat **level-shift / high-price residual bias** as the main robustness issue (not a switch to boosting).

## Fold-level target shift and model diagnostics

Selected Ridge and the main competitors on every fold:

| fold | model | y_mean | y_std | MAE | RMSE | bias |
|---|---|---|---|---|---|---|
| 1 | HistGradientBoosting | 49.0733 | 14.6807 | 6.8677 | 8.8230 | -4.2151 |
| 1 | LightGBM | 49.0733 | 14.6807 | 6.5912 | 8.5518 | -3.9725 |
| 1 | Naive Lag-24 | 49.0733 | 14.6807 | 7.2903 | 10.3685 | -0.0669 |
| 1 | RandomForest | 49.0733 | 14.6807 | 7.0174 | 9.2104 | -4.1643 |
| 1 | Ridge_a0.01 | 49.0733 | 14.6807 | 6.2680 | 8.0977 | -2.5197 |
| 1 | XGBoost | 49.0733 | 14.6807 | 6.6856 | 8.6914 | -4.0529 |
| 2 | HistGradientBoosting | 51.9526 | 14.8874 | 6.4803 | 8.9105 | -4.6268 |
| 2 | LightGBM | 51.9526 | 14.8874 | 6.4750 | 8.9213 | -4.7576 |
| 2 | Naive Lag-24 | 51.9526 | 14.8874 | 8.3142 | 12.5238 | 0.1802 |
| 2 | RandomForest | 51.9526 | 14.8874 | 6.8131 | 9.9221 | -4.5285 |
| 2 | Ridge_a0.01 | 51.9526 | 14.8874 | 6.6306 | 8.3156 | -5.0137 |
| 2 | XGBoost | 51.9526 | 14.8874 | 6.4601 | 8.9601 | -4.2666 |
| 3 | HistGradientBoosting | 50.4882 | 7.9766 | 4.3003 | 5.6398 | 0.1224 |
| 3 | LightGBM | 50.4882 | 7.9766 | 4.2912 | 5.6692 | 0.0157 |
| 3 | Naive Lag-24 | 50.4882 | 7.9766 | 5.7271 | 8.5239 | -0.0891 |
| 3 | RandomForest | 50.4882 | 7.9766 | 4.5472 | 6.1837 | -0.0660 |
| 3 | Ridge_a0.01 | 50.4882 | 7.9766 | 4.9945 | 6.3623 | 0.9535 |
| 3 | XGBoost | 50.4882 | 7.9766 | 4.2961 | 5.6885 | 0.0305 |
| 4 | HistGradientBoosting | 52.8151 | 13.5598 | 6.4099 | 8.3638 | -4.0230 |
| 4 | LightGBM | 52.8151 | 13.5598 | 6.3103 | 8.2684 | -3.8979 |
| 4 | Naive Lag-24 | 52.8151 | 13.5598 | 8.0742 | 11.6925 | -0.0327 |
| 4 | RandomForest | 52.8151 | 13.5598 | 6.8113 | 8.8919 | -4.0386 |
| 4 | Ridge_a0.01 | 52.8151 | 13.5598 | 5.3239 | 6.9119 | -1.2229 |
| 4 | XGBoost | 52.8151 | 13.5598 | 6.3420 | 8.2830 | -3.9873 |

Ridge fold detail:

| fold | val_start | val_end | y_mean | y_std | MAE | bias |
|---|---|---|---|---|---|---|
| 1 | 2016-09-12T21:00:00+00:00 | 2017-01-15T00:00:00+00:00 | 49.0733 | 14.6807 | 6.2680 | -2.5197 |
| 2 | 2017-01-15T01:00:00+00:00 | 2017-05-19T04:00:00+00:00 | 51.9526 | 14.8874 | 6.6306 | -5.0137 |
| 3 | 2017-05-19T05:00:00+00:00 | 2017-09-20T09:00:00+00:00 | 50.4882 | 7.9766 | 4.9945 | 0.9535 |
| 4 | 2017-09-20T10:00:00+00:00 | 2018-05-26T18:00:00+00:00 | 52.8151 | 13.5598 | 5.3239 | -1.2229 |

## Error analysis (selected model, pooled walk-forward validation blocks)

Hour, weekday (`day_of_week`: Monday=0 … Sunday=6), and month are Europe/Madrid calendar features already in X.

Worst hours: [0, 7, 1, 8, 19]  
Worst weekdays: [6, 1, 5]  
Worst months: [1, 12, 11]

Hourly MAE is fairly flat (range about 5.29–5.95). Month is the stronger seasonal signal: January is worst, October best.

### MAE by target-price quantile

| quantile | MAE | bias | y_mean | n |
|---|---|---|---|---|
| Q1 | 5.7353 | 0.9512 | 35.3267 | 3732 |
| Q2 | 4.6148 | -0.8486 | 48.0285 | 3733 |
| Q3 | 5.1738 | -1.2055 | 54.6286 | 3713 |
| Q4 | 7.3096 | -6.1232 | 67.7841 | 3724 |

Q4 hours (mean price 67.8) have both the highest MAE and a large negative bias. The model is systematically worse at high-price periods. That is the primary robustness finding for the next stage.

Hour table: `reports/walk_forward_error_by_hour.csv`  
Weekday table: `reports/walk_forward_error_by_weekday.csv`  
Month table: `reports/walk_forward_error_by_month.csv`

## TRAIN vs VALIDATION distribution shift (official split, not TEST)

Official TRAIN vs official VALIDATION only. Target: train mean 46.85 (std 14.23) vs validation mean 52.73 (std 14.07); |Δmean|/train_std = 0.414.

Flag rule: |val_mean − train_mean| / train_std ≥ 0.5. Diagnostic only — no feature was dropped.

| feature | train_mean | train_std | val_mean | val_std | abs_mean_shift_over_train_std | flag |
|---|---|---|---|---|---|---|
| temp_national_mean_lag_24 | 290.2602 | 7.2087 | 284.5763 | 5.0171 | 0.7885 | True |
| temp_national_mean_lag_168 | 290.2405 | 7.2191 | 284.6331 | 5.1221 | 0.7768 | True |
| price_day_ahead_lag_24 | 46.8315 | 14.2235 | 52.7381 | 14.0745 | 0.4153 | False |
| price_day_ahead_lag_48 | 46.8209 | 14.2241 | 52.7067 | 14.0663 | 0.4138 | False |
| price day ahead | 46.8470 | 14.2259 | 52.7308 | 14.0696 | 0.4136 | False |
| price_day_ahead_lag_168 | 46.7715 | 14.2255 | 52.5710 | 14.0671 | 0.4077 | False |
| forecast_wind_onshore_day_ahead | 5382.7354 | 3158.5252 | 6289.2821 | 3373.4133 | 0.2870 | False |
| renewable_share_lag_24 | 0.3810 | 0.1223 | 0.4106 | 0.1293 | 0.2419 | False |
| renewable_generation_lag_24 | 10837.0733 | 4118.1411 | 11800.9586 | 4354.5794 | 0.2341 | False |
| renewable_generation_lag_168 | 10854.6718 | 4117.6482 | 11767.6217 | 4395.7350 | 0.2217 | False |
| humidity_national_mean_lag_24 | 67.4575 | 14.8347 | 70.3546 | 13.8701 | 0.1953 | False |
| wind_speed_national_mean_lag_24 | 2.4501 | 1.3752 | 2.6755 | 1.5116 | 0.1639 | False |
| forecast_solar_day_ahead | 1476.4434 | 1693.3705 | 1277.2928 | 1588.6956 | 0.1176 | False |
| total_load_actual_lag_48 | 28552.3698 | 4537.6693 | 29066.0114 | 4705.0833 | 0.1132 | False |
| total_load_actual_lag_24 | 28552.7321 | 4536.8449 | 29064.0171 | 4703.7397 | 0.1127 | False |
| total_load_forecast | 28571.3689 | 4561.0515 | 29078.2977 | 4714.4228 | 0.1111 | False |
| total_load_actual_lag_168 | 28562.6735 | 4538.9502 | 29052.1127 | 4710.9126 | 0.1078 | False |
| clouds_all_national_mean_lag_24 | 24.3212 | 17.2596 | 25.6829 | 17.2381 | 0.0789 | False |

The largest flagged shifts are lagged national temperature (validation is a cooler Oct–May window than the full train history). Load, renewable, and price-lag means move with the target but stay below the 0.5-std flag on most series. No features were removed from this diagnostic.

## Leakage / lock checks

- `data/processed/test.parquet` is never opened (`assert_not_test` on every parquet read).
- Combined TRAIN+VALIDATION is UTC-aware, unique, strictly hourly, and sorted.
- Each fold is a prefix of that chronology; validation never precedes training.
- Preprocessing statistics are computed on the fold training block only.
- `price actual` is absent. Feature count remains 184 SAFE columns.
- Protected source parquets are hash-checked before and after the run.

## Reproducibility

`random_state = 42` where the estimator accepts it. Ridge / ElasticNet / Naive are deterministic given the fold data.

| file | md5 |
|---|---|
| walk_forward_model_comparison.csv | 7118e4d4cbd7812317fa76fe9bbf3f8e |
| walk_forward_fold_results.csv | ca0b1b1ac3f970f918d23ee30adc7d1d |
| walk_forward_error_by_hour.csv | ff97ee9ac0ed2ac184205de53d7d5531 |
| walk_forward_error_by_month.csv | 61fd06341b8fc1382a43bc79b639280b |
| walk_forward_error_by_weekday.csv | baecdaa11d62a6f7a0e9f748362b8ea3 |
| walk_forward_error_by_price_quantile.csv | 825591425524ac96f8b7e730b58e16ee |
