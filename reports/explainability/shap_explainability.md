# SHAP / Linear Explainability

**SHAP_EXPLAINABILITY = PASS**

Locked model explained as-is. No retuning, no new features, no METHOD_B change,
no alpha change, no test-based selection.

## 1. Model and data

| item | value |
|---|---|
| Model | Ridge (closed-form NumPy) |
| Alpha | 0.001 |
| Residual correction | expanding_historical addend (part of locked METHOD_B) |
| Extra features | METHOD_B high-price fractions (`fraction_high_price_last_7d, fraction_high_price_last_14d, fraction_high_price_last_30d`) |
| Development data | official TRAIN + VALIDATION only |
| Test | never loaded |
| Folds | same 4 expanding windows as Ridge tuning / METHOD_B |

Walk-forward folds:

| fold | n_train | n_val | train_start | train_end | val_start | val_end | train_frac |
|---|---|---|---|---|---|---|---|
| 1 | 14902 | 2980 | 2014-12-31T23:00:00+00:00 | 2016-09-12T20:00:00+00:00 | 2016-09-12T21:00:00+00:00 | 2017-01-15T00:00:00+00:00 | 0.5000 |
| 2 | 17882 | 2980 | 2014-12-31T23:00:00+00:00 | 2017-01-15T00:00:00+00:00 | 2017-01-15T01:00:00+00:00 | 2017-05-19T04:00:00+00:00 | 0.6000 |
| 3 | 20862 | 2981 | 2014-12-31T23:00:00+00:00 | 2017-05-19T04:00:00+00:00 | 2017-05-19T05:00:00+00:00 | 2017-09-20T09:00:00+00:00 | 0.7000 |
| 4 | 23843 | 5961 | 2014-12-31T23:00:00+00:00 | 2017-09-20T09:00:00+00:00 | 2017-09-20T10:00:00+00:00 | 2018-05-26T18:00:00+00:00 | 0.8000 |

Fold METHOD_B validation MAE (reproduced, not re-selected):

| fold | n_train | n_val | train_p75 | MAE | bias | p75_n | p75_bias | addend |
|---|---|---|---|---|---|---|---|---|
| 1.0000 | 14902.0000 | 2980.0000 | 55.1875 | 5.5769 | -1.0825 | 1104.0000 | -5.2954 | -0.1597 |
| 2.0000 | 17882.0000 | 2980.0000 | 56.4975 | 6.1985 | -4.3348 | 789.0000 | -9.8726 | 1.0433 |
| 3.0000 | 20862.0000 | 2981.0000 | 56.6000 | 5.0611 | 1.7641 | 409.0000 | -6.3019 | 1.2374 |
| 4.0000 | 23843.0000 | 5961.0000 | 55.6900 | 5.1476 | -0.0061 | 2583.0000 | -2.1126 | -0.1273 |

## 2. Why not TreeSHAP

Ridge is a **linear** model. TreeSHAP explains tree ensembles and is the wrong
estimator here.

This stage uses the **exact linear SHAP** values for a model
`ŷ = x_scaled · w + ȳ + addend`:

- After fold-train standardization, training features are centered, so
  `φ_i = w_i · x_scaled,i`
- `Σ φ_i = ŷ_ridge − ȳ`
- The METHOD_B expanding addend is a **single fold-train residual constant**.
  It shifts the base value; it is not a per-feature SHAP term.

Standardized coefficients `w` are reported because they are already on a
common scale (the preprocessor is fit on fold train only). Permutation
importance (MAE increase after a deterministic column shuffle on that fold's
validation block) is a complementary, model-agnostic check. It is not used
to change the model.

These numbers are **predictive attributions** for the locked forecast. They
are not causal effects.

## 3. Method (per fold)

1. Load TRAIN+VALIDATION; do not open `test.parquet`.
2. Recreate the locked METHOD_B matrix: 184 SAFE features + 3 causal
   high-price fractions. The P75 threshold is `quantile(y_fold_train, 0.75)`.
3. Fit median impute + standard scale on **fold train only**.
4. Fit Ridge `α=0.001` on fold train.
5. Predict fold validation; add the frozen expanding-historical addend
   computed from fold-train residuals only.
6. Compute linear SHAP on the scaled validation matrix.
7. Compute permutation MAE increases on validation (seed
   `42 + fold·10000 + feature_index`).
8. Average attributions across the 4 folds. Do not declare a winner from
   a single fold.

## 4. Top 20 features (mean |linear SHAP| across folds)

| rank_mean_abs_shap | feature | feature_group | mean_abs_shap | std_abs_shap | mean_shap | mean_std_coef | direction | mean_perm_mae_increase |
|---|---|---|---|---|---|---|---|---|
| 1 | month | calendar | 203.7116 | 71.6331 | 27.7457 | 208.5929 | + | 151.9518 |
| 2 | day_of_year | calendar | 203.4352 | 72.7498 | -27.6052 | -207.7710 | - | 151.9182 |
| 3 | generation_wind_onshore_lag_24 | historical_generation | 24.6795 | 2.5099 | 0.5863 | -31.3741 | - | 28.0568 |
| 4 | renewable_generation_lag_24 | historical_generation | 21.0116 | 2.6465 | -6.7703 | 26.5554 | + | 21.8447 |
| 5 | clouds_all_national_mean_lag_24 | weather_aggregate | 20.9588 | 3.0858 | 1.7618 | -26.9996 | - | 23.0428 |
| 6 | generation_hydro_water_reservoir_lag_24 | historical_generation | 16.3841 | 1.1528 | 8.1699 | -20.1021 | - | 12.5266 |
| 7 | price_mean_lag24_lag48 | historical_target | 15.9521 | 4.5422 | -9.2036 | -22.0862 | - | 15.1611 |
| 8 | day_of_month | calendar | 15.6399 | 1.1470 | 0.3697 | 18.2763 | + | 15.9232 |
| 9 | generation_solar_lag_24 | historical_generation | 13.4225 | 0.8118 | 0.4658 | -15.7185 | - | 12.3146 |
| 10 | total_generation_lag_24 | historical_generation | 13.0407 | 0.5910 | -0.8753 | 14.7321 | + | 13.4931 |
| 11 | price_day_ahead_lag_24 | historical_target | 11.1256 | 2.7423 | 5.8880 | 15.4532 | + | 10.5047 |
| 12 | clouds_all_bilbao_lag_24 | historical_weather | 9.7830 | 0.9980 | -0.5352 | 10.6509 | + | 7.9604 |
| 13 | clouds_all_national_mean_lag_168 | weather_aggregate | 8.5938 | 2.2314 | 0.9393 | -11.1110 | - | 7.4682 |
| 14 | wind_speed_national_mean_lag_24 | weather_aggregate | 7.9910 | 1.6046 | 1.0869 | -9.9846 | - | 6.1840 |
| 15 | generation_wind_onshore_lag_168 | historical_generation | 7.8643 | 1.5090 | -0.0215 | -9.8388 | - | 6.6661 |
| 16 | renewable_generation_lag_168 | historical_generation | 7.7513 | 1.2805 | -2.2935 | 9.6666 | + | 5.9796 |
| 17 | clouds_all_madrid_lag_24 | historical_weather | 7.6355 | 1.2689 | -0.2515 | 9.2375 | + | 5.7102 |
| 18 | price_day_ahead_lag_48 | historical_target | 7.5489 | 2.2783 | 4.0252 | 10.5018 | + | 6.1921 |
| 19 | clouds_all_barcelona_lag_24 | historical_weather | 7.3787 | 1.0264 | -0.2944 | 8.2679 | + | 5.9674 |
| 20 | clouds_all_seville_lag_24 | historical_weather | 6.3076 | 0.8679 | -0.5196 | 8.3295 | + | 4.3234 |

`direction` is the sign of the mean standardized coefficient:
`+` means a higher feature value is associated with a **higher predicted**
price, holding other standardized features fixed.

**Do not read row 1–2 as “month is the price driver.”** `month` and
`day_of_year` are almost collinear calendar encodings. Their standardized
coefficients are huge and opposite (~+209 vs ~−208). Mean SHAP on
validation is about +27.7 and −27.6, so the **net** contribution of the
pair is near zero. |SHAP| ranks them first because each term is large,
not because the locked model uses calendar season as a single lever.
Later rows (generation lags, target lags, cloud/wind aggregates) are the
stable attributions.

## 5. Direction: what pulls the forecast up vs down

Largest-magnitude features with a **positive** coefficient (forecast up when
the feature is high):

| feature | feature_group | mean_std_coef | mean_abs_shap | mean_shap |
|---|---|---|---|---|
| month | calendar | 208.5929 | 203.7116 | 27.7457 |
| renewable_generation_lag_24 | historical_generation | 26.5554 | 21.0116 | -6.7703 |
| day_of_month | calendar | 18.2763 | 15.6399 | 0.3697 |
| total_generation_lag_24 | historical_generation | 14.7321 | 13.0407 | -0.8753 |
| price_day_ahead_lag_24 | historical_target | 15.4532 | 11.1256 | 5.8880 |
| clouds_all_bilbao_lag_24 | historical_weather | 10.6509 | 9.7830 | -0.5352 |
| renewable_generation_lag_168 | historical_generation | 9.6666 | 7.7513 | -2.2935 |
| clouds_all_madrid_lag_24 | historical_weather | 9.2375 | 7.6355 | -0.2515 |
| price_day_ahead_lag_48 | historical_target | 10.5018 | 7.5489 | 4.0252 |
| clouds_all_barcelona_lag_24 | historical_weather | 8.2679 | 7.3787 | -0.2944 |

Largest-magnitude features with a **negative** coefficient (forecast down
when the feature is high):

| feature | feature_group | mean_std_coef | mean_abs_shap | mean_shap |
|---|---|---|---|---|
| day_of_year | calendar | -207.7710 | 203.4352 | -27.6052 |
| generation_wind_onshore_lag_24 | historical_generation | -31.3741 | 24.6795 | 0.5863 |
| clouds_all_national_mean_lag_24 | weather_aggregate | -26.9996 | 20.9588 | 1.7618 |
| generation_hydro_water_reservoir_lag_24 | historical_generation | -20.1021 | 16.3841 | 8.1699 |
| price_mean_lag24_lag48 | historical_target | -22.0862 | 15.9521 | -9.2036 |
| generation_solar_lag_24 | historical_generation | -15.7185 | 13.4225 | 0.4658 |
| clouds_all_national_mean_lag_168 | weather_aggregate | -11.1110 | 8.5938 | 0.9393 |
| wind_speed_national_mean_lag_24 | weather_aggregate | -9.9846 | 7.9910 | 1.0869 |
| generation_wind_onshore_lag_168 | historical_generation | -9.8388 | 7.8643 | -0.0215 |
| forecast_wind_share_of_load | day_ahead_forecast | -7.8317 | 6.0211 | 0.3436 |

## 6. Answers

### Historical target lags

| feature | mean_abs_shap | std_abs_shap | mean_std_coef | direction | mean_perm_mae_increase |
|---|---|---|---|---|---|
| price_day_ahead_lag_168 | 0.3193 | 0.1784 | -0.4317 | - | 0.0322 |
| price_day_ahead_lag_24 | 11.1256 | 2.7423 | 15.4532 | + | 10.5047 |
| price_day_ahead_lag_48 | 7.5489 | 2.2783 | 10.5018 | + | 6.1921 |

Historical-target group sum of mean |SHAP| = 42.1112.
The allowed lags (t−24 / t−48 / t−168 and their element-wise stats) are
among the strongest predictive associations. This is persistence of the
auction curve, not a causal claim.

### Load forecast

| feature | mean_abs_shap | mean_std_coef | direction | mean_perm_mae_increase |
|---|---|---|---|---|
| total_load_forecast | 4.4673 | 5.2316 | + | 2.5968 |

Day-ahead forecast group sum of mean |SHAP| = 13.9419.

### Renewable forecast

| feature | mean_abs_shap | mean_std_coef | direction | mean_perm_mae_increase |
|---|---|---|---|---|
| forecast_wind_share_of_load | 6.0211 | -7.8317 | - | 3.9978 |
| forecast_solar_day_ahead | 1.4410 | -1.6602 | - | 0.4666 |
| forecast_solar_share_of_load | 1.2210 | -1.4585 | - | 0.3688 |
| forecast_wind_onshore_day_ahead | 0.7084 | 0.8962 | + | 0.2092 |
| renewable_forecast_total | 0.0831 | 0.0248 | + | 0.0045 |

Higher renewable-forecast values are typically associated with a **lower**
predicted price when the standardized coefficient is negative. That is a
forecasting association (merit-order style correlation in the training
window), not proof that renewables cause the price.

### Weather

| feature_group | n_features | mean_abs_shap_sum | std_abs_shap_sum | mean_abs_shap_mean | mean_perm_mae_increase_sum |
|---|---|---|---|---|---|
| historical_weather | 100 | 86.9805 | 12.7152 | 0.8698 | 44.2013 |
| weather_aggregate | 10 | 49.7765 | 7.9040 | 4.9776 | 42.3344 |

Weather is **not** noise. National cloud-cover and wind-speed aggregates
and several city cloud lags enter the top 20. The 100 city-level weather
lags have a low **per-feature** mean |SHAP| (0.87), so most individual
city series are weak; the **aggregates** (group mean 4.98) are the
meaningful weather signal. That is still a predictive association with
lagged weather, not a claim that clouds cause the auction price.

### Calendar

Calendar group sum of mean |SHAP| = 435.4444;
per-feature mean = 21.7722.
That sum is dominated by the collinear `month` / `day_of_year` pair
discussed above. After that pair, calendar is a moderate contributor
(`day_of_month` is next). Cyclic hour/weekday terms are smaller than
target and generation lags.

### METHOD_B high-price fractions (`diğer`)

| feature | mean_abs_shap | std_abs_shap | mean_std_coef | direction | mean_perm_mae_increase |
|---|---|---|---|---|---|
| fraction_high_price_last_14d | 1.1413 | 0.3464 | -1.4745 | - | 0.1113 |
| fraction_high_price_last_30d | 1.6999 | 0.4843 | 2.5248 | + | 0.3028 |
| fraction_high_price_last_7d | 0.8911 | 0.1720 | 1.0035 | + | 0.1411 |

Group `diğer` sum of mean |SHAP| = 3.7323.
These features are the locked METHOD_B extras, not new engineering.

## 7. Feature-group importance

| feature_group | n_features | mean_abs_shap_sum | std_abs_shap_sum | mean_abs_shap_mean | mean_perm_mae_increase_sum |
|---|---|---|---|---|---|
| calendar | 20 | 435.4444 | 166.2152 | 21.7722 | 322.2317 |
| historical_target | 8 | 42.1112 | 13.1695 | 5.2639 | 35.2646 |
| historical_load | 4 | 4.3831 | 0.0974 | 1.0958 | 1.3861 |
| historical_generation | 36 | 151.7809 | 12.2965 | 4.2161 | 121.6064 |
| historical_weather | 100 | 86.9805 | 12.7152 | 0.8698 | 44.2013 |
| day_ahead_forecast | 6 | 13.9419 | 1.1795 | 2.3236 | 7.6437 |
| weather_aggregate | 10 | 49.7765 | 7.9040 | 4.9776 | 42.3344 |
| diğer | 3 | 3.7323 | 1.1075 | 1.2441 | 0.5552 |

`mean_abs_shap_sum` totals attribution mass. `mean_abs_shap_mean` is fairer
when groups have very different widths (weather has ~100 columns).

## 8. Fold stability

- Features with the same coefficient sign in all 4 folds: **155 / 187**
- `std_abs_shap` in the importance table is the fold-to-fold spread
- A feature is called important only if mean |SHAP| is high **and** the
  direction is not a one-fold artifact

| feature | mean_abs_shap | std_abs_shap | n_folds_pos_coef | n_folds_neg_coef | direction |
|---|---|---|---|---|---|
| month | 203.7116 | 71.6331 | 4 | 0 | + |
| day_of_year | 203.4352 | 72.7498 | 0 | 4 | - |
| generation_wind_onshore_lag_24 | 24.6795 | 2.5099 | 0 | 4 | - |
| renewable_generation_lag_24 | 21.0116 | 2.6465 | 4 | 0 | + |
| clouds_all_national_mean_lag_24 | 20.9588 | 3.0858 | 0 | 4 | - |
| generation_hydro_water_reservoir_lag_24 | 16.3841 | 1.1528 | 0 | 4 | - |
| price_mean_lag24_lag48 | 15.9521 | 4.5422 | 0 | 4 | - |
| day_of_month | 15.6399 | 1.1470 | 4 | 0 | + |
| generation_solar_lag_24 | 13.4225 | 0.8118 | 0 | 4 | - |
| total_generation_lag_24 | 13.0407 | 0.5910 | 4 | 0 | + |
| price_day_ahead_lag_24 | 11.1256 | 2.7423 | 4 | 0 | + |
| clouds_all_bilbao_lag_24 | 9.7830 | 0.9980 | 4 | 0 | + |

## 9. High-price hours (development only)

Thresholds are fold-train P75 applied to that fold's validation *y* **after**
predictions exist. This slice is diagnostic. It does not change ranks used
for the top-20 table and does not use the locked test set.

Mean SHAP on validation hours with `y ≥ train_P75` minus the rest
(largest absolute differences):

| feature | feature_group | mean_delta_shap | std_delta_shap | mean_shap_hp | mean_shap_rest |
|---|---|---|---|---|---|
| price_mean_lag24_lag48 | historical_target | -19.9513 | 9.0708 | -22.5555 | -2.6043 |
| price_day_ahead_lag_24 | historical_target | 14.8506 | 4.8521 | 15.8937 | 1.0431 |
| total_generation_lag_24 | historical_generation | 11.0996 | 3.9246 | 6.9943 | -4.1053 |
| month | calendar | -10.8052 | 49.3977 | 19.6050 | 30.4102 |
| day_of_year | calendar | 10.0873 | 48.4452 | -19.2958 | -29.3831 |
| price_day_ahead_lag_48 | historical_target | 7.3039 | 4.6123 | 8.8664 | 1.5626 |
| generation_wind_onshore_lag_24 | historical_generation | 5.7935 | 9.6538 | 3.8173 | -1.9762 |
| generation_fossil_gas_lag_24 | historical_generation | -5.6056 | 2.7096 | -6.0767 | -0.4712 |
| price_mean_lag24_lag48_lag168 | historical_target | 5.5555 | 3.1209 | 6.8556 | 1.3001 |
| total_load_forecast | day_ahead_forecast | 4.7482 | 1.1657 | 3.6460 | -1.1022 |
| forecast_wind_share_of_load | day_ahead_forecast | 4.2619 | 3.1655 | 3.0158 | -1.2461 |
| clouds_all_national_mean_lag_24 | weather_aggregate | -3.9534 | 4.2231 | -0.9867 | 2.9667 |
| generation_fossil_hard_coal_lag_24 | historical_generation | -3.8083 | 1.4945 | -1.9605 | 1.8478 |
| total_generation_lag_168 | historical_generation | 3.1797 | 1.9356 | 1.8263 | -1.3534 |
| generation_hydro_water_reservoir_lag_24 | historical_generation | -3.0498 | 3.8862 | 5.8452 | 8.8951 |

Reading (association only):

- If METHOD_B fraction features have a positive coefficient, a higher recent
  share of expensive hours raises the **forecast**. On development folds the
  locked model still underpredicted many high-price hours; extra level from
  these fractions reduced but did not remove that gap.
- Target lags and load forecast often show a larger positive SHAP mass on
  high-price validation hours because those hours also have higher lags and
  higher load forecasts — the model attributes the level through features
  that move with the expensive regime.
- This does **not** say the high-price bias observed on the locked test
  (where P75+ bias flipped positive) is explained or fixed. Test rows were
  not used here.

## 10. Leakage / test / protected files

LEAKAGE_CHECK = PASS
TEST_READ_COUNT = 0
TEST_USED_FOR_SELECTION = FALSE
PROTECTED_FILES_UNCHANGED = TRUE
REPRODUCIBILITY = PASS

- Chronological expanding folds; no shuffle of time
- Preprocess fit on fold train only
- METHOD_B P75 from fold-train *y* only
- Expanding addend from fold-train residuals only
- Test parquet never opened; test predictions unused
- Full-data fit was not used
- Existing modeling / final reports and prediction parquets were not overwritten

| file | md5_before | md5_after | unchanged |
|---|---|---|---|
| data/processed/merged/merged_energy_weather.parquet | ae4a12026b1a9682d6bbb58ef7471fa1 | ae4a12026b1a9682d6bbb58ef7471fa1 | True |
| data/processed/features/model_features.parquet | c9f07ac0f95e0f51fff5472129c1f9ad | c9f07ac0f95e0f51fff5472129c1f9ad | True |
| data/processed/splits/train.parquet | 278666dcdb30990b55a6aa5c882f21ee | 278666dcdb30990b55a6aa5c882f21ee | True |
| data/processed/splits/validation.parquet | cba753fa9327955d506139d25fdaae4d | cba753fa9327955d506139d25fdaae4d | True |
| data/processed/splits/test.parquet | 069afbe9c766426d2e095282ece93a69 | 069afbe9c766426d2e095282ece93a69 | True |
| reports/modeling/baseline_modeling.md | c95fc2d5542db5ab61bf61441c617729 | c95fc2d5542db5ab61bf61441c617729 | True |
| reports/modeling/ridge_tuning.md | edc2e471bf130d2591d081a22af9cfa6 | edc2e471bf130d2591d081a22af9cfa6 | True |
| reports/modeling/residual_correction.md | c3432b98b61ab1865ccbdd14653eee70 | c3432b98b61ab1865ccbdd14653eee70 | True |
| reports/modeling/high_price_strategy_comparison.md | 411ce6d06ca2a239e81f1056e9d3b3cb | 411ce6d06ca2a239e81f1056e9d3b3cb | True |
| reports/final/final_test_evaluation.md | 4e66d3e815f90cac5b0276653b947cf0 | 4e66d3e815f90cac5b0276653b947cf0 | True |
| reports/final/final_test_metrics.csv | 1ff84294518988ad83186f0ebdfec9cf | 1ff84294518988ad83186f0ebdfec9cf | True |
| data/processed/predictions/final_test_predictions.parquet | 586d26918ee347ee74d70a25535836eb | 586d26918ee347ee74d70a25535836eb | True |
| data/processed/predictions/high_price_strategy_predictions.parquet | c513ed3bb24fb647a5cc5ee79d576b2f | c513ed3bb24fb647a5cc5ee79d576b2f | True |

| new output | md5 |
|---|---|
| shap_feature_importance.csv | ba8ec7a4b906d4f133f37cdde0bd3223 |
| shap_fold_results.csv | 808a0f398013571ff645f1fb72a3a25d |
| shap_feature_groups.csv | 4c7b85c43de70f47cc5d4296e62af08d |
| shap_explainability.md | 3f3ba0bc965f21614244adefbd8abdf6 |

Figures written: summary=True importance=True groups=True

## 11. Limitations

- Linear SHAP assigns credit under a linear, standardized model. Correlated
  lags and lag-statistics split that credit.
- Permutation importance also suffers when features are collinear.
- Attributions describe the locked forecast, not the real-world price
  mechanism. This study does not establish causality.
- High-price diagnostics use development validation *y* only as a slice
  label after the fact.

SHAP_EXPLAINABILITY = PASS
TEST_READ_COUNT = 0
TEST_USED_FOR_SELECTION = FALSE
LEAKAGE_CHECK = PASS
REPRODUCIBILITY = PASS
PROTECTED_FILES_UNCHANGED = TRUE
