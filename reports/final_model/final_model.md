# Final Model Pipeline

**FINAL_MODEL = PASS**

Final model was selected before accessing the locked test target.

This stage archives the already-locked pipeline. It does **not** search
models, alphas, features, or METHOD_B settings.

## 1. Final model

Ridge(`alpha=0.001`), closed-form NumPy `(X'X + αI)w = X'y`.

## 2. Residual correction

**METHOD_B + AR(1)** as selected on walk-forward:

- 184 SAFE features (unchanged)
- three causal high-price fractions vs development P75
- expanding-historical addend from **development residuals only**
- addend = 1.471482
- AR(1) φ = 0.911802 on the last 1440 development residuals
- 24-hour block residual forecasts; that day's actual residuals then update state
- development P75 threshold = 57.582500

## 3. Development dataset

TRAIN + VALIDATION only.

| split | rows | start UTC | end UTC |
|---|---:|---|---|
| train | 24544 | 2014-12-31T23:00:00+00:00 | 2017-10-19T14:00:00+00:00 |
| validation | 5260 | 2017-10-19T15:00:00+00:00 | 2018-05-26T18:00:00+00:00 |
| **development** | **29804** | 2014-12-31T23:00:00+00:00 | 2018-05-26T18:00:00+00:00 |

## 4. Test

Completely locked holdout. Test was **not** used to fit the preprocessor,
Ridge, P75 threshold, or residual addend. Test rows were read only after
those objects were frozen, for inference comparison with the existing
locked evaluation.

| | value |
|---|---|
| TEST_USED_FOR_FITTING | FALSE |
| TEST_USED_FOR_SELECTION | FALSE |
| inference rows | 5260 |
| max \|new pred − locked pred\| | 0.000e+00 |
| matches locked evaluation | TRUE |

## 5. Feature count

- SAFE features = **184**
- METHOD_B extras = 3 (`fraction_high_price_last_7d, fraction_high_price_last_14d, fraction_high_price_last_30d`)
- Design matrix columns = 187

No new feature was created beyond the locked METHOD_B definition.

## 6. Preprocessing

Median impute + standard scale, fit on development X only
(29804 rows × 187 columns).
Test statistics were never used.

## 7. Model fitting

Ridge `α=0.001` fit on the scaled development matrix. Intercept =
training-target mean on development = 47.885429.
METHOD_B addend then added. Solver is deterministic.

Official **walk-forward** METHOD_B+AR(1) (not re-selected on test):
MAE = 4.529617, bias ≈ -0.71.

Development **in-sample** score after this full-dev fit is optimistic
(MAE = 4.472012, bias = 1.471482)
and is **not** used for selection.

## 8. Leakage audit

LEAKAGE_CHECK = PASS

- chronological development block; no shuffle
- `price actual` absent
- target at hour t not in X
- no t−1 target lag
- METHOD_B fractions: `y` shifted 24h, then 168/336/720 windows
- P75 from development *y* only
- addend from development residuals only
- preprocess fit on development only
- test unused for fitting / selection / thresholds

## 9. Reproducibility

REPRODUCIBILITY = PASS

The script fits twice in one process. Metadata, coefficients, and the
new inference parquet must match.

## 10. Locked test evaluation (reference only)

These numbers are the **already published** holdout. They are not a
reason to change the model.

| metric | locked test |
|---|---:|
| MAE | 3.990091 |
| RMSE | 5.878929 |
| R² | 0.668961 |
| sMAPE | 7.314412 |
| Bias | +1.419419 |

This run's inference: MAE = 3.990091, RMSE = 5.878929,
R² = 0.668961, sMAPE = 7.314412, bias = 1.419419.

## High-price (development in-sample)

Development quantiles of *y*, applied to development predictions. Diagnostic
only. Not used to retune.

| regime | q | threshold | n | MAE | bias |
|---|---|---|---|---|---|
| P75+ | 0.7500 | 57.5825 | 7451 | 4.0589 | -1.4849 |
| P90+ | 0.9000 | 65.0000 | 3025 | 4.6228 | -2.5382 |
| P95+ | 0.9500 | 69.2500 | 1492 | 5.5269 | -3.8573 |

Official walk-forward (causal fold-train) P75+ bias = -5.895626,
P90+ = -8.211142. In-sample development bias is smaller because
the same rows were used to fit the intercept and addend.

## High-price (locked test, reference only)

Previously recorded, **not** used for tuning:

| regime | locked bias |
|---|---:|
| P75+ | +0.840776 |
| P90+ | +0.920483 |
| P95+ | +0.556602 |

This run (same frozen objects):

| regime | q | threshold | n | MAE | bias |
|---|---|---|---|---|---|
| P75+ | 0.7500 | 57.5825 | 3747 | 3.2625 | 0.8408 |
| P90+ | 0.9000 | 65.0000 | 2005 | 3.3735 | 0.9205 |
| P95+ | 0.9500 | 69.2500 | 1014 | 3.2544 | 0.5566 |

## Standardized coefficients (largest |w|)

| feature | feature_group | coefficient | abs_coefficient |
|---|---|---|---|
| day_of_year | calendar | -201.1754 | 201.1754 |
| month | calendar | 200.1306 | 200.1306 |
| price_mean_lag24_lag48 | historical_target | -31.6569 | 31.6569 |
| generation_wind_onshore_lag_24 | historical_generation | -30.7356 | 30.7356 |
| renewable_generation_lag_24 | historical_generation | 21.4846 | 21.4846 |
| price_day_ahead_lag_24 | historical_target | 21.1086 | 21.1086 |
| generation_hydro_water_reservoir_lag_24 | historical_generation | -19.2509 | 19.2509 |
| total_generation_lag_24 | historical_generation | 18.8722 | 18.8722 |
| clouds_all_national_mean_lag_168 | weather_aggregate | -17.3837 | 17.3837 |
| day_of_month | calendar | 16.8506 | 16.8506 |
| price_day_ahead_lag_48 | historical_target | 15.3747 | 15.3747 |
| generation_solar_lag_24 | historical_generation | -14.9558 | 14.9558 |
| clouds_all_national_mean_lag_24 | weather_aggregate | -14.5608 | 14.5608 |
| wind_speed_national_mean_lag_24 | weather_aggregate | -12.7790 | 12.7790 |
| generation_wind_onshore_lag_168 | historical_generation | -10.8935 | 10.8935 |

Coefficients are on development-standardized features. They are predictive
weights, not causal effects. `month` / `day_of_year` remain a collinear pair.

## Artifacts

| path | md5 |
|---|---|
| model.joblib | 4241f2fdcca596c7a59464c9f95a30c2 |
| preprocessing.joblib | 75a56ce75b8c5da86d28a80d3ee19d99 |
| feature_manifest.json | 94eb94ecf340e479a9c156e5f7699979 |
| model_metadata.json | ec9441100d3fe9e044705b7f1e8209bf |
| method_b_parameters.json | f3911b778cbfc7c0a5bc6ce1d84c6a94 |
| final_model_coefficients.csv | 17ae6db8939c3a9ef866bd099697154e |
| final_model_test_predictions.parquet | 2b95a1c95aebb62c9b480c6da3765a24 |

Locked files (`final_test_predictions.parquet`, `final_test_metrics.csv`,
`final_test_evaluation.md`) were not rewritten.

PROTECTED_FILES_UNCHANGED = TRUE

| file | md5_before | md5_after | unchanged |
|---|---|---|---|
| energy_dataset.csv | 63afe36eed077c06fe342a7274d0e2e3 | 63afe36eed077c06fe342a7274d0e2e3 | True |
| weather_features.csv | 88e107d38ff66af919e781d339314cfe | 88e107d38ff66af919e781d339314cfe | True |
| energy_dataset.csv | 63afe36eed077c06fe342a7274d0e2e3 | 63afe36eed077c06fe342a7274d0e2e3 | True |
| weather_features.csv | 88e107d38ff66af919e781d339314cfe | 88e107d38ff66af919e781d339314cfe | True |
| merged_energy_weather.parquet | ae4a12026b1a9682d6bbb58ef7471fa1 | ae4a12026b1a9682d6bbb58ef7471fa1 | True |
| model_features.parquet | c9f07ac0f95e0f51fff5472129c1f9ad | c9f07ac0f95e0f51fff5472129c1f9ad | True |
| train.parquet | 278666dcdb30990b55a6aa5c882f21ee | 278666dcdb30990b55a6aa5c882f21ee | True |
| validation.parquet | cba753fa9327955d506139d25fdaae4d | cba753fa9327955d506139d25fdaae4d | True |
| test.parquet | 069afbe9c766426d2e095282ece93a69 | 069afbe9c766426d2e095282ece93a69 | True |
| final_test_predictions.parquet | 2b95a1c95aebb62c9b480c6da3765a24 | 2b95a1c95aebb62c9b480c6da3765a24 | True |
| final_test_metrics.csv | 9605a651aaedd892a9130ec0d631485e | 9605a651aaedd892a9130ec0d631485e | True |
| final_test_evaluation.md | c434ec54e833230836d69a42fe5cc64d | c434ec54e833230836d69a42fe5cc64d | True |
| high_price_strategy_comparison.md | 411ce6d06ca2a239e81f1056e9d3b3cb | 411ce6d06ca2a239e81f1056e9d3b3cb | True |
| shap_explainability.md | d8af078308919b7d61887fa27fe30c7b | d8af078308919b7d61887fa27fe30c7b | True |

FINAL_MODEL = PASS
MODEL = Ridge(alpha=0.001)
METHOD = METHOD_B
DEVELOPMENT_ROWS = 29804
FEATURE_COUNT = 184
TEST_USED_FOR_FITTING = FALSE
TEST_USED_FOR_SELECTION = FALSE
LEAKAGE_CHECK = PASS
REPRODUCIBILITY = PASS
PROTECTED_FILES_UNCHANGED = TRUE
