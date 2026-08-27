# Final Test Evaluation (locked holdout)

**FINAL_TEST_EVALUATION = PASS**

Test parquet was read **once the pipeline was frozen**. It was not used to
select the model, alpha, METHOD_B, AR(1) phi, thresholds, or residual addend.

## 1. Selected model

Ridge, closed-form NumPy, `alpha = 0.001`, plus METHOD_B and AR(1) on
Ridge+METHOD_B residuals. `random_state = 42` is recorded for
pipeline consistency; Ridge and AR(1) are deterministic.

## 2. Selected method

**METHOD_B + AR(1)** as selected on walk-forward:

- 184 SAFE features from `train`/`validation`/`test` parquets
- three causal high-price-frequency features (`y` shifted 24h, then 168/336/720-hour windows)
- high-price flag threshold = **development P75** = 57.5825 (`quantile(y_train+y_val, 0.75)` only)
- frozen `expanding_historical` addend from development residuals only (addend = 1.471482)
- AR(1) on the last 1440 development residuals of Ridge+METHOD_B; 24-hour
  block forecasts, then that day's actual residuals update the state

No alpha, feature, threshold, AR(1) order, or correction search was run on test.

## 3. Walk-forward validation (frozen numbers)

| metric | value |
|---|---|
| mean MAE | 4.529617 |
| MAE std | 0.561220 |
| mean bias | -0.71 |
| P75+ bias (causal fold-train) | -5.895626 |
| P90+ bias (causal fold-train) | -8.211142 |

## 4. Final test performance

Test rows scored: **5260**. Rows dropped: **0**.

| metric | test |
|---|---|
| MAE | 3.990091 |
| RMSE | 5.878929 |
| R² | 0.668961 |
| sMAPE | 7.314412 |
| bias | 1.419419 |

## 5. Naive Lag-24 (same test rows)

| | Ridge+METHOD_B+AR(1) | Naive Lag-24 |
|---|---:|---:|
| MAE | 3.990091 | 6.045924 |
| RMSE | 5.878929 | 9.030375 |
| bias | 1.419419 | -0.003076 |

MODEL_BEATS_NAIVE = TRUE

This comparison does **not** change the selected model.

## 6. High-price regimes (development quantiles applied to test y)

Thresholds come from train+validation only.

| regime | q | threshold | n | MAE | bias | y_mean | y_pred_mean |
|---|---|---|---|---|---|---|---|
| P25+ | 0.2500 | 40.0000 | 5065 | 3.7149 | 1.1684 | 62.3643 | 63.5328 |
| P50+ | 0.5000 | 48.6500 | 4771 | 3.5510 | 1.0344 | 63.4289 | 64.4633 |
| P75+ | 0.7500 | 57.5825 | 3747 | 3.2625 | 0.8408 | 66.1422 | 66.9830 |
| P90+ | 0.9000 | 65.0000 | 2005 | 3.3735 | 0.9205 | 69.9211 | 70.8416 |
| P95+ | 0.9500 | 69.2500 | 1014 | 3.2544 | 0.5566 | 72.7910 | 73.3476 |

TEST_P75_BIAS = 0.840776
TEST_P90_BIAS = 0.920483
TEST_P95_BIAS = 0.556602

## 7. Bias analysis

Overall test bias = 1.419 (validation walk-forward mean bias -0.71).
The sign flipped: walk-forward underpredicted on average; the frozen holdout
**overpredicts**. P75+/P90+/P95+ biases are also positive on test.

## 8. Validation → test

| | walk-forward val | test | delta (test − val) |
|---|---:|---:|---:|
| MAE | 4.529617 | 3.990091 | -0.539526 |
| bias | -0.71 | 1.419419 | 2.129419 |
| P75+ bias | -5.895626 | 0.840776 | 6.736402 |
| P90+ bias | -8.211142 | 0.920483 | 9.131625 |

Test MAE is not materially worse than walk-forward MAE.

Do not retune on this gap.

## 9. Distribution-shift observations

Associated differences (not causal claims):

| | development (train+val) | test |
|---|---:|---:|
| target mean | 47.885 | 61.144 |
| target std | 14.375 | 10.218 |
| load forecast mean | 28660.835 | 29002.776 |
| renewable forecast mean | 6984.025 | 6492.454 |

Development mean 47.89 vs test mean 61.14,
with **lower** test volatility (10.22 vs 14.37).
Most test hours sit above the development P75 (57.58), so METHOD_B
frequency features are often saturated. Together with the positive development
addend, that is consistent with the observed **test overprediction**, not with
the walk-forward underprediction pattern. This is an association, not a cause.

## 10. Limitations

- High-price bias was never solved in walk-forward; the holdout is not a new search.
- METHOD_B frequency features use past prices vs a **development** P75. If the test
  level sits well above that P75, the features saturate and cannot express a new extreme.
- Expanding-historical addend is a single development statistic; it cannot track a
  further test-only level jump.
- Calendar MAE tables describe error concentration; they were not used for selection.

## 11. Test-use statement

TEST_USED_FOR_TUNING = FALSE
TEST_USED_FOR_SELECTION = FALSE

Test `y` entered the pipeline only to score already-frozen predictions (and, like the
precomputed `price_day_ahead_lag_24` already stored on test rows, as **past** values
inside causal 24h-shifted windows for later test hours). The P75 threshold, scaler,
Ridge weights, and residual addend used **no test target**.

Hour / weekday / month MAE:

Worst hours: [1, 2, 4, 3, 8]
Worst months: [10, 12, 11]
Worst weekdays (Mon=0): [0, 5, 6]

## Reproducibility and protected files

REPRODUCIBILITY = PASS
PROTECTED_FILES_UNCHANGED = TRUE

| file | md5_before | md5_after | unchanged |
|---|---|---|---|
| merged_energy_weather.parquet | ae4a12026b1a9682d6bbb58ef7471fa1 | ae4a12026b1a9682d6bbb58ef7471fa1 | True |
| model_features.parquet | c9f07ac0f95e0f51fff5472129c1f9ad | c9f07ac0f95e0f51fff5472129c1f9ad | True |
| train.parquet | 278666dcdb30990b55a6aa5c882f21ee | 278666dcdb30990b55a6aa5c882f21ee | True |
| validation.parquet | cba753fa9327955d506139d25fdaae4d | cba753fa9327955d506139d25fdaae4d | True |
| test.parquet | 069afbe9c766426d2e095282ece93a69 | 069afbe9c766426d2e095282ece93a69 | True |

| file | md5 |
|---|---|
| final_test_predictions.parquet | 2b95a1c95aebb62c9b480c6da3765a24 |
| final_test_metrics.csv | 9605a651aaedd892a9130ec0d631485e |
| final_test_error_by_hour.csv | 2b5f1bc46b9b17f0d5e3a74cf754ca78 |
| final_test_error_by_month.csv | 1961a065d4d8046d25df1f61d464e634 |
| final_test_error_by_weekday.csv | ff03ba05f30e371e34a70a8d04b55e30 |
| final_test_error_by_price_quantile.csv | 1e292c5e363262e2d2e9f51c73b34e1f |

## Machine-readable summary

FINAL_TEST_EVALUATION = PASS
SELECTED_MODEL = Ridge(alpha=0.001)+METHOD_B+AR(1)
SELECTED_METHOD = METHOD_B_AR1
VALIDATION_MAE = 4.529617
TEST_MAE = 3.990091
TEST_RMSE = 5.878929
TEST_R2 = 0.668961
TEST_SMAPE = 7.314412
TEST_BIAS = 1.419419
TEST_P75_BIAS = 0.840776
TEST_P90_BIAS = 0.920483
TEST_P95_BIAS = 0.556602
NAIVE_TEST_MAE = 6.045924
MODEL_BEATS_NAIVE = TRUE
TEST_USED_FOR_TUNING = FALSE
TEST_USED_FOR_SELECTION = FALSE
REPRODUCIBILITY = PASS
PROTECTED_FILES_UNCHANGED = TRUE
