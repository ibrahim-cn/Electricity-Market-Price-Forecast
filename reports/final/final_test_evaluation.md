# Final Test Evaluation (locked holdout)

**FINAL_TEST_EVALUATION = PASS**

Test parquet was read **once the pipeline was frozen**. It was not used to
select the model, alpha, METHOD_B, thresholds, or residual addend.

## 1. Selected model

Ridge, closed-form NumPy, `alpha = 0.001`. `random_state = 42` is
recorded for pipeline consistency; the solver is deterministic.

## 2. Selected method

**METHOD_B** as selected on walk-forward:

- 184 SAFE features from `train`/`validation`/`test` parquets
- three causal high-price-frequency features (`y` shifted 24h, then 168/336/720-hour windows)
- high-price flag threshold = **development P75** = 57.5825 (`quantile(y_train+y_val, 0.75)` only)
- frozen `expanding_historical` addend from development residuals only (addend = 1.471482)

No alpha, feature, threshold, or correction search was run on test.

## 3. Walk-forward validation (frozen numbers)

| metric | value |
|---|---|
| mean MAE | 5.496022 |
| MAE std | 0.450160 |
| mean bias | -0.91 |
| P75+ bias (causal fold-train) | -5.895626 |
| P90+ bias (causal fold-train) | -8.211142 |

## 4. Final test performance

Test rows scored: **5260**. Rows dropped: **0**.

| metric | test |
|---|---|
| MAE | 4.329544 |
| RMSE | 6.136183 |
| R² | 0.639356 |
| sMAPE | 7.739424 |
| bias | 2.133567 |

## 5. Naive Lag-24 (same test rows)

| | Ridge+METHOD_B | Naive Lag-24 |
|---|---:|---:|
| MAE | 4.329544 | 6.045924 |
| RMSE | 6.136183 | 9.030375 |
| bias | 2.133567 | -0.003076 |

MODEL_BEATS_NAIVE = TRUE

This comparison does **not** change the selected model.

## 6. High-price regimes (development quantiles applied to test y)

Thresholds come from train+validation only.

| regime | q | threshold | n | MAE | bias | y_mean | y_pred_mean |
|---|---|---|---|---|---|---|---|
| P25+ | 0.2500 | 40.0000 | 5065 | 4.0397 | 1.7968 | 62.3643 | 64.1612 |
| P50+ | 0.5000 | 48.6500 | 4771 | 3.8728 | 1.5955 | 63.4289 | 65.0245 |
| P75+ | 0.7500 | 57.5825 | 3747 | 3.6536 | 1.3109 | 66.1422 | 67.4531 |
| P90+ | 0.9000 | 65.0000 | 2005 | 3.8676 | 1.2800 | 69.9211 | 71.2011 |
| P95+ | 0.9500 | 69.2500 | 1014 | 3.7506 | 0.6778 | 72.7910 | 73.4688 |

TEST_P75_BIAS = 1.310873
TEST_P90_BIAS = 1.279965
TEST_P95_BIAS = 0.677799

## 7. Bias analysis

Overall test bias = 2.134 (validation walk-forward mean bias -0.91).
The sign flipped: walk-forward underpredicted on average; the frozen holdout
**overpredicts**. P75+/P90+/P95+ biases are also positive on test.

## 8. Validation → test

| | walk-forward val | test | delta (test − val) |
|---|---:|---:|---:|
| MAE | 5.496022 | 4.329544 | -1.166478 |
| bias | -0.91 | 2.133567 | 3.043567 |
| P75+ bias | -5.895626 | 1.310873 | 7.206499 |
| P90+ bias | -8.211142 | 1.279965 | 9.491107 |

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

Worst hours: [8, 3, 2, 4, 1]
Worst months: [10, 7, 12]
Worst weekdays (Mon=0): [0, 5, 2]

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
| final_test_predictions.parquet | 586d26918ee347ee74d70a25535836eb |
| final_test_metrics.csv | 1ff84294518988ad83186f0ebdfec9cf |
| final_test_error_by_hour.csv | 603a36669904e22fdb33afad370dd386 |
| final_test_error_by_month.csv | 55d406a7d4ac283c79cd0cefadad0861 |
| final_test_error_by_weekday.csv | 6509391d47483f448a4c0158f28a91c8 |
| final_test_error_by_price_quantile.csv | 61178fd22f32bda3192173afcaba7c46 |

## Machine-readable summary

FINAL_TEST_EVALUATION = PASS
SELECTED_MODEL = Ridge(alpha=0.001)
SELECTED_METHOD = METHOD_B
VALIDATION_MAE = 5.496022
TEST_MAE = 4.329544
TEST_RMSE = 6.136183
TEST_R2 = 0.639356
TEST_SMAPE = 7.739424
TEST_BIAS = 2.133567
TEST_P75_BIAS = 1.310873
TEST_P90_BIAS = 1.279965
TEST_P95_BIAS = 0.677799
NAIVE_TEST_MAE = 6.045924
MODEL_BEATS_NAIVE = TRUE
TEST_USED_FOR_TUNING = FALSE
TEST_USED_FOR_SELECTION = FALSE
REPRODUCIBILITY = PASS
PROTECTED_FILES_UNCHANGED = TRUE
