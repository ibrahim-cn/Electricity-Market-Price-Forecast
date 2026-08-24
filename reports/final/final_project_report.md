# Final Project Report

Spanish hourly day-ahead electricity price forecasting.  
This document records the **locked** pipeline, selections, and holdout results.  
No model, alpha, method, feature, threshold, or preprocessing choice in this report is open for revision.

**Locked model:** Ridge(`alpha=0.001`) + METHOD_B  
**Locked test role:** single final holdout evaluation (5260 rows). Not used for selection or tuning.

---

# 1. Project Overview

This project builds a leakage-safe forecasting pipeline for the Spanish day-ahead electricity market (OMIE Mercado Diario). The deliverable is a reproducible, chronological pipeline from raw energy and weather files to a single frozen holdout evaluation.

Development used only train and validation (plus expanding-window folds on their chronological concatenation). The test set was scored once after the model and method were locked.

This study reports **forecasting performance** and **predictive association**. It does not establish causality.

---

# 2. Research Question

Can a leakage-safe linear model, using only information that would be available before delivery hour *t*, forecast the Spanish hourly day-ahead price (`price day ahead`) better than a Naive Lag-24 baseline on a locked chronological holdout?

Secondary development questions (answered on train/validation only):

- Which model family is most stable under expanding-window validation?
- Does a small Ridge penalty improve walk-forward MAE?
- Does a causal residual correction reduce level-shift error?
- Do causal high-price-frequency features improve walk-forward MAE?

Those development questions were closed before the locked test was evaluated. Test metrics were not used to answer them.

---

# 3. Dataset

Two original CSVs (never overwritten):

| File | Rows | Columns | Role |
|---|---:|---:|---|
| `energy_dataset.csv` | 35,064 | 29 | Hourly Spanish generation, load, day-ahead forecasts, and prices |
| `weather_features.csv` | 178,396 | 17 | Hourly weather for five Spanish cities |

Coverage is hourly from `2014-12-31T23:00:00+00:00` through `2018-12-31T22:00:00+00:00` (35,064 consecutive UTC hours).

**Target:** `price day ahead` (€/MWh), the day-ahead market price for delivery hour *t*.

**Forbidden as a feature:** `price actual` (real-time / imbalance-adjacent realized price). It is never used as a predictor, never lagged, and never joined into the feature matrix.

Copies used by the pipeline live under `data/raw/`. The project-root CSVs remain the original source files.

After merge and cleaning: 35,064 hourly UTC rows. After feature engineering: 184 SAFE features plus `timestamp_utc`. The target is joined later by timestamp.

---

# 4. Data Cleaning

Raw CSVs were not modified. Cleaning ran on `data/raw/` copies.

**Dropped (100% empty):**

- `generation hydro pumped storage aggregated`
- `forecast wind offshore eday ahead`

Zero-only generation columns were kept in the merged table and excluded later from feature construction.

**Timestamps:** timezone-aware UTC hourly index. Weather city rows were aligned to the same hourly grid. Duplicate weather rows were audited and resolved before the merge.

**Missing generation / actual load:** no blind zero-fill. Interior gaps of at most 3 hours were time-interpolated. Longer gaps and edge gaps were left as NaN.

**Target:** `price day ahead` had no missing values and was not interpolated.

Weather outliers were audited and documented; the merge output is `data/processed/merged/merged_energy_weather.parquet`.

---

# 5. Data Leakage Prevention

The information boundary is: predict the day-ahead price for delivery hour *t* **before** delivery-hour actuals exist. Same-hour realized generation, load, weather, and `price actual` are not features.

Leakage controls that were enforced:

- Chronological split only. **No shuffle.**
- The target at hour *t* is never used as a feature for hour *t*.
- `price actual` is never used.
- Same-hour actual generation, `total load actual`, and weather are never used.
- Target lags are **only** t−24, t−48, and t−168.
- **No t−1 target lag.**
- **No rolling window on the raw target** in the original 184-feature matrix. METHOD_B later adds causal high-price **fractions** from *y* shifted 24 hours, then rolled on that shifted series only.
- Preprocessing (median impute + standard scale) is fit **inside each fold** on that fold’s training block only. The final model fits preprocess on train+validation only.
- Test was **not** used for model selection, hyperparameter tuning, method selection, thresholds, or residual addend.
- Final test was evaluated **once**, after the pipeline was frozen.

A column-level leakage audit is in `reports/data/leakage_audit.md`. Feature-engineering scripts fail on forbidden columns, negative shifts, lag-1 names, and `.rolling(` on the original feature-construction source.

---

# 6. Feature Engineering

Source: `data/processed/merged/merged_energy_weather.parquet` (read-only).  
Output: `data/processed/features/model_features.parquet` — 184 SAFE features + `timestamp_utc`. Target is not stored in this file.

| Group | Count | Contents |
|---|---:|---|
| Calendar | 20 | Europe/Madrid hour, weekday, month, cyclic encodings |
| Day-ahead forecasts | 6 | Load / solar / wind forecasts for hour *t*, plus derived shares |
| Historical target | 8 | Lags t−24 / t−48 / t−168 and element-wise stats of those lags |
| Historical load | 4 | Actual-load lags and lag-24 forecast error |
| Historical generation | 36 | Generation lags t−24 / t−168 and lagged aggregates |
| Historical weather | 100 | City weather lags t−24 / t−168 (degrees as sin/cos before lag) |
| Weather aggregates | 10 | National means / max rain at t−24 and t−168 |
| **SAFE features** | **184** | |

METHOD_B (selected later on walk-forward) adds three causal features at fit time: the fraction of past hours above the **fold-train** (or, for the locked test, **development**) P75 of *y*, computed after a 24-hour shift, over 7 / 14 / 30 day windows. Those features are not stored in `model_features.parquet`; they are built from past target values only.

---

# 7. Chronological Train / Validation / Test Split

Deterministic 70 / 15 / 15 cut on the sorted hourly series (`n = 35064`). No random split.

| Split | Rows | Start (UTC) | End (UTC) | Target mean |
|---|---:|---|---|---:|
| Train | 24,544 | 2014-12-31T23:00Z | 2017-10-19T14:00Z | 46.85 |
| Validation | 5,260 | 2017-10-19T15:00Z | 2018-05-26T18:00Z | 52.73 |
| Test | 5,260 | 2018-05-26T19:00Z | 2018-12-31T22:00Z | 61.14 |

Checks: `max(train) < min(validation) < min(test)`. Zero overlap. Each split file contains `timestamp_utc`, 184 SAFE features, and `price day ahead`.

Target-level shift is already visible: test mean (61.14) is well above train (46.85) and validation (52.73). Test volatility is lower (std 10.22 vs train 14.23).

---

# 8. Baseline Models

Compared on the official validation split (test not used for selection):

| Model | Validation MAE | Notes |
|---|---:|---|
| Naive Lag-24 | 8.3036 | Copies `price_day_ahead_lag_24` |
| Mean Lag-24/48 | 8.8228 | |
| Mean Lag-24/48/168 | 9.0786 | |
| HistGradientBoosting | 6.6498 | Native NaNs |
| Ridge (`α=0.1` on this grid) | **5.4868** | Closed-form NumPy; median impute + scale fit on train only |

Ridge won the single-split baseline stage on validation MAE. Naive Lag-24 bias on validation was near zero (+0.007); Ridge already showed underprediction (bias −1.72).

A Naive Lag-24 score was also recorded on test as a **pre-specified** baseline (MAE 6.0459, bias ≈ 0). That number was not used to choose a model.

---

# 9. Walk-Forward Validation

Expanding windows on **train + validation only** (29,804 hours). Test parquet was never loaded.

| Fold | n_train | n_val | Train fraction |
|---|---:|---:|---:|
| 1 | 14,902 | 2,980 | 0.50 |
| 2 | 17,882 | 2,980 | 0.60 |
| 3 | 20,862 | 2,981 | 0.70 |
| 4 | 23,843 | 5,961 | 0.80 |

Each fold: `max(train timestamp) < min(validation timestamp)`. Impute and scale fit on that fold’s train block only.

Families compared: Naive Lag-24, Ridge (several α), ElasticNet, HistGradientBoosting, RandomForest, LightGBM, XGBoost.

**Winner of this stage:** Ridge `α=0.01` — mean MAE 5.804, MAE std 0.668, mean bias −1.95.  
Naive Lag-24 mean MAE was 7.351 (near-zero bias). Trees did not beat Ridge on mean MAE.

---

# 10. Ridge Hyperparameter Tuning

Same four expanding folds. Alpha grid fixed before any test look:

`[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]`

| α | Mean MAE | MAE std | Mean bias |
|---|---:|---:|---:|
| **0.001** | **5.7966** | 0.6489 | −1.90 |
| 0.003 | 5.7982 | 0.6531 | −1.91 |
| 0.01 | 5.8043 | 0.6679 | −1.95 |

**BEST_ALPHA = 0.001** (lowest mean walk-forward MAE). This alpha is **locked**.

Pooled walk-forward P75+ bias for this alpha was −6.12 (development high-price underprediction).

---

# 11. Residual Correction

Ridge `α=0.001` held fixed. Corrections used **fold-train residuals only**.

| Method | Mean MAE | Mean bias |
|---|---:|---:|
| no_correction | 5.7966 | −1.90 |
| fold_train_bias | 5.7966 | −1.90 |
| **expanding_historical** (τ = 720 h) | **5.7025** | −1.63 |
| regime_aware | 5.7793 | −1.94 |
| linear_calibration | 5.7966 | −1.90 |

**Selected correction:** expanding historical addend. High-price P75+ bias did **not** improve (moved from −6.12 to −6.31). The correction was kept because mean MAE improved by more than the pre-set 0.01 threshold.

For the locked test, this addend was frozen from **development residuals only** (addend = +1.471482). It was not re-estimated on test.

---

# 12. High-Price Analysis

Same four folds. Ridge `α=0.001` + expanding_historical held as CURRENT_BEST. Challengers were required to beat CURRENT_BEST by at least 0.01 mean MAE.

| Method | Mean MAE | Mean bias | Causal P75+ bias |
|---|---:|---:|---:|
| CURRENT_BEST | 5.7025 | −1.63 | −6.58 |
| METHOD_A | 5.5298 | −1.18 | −6.03 |
| **METHOD_B** | **5.4960** | **−0.91** | −5.90 |
| METHOD_C | 5.5223 | −1.21 | −5.91 |
| METHOD_D | 5.7477 | −1.65 | −6.79 |
| METHOD_E | 5.6762 | −1.64 | −6.60 |

**BEST_METHOD = METHOD_B** (184 SAFE features + causal 7/14/30-day high-price fractions + the same expanding addend).

Walk-forward validation of the selected method:

- MAE = 5.496022
- MAE std = 0.450160
- Bias ≈ −0.91

On **development** folds, high-price hours were still underpredicted (causal P75+ bias −5.90, P90+ −8.21). That development pattern is **not** restated as a test finding.

---

# 13. Final Locked Test Evaluation

The test parquet was opened **after** Ridge `α=0.001`, METHOD_B, the development P75 threshold (57.5825), preprocess statistics, Ridge weights, and the expanding addend were frozen.

Test rows scored: **5260**. Rows dropped: **0**.

Ridge(`alpha=0.001`) + METHOD_B:

| Metric | Test |
|---|---:|
| MAE | 4.3295 |
| RMSE | 6.1362 |
| R² | 0.6394 |
| sMAPE | 7.7394% |
| Bias | +2.1336 |

Exact stored values: MAE = 4.329544, RMSE = 6.136183, R² = 0.639356, sMAPE = 7.739424, bias = +2.133567.

Validation (walk-forward, METHOD_B): MAE = 5.4960, bias ≈ −0.91.

**Bias sign change (do not ignore):**

| Period | Bias |
|---|---|
| Walk-forward validation | ≈ −0.91 (average **underprediction**) |
| Locked test | +2.13 (average **overprediction**) |

High-price slice biases on the locked test (development quantiles, applied to test *y* only for reporting):

| Slice | Bias |
|---|---:|
| P75+ | +1.31 |
| P90+ | +1.28 |
| P95+ | +0.68 |

Exact: P75 = +1.310873, P90 = +1.279965, P95 = +0.677799. All three are **positive**.

High-price bias observed during development did not persist with the same sign on the locked test set; P75+/P90+/P95+ biases became positive.

Associated (not causal) context: development target mean 47.89 vs test mean 61.14; test std is lower (10.22 vs 14.38). Most test hours sit above the development P75, so METHOD_B frequency features are often saturated and the frozen positive addend is applied in a higher-price regime.

Predictions: `data/processed/predictions/final_test_predictions.parquet`.  
Metrics: `reports/final/final_test_metrics.csv`, `reports/final/final_test_evaluation.md`.

---

# 14. Model vs Naive Baseline

Same 5260 locked-test rows:

| Model | Test MAE | Test RMSE | Test bias |
|---|---:|---:|---:|
| Naive Lag-24 | 6.0459 | 9.0304 | ≈ 0 (−0.003) |
| Ridge + METHOD_B | 4.3295 | 6.1362 | +2.1336 |

MAE improvement vs naive:

\[
\frac{6.045924 - 4.329544}{6.045924} \times 100 = 28.39\% \approx 28.4\%
\]

The final model **beats the naive baseline on MAE**. MODEL_BEATS_NAIVE = TRUE.

The naive model is **more unbiased** on this holdout (bias ≈ 0 versus +2.13). Lower MAE does not imply a better-calibrated level. That trade-off is part of the locked result, not a reason to change the model.

---

# 15. Error Analysis

Diagnostic only. These tables were not used to retune, reselect, or change thresholds.

**By hour (test MAE):** worst hours 8, 3, 2, 4, 1 (MAE ≈ 4.53–4.46). Best hours are late evening (21–23, MAE ≈ 4.15–4.20). Hourly MAE is relatively flat.

**By month:** worst October (5.52), then July (4.88). May fragment of the holdout is easiest (1.88) but covers only the last days of May.

**By weekday (Mon=0):** Monday MAE 5.90 and Saturday 5.12 are clearly worse than mid-week (Thursday 3.45, Wednesday 3.63).

**By price quantile (test *y*, after predictions existed):**

| Band | n | MAE | Bias |
|---|---:|---:|---:|
| Below P25 | 197 | 11.80 | +10.81 |
| P25–P50 | 292 | 6.75 | +5.07 |
| P50–P75 | 1,024 | 4.68 | +2.64 |
| P75+ | 3,747 | 3.65 | +1.31 |

On this holdout, **low-price hours have the largest overprediction**. High-price hours have smaller MAE and still **positive** bias. This is the opposite sign of the development high-price residual pattern.

---

# 16. Limitations

- The test period has a clear **target distribution shift** (mean 61.14 vs development 47.89; lower volatility).
- Validation → test **bias sign change**: underprediction on walk-forward (−0.91) versus overprediction on test (+2.13).
- Reported test performance is a **single holdout period**. It is not a multi-year out-of-sample distribution of error.
- External variables and exogenous shocks (fuel prices, unplanned outages, policy, cross-border constraints) are missing or only indirectly associated through generation/load/weather history.
- High-price regime behavior is **not fully resolved**. Development underprediction did not continue with the same sign on test; the locked model overpredicted instead. That is a sign flip, not a solved regime problem.
- Predictive performance is not causality. Associations between features and the target are forecasting relationships only.

---

# 17. Reproducibility

Pipeline order (do not re-run for selection; artifacts are already written):

1. `src/data_preparation.py`
2. `src/feature_engineering.py`
3. `src/time_series_split.py`
4. `src/baseline_models.py`
5. `src/walk_forward_validation.py`
6. `src/ridge_tuning.py`
7. `src/residual_correction.py`
8. `src/high_price_analysis.py`
9. `src/final_test_evaluation.py`

Ridge is a closed-form NumPy solve `(X'X + αI)w = X'y`, not `sklearn.Ridge.fit`. `random_state=42` is recorded for pipeline consistency; the locked Ridge path is deterministic.

Protected processed hashes (must remain unchanged):

| File | MD5 |
|---|---|
| `data/processed/merged/merged_energy_weather.parquet` | `ae4a12026b1a9682d6bbb58ef7471fa1` |
| `data/processed/features/model_features.parquet` | `c9f07ac0f95e0f51fff5472129c1f9ad` |
| `data/processed/splits/train.parquet` | `278666dcdb30990b55a6aa5c882f21ee` |
| `data/processed/splits/validation.parquet` | `cba753fa9327955d506139d25fdaae4d` |
| `data/processed/splits/test.parquet` | `069afbe9c766426d2e095282ece93a69` |

Locked evaluation artifacts:

| File | MD5 |
|---|---|
| `data/processed/predictions/final_test_predictions.parquet` | `586d26918ee347ee74d70a25535836eb` |
| `reports/final/final_test_metrics.csv` | `1ff84294518988ad83186f0ebdfec9cf` |

There is no `tests/` suite of `test_*.py` files in this repository. Reproducibility is documented by script order, locked hashes, and compile checks.

---

# 18. Conclusion

The locked model is Ridge(`alpha=0.001`) with METHOD_B. On the locked test set it achieves MAE 4.3295 versus Naive Lag-24 MAE 6.0459, an improvement of about **28.4%** in MAE. It also improves RMSE and R² relative to that naive copy-yesterday rule.

Those numbers are **forecasting performance**. They show a predictive association between the leakage-safe feature set (plus METHOD_B’s causal high-price frequencies and a frozen residual addend) and the day-ahead price. They do **not** show that any feature causes the price to rise or fall.

**This study does not establish causality.**

Validation and test do not tell the same bias story. During development the model underpredicted on average (bias ≈ −0.91) and underpredicted high-price hours. On the locked test it overpredicted on average (bias = +2.13), and P75+/P90+/P95+ biases were positive. High-price bias observed during development did not persist with the same sign on the locked test set; P75+/P90+/P95+ biases became positive.

The project is therefore closed as: finalize, document, audit. The model is not reopened for tuning.
