# Final Project Report

Spanish hourly day-ahead electricity price forecasting (`price day ahead`, €/MWh).

This document is the delivery write-up for the **locked** pipeline. No model, alpha, METHOD_B setting, feature, threshold, or preprocessing choice here is open for revision. Numbers are taken from existing reports and CSVs. No new experiment was run for this document.

The assignment prompt used an “ETH/USDT” title and mentioned Reddit comments. Those series are **not** in this repository. Artifacts, target, and dashboard are Spanish DAM electricity.

**Locked model:** Ridge(`alpha=0.001`) + METHOD_B  
**Development:** TRAIN + VALIDATION, 29,804 rows  
**Locked test:** 5,260 rows, scored once after freeze  
**24-hour forecasting:** PRODUCTION_READY = FALSE

This study reports forecasting performance and predictive association. It does not establish causality.

---

## 1. Executive Summary

The project builds a leakage-safe chronological pipeline from two raw CSVs (energy + five-city weather) to a single frozen holdout evaluation and a read-only Streamlit dashboard.

Selection used walk-forward validation on development data only. The locked model is Ridge with `alpha=0.001` and METHOD_B (184 SAFE features + three causal high-price fractions + a frozen expanding-historical residual addend).

On the locked test set:

| Metric | Ridge + METHOD_B | Naive Lag-24 |
|---|---:|---:|
| MAE | 4.329544 | 6.045924 |
| RMSE | 6.136183 | 9.030375 |
| R² | 0.639356 | 0.218923 |
| sMAPE | 7.739424% | 11.675683% |
| Bias | +2.133567 | −0.003076 |

MODEL_BEATS_NAIVE = TRUE. That is an MAE comparison, not a claim that the model is unbiased or that high-price error is solved.

Walk-forward METHOD_B MAE was 5.496022 with bias ≈ −0.91 (underprediction), including remaining high-price underprediction (P75+ bias −5.90). On the frozen test the bias sign flipped to overprediction (+2.13), and P75+/P90+/P95+ biases are positive. Those two periods must not be collapsed into one story.

Explainability used exact linear SHAP on development folds. TreeSHAP was not used. SHAP is not causality.

STRICT 24-hour production forecasting is **not ready**: 106 SAFE / 6 UNKNOWN / 75 FORBIDDEN of 187 columns at a D-1 ~12:00 CET origin. STRICT `y_pred` is empty by design.

---

## 2. Business / Technical Problem

**Business framing.** A Spanish day-ahead market participant (or a research proxy for one) needs an hourly price forecast for delivery day D before the auction / before delivery. The target is the published day-ahead price, not the real-time `price actual`.

**Technical problem.** Forecast `price day ahead` at hour *t* using only information that would be available before delivery-hour actuals exist: past auction prices, TSO-style day-ahead load and renewable forecasts (publication time **not verified in-file**), and lagged load, generation, and weather.

**Not in scope.** Crypto (ETH/USDT) prediction, Reddit sentiment, causal identification of what “drives” the price, and a production 24-hour API.

**Research question (closed before test).** Can a leakage-safe linear model beat Naive Lag-24 on a locked chronological holdout? Secondary development questions (train/validation only): model family, Ridge α, residual correction, METHOD_B vs alternatives.

---

## 3. Data

Two original files (never overwritten):

| File | Rows | Columns | Role |
|---|---:|---:|---|
| `energy_dataset.csv` | 35,064 | 29 | Hourly generation, load, day-ahead forecasts, prices |
| `weather_features.csv` | 178,396 | 17 | Hourly weather for Madrid, Barcelona, Bilbao, Seville, Valencia |

Pipeline copies: `data/raw/`. Merge output: `data/processed/merged/merged_energy_weather.parquet` (35,064 × 103).

| Item | Value |
|---|---|
| Time range | 2014-12-31T23:00:00+00:00 → 2018-12-31T22:00:00+00:00 |
| Frequency | Hourly UTC, 0 missing hours, monotonic |
| Target | `price day ahead` (€/MWh) |
| Target missing | 0 |
| Target min / max | 2.06 / 101.99 |
| Forbidden feature | `price actual` (present in energy file, excluded from X) |

After feature engineering: 184 SAFE features + `timestamp_utc` in `model_features.parquet`. The target is joined later by timestamp. METHOD_B’s three fractions are built at fit time from past *y* only.

---

## 4. Data Quality

Source: `reports/data/final_data_quality.md`, `data_cleaning.md`, weather duplicate/outlier notes.

Raw CSVs were not modified. Cleaning ran on `data/raw/` copies.

**Dropped (100% empty):** `generation hydro pumped storage aggregated`, `forecast wind offshore eday ahead`. Zero-only generation columns stayed in the merge and were excluded from feature construction.

**Timestamps.** Timezone-aware UTC hourly index. Weather city rows aligned to the same grid. Duplicate weather rows audited before merge.

**Missing generation / actual load.** No blind zero-fill. Interior gaps of at most 3 hours were time-interpolated. Longer and edge gaps left as NaN.

**Target.** `price day ahead` had no missing values and was not interpolated.

Quality checks recorded as PASS: row count 35,064, no duplicate timestamps, timezone-aware UTC, weather match rate 1.0, no infinities, `price actual` present but excluded from features.

---

## 5. Leakage Audit

Column-level catalog: `reports/data/feature_availability.csv`, narrative `reports/data/leakage_audit.md`. Feature scripts fail on forbidden columns, negative shifts, lag-1 target names, and `.rolling(` on the original construction source.

Information boundary: predict the day-ahead price for delivery hour *t* **before** delivery-hour actuals exist.

Enforced controls:

- Chronological 70/15/15 split. No shuffle. No random split.
- Target at *t* is not a feature for *t*.
- `price actual` unused.
- No same-hour actual generation, `total load actual`, or weather.
- Target lags t−24, t−48, t−168 only. No t−1.
- No rolling window on the raw target in the 184-feature matrix.
- METHOD_B: *y* shifted 24 hours, then 168/336/720 windows on the shifted series; threshold = fold-train (or development) P75.
- Preprocess fit inside each fold; final preprocess on train+validation only.
- Test unused for selection, tuning, thresholds, or addend. Scored once.

A stricter **operational** DAM reading (D-1 ~12:00 CET) is tighter than the row-aligned hourly feature matrix. That gap is why 24-hour STRICT production is FALSE (Section 15), not a reason to change the locked model.

---

## 6. Feature Engineering

Source parquet read-only. Output: `data/processed/features/model_features.parquet`.

| Group | Count | Contents |
|---|---:|---|
| Calendar | 20 | Europe/Madrid hour, weekday, month, cyclic encodings |
| Day-ahead forecasts | 6 | Load / solar / wind forecasts for hour *t*, plus derived shares |
| Historical target | 8 | Lags t−24 / t−48 / t−168 and element-wise stats of those lags |
| Historical load | 4 | Actual-load lags and lag-24 forecast error |
| Historical generation | 36 | Generation lags t−24 / t−168 and lagged aggregates |
| Historical weather | 100 | City weather lags t−24 / t−168 (angles as sin/cos before lag) |
| Weather aggregates | 10 | National means / max rain at t−24 and t−168 |
| **SAFE features** | **184** | |

METHOD_B adds `fraction_high_price_last_7d`, `_14d`, `_30d` at fit time. They are not stored in `model_features.parquet`.

---

## 7. Split Strategy

Deterministic 70 / 15 / 15 on the sorted hourly series (`n = 35,064`). `max(train) < min(validation) < min(test)`. Zero overlap.

| Split | Rows | Start (UTC) | End (UTC) | Target mean |
|---|---:|---|---|---:|
| Train | 24,544 | 2014-12-31T23:00Z | 2017-10-19T14:00Z | 46.85 |
| Validation | 5,260 | 2017-10-19T15:00Z | 2018-05-26T18:00Z | 52.73 |
| Test | 5,260 | 2018-05-26T19:00Z | 2018-12-31T22:00Z | 61.14 |

Development = train + validation = **29,804** hours (through 2018-05-26T18:00Z).

Target-level shift is already visible: test mean 61.14 vs train 46.85 and validation 52.73. Test volatility is lower (std 10.22 vs train 14.23). This shift is context for the later bias sign change, not a reason to retune on test.

---

## 8. Baselines

Official **validation** split only for model comparison at this stage (`reports/modeling/baseline_model_comparison.csv`). Test Naive Lag-24 was recorded as a **pre-specified** baseline and was not used to choose a family.

| Model | Dataset | MAE | RMSE | R² | sMAPE | Bias |
|---|---|---:|---:|---:|---:|---:|
| Naive Lag-24 | validation | 8.303586 | 11.927336 | 0.281338 | 20.065253 | +0.007293 |
| Mean Lag-24/48 | validation | 8.822775 | 12.364521 | 0.227688 | 20.511290 | −0.008429 |
| Mean Lag-24/48/168 | validation | 9.078571 | 12.388198 | 0.224728 | 20.409798 | −0.058890 |
| HistGradientBoosting | validation | 6.649754 | 8.679889 | 0.619402 | 15.761973 | −4.298761 |
| Ridge (α=0.1 on this grid) | validation | 5.486840 | 7.123249 | 0.743673 | 13.090893 | −1.717725 |
| Naive Lag-24 | test (pre-specified) | 6.045924 | 9.030375 | 0.218923 | 11.675683 | −0.003076 |

Ridge won the single-split baseline stage on validation MAE. Naive validation bias was near zero; Ridge already underpredicted (bias −1.72).

---

## 9. Walk-forward Validation

Expanding windows on **train + validation only** (29,804 hours). `test.parquet` was never loaded (`reports/modeling/walk_forward_model_comparison.csv`).

| Fold | n_train | n_val | Train fraction |
|---|---:|---:|---:|
| 1 | 14,902 | 2,980 | 0.50 |
| 2 | 17,882 | 2,980 | 0.60 |
| 3 | 20,862 | 2,981 | 0.70 |
| 4 | 23,843 | 5,961 | 0.80 |

Each fold: `max(train timestamp) < min(validation timestamp)`. Impute and scale fit on that fold’s train block only.

Why walk-forward: a single 2017–2018 validation cut is one regime. Expanding windows test stability as the training origin moves, which is the right place to compare families and later to pick α and METHOD_B.

Families: Naive Lag-24, Ridge (several α), ElasticNet, HistGradientBoosting, RandomForest, LightGBM, XGBoost.

| Model | Mean MAE | MAE std | Mean bias |
|---|---:|---:|---:|
| Naive Lag-24 | 7.351456 | 1.011364 | −0.002130 |
| Ridge α=0.01 (this stage’s best Ridge on the coarse grid) | 5.804256 | 0.667880 | −1.950689 |
| LightGBM | 5.916945 | 0.943894 | −3.153088 |
| XGBoost | 5.945953 | 0.960483 | −3.069063 |
| HistGradientBoosting | 6.014534 | 1.004972 | −3.185623 |

Trees did not beat Ridge on mean MAE. Walk-forward is also why a later dedicated α grid was run (Section 10): the coarse walk-forward Ridge winner was α=0.01; the finer grid selected α=0.001.

---

## 10. Hyperparameter Tuning

Same four expanding folds. Alpha grid fixed before any test look (`reports/modeling/ridge_alpha_comparison.csv`):

`[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]`

| α | Mean MAE | MAE std | Mean bias |
|---|---:|---:|---:|
| **0.001** | **5.796612** | 0.648924 | −1.896630 |
| 0.003 | 5.798230 | 0.653128 | −1.909385 |
| 0.01 | 5.804256 | 0.667880 | −1.950689 |

**BEST_ALPHA = 0.001** (lowest mean walk-forward MAE). Locked. Pooled walk-forward P75+ bias for this alpha was −6.12 (development high-price underprediction). Test was not used.

---

## 11. Residual Correction

Ridge `α=0.001` held fixed. Corrections used **fold-train residuals only** (`reports/modeling/residual_correction_comparison.csv`).

| Method | Mean MAE | Mean bias |
|---|---:|---:|
| no_correction | 5.796612 | −1.896630 |
| fold_train_bias | 5.796612 | −1.896630 |
| **expanding_historical** (τ = 720 h) | **5.702475** | −1.629286 |
| regime_aware | 5.779327 | −1.935752 |
| linear_calibration | 5.796623 | −1.896528 |

Selected: expanding historical addend. High-price P75+ bias did **not** improve (moved from −6.12 to −6.31). The correction was kept because mean MAE improved by more than the pre-set 0.01 threshold.

For the locked test, the addend was frozen from **development residuals only** (addend = +1.471482). It was not re-estimated on test.

---

## 12. High-price Diagnostic

Same four folds. Ridge `α=0.001` + expanding_historical = CURRENT_BEST. Challengers had to beat CURRENT_BEST by at least 0.01 mean MAE (`reports/modeling/high_price_strategy_comparison.csv`).

| Method | Mean MAE | Mean bias | Causal P75+ bias | Causal P90+ bias |
|---|---:|---:|---:|---:|
| CURRENT_BEST | 5.702475 | −1.629286 | −6.580579 | −8.663258 |
| METHOD_A | 5.529761 | −1.180484 | −6.030362 | −8.146420 |
| **METHOD_B** | **5.496022** | **−0.914816** | **−5.895626** | **−8.211142** |
| METHOD_C | 5.522296 | −1.214697 | −5.907973 | −7.884586 |
| METHOD_D | 5.747654 | −1.646631 | −6.791716 | −8.889774 |
| METHOD_E | 5.676209 | −1.637131 | −6.600464 | −8.725618 |

**BEST_METHOD = METHOD_B.** Walk-forward MAE std = 0.450160.

On **development** folds, high-price hours were still underpredicted. Leakage-safe frequency features and the residual addend reduced mean MAE but **did not fully resolve** the high-price underprediction problem. That development pattern is **not** restated as a test finding.

---

## 13. Final Test

The test parquet was opened **after** Ridge `α=0.001`, METHOD_B, development P75 (57.5825), preprocess statistics, Ridge weights, and the expanding addend were frozen.

Test rows scored: **5,260**. Rows dropped: **0**.  
Source: `reports/final/final_test_metrics.csv`, `final_test_evaluation.md`.

| Metric | Ridge + METHOD_B | Naive Lag-24 |
|---|---:|---:|
| MAE | 4.329544 | 6.045924 |
| RMSE | 6.136183 | 9.030375 |
| R² | 0.639356 | 0.218923 |
| sMAPE | 7.739424% | 11.675683% |
| Bias | +2.133567 | −0.003076 |

MAE improvement vs naive ≈ 28.4%. MODEL_BEATS_NAIVE = TRUE.

High-price slice biases on the locked test (development quantiles, applied to test *y* only for reporting):

| Slice | Bias |
|---|---:|
| P75+ | +1.310873 |
| P90+ | +1.279965 |
| P95+ | +0.677799 |

All three are **positive**.

Associated (not causal) context: development target mean 47.89 vs test mean 61.14; test std lower (10.22 vs 14.38). Many test hours sit above the development P75, so METHOD_B frequencies are often saturated and the frozen positive addend is applied in a higher-price regime.

Predictions: `data/processed/predictions/final_test_predictions.parquet` (same values as `data/processed/final_model/final_model_test_predictions.parquet`, max |Δpred| = 0).

---

## 14. Explainability

Source: `reports/explainability/shap_explainability.md`. TRAIN+VALIDATION folds only. Test never loaded.

Ridge is linear, so **exact linear SHAP** is used: after fold-train standardization, `φ_i = w_i · x_scaled,i`. TreeSHAP is the wrong estimator.

The METHOD_B expanding addend is a single fold-train residual constant. It shifts the base value; it is not a per-feature SHAP term.

**Stable signals (do not over-read calendar |SHAP|):**

- `price_day_ahead_lag_24` mean |SHAP| 11.13, positive coefficient.
- `price_day_ahead_lag_48` mean |SHAP| 7.55, positive coefficient.
- Generation lags (wind, renewable, hydro, solar, total) rank in the top 10 after the collinear calendar pair.
- `total_load_forecast`: mean |SHAP| 4.47, standardized coefficient +5.23 (positive).
- `forecast_wind_share_of_load`: mean |SHAP| 6.02, negative coefficient.
- National cloud and wind aggregates contribute; city weather is weaker per column.

**Collinearity warning.** `month` and `day_of_year` have the largest |SHAP| (~203.7 and ~203.4) and huge opposite standardized coefficients (~+208.6 vs ~−207.8). Mean SHAP on validation is about +27.7 and −27.6, so the **net** pair is near zero. They are not independent “price drivers.”

**Group sums of mean |SHAP|:** calendar 435.4 (inflated by that pair), historical generation 151.8, historical weather 87.0 (100 columns), weather aggregates 49.8, historical target 42.1, day-ahead forecasts 13.9, METHOD_B fractions 3.7.

SHAP and coefficients are **association / explanation** for the locked forecast. They are not causal effects.

---

## 15. 24h Forecasting Audit

Source: `reports/forecasting/forecasting_24h.md`. Locked model unchanged. Test parquet not opened.

> 24-hour forecasting is only production-ready if every required feature is available at the forecast origin.

**PRODUCTION_READY = FALSE**

| classification_strict | count |
|---|---:|
| SAFE | 106 |
| UNKNOWN | 6 |
| FORBIDDEN | 75 |

UNKNOWN: named day-ahead forecast columns have **no publication timestamp** in the file.  
FORBIDDEN: at D-1 ~12:00 CET, `lag_24` actual load/generation/weather for later hours of D are not yet observed.

STRICT path: UNKNOWN is not imputed and not read from a future row. The locked model needs all 187 columns. Therefore STRICT `forecasting_predictions.csv` leaves `y_pred` empty. Empty is preferred to a leaked or silently filled path. Recursive prediction was not used.

Assumed scenario (`forecasting_predictions_assumed.csv`) uses unverified forecast-column timing and a late origin so that `t−24 ≤ origin`. It is labeled **NOT PRODUCTION-READY** and must not be used to retune.

To become production-ready later (not implemented):

1. Independently verified publish times for load/solar/wind day-ahead forecasts.
2. A forecast origin at which every required historical lag is already observed.
3. Either a reduced feature set that is entirely SAFE at that origin, or a new model selected without using test.
4. No silent fill of UNKNOWN/FORBIDDEN columns.
5. Operational monitoring of bias under regime shift.

---

## 16. Streamlit Dashboard

`app.py` is read-only. It does not fit Ridge, search α, change METHOD_B, write predictions, or set production-ready TRUE.

| Page (UI) | Content |
|---|---|
| Genel bakış | Locked KPIs; MODEL DURUMU = KİLİTLİ |
| Model performansı | Locked test vs naive; actual vs predicted; residuals |
| Hata analizi | Quantile / hour / month error; development vs test bias note |
| Açıklanabilirlik | Linear SHAP from saved CSVs/figure |
| 24 saatlik tahmin | STRICT empty (`—`); assumed labeled not production-ready |
| Veri ve sızıntı denetimi | Split table; 106/6/75; leakage PASS badges |
| Proje bilgisi | Pipeline and stage checklist |

Displayed KPIs match the locked holdout: MAE 4.33, RMSE 6.14, R² 0.639, sMAPE 7.74%, Naive MAE 6.05.

---

## 17. Limitations

- **Single frozen test period.** Reported test metrics are one holdout, not a sampling distribution of future error.
- **Historical dataset.** 2015–2018 Spanish DAM + city weather. No live feed.
- **Forecast production availability** is not fully verified. Publication times are missing.
- **High-price regime shift is not fully resolved.** Development underprediction vs test overprediction. Leakage-safe corrections did not “fix” the regime.
- **Association, not causality.** SHAP, coefficients, and MAE gains do not identify causes.
- **Missing exogenous variables.** Fuel prices, outages, policy, interconnectors are absent or only indirectly associated.
- The assignment prompt’s ETH/USDT / Reddit framing does not apply to these files.

---

## 18. Reproducibility

Pipeline order (artifacts already written; do not re-run for selection):

1. `src/data_preparation.py`
2. `src/feature_engineering.py`
3. `src/time_series_split.py`
4. `src/baseline_models.py`
5. `src/walk_forward_validation.py`
6. `src/ridge_tuning.py`
7. `src/residual_correction.py`
8. `src/high_price_analysis.py`
9. `src/shap_explainability.py`
10. `src/final_model.py`
11. `src/final_test_evaluation.py`
12. `src/forecast_24h.py`

Ridge is a closed-form NumPy solve. `random_state=42` is recorded for HGB/RF/LightGBM/XGBoost and permutation SHAP checks. The locked Ridge path is deterministic given the development matrix.

Protected processed hashes are listed in `reports/final_project_audit.md` and `reports/final/final_project_audit.md`.

There is no `tests/` suite of `test_*.py` files. Reproducibility is script order, locked hashes, and compile/import checks.

Dashboard:

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

---

## 19. Final Conclusion

The locked delivery is Ridge(`alpha=0.001`) + METHOD_B, evaluated once on a frozen chronological holdout, documented with linear SHAP, and shown in a read-only dashboard.

What the project **does** show:

- A leakage-safe pipeline (no shuffle, no `price actual`, no same-hour actuals, no t−1 target lag).
- Chronological evaluation and walk-forward validation on development data.
- A frozen holdout the model was not selected on.
- Test MAE better than Naive Lag-24 (4.329544 vs 6.045924).
- Reproducible artifacts and hashes.
- Explainability that is honest about collinearity and non-causality.

What it **does not** show:

- That “the model is successful because MAE is 4.33.”
- That high-price underprediction was solved (development vs test bias signs disagree).
- That any feature causes the price.
- That 24-hour operational forecasting is production-ready. **It is not.**

**This study does not establish causality.** The project is closed as: finalize, document, audit. The model is not reopened for tuning.
