# Elektrik Piyasası Fiyat Tahmini

Locked final project. The model is Ridge α=0.001 + METHOD_B + AR(1) on residuals. The 24-hour forecasting path remains **not production-ready**.

The assignment prompt used an “ETH/USDT” label. The data, target, features, and dashboard are **Spanish hourly day-ahead electricity** (`price day ahead`, €/MWh). There is no crypto or Reddit series in this repository.

---

## Project Overview

This project forecasts the Spanish day-ahead electricity auction price for each delivery hour using only leakage-safe information: calendar, named day-ahead load/renewable forecasts, historical target lags, historical load/generation/weather, national weather aggregates, and METHOD_B high-price frequency features.

The locked model is **Ridge(`alpha=0.001`) + METHOD_B + AR(1)**, selected on chronological walk-forward validation of TRAIN + VALIDATION (29,804 hours). The test set (5,260 hours) was scored once after this freeze.

A read-only Streamlit dashboard (`app.py`) displays the locked artifacts. It does not retrain, retune, or emit a production 24-hour forecast.

This study reports **forecasting performance** and **predictive association**. It does not establish causality.

---

## Repository layout

```
data/raw/          ham CSV (tek kopya)
data/processed/    birleşik parquet, özellikler, bölmeler, tahminler, kilitli model
src/               pipeline betikleri
reports/           aşama raporları, CSV’ler, bitirme şekilleri
outputs/figures/   SHAP ve 24s şekilleri
docs/              makale ve bitirme raporu (docx/pdf)
app.py             salt-okunur Streamlit panosu
```

---

## Problem Definition

Estimate the hourly day-ahead market price (`price day ahead`) before delivery hour *t*, using:

- past published auction prices (lags t−24 / t−48 / t−168 only)
- energy-market day-ahead load and renewable forecasts
- weather and generation **history** that would already be observed
- leakage-safe engineered features (no same-hour actuals, no `price actual`)

The operational question is whether a leakage-safe model can beat a Naive Lag-24 copy of yesterday’s auction price on a **frozen chronological holdout**.

---

## Dataset

Two original CSVs under `data/raw/` (never overwritten):

| File | Rows | Content |
|---|---:|---|
| `data/raw/energy_dataset.csv` | 35,064 | Hourly Spanish generation, load, day-ahead forecasts, and prices |
| `data/raw/weather_features.csv` | 178,396 | Hourly weather for five Spanish cities |

Coverage is hourly UTC from `2014-12-31T23:00:00+00:00` through `2018-12-31T22:00:00+00:00` (35,064 consecutive hours after merge). Target range: 2.06–101.99 €/MWh. `price actual` is present in the energy file and is **never** used as a feature.

There is no ETH/USDT price file and no Reddit comment file in this repository.

Chronological 70 / 15 / 15 split (no shuffle):

| Split | Rows | UTC start | UTC end |
|---|---:|---|---|
| Train | 24,544 | 2014-12-31T23:00Z | 2017-10-19T14:00Z |
| Validation | 5,260 | 2017-10-19T15:00Z | 2018-05-26T18:00Z |
| Test | 5,260 | 2018-05-26T19:00Z | 2018-12-31T22:00Z |

---

## Data Pipeline

```
CSV
→ data quality
→ merge
→ EDA
→ leakage audit
→ feature engineering
→ chronological split
→ baseline
→ walk-forward validation
→ tuning
→ residual / high-price analysis
→ explainability
→ final model archive
→ locked test
→ 24h forecasting audit
→ dashboard
```

Scripts, in order:

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

Later stages must not be re-run to “improve” the locked test score.

---

## Leakage Prevention

- Chronological split only. **No random split. No shuffle.**
- The target at hour *t* is never a feature for hour *t*.
- `price actual` is unused (not lagged, not joined into X).
- Same-hour actual generation, load, and weather are unused.
- Target lags are **only** t−24, t−48, t−168. **No t−1.**
- No rolling window on the raw target in the 184-feature matrix.
- METHOD_B fractions use *y* shifted 24 hours, then windows on that shifted series.
- Median impute + standard scale are fit **inside each fold** (final fit: train+validation only).
- Test was **not** used for fitting, tuning, method selection, thresholds, or the residual addend. It was scored once after freeze.

---

## Feature Engineering

184 SAFE features plus 3 METHOD_B fractions at fit time (187 columns):

| Group | Role |
|---|---|
| Calendar | Europe/Madrid hour, weekday, month, cyclic encodings |
| Day-ahead forecasts | Load / solar / wind forecasts for hour *t*, plus shares |
| Historical target | Lags t−24 / t−48 / t−168 and element-wise stats of those lags |
| Historical load | Actual-load lags and lag-24 forecast error |
| Historical generation | Generation lags t−24 / t−168 and lagged aggregates |
| Historical weather | City weather lags t−24 / t−168 |
| Aggregated weather | National means / max rain at t−24 and t−168 |
| METHOD_B | Causal 7 / 14 / 30 day high-price fractions vs development P75 |

---

## Model Selection

Compared on **validation** and/or **walk-forward TRAIN+VALIDATION** only. Test was not used to pick a family.

| Family | Where compared | Result (selection, not test) |
|---|---|---|
| Naive Lag-24 | Validation + walk-forward | Strong unbiased baseline; higher MAE |
| Ridge | Validation + walk-forward + α grid | Selected family |
| HistGradientBoosting | Validation + walk-forward | Did not beat Ridge on mean MAE |
| LightGBM | Walk-forward | Mean MAE 5.917; worse than Ridge |
| XGBoost | Walk-forward | Mean MAE 5.946; worse than Ridge |

Walk-forward (four expanding windows on 29,804 development hours) is used because a single validation cut can hide time-varying error and would be the wrong place to tune a time series. Each fold fits preprocess on that fold’s training block only.

Best walk-forward Ridge α = **0.001** (mean MAE 5.7966). METHOD_B then won the high-price-method search (mean MAE **5.496022**). Residual AR(1) with 24-hour block updates was added on walk-forward (mean MAE **4.529617**).

---

## Final Model

**Ridge(`alpha=0.001`) + METHOD_B + AR(1)**

- Closed-form NumPy Ridge `(X'X + αI)w = X'y` (not `sklearn.Ridge.fit`)
- 184 SAFE features + 3 causal high-price-frequency features
- Frozen expanding-historical residual addend from development residuals only (+1.471482)
- AR(1) on the last 1,440 development residuals of Ridge+METHOD_B; 24-hour block forecasts
- Development P75 threshold = 57.5825
- Fit on 29,804 development rows
- **MODEL STATUS = LOCKED**

---

## Final Test Results

Locked holdout, 5,260 hours. Exact stored values:

| Metric | Ridge+METHOD_B+AR(1) | Naive Lag-24 |
|---|---:|---:|
| MAE | 3.990091 | 6.045924 |
| RMSE | 5.878929 | 9.030375 |
| R² | 0.668961 | 0.218923 |
| sMAPE | 7.314412% | 11.675683% |
| Bias | +1.419419 | −0.003076 |

MODEL_BEATS_NAIVE = TRUE (~34.0% MAE reduction). The naive model is closer to unbiased on this holdout. Lower MAE is not a claim that the model is well calibrated.

High-price **test** slice bias (development quantiles, reporting only): P75+ +0.840776, P90+ +0.920483, P95+ +0.556602.

---

## Error Analysis

**Development / walk-forward:** METHOD_B+AR(1) mean MAE 4.529617, bias ≈ −0.71. METHOD_B alone was 5.496022, bias ≈ −0.91. Causal P75+ bias on METHOD_B was −5.90, P90+ −8.21. Leakage-safe correction methods **did not fully remove** high-price underprediction on development folds.

**Frozen test:** overall bias flipped to **+2.13 (overprediction)**. P75+/P90+/P95+ biases are also positive. Do not write that the model “solved” high-price underprediction. The correct statement is:

> High-price underprediction was observed during validation/development; leakage-safe corrections did not fully resolve it; on the frozen test period the error sign moved to overprediction.

Validation and test bias stories must not be mixed.

---

## Explainability

Exact **linear SHAP** on the four walk-forward folds (TRAIN+VALIDATION only). TreeSHAP was not used because the locked model is Ridge.

- `price_day_ahead_lag_24` and `lag_48` are strong predictive signals.
- `total_load_forecast` has a positive standardized coefficient.
- Renewable/wind share and some weather aggregates contribute.
- `month` and `day_of_year` are nearly collinear; large opposite coefficients do **not** mean they are independent price drivers.
- Weather attribution is spread across many columns.
- SHAP is association / explanation for the forecast. **It is not causality.**

---

## 24-Hour Forecasting

**Production-ready = FALSE.** This decision is locked.

At a D-1 ~12:00 CET origin, of 187 model columns:

| Class | Count |
|---|---:|
| SAFE | 106 |
| UNKNOWN | 6 |
| FORBIDDEN | 75 |

Reasons:

- Day-ahead forecast **publication times are not in the file** (UNKNOWN).
- Some actual `lag_24` load/generation/weather values are not yet observed at that origin (FORBIDDEN).
- STRICT mode does not use UNKNOWN/FORBIDDEN columns and does **not** silently impute them.
- The locked 187-column model therefore cannot emit a filled STRICT `y_pred`. Empty predictions are intentional.

The assumed scenario (`forecasting_predictions_assumed.csv`) is demonstration only and remains **NOT PRODUCTION-READY**. No recursive multi-step feedback was used.

To become production-ready later (not done here): verified publish timestamps for day-ahead forecasts; an origin where every required lag is observed; a model that uses only then-available columns; no silent fill of unknown features.

---

## Streamlit Dashboard

Read-only. No retraining, no test-based selection, no production forecast.

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Pages:

1. Genel bakış — locked KPIs (MAE 3.99, RMSE 5.88, R² 0.669, sMAPE 7.31%, Naive MAE 6.05)
2. Model karşılaştırması — Ridge, LightGBM, XGBoost, ARIMA, Ridge+B+AR(1)
3. Model performansı — locked test vs naive, actual vs predicted
4. Hata analizi — hour/month/quantile residuals; bias sign-flip note
5. Açıklanabilirlik — linear SHAP (development folds)
6. 24 saatlik tahmin — STRICT empty; assumed labeled not production-ready
7. Veri ve sızıntı denetimi — split, SAFE/UNKNOWN/FORBIDDEN
8. Proje bilgisi — pipeline and stage status

**MODEL STATUS = LOCKED.** Forecast status = **NOT PRODUCTION READY.**

---

## Reproducibility

Python 3 and `requirements.txt`. Ridge is deterministic (closed-form NumPy). `random_state=42` is recorded for tree comparisons and permutation checks; the locked Ridge path does not depend on shuffle.

Final inference reads archived artifacts (`data/processed/final_model/`, locked prediction parquet, locked metrics CSV). Protected hashes must stay unchanged — see `reports/final_project_audit.md`.

There is no `tests/` unit-test suite.

---

## Limitations

- Test is a **single frozen period**, not a multi-year error distribution.
- The dataset is **historical** (2015–2018 Spanish DAM + weather).
- Forecast **production availability** is not fully verified in-file.
- High-price **regime-shift** behavior is not fully resolved (development underprediction vs test overprediction).
- The model does association / prediction. **No causality claim.**
- Missing exogenous drivers (fuel prices, outages, policy, cross-border constraints) limit how far price dynamics can be explained from this feature set alone.

---

## Conclusion

The project is a leakage-safe, chronological, walk-forward-validated pipeline with a frozen holdout and a reproducible Ridge + METHOD_B + AR(1) archive.

On the locked test the model **beats Naive Lag-24 on MAE** (3.990091 vs 6.045924) and is explainable with linear SHAP. That is forecasting performance, not a claim that high-price error is solved.

**24-hour production forecasting is not ready.**

This study does not establish causality.
