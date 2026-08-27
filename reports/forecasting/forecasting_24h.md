# 24-Hour Forecasting

**FORECASTING_24H = PASS**

Locked model: Ridge(`alpha=0.001`) + METHOD_B. Not retuned.

> 24-hour forecasting is only production-ready if every required feature is
> available at the forecast origin.

**PRODUCTION_READY = FALSE**

Forecast availability timestamp is not independently verified.

## 1. Forecast origin and horizon

Spanish DAM framing: origin **τ = D-1 ~12:00 CET**, target = 24 hourly
`price day ahead` values for delivery day D.

This simulation's last known leakage-safe timestamp (end of development,
test never opened):

| item | value |
|---|---|
| forecast_origin | 2018-05-26T18:00:00+00:00 |
| horizons | h+1 … h+24 |
| first target hour | 2018-05-26T19:00:00+00:00 |
| last target hour | 2018-05-27T18:00:00+00:00 |
| recursive forecasting | not used |
| test.parquet read | no |

Direct 24-hour forecasting: each hour's lags and METHOD_B fractions use
**published history ≤ origin** (or ≤ t−24, which is ≤ origin for this
window). Model predictions are not fed back as future features.

## 2. Feature availability audit

DAM origin **D-1 12:00 CET** (the real operational origin), not the clock
of the last development row.

| classification_strict | count |
|---|---:|
| SAFE | 106 |
| CONDITIONAL | 0 |
| UNKNOWN | 6 |
| FORBIDDEN | 75 |

Full table: `forecasting_availability_audit.csv`.

### Requested feature families

| family | STRICT at D-1 noon | why |
|---|---|---|
| `price_day_ahead_lag_24/48/168` | SAFE | Prior auction curves, published on D-2 |
| `total_load_forecast` | UNKNOWN | Named day-ahead; no publish timestamp in file |
| `forecast_solar_day_ahead` | UNKNOWN | same |
| `forecast_wind_onshore_day_ahead` | UNKNOWN | same |
| historical load / generation / weather `lag_24` | FORBIDDEN for part of D | t−24 for evening hours of D is still in D-1 afternoon |
| historical `lag_48` / `lag_168` | SAFE | complete prior days |
| weather aggregates `lag_24` | FORBIDDEN for part of D | same as city weather lag_24 |
| calendar | SAFE | deterministic |
| METHOD_B fractions | SAFE | past published prices vs development P75 |

### Scenario A — STRICT / verified

Only SAFE features may enter X. UNKNOWN is not imputed and not read from a
future row. The locked model needs **all 187 columns**, including UNKNOWN
day-ahead forecasts and FORBIDDEN-at-noon `lag_24` actuals.

Blockers:

- Day-ahead load/solar/wind forecasts have no publication timestamp (UNKNOWN).
- At a D-1 12:00 CET origin, historical lag_24 actuals for later hours of D are not yet observed (FORBIDDEN).
- The locked 187-column model cannot be scored on STRICT features alone without changing the model.

Therefore STRICT does **not** emit a filled `y_pred`. Empty predictions are
preferred to a leaked 24-hour path.

### Scenario B — dataset-assumed day-ahead forecasts

If we **assume** the named forecast columns for hour t were published before
τ, those six columns become CONDITIONAL. That assumption is not verified.
Even then, a true D-1 noon origin still cannot lawfully fill `lag_24`
actuals for later hours of D.

An illustration with a **late** origin (2018-05-25T18:00:00+00:00, 24h
before the last development stamp) makes `t-24 ≤ origin` for all 24 hours,
so historical lags are available. Forecast columns are taken from those
development rows only (test still closed). That run is labeled
**NOT PRODUCTION-READY**.

## 3. STRICT production table

| timestamp_utc | forecast_horizon | y_pred | forecast_origin | model | alpha | method | scenario | production_ready | blocking_reason |
|---|---|---|---|---|---|---|---|---|---|
| 2018-05-26T19:00:00+00:00 | h+1 | nan | 2018-05-26T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | STRICT | False | required UNKNOWN/FORBIDDEN features at forecast origin |
| 2018-05-26T20:00:00+00:00 | h+2 | nan | 2018-05-26T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | STRICT | False | required UNKNOWN/FORBIDDEN features at forecast origin |
| 2018-05-26T21:00:00+00:00 | h+3 | nan | 2018-05-26T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | STRICT | False | required UNKNOWN/FORBIDDEN features at forecast origin |
| 2018-05-26T22:00:00+00:00 | h+4 | nan | 2018-05-26T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | STRICT | False | required UNKNOWN/FORBIDDEN features at forecast origin |
| 2018-05-26T23:00:00+00:00 | h+5 | nan | 2018-05-26T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | STRICT | False | required UNKNOWN/FORBIDDEN features at forecast origin |
| 2018-05-27T00:00:00+00:00 | h+6 | nan | 2018-05-26T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | STRICT | False | required UNKNOWN/FORBIDDEN features at forecast origin |
| 2018-05-27T01:00:00+00:00 | h+7 | nan | 2018-05-26T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | STRICT | False | required UNKNOWN/FORBIDDEN features at forecast origin |
| 2018-05-27T02:00:00+00:00 | h+8 | nan | 2018-05-26T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | STRICT | False | required UNKNOWN/FORBIDDEN features at forecast origin |
| 2018-05-27T03:00:00+00:00 | h+9 | nan | 2018-05-26T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | STRICT | False | required UNKNOWN/FORBIDDEN features at forecast origin |
| 2018-05-27T04:00:00+00:00 | h+10 | nan | 2018-05-26T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | STRICT | False | required UNKNOWN/FORBIDDEN features at forecast origin |
| 2018-05-27T05:00:00+00:00 | h+11 | nan | 2018-05-26T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | STRICT | False | required UNKNOWN/FORBIDDEN features at forecast origin |
| 2018-05-27T06:00:00+00:00 | h+12 | nan | 2018-05-26T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | STRICT | False | required UNKNOWN/FORBIDDEN features at forecast origin |
| 2018-05-27T07:00:00+00:00 | h+13 | nan | 2018-05-26T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | STRICT | False | required UNKNOWN/FORBIDDEN features at forecast origin |
| 2018-05-27T08:00:00+00:00 | h+14 | nan | 2018-05-26T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | STRICT | False | required UNKNOWN/FORBIDDEN features at forecast origin |
| 2018-05-27T09:00:00+00:00 | h+15 | nan | 2018-05-26T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | STRICT | False | required UNKNOWN/FORBIDDEN features at forecast origin |
| 2018-05-27T10:00:00+00:00 | h+16 | nan | 2018-05-26T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | STRICT | False | required UNKNOWN/FORBIDDEN features at forecast origin |
| 2018-05-27T11:00:00+00:00 | h+17 | nan | 2018-05-26T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | STRICT | False | required UNKNOWN/FORBIDDEN features at forecast origin |
| 2018-05-27T12:00:00+00:00 | h+18 | nan | 2018-05-26T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | STRICT | False | required UNKNOWN/FORBIDDEN features at forecast origin |
| 2018-05-27T13:00:00+00:00 | h+19 | nan | 2018-05-26T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | STRICT | False | required UNKNOWN/FORBIDDEN features at forecast origin |
| 2018-05-27T14:00:00+00:00 | h+20 | nan | 2018-05-26T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | STRICT | False | required UNKNOWN/FORBIDDEN features at forecast origin |
| 2018-05-27T15:00:00+00:00 | h+21 | nan | 2018-05-26T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | STRICT | False | required UNKNOWN/FORBIDDEN features at forecast origin |
| 2018-05-27T16:00:00+00:00 | h+22 | nan | 2018-05-26T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | STRICT | False | required UNKNOWN/FORBIDDEN features at forecast origin |
| 2018-05-27T17:00:00+00:00 | h+23 | nan | 2018-05-26T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | STRICT | False | required UNKNOWN/FORBIDDEN features at forecast origin |
| 2018-05-27T18:00:00+00:00 | h+24 | nan | 2018-05-26T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | STRICT | False | required UNKNOWN/FORBIDDEN features at forecast origin |

`y_pred` is empty because required features are UNKNOWN or FORBIDDEN at
the DAM origin / unverified at this origin.

## 4. ASSUMED illustration (not production)

Origin = 2018-05-25T18:00:00+00:00 (development only).

| timestamp_utc | forecast_horizon | y_pred | forecast_origin | model | alpha | method | scenario | production_ready | blocking_reason |
|---|---|---|---|---|---|---|---|---|---|
| 2018-05-25T19:00:00+00:00 | h+1 | 72.1579 | 2018-05-25T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | DATASET_ASSUMED | False | forecast publication time not verified; not a DAM-noon origin |
| 2018-05-25T20:00:00+00:00 | h+2 | 70.8079 | 2018-05-25T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | DATASET_ASSUMED | False | forecast publication time not verified; not a DAM-noon origin |
| 2018-05-25T21:00:00+00:00 | h+3 | 70.2928 | 2018-05-25T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | DATASET_ASSUMED | False | forecast publication time not verified; not a DAM-noon origin |
| 2018-05-25T22:00:00+00:00 | h+4 | 69.6623 | 2018-05-25T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | DATASET_ASSUMED | False | forecast publication time not verified; not a DAM-noon origin |
| 2018-05-25T23:00:00+00:00 | h+5 | 68.4337 | 2018-05-25T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | DATASET_ASSUMED | False | forecast publication time not verified; not a DAM-noon origin |
| 2018-05-26T00:00:00+00:00 | h+6 | 66.2878 | 2018-05-25T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | DATASET_ASSUMED | False | forecast publication time not verified; not a DAM-noon origin |
| 2018-05-26T01:00:00+00:00 | h+7 | 66.7927 | 2018-05-25T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | DATASET_ASSUMED | False | forecast publication time not verified; not a DAM-noon origin |
| 2018-05-26T02:00:00+00:00 | h+8 | 65.3534 | 2018-05-25T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | DATASET_ASSUMED | False | forecast publication time not verified; not a DAM-noon origin |
| 2018-05-26T03:00:00+00:00 | h+9 | 66.1906 | 2018-05-25T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | DATASET_ASSUMED | False | forecast publication time not verified; not a DAM-noon origin |
| 2018-05-26T04:00:00+00:00 | h+10 | 65.3536 | 2018-05-25T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | DATASET_ASSUMED | False | forecast publication time not verified; not a DAM-noon origin |
| 2018-05-26T05:00:00+00:00 | h+11 | 63.5174 | 2018-05-25T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | DATASET_ASSUMED | False | forecast publication time not verified; not a DAM-noon origin |
| 2018-05-26T06:00:00+00:00 | h+12 | 63.9916 | 2018-05-25T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | DATASET_ASSUMED | False | forecast publication time not verified; not a DAM-noon origin |
| 2018-05-26T07:00:00+00:00 | h+13 | 64.2464 | 2018-05-25T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | DATASET_ASSUMED | False | forecast publication time not verified; not a DAM-noon origin |
| 2018-05-26T08:00:00+00:00 | h+14 | 63.6881 | 2018-05-25T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | DATASET_ASSUMED | False | forecast publication time not verified; not a DAM-noon origin |
| 2018-05-26T09:00:00+00:00 | h+15 | 61.8242 | 2018-05-25T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | DATASET_ASSUMED | False | forecast publication time not verified; not a DAM-noon origin |
| 2018-05-26T10:00:00+00:00 | h+16 | 60.9349 | 2018-05-25T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | DATASET_ASSUMED | False | forecast publication time not verified; not a DAM-noon origin |
| 2018-05-26T11:00:00+00:00 | h+17 | 59.2043 | 2018-05-25T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | DATASET_ASSUMED | False | forecast publication time not verified; not a DAM-noon origin |
| 2018-05-26T12:00:00+00:00 | h+18 | 57.7667 | 2018-05-25T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | DATASET_ASSUMED | False | forecast publication time not verified; not a DAM-noon origin |
| 2018-05-26T13:00:00+00:00 | h+19 | 56.3971 | 2018-05-25T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | DATASET_ASSUMED | False | forecast publication time not verified; not a DAM-noon origin |
| 2018-05-26T14:00:00+00:00 | h+20 | 54.5851 | 2018-05-25T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | DATASET_ASSUMED | False | forecast publication time not verified; not a DAM-noon origin |
| 2018-05-26T15:00:00+00:00 | h+21 | 55.5746 | 2018-05-25T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | DATASET_ASSUMED | False | forecast publication time not verified; not a DAM-noon origin |
| 2018-05-26T16:00:00+00:00 | h+22 | 58.5644 | 2018-05-25T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | DATASET_ASSUMED | False | forecast publication time not verified; not a DAM-noon origin |
| 2018-05-26T17:00:00+00:00 | h+23 | 61.5462 | 2018-05-25T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | DATASET_ASSUMED | False | forecast publication time not verified; not a DAM-noon origin |
| 2018-05-26T18:00:00+00:00 | h+24 | 62.4334 | 2018-05-25T18:00:00+00:00 | Ridge | 0.0010 | METHOD_B | DATASET_ASSUMED | False | forecast publication time not verified; not a DAM-noon origin |

These numbers must not be used to retune, pick features, or change METHOD_B.

## 5. Leakage asserts

FUTURE_LEAKAGE = PASS

- future actual weather: not used
- future actual generation: not used
- future actual load: not used
- future `price actual`: not used
- future target: not used (METHOD_B history truncated at origin)
- test target: not used
- test.parquet: not opened
- no t−1 same-auction price lag
- no recursive ŷ → feature loop

TEST_USED_FOR_TUNING = FALSE
TEST_USED_FOR_SELECTION = FALSE

## 6. Reproducibility and protected files

REPRODUCIBILITY = PASS
PROTECTED_FILES_UNCHANGED = TRUE
FEATURE_AVAILABILITY_AUDIT = PASS

| file | unchanged |
|---|---|
| energy_dataset.csv | True |
| weather_features.csv | True |
| energy_dataset.csv | True |
| weather_features.csv | True |
| merged_energy_weather.parquet | True |
| model_features.parquet | True |
| train.parquet | True |
| validation.parquet | True |
| test.parquet | True |
| final_test_predictions.parquet | True |
| final_test_metrics.csv | True |
| final_test_evaluation.md | True |
| final_model.md | True |

| file | md5 |
|---|---|
| forecasting_availability_audit.csv | 5e680a1608e4c6e9651d82678ff3b108 |
| forecasting_predictions.csv | f53676704bb3c681d76e4cc2a368bfb0 |
| forecasting_predictions_assumed.csv | 8b6dfa93e5daddd748f3bb167823c651 |

Locked test evaluation files were not rewritten.

FORECASTING_24H = PASS
FEATURE_AVAILABILITY_AUDIT = PASS
FUTURE_LEAKAGE = PASS
TEST_USED_FOR_TUNING = FALSE
TEST_USED_FOR_SELECTION = FALSE
PROTECTED_FILES_UNCHANGED = TRUE
REPRODUCIBILITY = PASS
PRODUCTION_READY = FALSE
