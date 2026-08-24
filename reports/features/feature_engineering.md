# Feature Engineering Report

## 1. Feature engineering objective

Build a leakage-safe feature matrix for forecasting `price day ahead` at delivery hour t.
No model is trained. No split, tuning, or evaluation is performed.
Source `data/processed/merged/merged_energy_weather.parquet` is read-only.

Output: `data/processed/features/model_features.parquet` (identifier `timestamp_utc` + SAFE features only).
The target stays in the source parquet and is joined later by `timestamp_utc`.

## 2. Information boundary

Predict `price_day_ahead(t)` before delivery day D.

Forbidden at t: `price actual`; current or future `price day ahead`; other hours of the same delivery-day auction; same-hour generation, `total load actual`, and weather; future-looking rolls; target-at-t stats; full-dataset target encoding; randomness.

Allowed target history: **t-24, t-48, t-168 only**. `price_day_ahead(t-1)` is not created.

Calendar fields use `timestamp_utc` converted to **Europe/Madrid** so hour-of-day follows the Spanish clock and DST. The UTC timestamp remains the index/alignment key and is not a numeric feature.

## 3. Calendar features

20 features: hour, day_of_week, day_of_month, month, quarter, day_of_year, week_of_year (ISO), weekend/month/year boundary flags, plus sin/cos for hour (period 24), weekday (7), month (12), day_of_year (365.25).

Raw `time` / `timestamp_utc` are not in X.

## 4. Day-ahead forecast features

Identity: `total_load_forecast`, `forecast_solar_day_ahead`, `forecast_wind_onshore_day_ahead`.

Derived (row t only, no future rows):

- `renewable_forecast_total` = solar_forecast(t) + wind_forecast(t)
- `forecast_wind_share_of_load` = wind_forecast(t) / load_forecast(t)
- `forecast_solar_share_of_load` = solar_forecast(t) / load_forecast(t)

Shares are NaN when load_forecast ≤ 0 or non-finite. No actual generation or load is used.

## 5. Historical target features

Created only from `price day ahead` via `shift(+k)`:

- `price_day_ahead_lag_24`
- `price_day_ahead_lag_48`
- `price_day_ahead_lag_168`

Statistics are element-wise over those three series (or the first two), **not** a rolling window on the raw target. `skipna=False`: if any required lag is NaN, the statistic is NaN.

- `price_mean_lag24_lag48`
- `price_mean_lag24_lag48_lag168`
- `price_std_lag24_lag48_lag168` (population std, ddof=0)
- `price_min_lag24_lag48_lag168`
- `price_max_lag24_lag48_lag168`

`price_day_ahead_lag_1` does not exist.

## 6. Historical load features

- `total_load_actual_lag_24`, `_48`, `_168`
- `load_forecast_error_lag_24` = actual(t-24) − forecast(t-24)

`total load actual` at t is not in the matrix.

## 7. Historical generation features

Individual `lag_24` and `lag_168` for meaningful generation columns (zero-only series omitted; 100% empty columns were already absent from the source).

**Renewable:** biomass, hydro run-of-river and poundage, hydro water reservoir, other renewable, solar, wind onshore.  
**Fossil:** lignite, fossil gas, hard coal, fossil oil.  
**Total production:** renewable + fossil + nuclear + other + waste.  
**Excluded from aggregates:** hydro pumped storage *consumption* (not production), zero-only columns, waste not in renewable.

Aggregates are computed at historical hour s, then lagged: `total_generation_lag_*`, `renewable_generation_lag_*`, `fossil_generation_lag_*`, `renewable_share_lag_*` (NaN if total≤0).

Hydro pumped storage consumption still has its own lags; it is not inside renewable/fossil/total production.

## 8. Historical weather features

Same-hour weather is excluded.

Per city, `lag_24` and `lag_168` for: temp, humidity, pressure, wind_speed, clouds_all, rain_1h, rain_3h, snow_3h.

Wind direction: `wind_deg` is converted to `sin`/`cos` **before** lagging. Degree values are never averaged. No t-1 weather lags.

## 9. Weather aggregation

At each historical hour s, then lagged to t-24 and t-168 only:

- `temp_national_mean_lag_*`
- `humidity_national_mean_lag_*`
- `wind_speed_national_mean_lag_*`
- `clouds_all_national_mean_lag_*`
- `rain_1h_national_max_lag_*` (max hourly rain across the five cities)

Current weather is not used. `wind_deg` is not spatially averaged.

## 10. Missing-value behavior

- No zero-fill unless a share denominator is invalid (then **NaN**, not 0).
- Target is not interpolated or forward-filled (target is not even written to this file).
- Leading NaNs on lags are expected: first 24 / 48 / 168 UTC hours have no history.
- Rows are **not** dropped. Final rows = 35064.

## 11. Leakage prevention

Automated checks (script fails on violation):

1. target not in output
2. `price actual` not in output
3. no unsuffixed target name in X
4. no same-hour actual generation / load / weather
5. no negative `shift`
6. no lag-1 features
7. no random APIs in this source file
8. no `.rolling(` in this source file
9. timestamps sorted ascending
10. timestamps unique
11. 35064 rows, `timestamp_utc` identical to source

`price actual` is never lagged.

## 12. Final feature count

| Group | Count |
|---|---:|
| calendar | 20 |
| day_ahead_forecast | 6 |
| historical_generation | 36 |
| historical_load | 4 |
| historical_target | 8 |
| historical_weather | 100 |
| weather_aggregate | 10 |
| **SAFE features (X)** | **184** |
| Identifier (`timestamp_utc`) | 1 |
| Target in this file | 0 |

## 13. Removed / forbidden features

Not copied into `model_features.parquet`:

- `price day ahead` (target; remains in source only)
- `price actual`
- `time`
- same-hour `total load actual` and all same-hour `generation *`
- same-hour weather (numeric and categorical)
- zero-only generation series (no lag features)
- `price_day_ahead_lag_1`

## 14. Validation results

| Check | Value |
|---|---|
| original rows | 35064 |
| final rows | 35064 |
| total features | 184 |
| SAFE features | 184 |
| forbidden features in output | 0 |
| first timestamp | 2014-12-31T23:00:00+00:00 |
| last timestamp | 2018-12-31T22:00:00+00:00 |
| min timestamp diff | 0 days 01:00:00 |
| max timestamp diff | 0 days 01:00:00 |

NaN cell counts by group (lag heads dominate):

| feature_group | NaN cells |
|---|---:|
| calendar | 0 |
| day_ahead_forecast | 0 |
| historical_generation | 3672 |
| historical_load | 320 |
| historical_target | 960 |
| historical_weather | 9600 |
| weather_aggregate | 960 |

Manifest: `reports/features/feature_manifest.csv`.
