# Project Status

Locked delivery checklist. Model, α, METHOD_B, test metrics, and the 24-hour production decision are not reopened.

## Completed

- [x] Data collection
- [x] Data quality
- [x] Merge
- [x] EDA
- [x] Leakage audit
- [x] Feature engineering
- [x] Time split
- [x] Baseline
- [x] Model comparison
- [x] Walk-forward validation
- [x] Hyperparameter tuning
- [x] Residual correction
- [x] High-price analysis
- [x] Explainability
- [x] Final model
- [x] Frozen test
- [x] Streamlit dashboard
- [x] README / report

## Open / blocked

- [!] 24-hour production forecasting  
      **NOT PRODUCTION READY**

STRICT availability at D-1 ~12:00 CET: SAFE = 106, UNKNOWN = 6, FORBIDDEN = 75 of 187 columns. STRICT `y_pred` is empty. The assumed 24-hour path is demonstration only.

## Locked snapshot

| Item | Value |
|---|---|
| Model | Ridge(`alpha=0.001`) |
| Method | METHOD_B |
| Development rows | 29,804 |
| Test MAE | 4.329544 |
| Test RMSE | 6.136183 |
| Test R² | 0.639356 |
| Test sMAPE | 7.739424 |
| Test bias | +2.133567 |
| Naive test MAE | 6.045924 |
| MODEL_BEATS_NAIVE | TRUE |
| Forecast production | FALSE |

## Next steps (production forecasting only)

These steps would be a **new** project stage. They must not use the locked test for selection or silently change METHOD_B / α.

1. Obtain independent publication timestamps for `total_load_forecast`, `forecast_solar_day_ahead`, and `forecast_wind_onshore_day_ahead`.
2. Fix a real forecast origin (Spanish DAM D-1 ~12:00 CET or a documented alternative) and re-classify every required column as SAFE at that origin.
3. Either drop UNKNOWN/FORBIDDEN columns or train a new leakage-safe model that uses only then-available features — on development data only.
4. Do not impute UNKNOWN/FORBIDDEN features from future rows or from zeros.
5. Do not use recursive ŷ as a substitute for unpublished actuals.
6. Re-check high-price / level-shift behavior on a **new** chronological holdout; the current test bias sign already flipped versus development.
7. Keep PRODUCTION_READY = FALSE until every required feature is verified available at origin.

## Not in this repository

- ETH/USDT price series
- Reddit comments
- `notebooks/`, `tests/`, `scripts/`
