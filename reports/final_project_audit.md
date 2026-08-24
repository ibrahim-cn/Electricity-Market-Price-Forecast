# Final Project Audit

Finalize + document + audit. No new experiment. Model, α, METHOD_B, locked test metrics, and the 24-hour production decision are unchanged.

```
PROJECT_FINALIZATION = PASS

README_UPDATED = TRUE
FINAL_REPORT_CREATED = TRUE
DASHBOARD_CHECK = PASS
TESTS = PASS
IMPORT_CHECK = PASS
ARTIFACT_CHECK = PASS
PROTECTED_FILES_UNCHANGED = TRUE
LEAKAGE_CHECK = PASS
REPRODUCIBILITY = PASS
FORECAST_PRODUCTION_READY = FALSE

FILES_DELETED = .pytest_cache/
FILES_CREATED = reports/final_project_report.md; reports/final_results_summary.csv; reports/project_status.md; reports/final_project_audit.md
FILES_UPDATED = README.md; requirements.txt

FINAL_MODEL = Ridge(alpha=0.001)
FINAL_METHOD = METHOD_B
FINAL_TEST_MAE = 4.329544
FINAL_TEST_RMSE = 6.136183
FINAL_TEST_R2 = 0.639356
FINAL_TEST_SMAPE = 7.739424
FINAL_TEST_BIAS = 2.133567
NAIVE_TEST_MAE = 6.045924
```

---

## Scope

This audit does **not** reopen model selection. Ridge(`alpha=0.001`) + METHOD_B and the locked test numbers above are treated as final. `app.py` was inspected and **not** modified (Turkish UI already matches the seven delivery pages; KPIs match locked metrics; forecast remains not production-ready).

## Prompt vs repository

The assignment prompt titled the project “ETH/USDT Day-Ahead Price Prediction” and mentioned Reddit comments. Those files are **not** in the repository. All locked artifacts use Spanish hourly `price day ahead` (€/MWh), energy + weather CSVs, and electricity feature groups. Documentation follows the artifacts, not the prompt title.

## Repository inventory

| Path | Role |
|---|---|
| `README.md` | Delivery README (updated this stage) |
| `requirements.txt` | Runtime deps (streamlit kept; joblib + matplotlib added because scripts import them) |
| `app.py` | Read-only Streamlit dashboard |
| `src/` | 12 pipeline scripts, all referenced |
| `data/raw/` | Pipeline copies of original CSVs |
| `data/processed/{merged,features,splits,predictions,final_model}/` | Protected / stage artifacts |
| `reports/{data,features,modeling,final,explainability,forecasting,final_model}/` | Stage reports |
| `outputs/figures/` | SHAP and assumed-forecast figures |
| `energy_dataset.csv`, `weather_features.csv` | Original raw files at repo root |
| `notebooks/` | Absent |
| `tests/` | Absent |
| `scripts/` | Absent |

Import graph (all used):

- `data_preparation` → (standalone)
- `feature_engineering` → (standalone)
- `time_series_split` → (standalone)
- `baseline_models` → sklearn HGB / metrics
- `walk_forward_validation` → sklearn; optional LightGBM / XGBoost
- `ridge_tuning` → sklearn metrics
- `residual_correction` → `ridge_tuning`
- `high_price_analysis` → `ridge_tuning`, `residual_correction`
- `final_test_evaluation` → `ridge_tuning`, `residual_correction`, `high_price_analysis`
- `shap_explainability` → `ridge_tuning`, `residual_correction`, `high_price_analysis`
- `final_model` → `ridge_tuning`, `residual_correction`, `high_price_analysis`, joblib
- `forecast_24h` → `ridge_tuning`, joblib, matplotlib
- `app.py` → reads artifacts only

LightGBM / XGBoost remain optional (`find_spec`). They were used in the locked walk-forward comparison CSV and are **not** removed from the story. They are not hard requirements for the dashboard or locked inference.

## Unused / duplicate / temporary (not deleted unless safe)

Detected, **kept** (uncertain or required):

- Root `energy_dataset.csv`, `weather_features.csv` (original sources; do not overwrite)
- `data/raw/*` (pipeline copies)
- `data/processed/merged/merged_energy_weather.csv` (same merge as parquet)
- Intermediate prediction parquets (`baseline_*`, `ridge_walk_forward_*`, `residual_correction_*`, `high_price_strategy_*`)
- `reports/final/final_project_report.md` (earlier 18-section write-up; superseded for delivery by `reports/final_project_report.md`)
- `.streamlit/` (dashboard email-skip config)

**Deleted (safe generated cache only):** `.pytest_cache/`

No smoke-test scripts, unused logs, or orphan pipeline modules were found.

## Leakage

LEAKAGE_CHECK = PASS (unchanged locked pipeline):

- chronological 70/15/15, no shuffle
- target at hour *t* not a feature
- `price actual` unused
- no same-hour actual generation / load / weather
- target lags t−24 / t−48 / t−168 only; no t−1
- no rolling window on the raw target in the 184-feature matrix
- preprocess fit inside each fold (final: train+validation only)
- test unused for tuning or selection; scored once after freeze
- STRICT 24h does not fill UNKNOWN/FORBIDDEN; `y_pred` all-null (24 rows)

## Dashboard

DASHBOARD_CHECK = PASS

| check | result |
|---|---|
| `ast.parse` / `compile` `app.py` | PASS |
| `import app` | PASS |
| `load_predictions()` | 5,260 rows; `timestamp_utc`, `y_true`, `y_pred`, `residual` |
| Required report CSVs | all present |
| Existing `streamlit` on :8501 | HTTP 200 |
| Headless `streamlit` on :8766 | HTTP 200; process stopped after check |
| MODEL STATUS | LOCKED (α=0.001, METHOD_B) |
| Forecast status | NOT PRODUCTION READY |
| `app.py` edited this stage | FALSE |

## Tests and imports

| check | result |
|---|---|
| Syntax compile of `src/*.py` + `app.py` | PASS |
| Import of all 12 `src` modules | PASS |
| `pytest -q` | no tests ran (exit 5); no failures |
| Artifact load via dashboard helpers | PASS |

TESTS = PASS means: no failing tests. There is still no `tests/` unit-test suite.

`python3 -m compileall` attempted to write `.pyc` under `~/Library/Caches` and hit a sandbox permission error. Syntax was verified with `ast.parse` + `compile(..., "exec")` instead. That is not a source-code failure.

## Protected files

| file | md5 | unchanged |
|---|---|---|
| `data/processed/merged/merged_energy_weather.parquet` | `ae4a12026b1a9682d6bbb58ef7471fa1` | TRUE |
| `data/processed/features/model_features.parquet` | `c9f07ac0f95e0f51fff5472129c1f9ad` | TRUE |
| `data/processed/splits/train.parquet` | `278666dcdb30990b55a6aa5c882f21ee` | TRUE |
| `data/processed/splits/validation.parquet` | `cba753fa9327955d506139d25fdaae4d` | TRUE |
| `data/processed/splits/test.parquet` | `069afbe9c766426d2e095282ece93a69` | TRUE |
| `data/processed/predictions/final_test_predictions.parquet` | `586d26918ee347ee74d70a25535836eb` | TRUE |
| `data/processed/final_model/final_model_test_predictions.parquet` | `586d26918ee347ee74d70a25535836eb` | TRUE (same bytes as locked preds) |
| `reports/final/final_test_metrics.csv` | `1ff84294518988ad83186f0ebdfec9cf` | TRUE |

Final-model archive hashes (not overwritten this stage):

| file | md5 |
|---|---|
| `data/processed/final_model/model.joblib` | `fe5038edced95c791bf2dcace597f48d` |
| `data/processed/final_model/preprocessing.joblib` | `75a56ce75b8c5da86d28a80d3ee19d99` |
| `data/processed/final_model/model_metadata.json` | `2af3c7beb4e661a7f5cb2adda9b7947b` |
| `data/processed/final_model/method_b_parameters.json` | `ee6f9608dfc92f1e5fda4222fef22cce` |
| `data/processed/final_model/feature_manifest.json` | `94eb94ecf340e479a9c156e5f7699979` |

ARTIFACT_CHECK = PASS  
PROTECTED_FILES_UNCHANGED = TRUE

## Requirements

`requirements.txt` now lists packages that the locked pipeline and dashboard actually import:

- numpy, pandas, pyarrow, scikit-learn, streamlit (already present)
- joblib, matplotlib (added; used by `final_model.py`, `forecast_24h.py`, `shap_explainability.py`)

LightGBM / XGBoost were **not** added as hard deps (optional in `walk_forward_validation.py`). They were not removed from the comparison narrative.

## Reproducibility

REPRODUCIBILITY = PASS on syntax/import + hash lock + artifact-based dashboard. Ridge remains closed-form NumPy. `random_state=42` is recorded for tree comparisons and permutation checks only.

## Test-use statement

TEST_USED_FOR_TUNING = FALSE  
TEST_USED_FOR_SELECTION = FALSE  

Test (5,260 rows) remains the locked holdout already recorded in `reports/final/final_test_evaluation.md`.

## Forecasting

FORECAST_PRODUCTION_READY = FALSE  
STRICT `y_pred` all missing. Assumed scenario still labeled not production-ready.

DATA_LOSS = NONE  
BROKEN_REFERENCES = 0
