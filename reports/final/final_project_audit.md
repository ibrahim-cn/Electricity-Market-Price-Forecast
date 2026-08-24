# Final Project Audit

Finalize + document + audit + clean. No new experiment. Model and test metrics unchanged.

FINAL_PROJECT_AUDIT = PASS
MODEL_LOCKED = TRUE
TEST_LOCKED = TRUE
TEST_USED_FOR_TUNING = FALSE
TEST_USED_FOR_SELECTION = FALSE
LEAKAGE_CHECK = PASS
REPRODUCIBILITY = PASS
PROTECTED_FILES_UNCHANGED = TRUE
BROKEN_REFERENCES = 0
DATA_LOSS = NONE

FINAL_MODEL = Ridge(alpha=0.001)
FINAL_METHOD = METHOD_B

FINAL_TEST_MAE = 4.329544
FINAL_TEST_RMSE = 6.136183
FINAL_TEST_R2 = 0.639356
FINAL_TEST_SMAPE = 7.739424
FINAL_TEST_BIAS = 2.133567

NAIVE_TEST_MAE = 6.045924
MODEL_BEATS_NAIVE = TRUE

---

## Scope

This audit does **not** reopen model selection. Ridge(`alpha=0.001`) + METHOD_B and the locked test numbers above are treated as final.

## Leakage

LEAKAGE_CHECK = PASS because the locked pipeline used:

- chronological 70/15/15 split, no shuffle
- target at hour *t* not used as a feature
- `price actual` unused
- no same-hour actual generation / load / weather
- target lags t−24, t−48, t−168 only; no t−1
- no rolling window on the raw target in the 184-feature matrix
- preprocess fit inside each fold (final model: train+validation only)
- test unused for tuning or selection; scored once after freeze

## Reproducibility

- `python3 -m compileall -f src` → all 9 files under `src/` compiled (exit 0)
- `pytest -q` → no test suite (`tests/` and `test_*.py` do not exist; pytest exit 5, “no tests ran”)
- Critical output hashes match the locked evaluation (see below)

REPRODUCIBILITY = PASS on compile + hash lock. There is no unit-test suite to re-run.

## Protected files

| file | md5 | unchanged |
|---|---|---|
| `data/processed/merged/merged_energy_weather.parquet` | `ae4a12026b1a9682d6bbb58ef7471fa1` | TRUE |
| `data/processed/features/model_features.parquet` | `c9f07ac0f95e0f51fff5472129c1f9ad` | TRUE |
| `data/processed/splits/train.parquet` | `278666dcdb30990b55a6aa5c882f21ee` | TRUE |
| `data/processed/splits/validation.parquet` | `cba753fa9327955d506139d25fdaae4d` | TRUE |
| `data/processed/splits/test.parquet` | `069afbe9c766426d2e095282ece93a69` | TRUE |
| `data/processed/predictions/final_test_predictions.parquet` | `586d26918ee347ee74d70a25535836eb` | TRUE |
| `reports/final/final_test_metrics.csv` | `1ff84294518988ad83186f0ebdfec9cf` | TRUE |

## Dependency / cleanup audit

Inventory: `src/` (9 scripts), `reports/{data,features,modeling,final}/`, `data/raw/` (2 CSVs), `data/processed/{merged,features,splits,predictions}/`, project-root original CSVs.

Import graph (all referenced):

- `data_preparation` → none of the later scripts
- `feature_engineering` → none
- `time_series_split` → none
- `baseline_models` → sklearn metrics / HGB
- `walk_forward_validation` → sklearn; optional LightGBM / XGBoost
- `ridge_tuning` → sklearn metrics
- `residual_correction` → `ridge_tuning`
- `high_price_analysis` → `ridge_tuning`, `residual_correction`
- `final_test_evaluation` → `ridge_tuning`, `residual_correction`, `high_price_analysis`

No unused script. No cache, log, tmp, or `__pycache__` in the repository.

**Not deleted (uncertain or required):**

- project-root `energy_dataset.csv`, `weather_features.csv` (original raw files)
- `data/raw/*` (pipeline copies)
- `data/processed/merged/merged_energy_weather.csv` (same merge as the parquet; official pipeline output)
- intermediate prediction parquets (`baseline_*`, `ridge_walk_forward_*`, `residual_correction_*`, `high_price_strategy_*`)
- all existing stage reports

**Deleted:** none. Conservative rule: do not delete UNCERTAIN files.

**Folder reorganization (later):** reports and processed files were moved into subfolders. File **contents** (MD5) were not changed. Script paths were updated to match.

DATA_LOSS = NONE  
BROKEN_REFERENCES = 0

## Test-use statement

TEST_USED_FOR_TUNING = FALSE  
TEST_USED_FOR_SELECTION = FALSE  

Test (5,260 rows) was used only for the locked holdout evaluation already recorded in `reports/final/final_test_evaluation.md`.
