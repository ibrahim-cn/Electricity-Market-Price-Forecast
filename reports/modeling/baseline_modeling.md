# Baseline Modeling

**BASELINE_MODELING = PASS**

No Optuna. No feature selection. Test was not used to choose a model, alpha, or preprocessor.
Protected split/feature/raw files were not modified.

## Target (`price day ahead`, untransformed)

| split | count | mean | median | std | min | max | NaN |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 24544 | 46.8470 | 47.5900 | 14.2259 | 2.3000 | 101.9900 | 0 |
| VALIDATION | 5260 | 52.7308 | 53.9900 | 14.0696 | 2.0600 | 88.4400 | 0 |
| TEST | 5260 | 61.1438 | 63.0000 | 10.2178 | 3.0000 | 81.8200 | 0 |

## NaN handling

Train X: 15512 NaN cells in 198 rows across 158 features.
Validation/test X: 0 NaN.

These are almost entirely lag warm-up (`lag_24` / `lag_48` / `lag_168`) plus a few extra historical-load/generation NaNs that were already in the source series. Rows were **not** dropped.

**Strategy**

- Ridge: column-median imputation **fit on X_train only**, then standardize **fit on imputed X_train only**, then closed-form Ridge `(X'X + αI)w = X'y` (NumPy).
- HistGradientBoosting: native NaN support; no imputer (train NaNs stay missing).
- Naive lag predictors: use existing lag columns; validation/test have complete lags.

Validation/test were never used to compute imputer or scaler statistics.

## Comparison (validation for all; test only where allowed)

See `reports/baseline_model_comparison.csv`.

| model | split | MAE | RMSE | R2 | sMAPE | bias |
|---|---|---|---|---|---|---|
| Naive Lag-24 | validation | 8.3036 | 11.9273 | 0.2813 | 20.0653 | 0.0073 |
| Mean Lag-24/48 | validation | 8.8228 | 12.3645 | 0.2277 | 20.5113 | -0.0084 |
| Mean Lag-24/48/168 | validation | 9.0786 | 12.3882 | 0.2247 | 20.4098 | -0.0589 |
| Naive Lag-24 | test | 6.0459 | 9.0304 | 0.2189 | 11.6757 | -0.0031 |
| Ridge | validation | 5.4868 | 7.1232 | 0.7437 | 13.0909 | -1.7177 |
| HistGradientBoosting | validation | 6.6498 | 8.6799 | 0.6194 | 15.7620 | -4.2988 |
| Ridge | test | 6.3654 | 7.5929 | 0.4478 | 11.6426 | -4.7049 |

Ridge alpha grid (validation MAE only): [{'alpha': 0.1, 'MAE': 5.486840378023054, 'RMSE': 7.1232488301070305, 'R2': 0.743673059431947, 'sMAPE': 13.090892988412092, 'bias': -1.7177250095188477}, {'alpha': 1.0, 'MAE': 5.563253116035369, 'RMSE': 7.229678599942421, 'R2': 0.7359561814188156, 'sMAPE': 13.320324960154725, 'bias': -2.0336178938088842}, {'alpha': 10.0, 'MAE': 5.608210091079411, 'RMSE': 7.290177376810229, 'R2': 0.7315185939539076, 'sMAPE': 13.430849911487918, 'bias': -2.1487222798771617}, {'alpha': 100.0, 'MAE': 5.668256382387095, 'RMSE': 7.364762145850047, 'R2': 0.725996901638062, 'sMAPE': 13.551778258177604, 'bias': -2.261525305386584}]. **Selected alpha = 0.1**.

## Answers

1. **Naive 24-hour baseline (validation):** MAE=8.3036, RMSE=11.9273, R2=0.2813, sMAPE=20.0653, bias=0.0073.  
   **Naive 24-hour baseline (test, pre-specified, not used for selection):** MAE=6.0459, RMSE=9.0304, R2=0.2189, sMAPE=11.6757, bias=-0.0031.
2. **Does ML beat naive lag-24 on validation MAE?** Yes. Winner MAE 5.4868 vs naive 8.3036.
3. **Best on VALIDATION:** Ridge
4. **Validation MAE:** 5.4868
5. **Validation RMSE:** 7.1232
6. **Validation R2:** 0.7437
7. **Validation sMAPE:** 13.0909
8. **Validation bias (mean y_pred − y_true):** -1.7177 (underprediction)
9. **Hour/month variation:** worst hours [3, 1, 2, 4, 5] (MAE [6.573, 6.456, 6.403, 6.349, 6.112]); best hours [22, 23, 21, 20, 19]; worst months [3, 5, 1]; worst weekdays [6, 5, 3]. Diagnostic only; model not changed.
10. **Selected baseline:** Ridge (lowest validation MAE).
11. **Untouched TEST (Ridge):** MAE=6.3654, RMSE=7.5929, R2=0.4478, sMAPE=11.6426, bias=-4.7049.

## Error analysis (selected model, validation)

Worst hours:

| hour | MAE |
|---|---|
| 3.0000 | 6.5732 |
| 1.0000 | 6.4564 |
| 2.0000 | 6.4034 |
| 4.0000 | 6.3487 |
| 5.0000 | 6.1117 |

MAE by month:

| month | MAE |
|---|---|
| 1.0000 | 5.9882 |
| 2.0000 | 4.9128 |
| 3.0000 | 6.8910 |
| 4.0000 | 5.6706 |
| 5.0000 | 6.1331 |
| 10.0000 | 2.6827 |
| 11.0000 | 4.5581 |
| 12.0000 | 5.3989 |

MAE by weekday (Monday=0):

| weekday | MAE |
|---|---|
| 0.0000 | 5.2485 |
| 1.0000 | 5.5343 |
| 2.0000 | 4.7612 |
| 3.0000 | 5.7163 |
| 4.0000 | 5.0775 |
| 5.0000 | 5.9757 |
| 6.0000 | 6.0917 |

## Reproducibility

`random_state = 42` on HistGradientBoosting. Ridge and naive are deterministic.

| file | md5 |
|---|---|
| baseline_validation_predictions.parquet | 2166687b71697ac08516391215eb5f26 |
| baseline_test_predictions.parquet | 6a88cd3330fea7e5cd3d665f74b2b0b4 |
| baseline_model_comparison.csv | b2babe5f3b034917b6dcc170f29ddcd8 |
