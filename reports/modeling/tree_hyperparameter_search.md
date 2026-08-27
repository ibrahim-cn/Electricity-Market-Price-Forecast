# Tree hyperparameter search (development only)

**TREE_HYPERPARAMETER_SEARCH = PASS**

TEST was not loaded. Locked model is unchanged:
Ridge(`alpha=0.001`) + METHOD_B.

## Protocol

- Same 4 expanding TRAIN+VALIDATION folds as Ridge tuning.
- Early stopping on the last 20% of **fold-train** (chronological).
- Outer fold validation is used only for scoring, not for stopping.
- Trees see native NaNs (same as the original family comparison).
- Grid: XGBoost depth×lr×subsample (12) and LightGBM leaves×lr×subsample (12).
- `n_estimators` cap 500, stopping rounds 40, then refit on full fold-train.

## Best configs

| competitor | mean MAE | MAE std | mean bias | mean trees |
|---|---:|---:|---:|---:|
| Best in this search (`LGBM_nl63_lr0.1_ss0.8`) | 5.859459 | 1.103850 | -3.196502 | 87.0 |
| Best XGBoost (`XGB_d6_lr0.1_ss0.8`) | 6.129207 | 1.246279 | -3.022390 | 43.8 |
| Best LightGBM (`LGBM_nl63_lr0.1_ss0.8`) | 5.859459 | 1.103850 | -3.196502 | 87.0 |
| Original tree XGBoost (untuned) | 5.945953 | — | — | 200 |
| Original LightGBM (untuned) | 5.916945 | — | — | 200 |
| Ridge α=0.001 (184 SAFE, no METHOD_B) | 5.796612 | — | — | — |
| Locked Ridge+METHOD_B | 5.496022 | — | — | — |

Beats Ridge α=0.001? **FALSE**  
Beats locked METHOD_B? **FALSE**

## Top 12 by mean MAE

| model | family | mean_MAE | std_MAE | mean_RMSE | mean_R2 | mean_sMAPE | mean_bias | mean_trees | n_folds |
|---|---|---|---|---|---|---|---|---|---|
| LGBM_nl63_lr0.1_ss0.8 | LightGBM | 5.8595 | 1.1039 | 7.8543 | 0.6084 | 12.8304 | -3.1965 | 87.0000 | 4 |
| LGBM_nl63_lr0.1_ss1.0 | LightGBM | 6.0245 | 1.2525 | 8.0497 | 0.5860 | 13.0539 | -3.1807 | 52.7500 | 4 |
| LGBM_nl31_lr0.1_ss0.8 | LightGBM | 6.0456 | 1.2265 | 8.0406 | 0.5889 | 13.1723 | -3.1182 | 66.0000 | 4 |
| LGBM_nl31_lr0.1_ss1.0 | LightGBM | 6.1077 | 1.2796 | 8.1101 | 0.5822 | 13.3185 | -3.1670 | 58.7500 | 4 |
| XGB_d6_lr0.1_ss0.8 | XGBoost | 6.1292 | 1.2463 | 8.1346 | 0.5790 | 13.2799 | -3.0224 | 43.7500 | 4 |
| XGB_d6_lr0.1_ss1.0 | XGBoost | 6.1355 | 1.2497 | 8.1766 | 0.5697 | 13.3343 | -3.1400 | 60.0000 | 4 |
| XGB_d8_lr0.1_ss1.0 | XGBoost | 6.1446 | 1.3596 | 8.1946 | 0.5805 | 13.3170 | -3.3061 | 81.2500 | 4 |
| LGBM_nl15_lr0.1_ss1.0 | LightGBM | 6.1530 | 1.3177 | 8.1065 | 0.5814 | 13.3851 | -3.2157 | 56.7500 | 4 |
| LGBM_nl63_lr0.05_ss0.8 | LightGBM | 6.1944 | 1.4456 | 8.1885 | 0.5743 | 13.3486 | -3.3532 | 99.5000 | 4 |
| LGBM_nl31_lr0.05_ss1.0 | LightGBM | 6.2053 | 1.4225 | 8.1477 | 0.5797 | 13.4470 | -3.3056 | 107.0000 | 4 |
| LGBM_nl63_lr0.05_ss1.0 | LightGBM | 6.2113 | 1.4328 | 8.2108 | 0.5713 | 13.4413 | -3.3286 | 143.0000 | 4 |
| LGBM_nl31_lr0.05_ss0.8 | LightGBM | 6.2280 | 1.4912 | 8.2209 | 0.5717 | 13.4590 | -3.3114 | 147.0000 | 4 |

## Decision

LOCKED_MODEL_UNCHANGED = TRUE  
TEST_USED_FOR_SELECTION = FALSE  
