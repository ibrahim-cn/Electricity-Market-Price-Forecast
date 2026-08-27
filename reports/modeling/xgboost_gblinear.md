# XGBoost gblinear (Ridge-like) comparison

**XGBOOST_GBLINEAR = PASS**

This is a **development-only** experiment. TEST was not loaded.
Locked model remains Ridge(`alpha=0.001`) + METHOD_B.

## What was fit

XGBoost `booster='gblinear'` with L2 (`reg_lambda`) on the same four
expanding TRAIN+VALIDATION folds and the same fold-train impute+scale
as NumPy Ridge. `reg_alpha=0` (no L1). `updater=coord_descent`.

This is the linear / Ridge-like XGBoost booster, not tree XGBoost.

## This-run gblinear grid

| model | reg_lambda | mean_MAE | std_MAE | mean_RMSE | mean_R2 | mean_sMAPE | mean_bias | n_folds |
|---|---|---|---|---|---|---|---|---|
| XGB_gblinear_l2_0.001 | 0.0010 | 6.0134 | 1.1571 | 7.6605 | 0.6045 | 13.4956 | -2.6694 | 4 |
| XGB_gblinear_l2_0.01 | 0.0100 | 6.0513 | 1.1373 | 7.7237 | 0.5990 | 13.5426 | -2.7367 | 4 |
| XGB_gblinear_l2_0.1 | 0.1000 | 6.3372 | 1.2646 | 8.1171 | 0.5672 | 13.9899 | -3.1255 | 4 |
| XGB_gblinear_l2_1.0 | 1.0000 | 6.8405 | 1.6758 | 9.0145 | 0.4814 | 14.6824 | -3.4598 | 4 |
| XGB_gblinear_l2_10.0 | 10.0000 | 8.1917 | 2.3236 | 10.9917 | 0.2533 | 17.1199 | -4.2312 | 4 |

Best gblinear: **XGB_gblinear_l2_0.001** mean MAE = 6.013412

## Versus locked family table (not refit)

| model | mean MAE | source |
|---|---:|---|
| Ridge α=0.001 | 5.796612 | ridge_alpha_comparison.csv |
| Tree XGBoost | 5.945953 | walk_forward_model_comparison.csv |
| Best XGB gblinear | 6.013412 | this run |

Beats locked Ridge α=0.001 on walk-forward MAE? **FALSE**

## Decision

LOCKED_MODEL_UNCHANGED = TRUE  
TEST_USED_FOR_SELECTION = FALSE  

Even if gblinear were better, the frozen holdout is not reopened here.
