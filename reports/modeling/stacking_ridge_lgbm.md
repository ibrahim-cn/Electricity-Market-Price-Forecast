# Stacking Ridge+METHOD_B + LightGBM-184 (development only)

**STACKING_RIDGE_LGBM = PASS**

TEST was not loaded. Locked model is unchanged:
Ridge(`alpha=0.001`) + METHOD_B.

## Protocol

Same four expanding TRAIN+VALIDATION folds.

Base models, fit on fold-train only:
- Ridge+METHOD_B: 184 SAFE + 3 causal high-price fractions + expanding addend
- LightGBM: original 200 / lr=0.05 / 31 leaves, **184 SAFE only** (METHOD_B off)

Meta-learner:
- Inner expanding OOF inside fold-train (cuts at 50% / 65% / 80%)
- Outer validation never used for OOF, stopping, or meta weights
- Small Ridge (`α=0.001`) on the two OOF prediction columns
  (median-impute + scale from OOF only)
- Also scored: simple 50/50 average (no learned meta)

## This-run comparison

| model | mean_MAE | std_MAE | mean_RMSE | mean_R2 | mean_sMAPE | mean_bias | n_folds |
|---|---|---|---|---|---|---|---|
| Average | 5.4061 | 0.7151 | 7.0989 | 0.6637 | 11.8998 | -2.0340 | 4 |
| Ridge_METHOD_B | 5.4960 | 0.5198 | 7.1509 | 0.6387 | 12.1184 | -0.9148 | 4 |
| LGBM_184 | 5.9169 | 1.0899 | 7.8527 | 0.6061 | 13.0035 | -3.1531 | 4 |
| Stack_RidgeMeta | 7.1559 | 2.3636 | 8.8404 | 0.4680 | 15.8517 | -2.9563 | 4 |

Stacked mean MAE = 7.155872  
Ridge+METHOD_B this run = 5.496022  
Average this run = 5.406119

Mean unscaled OLS mix (for reading, not the scaled Ridge weights):
Ridge+B weight ≈ 0.681, LightGBM weight ≈ 0.355

## Versus locked / prior

| competitor | mean MAE |
|---|---:|
| Stack Ridge-meta (this run) | 7.155872 |
| 50/50 average (this run) | 5.406119 |
| Ridge+METHOD_B (this run) | 5.496022 |
| Locked Ridge+METHOD_B | 5.496022 |
| LightGBM 184 (prior) | 5.916945 |

Beats this-run Ridge+METHOD_B? **FALSE**  
Beats locked METHOD_B? **FALSE**

## Meta coefficients by fold

| fold | n_oof | ols_intercept | ols_weight_ridge_b | ols_weight_lgbm | scaled_coef_ridge_b | scaled_coef_lgbm | oof_mae_ridge | oof_mae_lgbm |
|---|---|---|---|---|---|---|---|---|
| 1.0000 | 7451.0000 | -6.0409 | 0.0306 | 0.9711 | 0.7528 | 11.4501 | 9.5129 | 7.9189 |
| 2.0000 | 8941.0000 | -2.2129 | 0.8864 | 0.1222 | 11.0625 | 1.4537 | 6.9715 | 8.1414 |
| 3.0000 | 10431.0000 | -2.8850 | 0.9414 | 0.1485 | 12.1375 | 1.6803 | 6.3250 | 7.1547 |
| 4.0000 | 11922.0000 | -0.7584 | 0.8662 | 0.1783 | 10.3247 | 1.8951 | 5.6420 | 6.1900 |

## Decision

LOCKED_MODEL_UNCHANGED = TRUE  
TEST_USED_FOR_SELECTION = FALSE  
