# LightGBM + METHOD_B (development only)

**LGBM_METHOD_B = PASS**

TEST was not loaded. Locked model is unchanged:
Ridge(`alpha=0.001`) + METHOD_B.

## What was fit

Same four expanding TRAIN+VALIDATION folds. Fold-train P75 defines the
three causal high-price fractions (`y` shifted 24h, windows 168/336/720).
Expanding-historical addend is fit on fold-train residuals only.

| model | recipe |
|---|---|
| Ridge_METHOD_B | 184 SAFE + 3 fractions, Ridge α=0.001, expanding addend (reproduction) |
| LGBM_184 | original LightGBM (200, lr=0.05, leaves=31), 184 SAFE, no METHOD_B |
| LGBM_METHOD_B | same LightGBM, 184 + 3 fractions, no addend |
| LGBM_METHOD_B_addend | same LightGBM, 184 + 3 fractions, expanding addend |
| LGBM_tuned_METHOD_B_addend | tuned LightGBM (leaves=63, lr=0.1, subsample=0.8), fractions + addend; early stopping on last 20% of fold-train |

## This-run comparison

| model | mean_MAE | std_MAE | mean_RMSE | mean_R2 | mean_sMAPE | mean_bias | mean_p75_bias | mean_p90_bias | mean_trees | n_folds |
|---|---|---|---|---|---|---|---|---|---|---|
| Ridge_METHOD_B | 5.4960 | 0.5198 | 7.1509 | 0.6387 | 12.1184 | -0.9148 | -5.8956 | -8.2111 | nan | 4 |
| LGBM_184 | 5.9169 | 1.0899 | 7.8527 | 0.6061 | 13.0035 | -3.1531 | -8.3635 | -11.6165 | 200.0000 | 4 |
| LGBM_METHOD_B_addend | 6.1456 | 1.1492 | 7.9274 | 0.5994 | 13.5584 | -1.9181 | -7.1004 | -10.3585 | 200.0000 | 4 |
| LGBM_METHOD_B | 6.1508 | 1.1901 | 7.9416 | 0.5988 | 13.5823 | -1.9784 | -7.1607 | -10.4188 | 200.0000 | 4 |
| LGBM_tuned_METHOD_B_addend | 6.2221 | 1.1301 | 8.1864 | 0.5694 | 13.6004 | -2.0199 | -8.4662 | -12.1378 | 66.7500 | 4 |

Best LightGBM in this run: **LGBM_184** mean MAE = 5.916945

Ridge METHOD_B this run: 5.496022 (locked report 5.496022)

## Versus prior numbers (not refit)

| competitor | mean MAE |
|---|---:|
| Best LightGBM + METHOD_B (this run) | 5.916945 |
| Ridge + METHOD_B (this run) | 5.496022 |
| Locked Ridge+METHOD_B | 5.496022 |
| LightGBM 184 (original family table) | 5.916945 |
| LightGBM tuned, 184 only | 5.859459 |
| Ridge α=0.001, 184 only | 5.796612 |

Beats this-run Ridge METHOD_B? **FALSE**  
Beats locked METHOD_B? **FALSE**

## Decision

LOCKED_MODEL_UNCHANGED = TRUE  
TEST_USED_FOR_SELECTION = FALSE  
