# XGBoost gblinear tuned search (development only)

**XGBOOST_GBLINEAR_TUNED = PASS**

TEST was not loaded. Locked model is unchanged:
Ridge(`alpha=0.001`) + METHOD_B.

## What was tuned

XGBoost `booster='gblinear'` (linear / Ridge-like booster), not tree XGBoost.

Fixed Ridge-like choices:
- `updater=coord_descent`
- `reg_alpha=0` (L2 only)
- same fold-train median impute + scale as NumPy Ridge
- `base_score` = fold-inner y mean

Searched:
- `reg_lambda` ∈ (1e-06, 1e-05, 0.0001, 0.001, 0.003, 0.01)
- `learning_rate` ∈ (0.1, 0.3, 1.0)
- `feature_selector` ∈ ('cyclic', 'shuffle')
- `n_estimators` via early stopping (cap 300, patience 40)
  on the last 20% of **fold-train** only, then refit on full fold-train.

Outer fold validation is scoring only. 36 configs × 4 folds.

## Best config

| field | value |
|---|---|
| model | `gblin_1e-06_eta1p0_cyclic` |
| reg_lambda | 1e-06 |
| learning_rate | 1.0 |
| feature_selector | cyclic |
| mean rounds | 121.8 |
| mean MAE | 6.025908 |
| MAE std | 1.195205 |
| mean bias | -2.650201 |

## Versus locked / prior numbers (not refit)

| competitor | mean MAE |
|---|---:|
| Best tuned gblinear | 6.025908 |
| Coarse gblinear (λ grid only, 100 rounds) | 6.013412 |
| Ridge α=0.001 (184 SAFE, no METHOD_B) | 5.796612 |
| Locked Ridge+METHOD_B | 5.496022 |

Beats coarse gblinear? **FALSE**  
Beats Ridge α=0.001? **FALSE**  
Beats locked METHOD_B? **FALSE**

## Top 15 by mean MAE

| model | reg_lambda | learning_rate | feature_selector | mean_MAE | std_MAE | mean_RMSE | mean_R2 | mean_sMAPE | mean_bias | mean_rounds | n_folds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gblin_1e-06_eta1p0_cyclic | 0.0000 | 1.0000 | cyclic | 6.0259 | 1.1952 | 7.6705 | 0.6027 | 13.5292 | -2.6502 | 121.7500 | 4 |
| gblin_1e-05_eta1p0_cyclic | 0.0000 | 1.0000 | cyclic | 6.0260 | 1.1952 | 7.6706 | 0.6027 | 13.5293 | -2.6503 | 121.7500 | 4 |
| gblin_1e-04_eta1p0_cyclic | 0.0001 | 1.0000 | cyclic | 6.0262 | 1.1954 | 7.6711 | 0.6027 | 13.5299 | -2.6523 | 122.0000 | 4 |
| gblin_1e-03_eta1p0_cyclic | 0.0010 | 1.0000 | cyclic | 6.0333 | 1.1953 | 7.6806 | 0.6019 | 13.5427 | -2.6644 | 121.2500 | 4 |
| gblin_3e-03_eta1p0_cyclic | 0.0030 | 1.0000 | cyclic | 6.0472 | 1.1953 | 7.6994 | 0.6004 | 13.5671 | -2.6923 | 120.2500 | 4 |
| gblin_1e-06_eta1p0_shuffle | 0.0000 | 1.0000 | shuffle | 6.0713 | 0.9644 | 7.7935 | 0.5895 | 13.6299 | -2.9642 | 10.2500 | 4 |
| gblin_1e-05_eta1p0_shuffle | 0.0000 | 1.0000 | shuffle | 6.0713 | 0.9645 | 7.7935 | 0.5895 | 13.6300 | -2.9642 | 10.2500 | 4 |
| gblin_1e-04_eta1p0_shuffle | 0.0001 | 1.0000 | shuffle | 6.0716 | 0.9649 | 7.7938 | 0.5895 | 13.6304 | -2.9645 | 10.2500 | 4 |
| gblin_1e-03_eta1p0_shuffle | 0.0010 | 1.0000 | shuffle | 6.0747 | 0.9688 | 7.7971 | 0.5893 | 13.6347 | -2.9675 | 10.2500 | 4 |
| gblin_1e-02_eta1p0_cyclic | 0.0100 | 1.0000 | cyclic | 6.0748 | 1.1956 | 7.7462 | 0.5967 | 13.6010 | -2.7930 | 65.5000 | 4 |
| gblin_3e-03_eta1p0_shuffle | 0.0030 | 1.0000 | shuffle | 6.0815 | 0.9774 | 7.8045 | 0.5890 | 13.6444 | -2.9740 | 10.2500 | 4 |
| gblin_1e-02_eta1p0_shuffle | 0.0100 | 1.0000 | shuffle | 6.1053 | 1.0059 | 7.8305 | 0.5876 | 13.6787 | -2.9954 | 10.2500 | 4 |
| gblin_1e-06_eta0p3_shuffle | 0.0000 | 0.3000 | shuffle | 6.1637 | 1.2148 | 7.9639 | 0.5749 | 13.6640 | -2.8318 | 93.0000 | 4 |
| gblin_1e-05_eta0p3_shuffle | 0.0000 | 0.3000 | shuffle | 6.1637 | 1.2148 | 7.9639 | 0.5749 | 13.6640 | -2.8319 | 93.0000 | 4 |
| gblin_1e-04_eta0p3_shuffle | 0.0001 | 0.3000 | shuffle | 6.1640 | 1.2148 | 7.9643 | 0.5748 | 13.6644 | -2.8325 | 93.0000 | 4 |

## Decision

LOCKED_MODEL_UNCHANGED = TRUE  
TEST_USED_FOR_SELECTION = FALSE  

Even if tuned gblinear were better, the frozen holdout is not reopened here.
