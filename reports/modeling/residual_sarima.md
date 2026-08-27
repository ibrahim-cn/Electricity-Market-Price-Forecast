# Residual SARIMA on Ridge+METHOD_B (development only)

**RESIDUAL_SARIMA = PASS**

TEST was not loaded. Locked model is unchanged:
Ridge(`alpha=0.001`) + METHOD_B.

## Why this form

Ridge already uses lag-24/48/168, calendar, and day-ahead load/renewable
forecasts. The time-series model is fit on **Ridge+METHOD_B residuals**,
not on raw price. Seasonal period **24** only (hourly day). Period 168
was not fit: too many seasonal parameters and much slower MLE.

Fit window: last **1440** fold-train residual hours.
Order selection: **lowest mean train AIC**, then fewer mean-parameters.

Parsimonious candidates:

| name | order | seasonal | mean-params |
|---|---|---|---:|
| AR1 | (1,0,0) | none | 1 |
| ARMA11 | (1,0,1) | none | 2 |
| SAR_AR1_s24 | (1,0,0) | (1,0,0,24) | 2 |
| SARIMA1111_s24 | (1,0,1) | (1,0,1,24) | 4 |

AIC winner: **SARIMA1111_s24**

## Protocols

- **DAM_24**: each origin forecasts 24 residual hours, then that day's
  actual residuals are appended (`refit=False`). This is the day-ahead
  comparison.
- **STEP_1**: one-step updates. Uses lag-1 residual; **not** a DAM noon
  origin. Diagnostic only.

## Train AIC

| model | mean_AIC | n_mean_params |
|---|---|---|
| SARIMA1111_s24 | 6512.7773 | 4 |
| SAR_AR1_s24 | 6531.3107 | 2 |
| ARMA11 | 6700.9711 | 2 |
| AR1 | 6708.3979 | 1 |

## Walk-forward MAE

| model | mean_MAE | std_MAE | mean_RMSE | mean_R2 | mean_sMAPE | mean_bias | n_folds |
|---|---|---|---|---|---|---|---|
| AR1_STEP_1_diagnostic | 1.8168 | 0.2036 | 2.7440 | 0.9488 | 4.2397 | -0.1183 | 4 |
| AR1_DAM_24 | 4.5296 | 0.5612 | 6.0706 | 0.7478 | 10.1461 | -0.7127 | 4 |
| ARMA11_DAM_24 | 4.5824 | 0.5502 | 6.1320 | 0.7425 | 10.2320 | -0.7241 | 4 |
| SAR_AR1_s24_DAM_24 | 4.7144 | 0.6274 | 6.2845 | 0.7328 | 10.4045 | -0.8375 | 4 |
| SARIMA1111_s24_DAM_24 | 4.8653 | 0.6003 | 6.4718 | 0.7166 | 10.7088 | -0.8887 | 4 |
| Ridge_METHOD_B | 5.4960 | 0.5198 | 7.1509 | 0.6387 | 12.1184 | -0.9148 | 4 |

AIC-winner DAM_24 MAE = 4.865296  
AIC-winner STEP_1 MAE = 1.816793  
Ridge+METHOD_B MAE = 5.496022 (locked report 5.496022)

DAM_24 beats Ridge+METHOD_B? **TRUE**  
STEP_1 beats Ridge+METHOD_B? **TRUE**

## Decision

LOCKED_MODEL_UNCHANGED = TRUE  
TEST_USED_FOR_SELECTION = FALSE  
