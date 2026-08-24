# High-Price Diagnostic

Baseline reproduction (CURRENT_BEST = Ridge α=0.001 + expanding_historical):
**BASELINE_REPRODUCED = TRUE**

Thresholds for P50+/P75+/P90+/P95+ are `quantile(y_fold_train, q)` applied to that
fold's validation block only. They are not computed on the pooled dataset.

## Folds

| fold | n_train | n_val | train_start | train_end | val_start | val_end |
|---|---|---|---|---|---|---|
| 1 | 14902 | 2980 | 2014-12-31T23:00:00+00:00 | 2016-09-12T20:00:00+00:00 | 2016-09-12T21:00:00+00:00 | 2017-01-15T00:00:00+00:00 |
| 2 | 17882 | 2980 | 2014-12-31T23:00:00+00:00 | 2017-01-15T00:00:00+00:00 | 2017-01-15T01:00:00+00:00 | 2017-05-19T04:00:00+00:00 |
| 3 | 20862 | 2981 | 2014-12-31T23:00:00+00:00 | 2017-05-19T04:00:00+00:00 | 2017-05-19T05:00:00+00:00 | 2017-09-20T09:00:00+00:00 |
| 4 | 23843 | 5961 | 2014-12-31T23:00:00+00:00 | 2017-09-20T09:00:00+00:00 | 2017-09-20T10:00:00+00:00 | 2018-05-26T18:00:00+00:00 |

## Causal high-price metrics (CURRENT_BEST)

| fold | regime | threshold | n | MAE | RMSE | bias | y_mean | y_pred_mean | residual_mean |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ALL | nan | 2980 | 6.4704 | 8.3041 | -3.1201 | 49.0733 | 45.9532 | -3.1201 |
| 1 | P50+ | 45.0000 | 1866 | 6.6716 | 8.6210 | -5.4059 | 57.9296 | 52.5237 | -5.4059 |
| 1 | P75+ | 55.1875 | 1104 | 7.9212 | 9.8700 | -7.5566 | 64.0504 | 56.4938 | -7.5566 |
| 1 | P90+ | 62.9300 | 601 | 9.0226 | 10.7982 | -8.9184 | 68.0283 | 59.1099 | -8.9184 |
| 1 | P95+ | 66.0095 | 391 | 10.4049 | 12.0617 | -10.3468 | 69.9697 | 59.6230 | -10.3468 |
| 2 | ALL | nan | 2980 | 5.7150 | 7.3395 | -3.0120 | 51.9526 | 48.9406 | -3.0120 |
| 2 | P50+ | 45.5500 | 2010 | 6.3690 | 8.0710 | -4.2931 | 58.7078 | 54.4146 | -4.2931 |
| 2 | P75+ | 56.4975 | 789 | 9.1860 | 10.6698 | -8.8184 | 71.2865 | 62.4681 | -8.8184 |
| 2 | P90+ | 63.7690 | 483 | 10.5968 | 11.9029 | -10.3362 | 78.6886 | 68.3524 | -10.3362 |
| 2 | P95+ | 67.1000 | 420 | 11.0309 | 12.2248 | -10.7921 | 80.6965 | 69.9044 | -10.7921 |
| 3 | ALL | nan | 2981 | 5.1580 | 6.5002 | 1.6270 | 50.4882 | 52.1152 | 1.6270 |
| 3 | P50+ | 46.3250 | 2158 | 5.2808 | 6.7618 | 1.2828 | 53.8698 | 55.1526 | 1.2828 |
| 3 | P75+ | 56.6000 | 409 | 6.9259 | 9.0907 | -5.8190 | 64.5775 | 58.7585 | -5.8190 |
| 3 | P90+ | 64.2000 | 167 | 10.8790 | 12.5882 | -10.5869 | 72.0171 | 61.4302 | -10.5869 |
| 3 | P95+ | 68.4080 | 115 | 12.7277 | 13.8872 | -12.6739 | 74.7263 | 62.0523 | -12.6739 |
| 4 | ALL | nan | 5961 | 5.4665 | 7.0971 | -2.0121 | 52.8151 | 50.8031 | -2.0121 |
| 4 | P50+ | 47.2500 | 4237 | 5.3234 | 6.8154 | -3.0123 | 59.3104 | 56.2980 | -3.0123 |
| 4 | P75+ | 55.6900 | 2583 | 5.7602 | 7.3241 | -4.1283 | 64.3030 | 60.1747 | -4.1283 |
| 4 | P90+ | 63.9800 | 1141 | 6.2243 | 7.8550 | -4.8116 | 69.7551 | 64.9436 | -4.8116 |
| 4 | P95+ | 68.1890 | 616 | 6.7772 | 8.4671 | -5.2465 | 73.0540 | 67.8076 | -5.2465 |

Pooled-qcut P75+ bias from the previous residual-correction report is a different
definition (pooled walk-forward y_true quartiles). This table uses fold-train
quantiles so validation regime labels do not depend on later validation prices.

## Association checks (diagnostic only)

These numbers are **associated with** high-price validation hours. They do **not**
establish that any driver causes the residual.

| fold | train_y_mean | train_y_std | val_y_mean | val_y_std | val_hp_y_mean | val_hp_pred_mean | train_load_fc | val_load_fc | val_hp_load_fc | train_ren_fc | val_ren_fc | val_hp_ren_fc | train_load_err_lag24 | val_hp_load_err_lag24 | train_lag24 | val_hp_lag24 | val_hp_hour | val_rest_hour | val_hp_n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0000 | 44.3420 | 14.5717 | 49.0733 | 14.6807 | 64.0504 | 56.4938 | 28473.4631 | 28332.2658 | 31068.0100 | 7040.1592 | 6122.3195 | 4817.2790 | -24.4663 | -23.4991 | 44.3231 | 60.4636 | 13.9312 | 10.0586 | 1104.0000 |
| 2.0000 | 45.1304 | 14.6960 | 51.9526 | 14.8874 | 71.2865 | 62.4681 | 28449.9328 | 28689.3752 | 32625.6426 | 6887.2030 | 7136.8339 | 6526.2510 | -24.6835 | -29.0304 | 45.1047 | 65.7584 | 13.2471 | 10.8581 | 789.0000 |
| 3.0000 | 46.1049 | 14.9158 | 50.4882 | 7.9766 | 64.5775 | 58.7585 | 28484.1356 | 29291.2365 | 31728.4474 | 6922.8611 | 6641.1932 | 6543.2787 | -19.4578 | -14.3521 | 46.1097 | 56.4509 | 14.0073 | 11.0964 | 409.0000 |
| 4.0000 | 46.6530 | 14.3081 | 52.8151 | 13.5598 | 64.3030 | 60.1747 | 28585.0444 | 28963.9844 | 31329.1111 | 6887.6453 | 7369.5259 | 6153.7638 | -20.9197 | -12.5180 | 46.6466 | 60.8241 | 13.3384 | 10.1063 | 2583.0000 |

Reading:

- Validation blocks have a higher target mean than the corresponding fold-train
  history. That is consistent with a **historical / recent price-level shift**.
- High-price validation hours (train-P75+) also show higher load-forecast and
  often lower renewable-forecast levels than the rest of the same validation
  block, which is consistent with a **tight-supply / high-demand regime**.
- `load_forecast_error_lag_24` differences, if present, suggest yesterday's
  load surprise is only weakly aligned with today's residual.
- Hour and month columns, when they differ, are consistent with **time-of-day
  and seasonal concentration** of expensive hours, not proof of a missing
  calendar feature (calendar is already in X).
- Because Ridge is unbiased on fold train (intercept), the P75+ gap is
  consistent with a **train-to-validation regime shift** that a global
  residual add-on cannot see.

No feature was dropped from this diagnostic.

## Files

`high_price_diagnostic.csv` hash: `ca7c0cac563f443f8b4a9c484785fe29`
