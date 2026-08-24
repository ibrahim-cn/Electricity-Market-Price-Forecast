# Model Readiness Audit

**MODEL_READY = PASS**

No model was trained. No imputer or scaler was fit. Protected source files were not written.

## Checklist

| status | check |
|---|---|
| [x] | Raw CSVs unchanged |
| [x] | Source merged parquet unchanged |
| [x] | Feature parquet unchanged |
| [x] | Timestamp UTC |
| [x] | Hourly frequency |
| [x] | No duplicate timestamps |
| [x] | No timestamp overlap between splits |
| [x] | Chronological split |
| [x] | No shuffle |
| [x] | Target excluded from X |
| [x] | price actual excluded |
| [x] | No forbidden features |
| [x] | Feature manifest all SAFE |
| [x] | No future features |
| [x] | Target has no NaN |
| [x] | NaNs not globally imputed |
| [x] | Train/validation/test chronological |
| [x] | Test occurs strictly after validation |
| [x] | Reproducible |

## Protected source hashes (unchanged)

| file | md5 |
|---|---|
| merged_energy_weather.parquet | ae4a12026b1a9682d6bbb58ef7471fa1 |
| model_features.parquet | c9f07ac0f95e0f51fff5472129c1f9ad |

## Target by split (`price day ahead`)

| split | count | NaN | min | max | mean | median | std |
|---|---|---|---|---|---|---|---|
| train | 24544 | 0 | 2.3000 | 101.9900 | 46.8470 | 47.5900 | 14.2259 |
| validation | 5260 | 0 | 2.0600 | 88.4400 | 52.7308 | 53.9900 | 14.0696 |
| test | 5260 | 0 | 3.0000 | 81.8200 | 61.1438 | 63.0000 | 10.2178 |

Train/validation/test means are reported only. The target was not transformed.

## NaN audit (features only; not imputed)

| split | NaN cells | NaN % of X cells | rows with any NaN |
|---|---|---|---|
| train | 15512 | 0.3435% | 198 |
| validation | 0 | 0.0000% | 0 |
| test | 0 | 0.0000% | 0 |

Highest NaN feature counts (expected lag heads, mainly train):

| split | feature | NaN count |
|---|---|---|
| train | total_load_actual_lag_168 | 182 |
| train | generation_wind_onshore_lag_168 | 174 |
| train | generation_nuclear_lag_168 | 174 |
| train | generation_fossil_oil_lag_168 | 174 |
| train | generation_hydro_run_of_river_and_poundage_lag_168 | 174 |
| train | generation_fossil_hard_coal_lag_168 | 174 |
| train | renewable_share_lag_168 | 174 |
| train | generation_fossil_gas_lag_168 | 174 |

## Potential distribution shifts

Rule: \|split_mean − train_mean\| / train_std ≥ 0.5. Report only; no feature dropped.

| feature | split | train_mean | split_mean | |Δmean|/train_std |
|---|---|---|---|---|
| generation_other_renewable_lag_168 | test | 80.5817 | 98.2367 | 1.343 |
| generation_other_renewable_lag_24 | test | 80.6786 | 98.0241 | 1.316 |
| price_mean_lag24_lag48_lag168 | test | 46.8552 | 61.1592 | 1.221 |
| generation_other_renewable_lag_168 | validation | 80.5817 | 96.1987 | 1.188 |
| generation_other_renewable_lag_24 | validation | 80.6786 | 96.3200 | 1.187 |
| price_min_lag24_lag48_lag168 | test | 39.0299 | 54.5901 | 1.108 |
| price_mean_lag24_lag48 | test | 46.8263 | 61.1545 | 1.098 |
| price_max_lag24_lag48_lag168 | test | 54.3843 | 66.9473 | 1.066 |
| temp_seville_lag_24 | validation | 294.5139 | 286.2758 | 1.046 |
| temp_seville_lag_168 | validation | 294.5007 | 286.4166 | 1.025 |
| price_day_ahead_lag_168 | test | 46.7715 | 61.1687 | 1.012 |
| price_day_ahead_lag_48 | test | 46.8209 | 61.1682 | 1.009 |
| price_day_ahead_lag_24 | test | 46.8315 | 61.1408 | 1.006 |
| month | test | 6.1834 | 8.9112 | 0.823 |
| day_of_year | test | 172.7475 | 255.9458 | 0.821 |
| temp_barcelona_lag_24 | validation | 290.2605 | 284.9815 | 0.815 |
| generation_waste_lag_24 | test | 257.4192 | 299.4817 | 0.814 |
| temp_barcelona_lag_168 | validation | 290.2438 | 285.0034 | 0.808 |
| generation_waste_lag_168 | validation | 257.0537 | 298.3536 | 0.800 |
| quarter | test | 2.4016 | 3.2597 | 0.794 |
| week_of_year | test | 25.2720 | 36.7548 | 0.792 |
| temp_national_mean_lag_24 | validation | 290.2602 | 284.5763 | 0.788 |
| temp_national_mean_lag_168 | validation | 290.2405 | 284.6331 | 0.777 |
| generation_waste_lag_168 | test | 257.0537 | 296.9888 | 0.774 |
| generation_biomass_lag_24 | validation | 402.7705 | 332.7527 | 0.765 |
| generation_biomass_lag_168 | validation | 403.0771 | 333.6776 | 0.757 |
| month_sin | test | 0.0200 | -0.5095 | 0.736 |
| generation_waste_lag_24 | validation | 257.4192 | 295.2956 | 0.733 |
| day_of_year_sin | test | 0.0391 | -0.4807 | 0.730 |
| temp_madrid_lag_24 | validation | 288.8934 | 282.2487 | 0.703 |
| temp_madrid_lag_168 | validation | 288.8757 | 282.2710 | 0.698 |
| day_of_year_cos | validation | -0.0540 | 0.4060 | 0.658 |
| generation_biomass_lag_24 | test | 402.7705 | 345.0587 | 0.631 |
| generation_hydro_run_of_river_and_poundage_lag_24 | validation | 903.4215 | 1136.6443 | 0.630 |
| generation_biomass_lag_168 | test | 403.0771 | 345.4577 | 0.629 |
| temp_bilbao_lag_24 | validation | 286.7390 | 282.4236 | 0.626 |
| generation_hydro_run_of_river_and_poundage_lag_168 | test | 906.4036 | 1137.2656 | 0.626 |
| generation_other_lag_168 | validation | 63.4159 | 49.9209 | 0.621 |
| generation_other_lag_24 | validation | 63.3689 | 49.9918 | 0.617 |
| temp_bilbao_lag_168 | validation | 286.7086 | 282.5075 | 0.609 |

## Split reminder

Train 24544 → validation 5260 → test 5260. Test starts after validation ends.
