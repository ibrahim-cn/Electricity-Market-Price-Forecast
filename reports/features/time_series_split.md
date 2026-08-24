# Time-Series Split

Chronological 70 / 15 / 15 split. No shuffle, no random split, no new features, no model.

Join key: `timestamp_utc` (UTC, hourly). Target `price day ahead` left-joined from the merged parquet. `price actual` was not joined.

## Split sizes

| split | rows | percentage | start_utc | end_utc | duration |
|---|---|---|---|---|---|
| train | 24544 | 69.9977% | 2014-12-31T23:00:00+00:00 | 2017-10-19T14:00:00+00:00 | 1022 days 15:00:00 |
| validation | 5260 | 15.0011% | 2017-10-19T15:00:00+00:00 | 2018-05-26T18:00:00+00:00 | 219 days 03:00:00 |
| test | 5260 | 15.0011% | 2018-05-26T19:00:00+00:00 | 2018-12-31T22:00:00+00:00 | 219 days 03:00:00 |

## Exact boundaries

| Boundary | Timestamp (UTC) |
|---|---|
| max(train) | 2017-10-19T14:00:00+00:00 |
| min(validation) | 2017-10-19T15:00:00+00:00 |
| max(validation) | 2018-05-26T18:00:00+00:00 |
| min(test) | 2018-05-26T19:00:00+00:00 |

Checks: max(train) < min(validation); max(validation) < min(test). Overlaps = 0. Multi-assigned rows = 0.

Cut indices (deterministic): train = `[:int(n*0.70)]`, validation = `[int(n*0.70):int(n*0.85)]`, test = `[int(n*0.85):]` with n = 35064.

## Target-lag alignment

| feature | lag | compared | max_abs_diff | pass |
|---|---|---|---|---|
| price_day_ahead_lag_24 | 24 | 35040 | 0.0 | True |
| price_day_ahead_lag_48 | 48 | 35016 | 0.0 | True |
| price_day_ahead_lag_168 | 168 | 34896 | 0.0 | True |

Leading lag NaNs were kept. No imputation.

## Output hashes

| file | md5 |
|---|---|
| data/processed/splits/train.parquet | 278666dcdb30990b55a6aa5c882f21ee |
| data/processed/splits/validation.parquet | cba753fa9327955d506139d25fdaae4d |
| data/processed/splits/test.parquet | 069afbe9c766426d2e095282ece93a69 |
