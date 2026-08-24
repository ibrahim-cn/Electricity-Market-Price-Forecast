# Final Data Quality

Dataset: `data/processed/merged/merged_energy_weather.csv` (+ parquet, üretildiyse).

## Checks

| check | sonuç | detay |
|---|---|---|
| row_count | PASS | {"actual": 35064, "expected": 35064} |
| column_count | PASS | {"actual": 103} |
| duplicate_timestamp | PASS | {"count": 0} |
| parseable_timestamp | PASS | {"unparsed": 0} |
| timezone_aware_utc | PASS | {"tz": "UTC"} |
| monotonic_timestamp | PASS | true |
| missing_timestamp | PASS | {"count": 0} |
| target_present | PASS | "price day ahead" |
| target_missing_count | PASS | {"count": 0} |
| target_min_max | PASS | {"min": 2.06, "max": 101.99} |
| weather_match_rate | PASS | {"rate": 1.0} |
| price_actual_present_but_excluded_from_features | PASS | true |
| original_time_kept | PASS | true |
| infinite_values | PASS | {"count": 0} |
| duplicate_rows | PASS | {"count": 0} |

## Shape

- Satır: **35064**
- Kolon: **103**

## Timestamp

- Kolon: `timestamp_utc` (timezone-aware, UTC)
- Orijinal energy kolonu korundu: `time`
- min: `2014-12-31T23:00:00+00:00`
- max: `2018-12-31T22:00:00+00:00`
- duplicate: 0
- missing hour: 0
- monotonic: evet

## Target

- Kolon: `price day ahead`
- missing: 0
- min: 2.06
- max: 101.99
- `price actual` dataset içinde **var**, feature listesinde **yok**

## Feature exclusion (henüz model yok)

Feature olarak kullanılmayacak kolonlar:

- `price actual`
- `time`
- `timestamp_utc`
- `price day ahead`

Aday feature sayısı (referans; feature engineering yapılmadı): 99

## Missing values by column (final)

| kolon | NaN | oran |
|---|---|---|
| total load actual | 14 | 0.0399% |
| generation hydro water reservoir | 6 | 0.0171% |
| generation wind onshore | 6 | 0.0171% |
| generation wind offshore | 6 | 0.0171% |
| generation waste | 6 | 0.0171% |
| generation solar | 6 | 0.0171% |
| generation other renewable | 6 | 0.0171% |
| generation other | 6 | 0.0171% |
| generation nuclear | 6 | 0.0171% |
| generation marine | 6 | 0.0171% |
| generation biomass | 6 | 0.0171% |
| generation fossil brown coal/lignite | 6 | 0.0171% |
| generation hydro pumped storage consumption | 6 | 0.0171% |
| generation geothermal | 6 | 0.0171% |
| generation fossil peat | 6 | 0.0171% |
| generation fossil oil shale | 6 | 0.0171% |
| generation fossil oil | 6 | 0.0171% |
| generation fossil hard coal | 6 | 0.0171% |
| generation fossil gas | 6 | 0.0171% |
| generation fossil coal-derived gas | 6 | 0.0171% |
| generation hydro run-of-river and poundage | 6 | 0.0171% |

## Infinite values

- {'count': 0}

## Duplicate rows

- {'count': 0}

## Weather match rate

- {'rate': 1.0}
