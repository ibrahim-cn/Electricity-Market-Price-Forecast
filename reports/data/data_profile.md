# Data Profile Report

Kaynak dosyalar (ham CSV, değiştirilmedi):

- `energy_dataset.csv`
- `weather_features.csv`

Bu rapor yalnızca mevcut kolonlar ve hesaplanan istatistiklere dayanır. Model eğitimi, feature engineering ve veri değişikliği yapılmamıştır.

---

## 1. Tespit edilen CSV dosyaları

Proje kökünde tam olarak **2** CSV vardır.

| Dosya | Boyut | Satır (data) | Kolon |
|---|---:|---:|---:|
| `energy_dataset.csv` | 6,273,009 byte | 35,064 | 29 |
| `weather_features.csv` | 19,918,887 byte | 178,396 | 17 |

Satır sayıları header hariç pandas `read_csv` ile doğrulanmıştır.

---

## 2. `energy_dataset.csv`

### 2.1 Kimlik

- Dosya adı: `energy_dataset.csv`
- Satır: **35064**
- Kolon: **29**
- Tam satır duplicate: **0**

### 2.2 Kolon isimleri ve pandas dtype

| # | Kolon | dtype |
|---|---|---|
| 1 | `time` | object |
| 2 | `generation biomass` | float64 |
| 3 | `generation fossil brown coal/lignite` | float64 |
| 4 | `generation fossil coal-derived gas` | float64 |
| 5 | `generation fossil gas` | float64 |
| 6 | `generation fossil hard coal` | float64 |
| 7 | `generation fossil oil` | float64 |
| 8 | `generation fossil oil shale` | float64 |
| 9 | `generation fossil peat` | float64 |
| 10 | `generation geothermal` | float64 |
| 11 | `generation hydro pumped storage aggregated` | float64 |
| 12 | `generation hydro pumped storage consumption` | float64 |
| 13 | `generation hydro run-of-river and poundage` | float64 |
| 14 | `generation hydro water reservoir` | float64 |
| 15 | `generation marine` | float64 |
| 16 | `generation nuclear` | float64 |
| 17 | `generation other` | float64 |
| 18 | `generation other renewable` | float64 |
| 19 | `generation solar` | float64 |
| 20 | `generation waste` | float64 |
| 21 | `generation wind offshore` | float64 |
| 22 | `generation wind onshore` | float64 |
| 23 | `forecast solar day ahead` | float64 |
| 24 | `forecast wind offshore eday ahead` | float64 |
| 25 | `forecast wind onshore day ahead` | float64 |
| 26 | `total load forecast` | float64 |
| 27 | `total load actual` | float64 |
| 28 | `price day ahead` | float64 |
| 29 | `price actual` | float64 |

`time` ham dosyada metin olarak duruyor; parse edilince timezone-aware datetime oluyor (aşağıda).

### 2.3 İlk 5 satır

`time` değerleri: `2015-01-01 00:00:00+01:00` … `2015-01-01 04:00:00+01:00`

| time | generation biomass | generation fossil brown coal/lignite | generation fossil coal-derived gas | generation fossil gas | generation fossil hard coal | generation fossil oil | generation fossil oil shale | generation fossil peat | generation geothermal | generation hydro pumped storage aggregated | generation hydro pumped storage consumption | generation hydro run-of-river and poundage | generation hydro water reservoir | generation marine | generation nuclear | generation other | generation other renewable | generation solar | generation waste | generation wind offshore | generation wind onshore | forecast solar day ahead | forecast wind offshore eday ahead | forecast wind onshore day ahead | total load forecast | total load actual | price day ahead | price actual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2015-01-01 00:00:00+01:00 | 447.0 | 329.0 | 0.0 | 4844.0 | 4821.0 | 162.0 | 0.0 | 0.0 | 0.0 | NaN | 863.0 | 1051.0 | 1899.0 | 0.0 | 7096.0 | 43.0 | 73.0 | 49.0 | 196.0 | 0.0 | 6378.0 | 17.0 | NaN | 6436.0 | 26118.0 | 25385.0 | 50.10 | 65.41 |
| 2015-01-01 01:00:00+01:00 | 449.0 | 328.0 | 0.0 | 5196.0 | 4755.0 | 158.0 | 0.0 | 0.0 | 0.0 | NaN | 920.0 | 1009.0 | 1658.0 | 0.0 | 7096.0 | 43.0 | 71.0 | 50.0 | 195.0 | 0.0 | 5890.0 | 16.0 | NaN | 5856.0 | 24934.0 | 24382.0 | 48.10 | 64.92 |
| 2015-01-01 02:00:00+01:00 | 448.0 | 323.0 | 0.0 | 4857.0 | 4581.0 | 157.0 | 0.0 | 0.0 | 0.0 | NaN | 1164.0 | 973.0 | 1371.0 | 0.0 | 7099.0 | 43.0 | 73.0 | 50.0 | 196.0 | 0.0 | 5461.0 | 8.0 | NaN | 5454.0 | 23515.0 | 22734.0 | 47.33 | 64.48 |
| 2015-01-01 03:00:00+01:00 | 438.0 | 254.0 | 0.0 | 4314.0 | 4131.0 | 160.0 | 0.0 | 0.0 | 0.0 | NaN | 1503.0 | 949.0 | 779.0 | 0.0 | 7098.0 | 43.0 | 75.0 | 50.0 | 191.0 | 0.0 | 5238.0 | 2.0 | NaN | 5151.0 | 22642.0 | 21286.0 | 42.27 | 59.32 |
| 2015-01-01 04:00:00+01:00 | 428.0 | 187.0 | 0.0 | 4130.0 | 3840.0 | 156.0 | 0.0 | 0.0 | 0.0 | NaN | 1826.0 | 953.0 | 720.0 | 0.0 | 7097.0 | 43.0 | 74.0 | 42.0 | 189.0 | 0.0 | 4935.0 | 9.0 | NaN | 4861.0 | 21785.0 | 20264.0 | 38.41 | 56.04 |

### 2.4 Son 5 satır

| time | generation biomass | generation fossil brown coal/lignite | generation fossil coal-derived gas | generation fossil gas | generation fossil hard coal | generation fossil oil | generation fossil oil shale | generation fossil peat | generation geothermal | generation hydro pumped storage aggregated | generation hydro pumped storage consumption | generation hydro run-of-river and poundage | generation hydro water reservoir | generation marine | generation nuclear | generation other | generation other renewable | generation solar | generation waste | generation wind offshore | generation wind onshore | forecast solar day ahead | forecast wind offshore eday ahead | forecast wind onshore day ahead | total load forecast | total load actual | price day ahead | price actual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2018-12-31 19:00:00+01:00 | 297.0 | 0.0 | 0.0 | 7634.0 | 2628.0 | 178.0 | 0.0 | 0.0 | 0.0 | NaN | 1.0 | 1135.0 | 4836.0 | 0.0 | 6073.0 | 63.0 | 95.0 | 85.0 | 277.0 | 0.0 | 3113.0 | 96.0 | NaN | 3253.0 | 30619.0 | 30653.0 | 68.85 | 77.02 |
| 2018-12-31 20:00:00+01:00 | 296.0 | 0.0 | 0.0 | 7241.0 | 2566.0 | 174.0 | 0.0 | 0.0 | 0.0 | NaN | 1.0 | 1172.0 | 3931.0 | 0.0 | 6074.0 | 62.0 | 95.0 | 33.0 | 280.0 | 0.0 | 3288.0 | 51.0 | NaN | 3353.0 | 29932.0 | 29735.0 | 68.40 | 76.16 |
| 2018-12-31 21:00:00+01:00 | 292.0 | 0.0 | 0.0 | 7025.0 | 2422.0 | 168.0 | 0.0 | 0.0 | 0.0 | NaN | 50.0 | 1148.0 | 2831.0 | 0.0 | 6076.0 | 61.0 | 94.0 | 31.0 | 286.0 | 0.0 | 3503.0 | 36.0 | NaN | 3404.0 | 27903.0 | 28071.0 | 66.88 | 74.30 |
| 2018-12-31 22:00:00+01:00 | 293.0 | 0.0 | 0.0 | 6562.0 | 2293.0 | 163.0 | 0.0 | 0.0 | 0.0 | NaN | 108.0 | 1128.0 | 2068.0 | 0.0 | 6075.0 | 61.0 | 93.0 | 31.0 | 287.0 | 0.0 | 3586.0 | 29.0 | NaN | 3273.0 | 25450.0 | 25801.0 | 63.93 | 69.89 |
| 2018-12-31 23:00:00+01:00 | 290.0 | 0.0 | 0.0 | 6926.0 | 2166.0 | 163.0 | 0.0 | 0.0 | 0.0 | NaN | 108.0 | 1069.0 | 1686.0 | 0.0 | 6075.0 | 61.0 | 92.0 | 31.0 | 287.0 | 0.0 | 3651.0 | 26.0 | NaN | 3117.0 | 24424.0 | 24455.0 | 64.27 | 69.88 |

### 2.5 Eksik değerler

| Kolon | Eksik adet | Oran |
|---|---:|---:|
| `time` | 0 | 0.0000% |
| `generation biomass` | 19 | 0.0542% |
| `generation fossil brown coal/lignite` | 18 | 0.0513% |
| `generation fossil coal-derived gas` | 18 | 0.0513% |
| `generation fossil gas` | 18 | 0.0513% |
| `generation fossil hard coal` | 18 | 0.0513% |
| `generation fossil oil` | 19 | 0.0542% |
| `generation fossil oil shale` | 18 | 0.0513% |
| `generation fossil peat` | 18 | 0.0513% |
| `generation geothermal` | 18 | 0.0513% |
| `generation hydro pumped storage aggregated` | 35064 | **100.0000%** |
| `generation hydro pumped storage consumption` | 19 | 0.0542% |
| `generation hydro run-of-river and poundage` | 19 | 0.0542% |
| `generation hydro water reservoir` | 18 | 0.0513% |
| `generation marine` | 19 | 0.0542% |
| `generation nuclear` | 17 | 0.0485% |
| `generation other` | 18 | 0.0513% |
| `generation other renewable` | 18 | 0.0513% |
| `generation solar` | 18 | 0.0513% |
| `generation waste` | 19 | 0.0542% |
| `generation wind offshore` | 18 | 0.0513% |
| `generation wind onshore` | 18 | 0.0513% |
| `forecast solar day ahead` | 0 | 0.0000% |
| `forecast wind offshore eday ahead` | 35064 | **100.0000%** |
| `forecast wind onshore day ahead` | 0 | 0.0000% |
| `total load forecast` | 0 | 0.0000% |
| `total load actual` | 36 | 0.1027% |
| `price day ahead` | 0 | 0.0000% |
| `price actual` | 0 | 0.0000% |

Kısmi eksik (yüzde 100 boş olmayan kolonlar) **47 satırda** görülür. Bunlardan **17 satırda** tüm dolu-olabilecek `generation *` kolonları aynı anda NaN’dır. `total load actual` 36 saatte NaN’dır. İki fiyat kolonu ve day-ahead forecast kolonlarının (offshore hariç) hiç eksiği yoktur.

Tüm generation kolonlarının birlikte boş olduğu saatler:

- 2015-01-05 03:00:00+01:00
- 2015-01-05 12:00:00+01:00 … 2015-01-05 17:00:00+01:00 (6 saat ardışık)
- 2015-01-19 19:00:00+01:00, 2015-01-19 20:00:00+01:00
- 2015-01-27 19:00:00+01:00
- 2015-01-28 13:00:00+01:00
- 2015-04-16 09:00:00+02:00
- 2015-04-23 21:00:00+02:00
- 2015-06-15 09:00:00+02:00
- 2015-10-02 11:00:00+02:00
- 2015-12-02 09:00:00+01:00
- 2018-07-11 09:00:00+02:00

### 2.6 Unique değer sayıları

`nunique` NaN hariç; parantez içi NaN dahil.

| Kolon | unique (dropna) | unique (NaN dahil) |
|---|---:|---:|
| `time` | 35064 | 35064 |
| `generation biomass` | 423 | 424 |
| `generation fossil brown coal/lignite` | 956 | 957 |
| `generation fossil coal-derived gas` | 1 | 2 |
| `generation fossil gas` | 8297 | 8298 |
| `generation fossil hard coal` | 7266 | 7267 |
| `generation fossil oil` | 321 | 322 |
| `generation fossil oil shale` | 1 | 2 |
| `generation fossil peat` | 1 | 2 |
| `generation geothermal` | 1 | 2 |
| `generation hydro pumped storage aggregated` | 0 | 1 |
| `generation hydro pumped storage consumption` | 3311 | 3312 |
| `generation hydro run-of-river and poundage` | 1684 | 1685 |
| `generation hydro water reservoir` | 7029 | 7030 |
| `generation marine` | 1 | 2 |
| `generation nuclear` | 2388 | 2389 |
| `generation other` | 103 | 104 |
| `generation other renewable` | 78 | 79 |
| `generation solar` | 5331 | 5332 |
| `generation waste` | 262 | 263 |
| `generation wind offshore` | 1 | 2 |
| `generation wind onshore` | 11465 | 11466 |
| `forecast solar day ahead` | 5356 | 5356 |
| `forecast wind offshore eday ahead` | 0 | 1 |
| `forecast wind onshore day ahead` | 11332 | 11332 |
| `total load forecast` | 14790 | 14790 |
| `total load actual` | 15127 | 15128 |
| `price day ahead` | 5747 | 5747 |
| `price actual` | 6653 | 6653 |

Tek non-null değeri `0.0` olan kolonlar (geri kalanı NaN):

- `generation fossil coal-derived gas` (35046 adet 0.0, 18 NaN)
- `generation fossil oil shale` (35046 adet 0.0, 18 NaN)
- `generation fossil peat` (35046 adet 0.0, 18 NaN)
- `generation geothermal` (35046 adet 0.0, 18 NaN)
- `generation marine` (35045 adet 0.0, 19 NaN)
- `generation wind offshore` (35046 adet 0.0, 18 NaN)

Tamamen boş kolonlar:

- `generation hydro pumped storage aggregated` (35064 NaN)
- `forecast wind offshore eday ahead` (35064 NaN; kolon adında `eday` yazıyor)

### 2.7 Sayısal kolon istatistikleri

NaN’lar istatistiğe dahil edilmedi. Tamamen boş kolonlarda min/max/mean/median/std tanımsızdır.

| Kolon | min | max | mean | median | std |
|---|---:|---:|---:|---:|---:|
| `generation biomass` | 0.00 | 592.00 | 383.513540 | 367.00 | 85.353943 |
| `generation fossil brown coal/lignite` | 0.00 | 999.00 | 448.059208 | 509.00 | 354.568590 |
| `generation fossil coal-derived gas` | 0.00 | 0.00 | 0.000000 | 0.00 | 0.000000 |
| `generation fossil gas` | 0.00 | 20034.00 | 5622.737488 | 4969.00 | 2201.830478 |
| `generation fossil hard coal` | 0.00 | 8359.00 | 4256.065742 | 4474.00 | 1961.601013 |
| `generation fossil oil` | 0.00 | 449.00 | 298.319789 | 300.00 | 52.520673 |
| `generation fossil oil shale` | 0.00 | 0.00 | 0.000000 | 0.00 | 0.000000 |
| `generation fossil peat` | 0.00 | 0.00 | 0.000000 | 0.00 | 0.000000 |
| `generation geothermal` | 0.00 | 0.00 | 0.000000 | 0.00 | 0.000000 |
| `generation hydro pumped storage aggregated` | — | — | — | — | — |
| `generation hydro pumped storage consumption` | 0.00 | 4523.00 | 475.577343 | 68.00 | 792.406614 |
| `generation hydro run-of-river and poundage` | 0.00 | 2000.00 | 972.116108 | 906.00 | 400.777536 |
| `generation hydro water reservoir` | 0.00 | 9728.00 | 2605.114735 | 2164.00 | 1835.199745 |
| `generation marine` | 0.00 | 0.00 | 0.000000 | 0.00 | 0.000000 |
| `generation nuclear` | 0.00 | 7117.00 | 6263.907039 | 6566.00 | 839.667958 |
| `generation other` | 0.00 | 106.00 | 60.228585 | 57.00 | 20.238381 |
| `generation other renewable` | 0.00 | 119.00 | 85.639702 | 88.00 | 14.077554 |
| `generation solar` | 0.00 | 5792.00 | 1432.665925 | 616.00 | 1680.119887 |
| `generation waste` | 0.00 | 357.00 | 269.452133 | 279.00 | 50.195536 |
| `generation wind offshore` | 0.00 | 0.00 | 0.000000 | 0.00 | 0.000000 |
| `generation wind onshore` | 0.00 | 17436.00 | 5464.479769 | 4849.00 | 3213.691587 |
| `forecast solar day ahead` | 0.00 | 5836.00 | 1439.066735 | 576.00 | 1677.703355 |
| `forecast wind offshore eday ahead` | — | — | — | — | — |
| `forecast wind onshore day ahead` | 237.00 | 17430.00 | 5471.216689 | 4855.00 | 3176.312853 |
| `total load forecast` | 18105.00 | 41390.00 | 28712.129962 | 28906.00 | 4594.100854 |
| `total load actual` | 18041.00 | 41015.00 | 28696.939905 | 28901.00 | 4574.987950 |
| `price day ahead` | 2.06 | 101.99 | 49.874341 | 50.52 | 14.618900 |
| `price actual` | 9.33 | 116.80 | 57.884023 | 58.02 | 14.204083 |

---

## 3. `weather_features.csv`

### 3.1 Kimlik

- Dosya adı: `weather_features.csv`
- Satır: **178396**
- Kolon: **17**
- Tam satır duplicate: **21** (aynı satırın birebir kopyası; `keep='first'` ile 21 fazla satır)

### 3.2 Kolon isimleri ve pandas dtype

| # | Kolon | dtype |
|---|---|---|
| 1 | `dt_iso` | object |
| 2 | `city_name` | object |
| 3 | `temp` | float64 |
| 4 | `temp_min` | float64 |
| 5 | `temp_max` | float64 |
| 6 | `pressure` | int64 |
| 7 | `humidity` | int64 |
| 8 | `wind_speed` | int64 |
| 9 | `wind_deg` | int64 |
| 10 | `rain_1h` | float64 |
| 11 | `rain_3h` | float64 |
| 12 | `snow_3h` | float64 |
| 13 | `clouds_all` | int64 |
| 14 | `weather_id` | int64 |
| 15 | `weather_main` | object |
| 16 | `weather_description` | object |
| 17 | `weather_icon` | object |

### 3.3 İlk 5 satır

Dosya şehir blokları halinde tutuluyor. İlk blok Valencia’dır.

| dt_iso | city_name | temp | temp_min | temp_max | pressure | humidity | wind_speed | wind_deg | rain_1h | rain_3h | snow_3h | clouds_all | weather_id | weather_main | weather_description | weather_icon |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 2015-01-01 00:00:00+01:00 | Valencia | 270.475 | 270.475 | 270.475 | 1001 | 77 | 1 | 62 | 0.0 | 0.0 | 0.0 | 0 | 800 | clear | sky is clear | 01n |
| 2015-01-01 01:00:00+01:00 | Valencia | 270.475 | 270.475 | 270.475 | 1001 | 77 | 1 | 62 | 0.0 | 0.0 | 0.0 | 0 | 800 | clear | sky is clear | 01n |
| 2015-01-01 02:00:00+01:00 | Valencia | 269.686 | 269.686 | 269.686 | 1002 | 78 | 0 | 23 | 0.0 | 0.0 | 0.0 | 0 | 800 | clear | sky is clear | 01n |
| 2015-01-01 03:00:00+01:00 | Valencia | 269.686 | 269.686 | 269.686 | 1002 | 78 | 0 | 23 | 0.0 | 0.0 | 0.0 | 0 | 800 | clear | sky is clear | 01n |
| 2015-01-01 04:00:00+01:00 | Valencia | 269.686 | 269.686 | 269.686 | 1002 | 78 | 0 | 23 | 0.0 | 0.0 | 0.0 | 0 | 800 | clear | sky is clear | 01n |

### 3.4 Son 5 satır

Son blok Seville’dir.

| dt_iso | city_name | temp | temp_min | temp_max | pressure | humidity | wind_speed | wind_deg | rain_1h | rain_3h | snow_3h | clouds_all | weather_id | weather_main | weather_description | weather_icon |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 2018-12-31 19:00:00+01:00 | Seville | 287.76 | 287.15 | 288.15 | 1028 | 54 | 3 | 30 | 0.0 | 0.0 | 0.0 | 0 | 800 | clear | sky is clear | 01n |
| 2018-12-31 20:00:00+01:00 | Seville | 285.76 | 285.15 | 286.15 | 1029 | 62 | 3 | 30 | 0.0 | 0.0 | 0.0 | 0 | 800 | clear | sky is clear | 01n |
| 2018-12-31 21:00:00+01:00 | Seville | 285.15 | 285.15 | 285.15 | 1028 | 58 | 4 | 50 | 0.0 | 0.0 | 0.0 | 0 | 800 | clear | sky is clear | 01n |
| 2018-12-31 22:00:00+01:00 | Seville | 284.15 | 284.15 | 284.15 | 1029 | 57 | 4 | 60 | 0.0 | 0.0 | 0.0 | 0 | 800 | clear | sky is clear | 01n |
| 2018-12-31 23:00:00+01:00 | Seville | 283.97 | 282.15 | 285.15 | 1029 | 70 | 3 | 50 | 0.0 | 0.0 | 0.0 | 0 | 800 | clear | sky is clear | 01n |

### 3.5 Eksik değerler

Tüm 17 kolonda eksik değer **0** (oran %0.0000).

### 3.6 Unique değer sayıları

| Kolon | unique |
|---|---:|
| `dt_iso` | 35064 |
| `city_name` | 5 |
| `temp` | 20743 |
| `temp_min` | 18553 |
| `temp_max` | 18591 |
| `pressure` | 190 |
| `humidity` | 100 |
| `wind_speed` | 36 |
| `wind_deg` | 361 |
| `rain_1h` | 7 |
| `rain_3h` | 89 |
| `snow_3h` | 66 |
| `clouds_all` | 97 |
| `weather_id` | 38 |
| `weather_main` | 12 |
| `weather_description` | 43 |
| `weather_icon` | 24 |

`city_name` ham değerleri (repr):

- `'Valencia'` — 35145 satır
- `'Madrid'` — 36267 satır
- `'Bilbao'` — 35951 satır
- `' Barcelona'` — 35476 satır (**başta boşluk var**)
- `'Seville'` — 35557 satır

`weather_main` dağılımı: clear 82685, clouds 68055, rain 17391, mist 3908, fog 2506, drizzle 1724, thunderstorm 1041, haze 435, dust 347, snow 270, smoke 33, squall 1.

`rain_1h` gözlenen değerler: `0.0, 0.25, 0.3, 0.9, 2.29, 3.0, 12.0`.

### 3.7 Sayısal kolon istatistikleri

| Kolon | min | max | mean | median | std |
|---|---:|---:|---:|---:|---:|
| `temp` | 262.24 | 315.600 | 289.618605 | 289.15 | 8.026199 |
| `temp_min` | 262.24 | 315.150 | 288.330442 | 288.15 | 7.955491 |
| `temp_max` | 262.24 | 321.150 | 291.091267 | 290.15 | 8.612454 |
| `pressure` | 0.00 | 1008371.000 | 1069.260740 | 1018.00 | 5969.631893 |
| `humidity` | 0.00 | 100.000 | 68.423457 | 72.00 | 21.902888 |
| `wind_speed` | 0.00 | 133.000 | 2.470560 | 2.00 | 2.095910 |
| `wind_deg` | 0.00 | 360.000 | 166.591190 | 177.00 | 116.611927 |
| `rain_1h` | 0.00 | 12.000 | 0.075492 | 0.00 | 0.398847 |
| `rain_3h` | 0.00 | 2.315 | 0.000380 | 0.00 | 0.007288 |
| `snow_3h` | 0.00 | 21.500 | 0.004763 | 0.00 | 0.222604 |
| `clouds_all` | 0.00 | 100.000 | 25.073292 | 20.00 | 30.774129 |
| `weather_id` | 200.00 | 804.000 | 759.831902 | 800.00 | 108.733223 |

`temp` aralığı 262.24–315.60’tır. Bu aralık Celsius değil, Kelvin ölçeği ile uyumludur (yaklaşık −10.91 °C … 42.45 °C). Dosyada birim yazısı yoktur; birim çıkarımı yalnızca değer aralığından yapılmıştır.

`pressure` median’ı 1018 iken max 1,008,371’dir. 45 satırda pressure > 2000, 2 satırda pressure = 0. Aykırı değerler büyük ölçüde Barcelona, 2015-02-20 … 2015-02-22 aralığındadır.

`wind_speed` max 133 (Valencia, 2017-05-11 12:00:00+02:00). 31 satırda `wind_speed > 20`.

`humidity == 0` olan 63 satır vardır.

### 3.8 Şehir blok sırası (dosya düzeni)

Weather dosyası saatlik tek tablo değil; şehirlere göre ardışık bloklardır:

| city_name (ham) | ilk index | son index | satır | ilk ts | son ts |
|---|---:|---:|---:|---|---|
| Valencia | 0 | 35144 | 35145 | 2015-01-01 00:00:00+01:00 | 2018-12-31 23:00:00+01:00 |
| Madrid | 35145 | 71411 | 36267 | 2015-01-01 00:00:00+01:00 | 2018-12-31 23:00:00+01:00 |
| Bilbao | 71412 | 107362 | 35951 | 2015-01-01 00:00:00+01:00 | 2018-12-31 23:00:00+01:00 |
| ` Barcelona` | 107363 | 142838 | 35476 | 2015-01-01 00:00:00+01:00 | 2018-12-31 23:00:00+01:00 |
| Seville | 142839 | 178395 | 35557 | 2015-01-01 00:00:00+01:00 | 2018-12-31 23:00:00+01:00 |

Her şehirde unique `dt_iso` = **35064**. Fazla satırlar aynı `(dt_iso, city)` için ek kayıtlardır.

---

## 4. Timestamp analizi

### 4.1 Tespit edilen tarih/saat kolonları

Kolon adı tahmini yapılmadı. Mevcut kolonlar incelendi:

| Dosya | Kolon | Gerekçe |
|---|---|---|
| `energy_dataset.csv` | `time` | Ham değerler `YYYY-MM-DD HH:MM:SS±HH:MM` formatında; 35064/35064 parse edildi |
| `weather_features.csv` | `dt_iso` | Aynı format; 178396/178396 parse edildi |

Başka datetime kolonu yoktur.

### 4.2 Timestamp formatı

Her iki kolonda da tüm değerler **25 karakter** ve şu kalıpta:

```
YYYY-MM-DD HH:MM:SS+0X:00
```

Örnekler:

- `2015-01-01 00:00:00+01:00`
- `2015-03-29 03:00:00+02:00`
- `2018-12-31 23:00:00+01:00`

Saniye her zaman `00`. `Z` soneki yok. Parse hatası yok (`utc=True` ile 0 NaT).

### 4.3 Timezone

Timezone bilgisi **vardır**: her satırda açık UTC offset vardır.

| Offset | energy `time` | weather `dt_iso` |
|---|---:|---:|
| `+01:00` | 14400 | 73947 |
| `+02:00` | 20664 | 104449 |

IANA adı (`Europe/Madrid` vb.) dosyada **yazılı değildir**. Offset geçişleri Avrupa yaz/kış saati (CET/CEST) takvimi ile örtüşür:

| Geçiş (energy index) | Ham değer | UTC |
|---|---|---|
| 0 | 2015-01-01 00:00:00+01:00 | 2014-12-31 23:00:00+00:00 |
| 2090 | 2015-03-29 03:00:00+02:00 | 2015-03-29 01:00:00+00:00 |
| 7130 | 2015-10-25 02:00:00+01:00 | 2015-10-25 01:00:00+00:00 |
| 10826 | 2016-03-27 03:00:00+02:00 | 2016-03-27 01:00:00+00:00 |
| 16034 | 2016-10-30 02:00:00+01:00 | 2016-10-30 01:00:00+00:00 |
| 19562 | 2017-03-26 03:00:00+02:00 | 2017-03-26 01:00:00+00:00 |
| 24770 | 2017-10-29 02:00:00+01:00 | 2017-10-29 01:00:00+00:00 |
| 28298 | 2018-03-25 03:00:00+02:00 | 2018-03-25 01:00:00+00:00 |
| 33506 | 2018-10-28 02:00:00+01:00 | 2018-10-28 01:00:00+00:00 |

İlkbahar geçişinde yerel saat `02:00` atlanır (`00, 01, 03, …`). Sonbahar geçişinde aynı yerel `02:00` iki kez vardır: `02:00:00+02:00` sonra `02:00:00+01:00`. Bu, eksik/duplicate timestamp değil; offset-aware saatlik serinin beklenen DST davranışıdır.

### 4.4 Saatlik frekans ve eksik saatler

`time` / `dt_iso` `utc=True` ile parse edilince:

- Energy UTC min: `2014-12-31 23:00:00+00:00`
- Energy UTC max: `2018-12-31 22:00:00+00:00`
- Weather UTC min/max: **aynı**
- Energy UTC ardışık fark: 35063 adet tam olarak `1 hour`
- Beklenen saatlik UTC nokta sayısı: 35064
- Gerçek unique UTC: 35064
- **Eksik UTC saati: 0**
- **Fazla UTC saati: 0**
- 2016-02-29 (artık gün) her iki dosyada 24 saat mevcut
- Toplam 35064 = 24 × (365×3 + 366), 2015–2018 tam takvim

Yerel duvar saati olarak ilkbahar günlerinde 23 satır, sonbahar günlerinde 25 satır vardır; UTC’de boşluk yoktur.

Energy `time` raw string olarak monotonic değildir (sonbahar `+02` / `+01` sözlük sırası). UTC parse edilmiş seri **monotonic ve unique**’tir.

### 4.5 Duplicate timestamp

**Energy**

- Duplicate raw `time`: **0**
- Duplicate UTC timestamp: **0**
- 1 satır = 1 saat

**Weather**

- Unique `dt_iso`: **35064** (energy ile aynı küme)
- Satır: 178396 → ortalama ~5.09 satır / timestamp (5 şehir)
- Aynı `dt_iso` için satır sayısı: min 5, max 10
  - 5 satır: 32514 timestamp
  - 6: 2114
  - 7: 366
  - 8: 55
  - 9: 10
  - 10: 5
- `(dt_iso, city_name)` tekrarları: 2798 grup, 5874 satır (2/3/4 kopya)
- Bunların **21 grubu** birebir aynı satır (tam duplicate)
- **2777 grubu** aynı sayısal hava değişkenlerine sahip, fakat `weather_id` / `weather_main` / `weather_description` farklı (ör. aynı saatte `mist` + `fog`, veya `rain` + `drizzle`)

Şehir bazında fazla satır (unique ts hep 35064):

| Şehir (strip sonrası) | satır | extra |
|---|---:|---:|
| Valencia | 35145 | 81 |
| Barcelona | 35476 | 412 |
| Seville | 35557 | 493 |
| Bilbao | 35951 | 887 |
| Madrid | 36267 | 1203 |

Her şehrin unique timestamp kümesi energy ile **birebir aynıdır**.

### 4.6 Başlangıç ve bitiş

Ham (offset’li yerel) aralık, her iki dosyada:

- Başlangıç: `2015-01-01 00:00:00+01:00`
- Bitiş: `2018-12-31 23:00:00+01:00`

UTC karşılığı:

- Başlangıç: `2014-12-31 23:00:00+00:00`
- Bitiş: `2018-12-31 22:00:00+00:00`

---

## 5. İki CSV zaman aralığı karşılaştırması

Ham string kümeleri ve UTC kümeleri ayrı ayrı karşılaştırıldı.

| Ölçüt | Sonuç |
|---|---|
| Energy unique timestamp | 35064 |
| Weather unique timestamp | 35064 |
| Ortak ham `time` ∩ `dt_iso` | **35064** |
| Sadece energy’de | **0** |
| Sadece weather’da | **0** |
| Ortak UTC timestamp | **35064** |
| Energy-only UTC | **0** |
| Weather-only UTC | **0** |
| Ortak takvim aralığı | 2015-01-01 00:00:00+01:00 → 2018-12-31 23:00:00+01:00 |
| Sadece CSV1 (`energy_dataset.csv`) aralığı | yok |
| Sadece CSV2 (`weather_features.csv`) aralığı | yok |

Zaman ekseni örtüşmesi tamdır. Birleştirme engeli aralık farkı değil, weather’ın uzun (şehir × saat, artı çoklu condition) formatıdır.

---

## 6. Kolonların anlamsal sınıflandırması

Sınıflandırma **yalnızca kolon adları ve gözlenen değerlere** göredir. Hiçbir kolon çıkarılmamıştır.

### 6.1 Target olabilecek fiyat kolonları

- `price actual`
- `price day ahead`

İkisi de eksiksiz, sayısal, €/MWh ölçeği ile uyumlu aralıktadır. Aynı seri değildir (aşağıda).

### 6.2 Tüketim / demand

- `total load actual`
- `total load forecast` (gerçekleşen değil, day-ahead yük tahmini)

### 6.3 Üretim (genel)

Adı `generation` ile başlayan tüm kolonlar:

- `generation biomass`
- `generation fossil brown coal/lignite`
- `generation fossil coal-derived gas`
- `generation fossil gas`
- `generation fossil hard coal`
- `generation fossil oil`
- `generation fossil oil shale`
- `generation fossil peat`
- `generation geothermal`
- `generation hydro pumped storage aggregated` (tamamen boş)
- `generation hydro pumped storage consumption`
- `generation hydro run-of-river and poundage`
- `generation hydro water reservoir`
- `generation marine`
- `generation nuclear`
- `generation other`
- `generation other renewable`
- `generation solar`
- `generation waste`
- `generation wind offshore`
- `generation wind onshore`

### 6.4 Yenilenebilir üretim

Kolon adında açıkça yenilenebilir kaynak geçenler:

- `generation biomass`
- `generation geothermal` (tümü 0 veya NaN)
- `generation hydro pumped storage aggregated` (tümü NaN)
- `generation hydro pumped storage consumption`
- `generation hydro run-of-river and poundage`
- `generation hydro water reservoir`
- `generation marine` (tümü 0 veya NaN)
- `generation other renewable`
- `generation solar`
- `generation wind offshore` (tümü 0 veya NaN)
- `generation wind onshore`

`generation waste` adında “renewable” geçmez; atık yakıtı ayrı bırakıldı (diğer üretim).

### 6.5 Hava durumu

`weather_features.csv` içindeki `dt_iso` ve `city_name` hariç tüm kolonlar:

- `temp`, `temp_min`, `temp_max`
- `pressure`, `humidity`
- `wind_speed`, `wind_deg`
- `rain_1h`, `rain_3h`, `snow_3h`
- `clouds_all`
- `weather_id`, `weather_main`, `weather_description`, `weather_icon`

### 6.6 Diğer

- `time`, `dt_iso` — timestamp anahtarları
- `city_name` — şehir boyutu (weather)
- `forecast solar day ahead`
- `forecast wind offshore eday ahead` (tamamen boş)
- `forecast wind onshore day ahead`

Bu forecast kolonları gerçekleşen üretim değil, day-ahead üretim tahminidir.

---

## 7. Fiyat kolonu adayları (varsayılmadı)

İki aday vardır. Proje metnindeki “saatlik piyasa takas fiyatı (€/MWh)” ifadesi tek bir kolona kilitlenemez; dosyada iki ayrı fiyat serisi durur.

| Aday | min | max | mean | median | std | eksik | unique |
|---|---:|---:|---:|---:|---:|---:|---:|
| `price actual` | 9.33 | 116.80 | 57.884 | 58.02 | 14.204 | 0 | 6653 |
| `price day ahead` | 2.06 | 101.99 | 49.874 | 50.52 | 14.619 | 0 | 5747 |

Gözlenen farklar:

- `price actual - price day ahead`: mean +8.01, min −61.04, max +77.58
- İki kolonun eşit olduğu satır: **2**
- Pearson korelasyon: **0.732**

Neden `price actual` adaydır:

- Adı gerçekleşen fiyatı işaret eder.
- Hiç eksiği yoktur.
- `price day ahead` ile sistematik olarak farklıdır; yani day-ahead kopyası değildir.

Neden `price day ahead` adaydır:

- Adı gün öncesi piyasa fiyatını işaret eder.
- İspanya’da “piyasa takas / casación” çoğu zaman gün öncesi saatlik fiyattır.
- Hiç eksiği yoktur.

Bu aşamada target seçilmedi. İleride hedef, tahmin ufkuna göre bu iki kolondan biri olarak netleştirilmelidir.

---

## 8. Potansiyel leakage riskleri (kolon çıkarılmadı)

Risk, **hedef kolona ve tahmin anına** bağlıdır. Şimdilik yalnızca işaretlendi.

Yüksek / dikkat riski:

- Aynı saatin `generation *` gerçekleşenleri — saat kapandıktan sonra bilinen gerçekleşmeler. Saat başı öncesi tahmin için leakage adayı.
- `total load actual` — aynı gerekçe.
- Hedef `price day ahead` ise `price actual` — gerçekleşen fiyat, gün öncesi takastan sonra/sonra bilinir.
- Hedef `price actual` ise aynı saatin `price day ahead` değeri — gün öncesi yayınlanmışsa leakage olmayabilir; yayın zamanı bu dosyada yok, bu yüzden risk olarak işaretlendi.
- Aynı saatin weather gözlemleri (`temp`, `pressure`, `wind_*`, `rain_*`, `snow_*`, `clouds_all`, kategorik weather) — gerçekleşen hava, saat öncesi bilinmeyebilir.
- `rain_3h`, `snow_3h` — 3 saatlik agregimler, pencere yönü dosyada yazmıyor.

Daha düşük / bağlama bağlı risk:

- `forecast solar day ahead`, `forecast wind onshore day ahead`, `total load forecast` — adları day-ahead tahmindir. Gün öncesi yayınlandıysa `price actual` tahmini için meşru olabilir. `price day ahead` ile aynı turda üretilmişlerse o hedef için riskli olabilir.
- `forecast wind offshore eday ahead` — tamamen boş; leakage değil, kullanılamaz kolon.

Bu kolonlar raporda kalır; drop edilmedi.

---

## 9. Birleştirilebilirlik

**Evet — timestamp üzerinden birleştirilebilir**, çünkü `energy.time` ve `weather.dt_iso` kümeleri özdeş (35064 ortak, 0 fark).

Düz `pd.merge` (energy ⨯ weather on timestamp) **doğrudan uygun değildir**:

- Energy: 1 satır / saat → 35064
- Weather: 5 şehir / saat + aynı şehirde çoklu weather condition → 178396
- Join sonrası energy satırları 5–10 kez çoğalır

Önerilen yön (henüz uygulanmadı):

1. Weather’da `city_name` başındaki boşluğu (`' Barcelona'`) normalize et (kopya üzerinde).
2. `(dt_iso, city)` için tek satıra indir: tam duplicate 21 grubu drop; 2777 çoklu-condition grubunda kural seç (ör. ilk kayıt, veya kategorikleri ayrı tutup sayıları tekilleştir — sayılar zaten aynı).
3. Weather’ı wide’a çevir: her şehir için `temp_Madrid`, `temp_Valencia`, … (5 × sayısal/kategorik).
4. Energy’yi sol tablo al, `time == dt_iso` (tercihen UTC normalize) ile **1:1 join**.
5. Beklenen sonuç: **35064 satır**.

Join anahtarı: ham string eşitliği yeterlidir (kümeler birebir). Yine de DST sonbahar `02:00+02` / `02:00+01` ayrımı için offset korunmalı veya UTC’ye çevrilmelidir; naive datetime’a (`+01/+02` atarak) parse **yapılmamalı**.

---

## Dataset Overview

- 2 ham CSV: `energy_dataset.csv` (35064 × 29) ve `weather_features.csv` (178396 × 17).
- Ortak saatlik pencere: 2015-01-01 00:00+01:00 … 2018-12-31 23:00+01:00 (UTC’de boşluksuz 35064 saat, 2016 artık gün dahil).
- Energy ulusal saatlik üretim / yük / fiyat; weather 5 İspanyol şehri için saatlik hava.
- Energy’de 2 kolon %100 boş; 6 üretim kolonu yalnızca 0 (veya NaN).
- Weather’da eksik yok; `(saat, şehir)` tekrarları ve pressure/rüzgar aykırı değerleri var.

## Column Analysis

- Energy: 1 object timestamp + 28 float64.
- Weather: 2 object anahtar (`dt_iso`, `city_name`), 4 object kategorik weather, 11 sayısal.
- `city_name` içinde `' Barcelona'` leading space.
- Weather `temp` aralığı Kelvin ile uyumlu; birim etiketi yok.
- `weather_id` / `weather_main` / `weather_description` / `weather_icon` birlikte OpenWeather-benzeri kodlardır (dosyada kaynak yazmıyor).

## Timestamp Analysis

- Kolonlar: energy `time`, weather `dt_iso`.
- Format: `YYYY-MM-DD HH:MM:SS+0X:00`, offset `+01:00` veya `+02:00`.
- Timezone: offset var; IANA adı yok.
- UTC saatlik frekans eksiksiz; duplicate energy timestamp yok.
- Weather unique timestamp = energy; satır fazlası şehir ve çoklu condition kaynaklı.
- DST yerel saat atlama/çiftleme var, UTC boşluğu yok.

## Missing Data Analysis

- Weather: 0 eksik.
- Energy: 2 kolon %100 NaN (`generation hydro pumped storage aggregated`, `forecast wind offshore eday ahead`).
- Energy üretim kolonları ~17–19 NaN (%0.05); çoğu aynı 17 saatte kümelenmiş.
- `total load actual`: 36 NaN (%0.10).
- Fiyat ve (offshore hariç) day-ahead forecast: 0 NaN.

## Duplicate Analysis

- Energy tam satır duplicate: 0. Timestamp duplicate: 0.
- Weather tam satır duplicate: 21.
- Weather `(dt_iso, city)` duplicate grup: 2798 (21 identik, 2777 farklı weather kodu, aynı sayısal ölçümler).
- Weather aynı timestamp’te 5–10 satır.

## Date Range Comparison

- Ortak aralık: 2015-01-01 00:00:00+01:00 — 2018-12-31 23:00:00+01:00.
- Sadece energy: yok.
- Sadece weather: yok.
- Ortak timestamp: 35064 (ham ve UTC).

## Potential Target

- Aday 1: `price actual` (gerçekleşen fiyat; eksiksiz; day-ahead’den farklı).
- Aday 2: `price day ahead` (gün öncesi fiyat; eksiksiz; “takas/casación” ile isimsel örtüşme).
- Bu aşamada seçim yapılmadı.

## Production Features

- 21 `generation *` kolonu.
- Kullanılabilir varyans: gas, hard coal, nuclear, hydro reservoir/run-of-river, wind onshore, solar, biomass, oil, lignite, waste, other, other renewable, hydro pumped storage consumption.
- Bilgi yok / sıfır: coal-derived gas, oil shale, peat, geothermal, marine, wind offshore, hydro pumped storage aggregated.

## Demand Features

- `total load actual` (36 eksik).
- `total load forecast` (0 eksik; gerçekleşenle korelasyon 0.995; 40 satırda eşit).

## Weather Features

- 5 şehir: Valencia, Madrid, Bilbao, Barcelona (leading space), Seville.
- Sayısal: temp/temp_min/temp_max, pressure, humidity, wind_speed, wind_deg, rain_1h, rain_3h, snow_3h, clouds_all.
- Kategorik: weather_id, weather_main, weather_description, weather_icon.
- Dikkat: Barcelona pressure aykırıları, Valencia wind_speed=133, humidity=0 (63 satır).

## Potential Leakage Risks

- Aynı saat gerçekleşen üretim, `total load actual`, weather gözlemleri.
- Hedefe göre diğer fiyat kolonu.
- Day-ahead forecast kolonları (yayın zamanı dosyada yok).
- `rain_3h` / `snow_3h` pencere yönü belirsiz.
- Hiçbir kolon drop edilmedi.

## Recommended Merge Strategy

1. Anahtar: `energy.time` = `weather.dt_iso` (offset’i koru veya UTC’ye çevir).
2. Weather’ı önce `(timestamp, city)` tekilliğine getir, sonra şehri wide pivot et.
3. Energy left join weather-wide → 35064 satır, 1:1.
4. Düz long join kullanma (satır patlaması).
5. Ham CSV’leri overwrite etme; birleşik tablo ayrı üretilmeli (şimdilik üretilmedi).

## Issues Requiring Attention

1. İki fiyat kolonu; target henüz seçilmedi.
2. 2 tamamen boş energy kolonu; 6 sıfır-only üretim kolonu.
3. Energy’de 17 saatte tüm generation NaN; 36 saatte load actual NaN.
4. Weather `(saat, şehir)` çoklu kayıt (çoğu farklı condition kodu).
5. `city_name == ' Barcelona'` leading space.
6. Barcelona pressure (0 … 1,008,371) ve Valencia wind_speed=133.
7. `temp` birimi dosyada yazmıyor; aralık Kelvin ile uyumlu.
8. Kolon adı typo: `forecast wind offshore eday ahead`.
9. Naive datetime parse DST sonbaharında iki `02:00`’ı çökertir.
10. Aynı saat actuals vs day-ahead forecasts leakage/kullanılabilirlik ayrımı netleştirilmeli.

## Recommended Next Step

Ham CSV’lere dokunmadan, kopya üzerinde: (1) target’ı `price actual` vs `price day ahead` olarak netleştir, (2) weather `(dt_iso, city)` çakışmalarını ve `city_name` boşluğunu belgeleyerek tekilleştir, (3) UTC-aware 1:1 merge ile birleşik analitik tabloyu **ayrı dosyaya** üret, (4) ancak ondan sonra eksik/aykırı değer stratejisi ve feature planına geç. Model eğitme.
