# Weather Outliers

Kaynak: normalize edilmiş weather kopyası (aggregation öncesi, satır düzeyi).
Bu aşamada outlier **silinmedi** ve **düzeltilmedi**.

Toplam incelenen satır: 178396

### `pressure > 2000`

| Ölçüt | Değer |
|---|---|
| Sayı | 45 |
| Oran | 0.0252% |
| Tarih aralığı (UTC) | 2015-02-20T08:00:00+00:00 → 2015-02-22T13:00:00+00:00 |

Şehir dağılımı:

| city_name | adet |
|---|---|
| Barcelona | 45 |

### `pressure <= 0`

| Ölçüt | Değer |
|---|---|
| Sayı | 2 |
| Oran | 0.0011% |
| Tarih aralığı (UTC) | 2015-02-22T19:00:00+00:00 → 2015-02-22T20:00:00+00:00 |

Şehir dağılımı:

| city_name | adet |
|---|---|
| Barcelona | 2 |

### `wind_speed > 20`

| Ölçüt | Değer |
|---|---|
| Sayı | 31 |
| Oran | 0.0174% |
| Tarih aralığı (UTC) | 2015-01-11T01:00:00+00:00 → 2017-05-11T10:00:00+00:00 |

Şehir dağılımı:

| city_name | adet |
|---|---|
| Valencia | 31 |

### `humidity == 0`

| Ölçüt | Değer |
|---|---|
| Sayı | 63 |
| Oran | 0.0353% |
| Tarih aralığı (UTC) | 2015-02-09T13:00:00+00:00 → 2015-03-30T11:00:00+00:00 |

Şehir dağılımı:

| city_name | adet |
|---|---|
| Barcelona | 61 |
| Madrid | 2 |


## Karar

Anomaliler belgelendi. Modelleme aşamasında clip, medyan yerine koyma veya satır/şehir özel kuralı ayrıca seçilecek.
