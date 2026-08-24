# Weather Duplicate Analysis

Kaynak: `data/raw/weather_features.csv` kopyası (`city_name` strip edilmiş).
Ham kök CSV overwrite edilmedi. `drop_duplicates()` ile sessiz çözüm uygulanmadı.

## Özet

| Ölçüt | Değer |
|---|---:|
| Toplam satır | 178396 |
| Toplam (timestamp_utc, city_name) group | 175320 |
| Toplam duplicate group (size > 1) | 2798 |
| Birebir duplicate group | 21 |
| Numeric aynı, weather category farklı group | 2777 |
| Numeric değerleri farklı group | 0 |
| Diğer duplicate group | 0 |
| Duplicate group oranı (group / tüm group) | 1.5959% |
| Duplicate group içindeki satır sayısı | 5874 |
| Duplicate group içindeki satır oranı | 3.2927% |
| Extra satır (len - unique groups) | 3076 |
| Extra satır oranı | 1.7243% |

## Multiplicity

| group size | group count |
|---|---|
| 2 | 2538 |
| 3 | 242 |
| 4 | 18 |

## Şehir bazında duplicate sayıları

| city_name | satır | unique timestamp | extra satır | duplicate group | extra satır oranı |
|---|---|---|---|---|---|
| Barcelona | 35476 | 35064 | 412 | 393 | 1.1613% |
| Bilbao | 35951 | 35064 | 887 | 843 | 2.4672% |
| Madrid | 36267 | 35064 | 1203 | 1011 | 3.3171% |
| Seville | 35557 | 35064 | 493 | 470 | 1.3865% |
| Valencia | 35145 | 35064 | 81 | 81 | 0.2305% |

## Yorum

- Duplicate'ler sessizce silinmedi.
- Çoğu tekrar, aynı sayısal ölçüm + farklı `weather_id` / `weather_main` / `weather_description` kombinasyonudur.
- Sonraki adımda deterministik aggregation uygulanır: numeric için mean/median/max/circular mean, kategorik için mode (eşitlikte lexical ilk değer).
