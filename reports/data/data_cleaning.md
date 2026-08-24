# Data Cleaning

Ham CSV'ler değiştirilmedi. İşlem `data/raw/` kopyaları üzerindedir.

## Kaldırılan kolonlar

| kolon | eksiklik | gerekçe |
|---|---|---|
| generation hydro pumped storage aggregated | 35064/35064 NaN (%100) | Hiç gözlem yok; bilgi taşımıyor. |
| forecast wind offshore eday ahead | 35064/35064 NaN (%100) | Hiç gözlem yok; kolon adında `eday` yazıyor. Kullanılamaz. |

Sıfır-only üretim kolonları **kaldırılmadı**. Feature selection sonraki aşamadadır:

- `generation fossil coal-derived gas`
- `generation fossil oil shale`
- `generation fossil peat`
- `generation geothermal`
- `generation marine`
- `generation wind offshore`

## Missing pattern (imputasyon öncesi)

- Herhangi bir generation / `total load actual` eksiği olan satır: **47**
- Tüm generation kolonlarının birlikte NaN olduğu satır: **17**

| kolon | NaN (önce) |
|---|---|
| generation biomass | 19 |
| generation fossil brown coal/lignite | 18 |
| generation fossil coal-derived gas | 18 |
| generation fossil gas | 18 |
| generation fossil hard coal | 18 |
| generation fossil oil | 19 |
| generation fossil oil shale | 18 |
| generation fossil peat | 18 |
| generation geothermal | 18 |
| generation hydro pumped storage consumption | 19 |
| generation hydro run-of-river and poundage | 19 |
| generation hydro water reservoir | 18 |
| generation marine | 19 |
| generation nuclear | 17 |
| generation other | 18 |
| generation other renewable | 18 |
| generation solar | 18 |
| generation waste | 19 |
| generation wind offshore | 18 |
| generation wind onshore | 18 |
| total load actual | 36 |

Tüm generation'ın boş olduğu UTC saatler:

- 2015-01-05T02:00:00+00:00
- 2015-01-05T11:00:00+00:00
- 2015-01-05T12:00:00+00:00
- 2015-01-05T13:00:00+00:00
- 2015-01-05T14:00:00+00:00
- 2015-01-05T15:00:00+00:00
- 2015-01-05T16:00:00+00:00
- 2015-01-19T18:00:00+00:00
- 2015-01-19T19:00:00+00:00
- 2015-01-27T18:00:00+00:00
- 2015-01-28T12:00:00+00:00
- 2015-04-16T07:00:00+00:00
- 2015-04-23T19:00:00+00:00
- 2015-06-15T07:00:00+00:00
- 2015-10-02T09:00:00+00:00
- 2015-12-02T08:00:00+00:00
- 2018-07-11T07:00:00+00:00

## Imputasyon stratejisi

Körü körüne 0 doldurma **yok**. NaN, fiziksel 0 üretim varsayılmadı.

Uygulanan kural:

1. Index: timezone-aware `timestamp_utc` (UTC).
2. Yalnızca `generation *` ve `total load actual`.
3. İçeride kalan (edge olmayan) NaN koşuları, uzunluk **≤ 3 saat** ise `method='time'` interpolasyon.
4. 4+ saatlik veya seri başı/sonu gap'ler NaN bırakılır.
5. `price day ahead` imputasyon yok (zaten eksik değil; kontrol edildi).
6. `price actual` ne target ne imputasyon kaynağıdır.

Gerekçe: 1–3 saatlik kopukluklar komşu saatlerden zaman ağırlıklı tahmin edilebilir. 6 saatlik blok (ör. 2015-01-05 öğleden sonra) kısa gap değildir; uydurma değer üretmek bias riski taşır.

## Imputasyon sonuçları

| kolon | NaN önce | doldurulan | NaN sonra |
|---|---|---|---|
| generation biomass | 19 | 13 | 6 |
| generation fossil brown coal/lignite | 18 | 12 | 6 |
| generation fossil coal-derived gas | 18 | 12 | 6 |
| generation fossil gas | 18 | 12 | 6 |
| generation fossil hard coal | 18 | 12 | 6 |
| generation fossil oil | 19 | 13 | 6 |
| generation fossil oil shale | 18 | 12 | 6 |
| generation fossil peat | 18 | 12 | 6 |
| generation geothermal | 18 | 12 | 6 |
| generation hydro pumped storage consumption | 19 | 13 | 6 |
| generation hydro run-of-river and poundage | 19 | 13 | 6 |
| generation hydro water reservoir | 18 | 12 | 6 |
| generation marine | 19 | 13 | 6 |
| generation nuclear | 17 | 11 | 6 |
| generation other | 18 | 12 | 6 |
| generation other renewable | 18 | 12 | 6 |
| generation solar | 18 | 12 | 6 |
| generation waste | 19 | 13 | 6 |
| generation wind offshore | 18 | 12 | 6 |
| generation wind onshore | 18 | 12 | 6 |
| total load actual | 36 | 22 | 14 |

### Doldurulan kısa gap örnekleri

| kolon | start_utc | end_utc | uzunluk (saat) |
|---|---|---|---|
| generation biomass | 2015-01-05T02:00:00+00:00 | 2015-01-05T02:00:00+00:00 | 1 |
| generation biomass | 2015-01-19T18:00:00+00:00 | 2015-01-19T19:00:00+00:00 | 2 |
| generation biomass | 2015-01-27T18:00:00+00:00 | 2015-01-27T18:00:00+00:00 | 1 |
| generation fossil brown coal/lignite | 2015-01-05T02:00:00+00:00 | 2015-01-05T02:00:00+00:00 | 1 |
| generation fossil brown coal/lignite | 2015-01-19T18:00:00+00:00 | 2015-01-19T19:00:00+00:00 | 2 |
| generation fossil brown coal/lignite | 2015-01-27T18:00:00+00:00 | 2015-01-27T18:00:00+00:00 | 1 |
| generation fossil coal-derived gas | 2015-01-05T02:00:00+00:00 | 2015-01-05T02:00:00+00:00 | 1 |
| generation fossil coal-derived gas | 2015-01-19T18:00:00+00:00 | 2015-01-19T19:00:00+00:00 | 2 |
| generation fossil coal-derived gas | 2015-01-27T18:00:00+00:00 | 2015-01-27T18:00:00+00:00 | 1 |
| generation fossil gas | 2015-01-05T02:00:00+00:00 | 2015-01-05T02:00:00+00:00 | 1 |
| generation fossil gas | 2015-01-19T18:00:00+00:00 | 2015-01-19T19:00:00+00:00 | 2 |
| generation fossil gas | 2015-01-27T18:00:00+00:00 | 2015-01-27T18:00:00+00:00 | 1 |
| generation fossil hard coal | 2015-01-05T02:00:00+00:00 | 2015-01-05T02:00:00+00:00 | 1 |
| generation fossil hard coal | 2015-01-19T18:00:00+00:00 | 2015-01-19T19:00:00+00:00 | 2 |
| generation fossil hard coal | 2015-01-27T18:00:00+00:00 | 2015-01-27T18:00:00+00:00 | 1 |
| generation fossil oil | 2015-01-05T02:00:00+00:00 | 2015-01-05T02:00:00+00:00 | 1 |
| generation fossil oil | 2015-01-19T18:00:00+00:00 | 2015-01-19T19:00:00+00:00 | 2 |
| generation fossil oil | 2015-01-27T18:00:00+00:00 | 2015-01-27T18:00:00+00:00 | 1 |
| generation fossil oil shale | 2015-01-05T02:00:00+00:00 | 2015-01-05T02:00:00+00:00 | 1 |
| generation fossil oil shale | 2015-01-19T18:00:00+00:00 | 2015-01-19T19:00:00+00:00 | 2 |
| generation fossil oil shale | 2015-01-27T18:00:00+00:00 | 2015-01-27T18:00:00+00:00 | 1 |
| generation fossil peat | 2015-01-05T02:00:00+00:00 | 2015-01-05T02:00:00+00:00 | 1 |
| generation fossil peat | 2015-01-19T18:00:00+00:00 | 2015-01-19T19:00:00+00:00 | 2 |
| generation fossil peat | 2015-01-27T18:00:00+00:00 | 2015-01-27T18:00:00+00:00 | 1 |
| generation geothermal | 2015-01-05T02:00:00+00:00 | 2015-01-05T02:00:00+00:00 | 1 |
| generation geothermal | 2015-01-19T18:00:00+00:00 | 2015-01-19T19:00:00+00:00 | 2 |
| generation geothermal | 2015-01-27T18:00:00+00:00 | 2015-01-27T18:00:00+00:00 | 1 |
| generation hydro pumped storage consumption | 2015-01-05T02:00:00+00:00 | 2015-01-05T02:00:00+00:00 | 1 |
| generation hydro pumped storage consumption | 2015-01-19T18:00:00+00:00 | 2015-01-19T19:00:00+00:00 | 2 |
| generation hydro pumped storage consumption | 2015-01-27T18:00:00+00:00 | 2015-01-27T18:00:00+00:00 | 1 |
| generation hydro run-of-river and poundage | 2015-01-05T02:00:00+00:00 | 2015-01-05T02:00:00+00:00 | 1 |
| generation hydro run-of-river and poundage | 2015-01-19T18:00:00+00:00 | 2015-01-19T19:00:00+00:00 | 2 |
| generation hydro run-of-river and poundage | 2015-01-27T18:00:00+00:00 | 2015-01-27T18:00:00+00:00 | 1 |
| generation hydro water reservoir | 2015-01-05T02:00:00+00:00 | 2015-01-05T02:00:00+00:00 | 1 |
| generation hydro water reservoir | 2015-01-19T18:00:00+00:00 | 2015-01-19T19:00:00+00:00 | 2 |
| generation hydro water reservoir | 2015-01-27T18:00:00+00:00 | 2015-01-27T18:00:00+00:00 | 1 |
| generation marine | 2015-01-05T02:00:00+00:00 | 2015-01-05T02:00:00+00:00 | 1 |
| generation marine | 2015-01-19T18:00:00+00:00 | 2015-01-19T19:00:00+00:00 | 2 |
| generation marine | 2015-01-27T18:00:00+00:00 | 2015-01-27T18:00:00+00:00 | 1 |
| generation nuclear | 2015-01-05T02:00:00+00:00 | 2015-01-05T02:00:00+00:00 | 1 |
| generation nuclear | 2015-01-19T18:00:00+00:00 | 2015-01-19T19:00:00+00:00 | 2 |
| generation nuclear | 2015-01-27T18:00:00+00:00 | 2015-01-27T18:00:00+00:00 | 1 |
| generation other | 2015-01-05T02:00:00+00:00 | 2015-01-05T02:00:00+00:00 | 1 |
| generation other | 2015-01-19T18:00:00+00:00 | 2015-01-19T19:00:00+00:00 | 2 |
| generation other | 2015-01-27T18:00:00+00:00 | 2015-01-27T18:00:00+00:00 | 1 |
| generation other renewable | 2015-01-05T02:00:00+00:00 | 2015-01-05T02:00:00+00:00 | 1 |
| generation other renewable | 2015-01-19T18:00:00+00:00 | 2015-01-19T19:00:00+00:00 | 2 |
| generation other renewable | 2015-01-27T18:00:00+00:00 | 2015-01-27T18:00:00+00:00 | 1 |
| generation solar | 2015-01-05T02:00:00+00:00 | 2015-01-05T02:00:00+00:00 | 1 |
| generation solar | 2015-01-19T18:00:00+00:00 | 2015-01-19T19:00:00+00:00 | 2 |
| generation solar | 2015-01-27T18:00:00+00:00 | 2015-01-27T18:00:00+00:00 | 1 |
| generation waste | 2015-01-05T02:00:00+00:00 | 2015-01-05T02:00:00+00:00 | 1 |
| generation waste | 2015-01-19T18:00:00+00:00 | 2015-01-19T19:00:00+00:00 | 2 |
| generation waste | 2015-01-27T18:00:00+00:00 | 2015-01-27T18:00:00+00:00 | 1 |
| generation wind offshore | 2015-01-05T02:00:00+00:00 | 2015-01-05T02:00:00+00:00 | 1 |
| generation wind offshore | 2015-01-19T18:00:00+00:00 | 2015-01-19T19:00:00+00:00 | 2 |
| generation wind offshore | 2015-01-27T18:00:00+00:00 | 2015-01-27T18:00:00+00:00 | 1 |
| generation wind onshore | 2015-01-05T02:00:00+00:00 | 2015-01-05T02:00:00+00:00 | 1 |
| generation wind onshore | 2015-01-19T18:00:00+00:00 | 2015-01-19T19:00:00+00:00 | 2 |
| generation wind onshore | 2015-01-27T18:00:00+00:00 | 2015-01-27T18:00:00+00:00 | 1 |
| total load actual | 2015-01-28T12:00:00+00:00 | 2015-01-28T12:00:00+00:00 | 1 |
| total load actual | 2015-02-01T06:00:00+00:00 | 2015-02-01T08:00:00+00:00 | 3 |
| total load actual | 2015-04-05T01:00:00+00:00 | 2015-04-05T01:00:00+00:00 | 1 |

### Bilinçli olarak NaN bırakılan gap'ler

| kolon | start_utc | end_utc | uzunluk (saat) | neden |
|---|---|---|---|---|
| generation biomass | 2015-01-05T11:00:00+00:00 | 2015-01-05T16:00:00+00:00 | 6 | long gap |
| generation fossil brown coal/lignite | 2015-01-05T11:00:00+00:00 | 2015-01-05T16:00:00+00:00 | 6 | long gap |
| generation fossil coal-derived gas | 2015-01-05T11:00:00+00:00 | 2015-01-05T16:00:00+00:00 | 6 | long gap |
| generation fossil gas | 2015-01-05T11:00:00+00:00 | 2015-01-05T16:00:00+00:00 | 6 | long gap |
| generation fossil hard coal | 2015-01-05T11:00:00+00:00 | 2015-01-05T16:00:00+00:00 | 6 | long gap |
| generation fossil oil | 2015-01-05T11:00:00+00:00 | 2015-01-05T16:00:00+00:00 | 6 | long gap |
| generation fossil oil shale | 2015-01-05T11:00:00+00:00 | 2015-01-05T16:00:00+00:00 | 6 | long gap |
| generation fossil peat | 2015-01-05T11:00:00+00:00 | 2015-01-05T16:00:00+00:00 | 6 | long gap |
| generation geothermal | 2015-01-05T11:00:00+00:00 | 2015-01-05T16:00:00+00:00 | 6 | long gap |
| generation hydro pumped storage consumption | 2015-01-05T11:00:00+00:00 | 2015-01-05T16:00:00+00:00 | 6 | long gap |
| generation hydro run-of-river and poundage | 2015-01-05T11:00:00+00:00 | 2015-01-05T16:00:00+00:00 | 6 | long gap |
| generation hydro water reservoir | 2015-01-05T11:00:00+00:00 | 2015-01-05T16:00:00+00:00 | 6 | long gap |
| generation marine | 2015-01-05T11:00:00+00:00 | 2015-01-05T16:00:00+00:00 | 6 | long gap |
| generation nuclear | 2015-01-05T11:00:00+00:00 | 2015-01-05T16:00:00+00:00 | 6 | long gap |
| generation other | 2015-01-05T11:00:00+00:00 | 2015-01-05T16:00:00+00:00 | 6 | long gap |
| generation other renewable | 2015-01-05T11:00:00+00:00 | 2015-01-05T16:00:00+00:00 | 6 | long gap |
| generation solar | 2015-01-05T11:00:00+00:00 | 2015-01-05T16:00:00+00:00 | 6 | long gap |
| generation waste | 2015-01-05T11:00:00+00:00 | 2015-01-05T16:00:00+00:00 | 6 | long gap |
| generation wind offshore | 2015-01-05T11:00:00+00:00 | 2015-01-05T16:00:00+00:00 | 6 | long gap |
| generation wind onshore | 2015-01-05T11:00:00+00:00 | 2015-01-05T16:00:00+00:00 | 6 | long gap |
| total load actual | 2015-01-05T11:00:00+00:00 | 2015-01-05T16:00:00+00:00 | 6 | long gap |
| total load actual | 2015-02-01T11:00:00+00:00 | 2015-02-01T18:00:00+00:00 | 8 | long gap |

## Target ve leakage notu

- Target: `price day ahead` — imputasyon yok.
- `price actual` dataset içinde tutulur, feature listesine girmez.
