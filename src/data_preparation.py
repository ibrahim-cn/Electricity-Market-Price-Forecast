"""Timezone-aware energy + weather merge pipeline.

Reads copies under data/raw/. Never writes to the original project-root CSVs.
Does not train a model, engineer features, or split train/test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MERGED_DIR = PROCESSED_DIR / "merged"
REPORTS_DIR = ROOT / "reports" / "data"

ENERGY_RAW_NAME = "energy_dataset.csv"
WEATHER_RAW_NAME = "weather_features.csv"

EXPECTED_ROWS = 35064
TARGET_COL = "price day ahead"
LEAKAGE_EXCLUDED_FEATURES = ("price actual",)
EMPTY_ENERGY_COLUMNS = (
    "generation hydro pumped storage aggregated",
    "forecast wind offshore eday ahead",
)
MAX_INTERPOLATION_GAP_HOURS = 3

NUMERIC_WEATHER_COLUMNS = (
    "temp",
    "temp_min",
    "temp_max",
    "pressure",
    "humidity",
    "wind_speed",
    "wind_deg",
    "rain_1h",
    "rain_3h",
    "snow_3h",
    "clouds_all",
)
CATEGORICAL_WEATHER_COLUMNS = (
    "weather_id",
    "weather_main",
    "weather_description",
    "weather_icon",
)
WEATHER_VALUE_COLUMNS = NUMERIC_WEATHER_COLUMNS + CATEGORICAL_WEATHER_COLUMNS

CITIES_ORDER = ("Barcelona", "Bilbao", "Madrid", "Seville", "Valencia")


class DataPreparationError(ValueError):
    """Raised when a hard data-quality check fails."""


def _md_table(rows: list[list[Any]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def _fmt_ts(value: Any) -> str:
    if value is pd.NaT or value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def circular_mean_degrees(values: pd.Series) -> float:
    """Vector mean of angles in degrees, mapped to [0, 360)."""
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    radians = np.deg2rad(arr)
    sine = np.mean(np.sin(radians))
    cosine = np.mean(np.cos(radians))
    return float(np.rad2deg(np.arctan2(sine, cosine)) % 360)


def deterministic_mode(values: pd.Series) -> Any:
    """Mode with lexical tie-break on string form of the original values."""
    cleaned = values.dropna()
    if cleaned.empty:
        return np.nan
    as_str = cleaned.astype(str)
    counts = as_str.value_counts()
    winners = sorted(counts[counts == counts.max()].index.tolist())
    winner = winners[0]
    return cleaned.loc[as_str == winner].iloc[0]


def setup_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for name in (ENERGY_RAW_NAME, WEATHER_RAW_NAME):
        dest = RAW_DIR / name
        src = ROOT / name
        if dest.exists():
            continue
        if not src.exists():
            raise DataPreparationError(f"Missing raw source file: {src}")
        dest.write_bytes(src.read_bytes())


def load_energy_data(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or (RAW_DIR / ENERGY_RAW_NAME)
    return pd.read_csv(csv_path)


def load_weather_data(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or (RAW_DIR / WEATHER_RAW_NAME)
    weather = pd.read_csv(csv_path)
    weather = weather.copy()
    weather["city_name"] = weather["city_name"].astype(str).str.strip()
    return weather


def normalize_timestamps(
    df: pd.DataFrame,
    source_col: str,
    utc_col: str = "timestamp_utc",
) -> pd.DataFrame:
    if source_col not in df.columns:
        raise DataPreparationError(f"Timestamp column not found: {source_col}")
    out = df.copy()
    parsed = pd.to_datetime(out[source_col], utc=True, errors="coerce")
    if parsed.dt.tz is None:
        raise DataPreparationError("Parsed timestamps are naive; utc=True is required.")
    out[utc_col] = parsed
    return out


def validate_hourly_utc_index(timestamps: pd.Series, name: str) -> None:
    ts = timestamps.copy()
    if ts.isna().any():
        raise DataPreparationError(
            f"{name}: parse edilemeyen timestamp = {int(ts.isna().sum())} (beklenen 0)"
        )
    if ts.dt.tz is None:
        raise DataPreparationError(f"{name}: naive datetime tespit edildi.")
    if str(ts.dt.tz) != "UTC":
        raise DataPreparationError(f"{name}: timezone UTC değil: {ts.dt.tz}")
    if int(ts.nunique()) != EXPECTED_ROWS:
        raise DataPreparationError(
            f"{name}: unique UTC timestamp = {int(ts.nunique())} (beklenen {EXPECTED_ROWS})"
        )
    if bool(ts.duplicated().any()):
        raise DataPreparationError(
            f"{name}: duplicate timestamp = {int(ts.duplicated().sum())} (beklenen 0)"
        )
    ordered = ts.sort_values()
    diffs = ordered.diff().dropna()
    if not bool((diffs == pd.Timedelta(hours=1)).all()):
        bad = diffs[diffs != pd.Timedelta(hours=1)]
        raise DataPreparationError(f"{name}: saatlik olmayan farklar var: {bad.head().to_dict()}")
    expected = pd.date_range(ordered.min(), ordered.max(), freq="h", tz="UTC")
    missing = expected.difference(pd.DatetimeIndex(ordered.unique()))
    if len(missing) != 0:
        raise DataPreparationError(f"{name}: eksik UTC timestamp = {len(missing)} (beklenen 0)")


def _group_key_frame(weather: pd.DataFrame) -> pd.DataFrame:
    return weather[["timestamp_utc", "city_name"]]


def analyze_weather_duplicates(weather: pd.DataFrame) -> dict[str, Any]:
    keys = ["timestamp_utc", "city_name"]
    sizes = weather.groupby(keys, sort=True).size()
    n_groups = int(len(sizes))
    n_dup_groups = int((sizes > 1).sum())
    multiplicity = {int(k): int(v) for k, v in sizes[sizes > 1].value_counts().sort_index().items()}

    exact_groups = 0
    numeric_same_cat_diff = 0
    numeric_diff = 0
    other_groups = 0

    if n_dup_groups:
        row_hash = pd.util.hash_pandas_object(weather, index=False)
        hash_nunique = weather.assign(_row_hash=row_hash).groupby(keys, sort=True)["_row_hash"].nunique()
        numeric_nunique = weather.groupby(keys, sort=True)[list(NUMERIC_WEATHER_COLUMNS)].nunique(dropna=False)
        cat_nunique = weather.groupby(keys, sort=True)[list(CATEGORICAL_WEATHER_COLUMNS)].nunique(dropna=False)
        dup_mask = sizes > 1
        numeric_same = numeric_nunique.max(axis=1) <= 1
        cat_same = cat_nunique.max(axis=1) <= 1
        exact = (hash_nunique == 1) & dup_mask
        numeric_same_cat_diff_mask = dup_mask & numeric_same & ~cat_same
        numeric_diff_mask = dup_mask & ~numeric_same
        classified = exact | numeric_same_cat_diff_mask | numeric_diff_mask
        exact_groups = int(exact.sum())
        numeric_same_cat_diff = int(numeric_same_cat_diff_mask.sum())
        numeric_diff = int(numeric_diff_mask.sum())
        other_groups = int((dup_mask & ~classified).sum())

    city_rows = []
    for city, city_df in weather.groupby("city_name", sort=True):
        n_rows = int(len(city_df))
        n_unique = int(city_df["timestamp_utc"].nunique())
        extra = n_rows - n_unique
        city_dup_groups = int((city_df.groupby("timestamp_utc").size() > 1).sum())
        city_rows.append(
            {
                "city_name": city,
                "rows": n_rows,
                "unique_timestamps": n_unique,
                "extra_rows": extra,
                "duplicate_groups": city_dup_groups,
                "extra_row_rate_pct": extra / n_rows * 100 if n_rows else 0.0,
            }
        )

    n_rows_in_dup_groups = int(weather.duplicated(keys, keep=False).sum())
    extra_rows = int(len(weather) - n_groups)
    summary = {
        "total_rows": int(len(weather)),
        "total_groups": n_groups,
        "total_duplicate_groups": n_dup_groups,
        "exact_duplicate_groups": exact_groups,
        "numeric_same_category_different_groups": numeric_same_cat_diff,
        "numeric_different_groups": numeric_diff,
        "other_duplicate_groups": other_groups,
        "rows_in_duplicate_groups": n_rows_in_dup_groups,
        "extra_rows": extra_rows,
        "duplicate_group_rate_pct": n_dup_groups / n_groups * 100 if n_groups else 0.0,
        "rows_in_duplicate_groups_rate_pct": n_rows_in_dup_groups / len(weather) * 100,
        "extra_row_rate_pct": extra_rows / len(weather) * 100,
        "multiplicity": dict(sorted(multiplicity.items())),
        "by_city": city_rows,
    }
    return summary


def write_weather_duplicate_report(summary: dict[str, Any], path: Path) -> None:
    city_table = _md_table(
        [
            [
                r["city_name"],
                r["rows"],
                r["unique_timestamps"],
                r["extra_rows"],
                r["duplicate_groups"],
                f"{r['extra_row_rate_pct']:.4f}%",
            ]
            for r in summary["by_city"]
        ],
        ["city_name", "satır", "unique timestamp", "extra satır", "duplicate group", "extra satır oranı"],
    )
    multiplicity_rows = [[k, v] for k, v in summary["multiplicity"].items()]
    multiplicity_table = _md_table(multiplicity_rows, ["group size", "group count"]) if multiplicity_rows else "_Yok_"

    text = f"""# Weather Duplicate Analysis

Kaynak: `data/raw/weather_features.csv` kopyası (`city_name` strip edilmiş).
Ham kök CSV overwrite edilmedi. `drop_duplicates()` ile sessiz çözüm uygulanmadı.

## Özet

| Ölçüt | Değer |
|---|---:|
| Toplam satır | {summary['total_rows']} |
| Toplam (timestamp_utc, city_name) group | {summary['total_groups']} |
| Toplam duplicate group (size > 1) | {summary['total_duplicate_groups']} |
| Birebir duplicate group | {summary['exact_duplicate_groups']} |
| Numeric aynı, weather category farklı group | {summary['numeric_same_category_different_groups']} |
| Numeric değerleri farklı group | {summary['numeric_different_groups']} |
| Diğer duplicate group | {summary['other_duplicate_groups']} |
| Duplicate group oranı (group / tüm group) | {summary['duplicate_group_rate_pct']:.4f}% |
| Duplicate group içindeki satır sayısı | {summary['rows_in_duplicate_groups']} |
| Duplicate group içindeki satır oranı | {summary['rows_in_duplicate_groups_rate_pct']:.4f}% |
| Extra satır (len - unique groups) | {summary['extra_rows']} |
| Extra satır oranı | {summary['extra_row_rate_pct']:.4f}% |

## Multiplicity

{multiplicity_table}

## Şehir bazında duplicate sayıları

{city_table}

## Yorum

- Duplicate'ler sessizce silinmedi.
- Çoğu tekrar, aynı sayısal ölçüm + farklı `weather_id` / `weather_main` / `weather_description` kombinasyonudur.
- Sonraki adımda deterministik aggregation uygulanır: numeric için mean/median/max/circular mean, kategorik için mode (eşitlikte lexical ilk değer).
"""
    path.write_text(text, encoding="utf-8")


def _aggregate_duplicate_weather(dup_part: pd.DataFrame) -> pd.DataFrame:
    grouped = dup_part.groupby(["timestamp_utc", "city_name"], sort=True, dropna=False)
    return grouped.agg(
        temp=("temp", "mean"),
        temp_min=("temp_min", "mean"),
        temp_max=("temp_max", "mean"),
        pressure=("pressure", "median"),
        humidity=("humidity", "mean"),
        wind_speed=("wind_speed", "mean"),
        wind_deg=("wind_deg", circular_mean_degrees),
        rain_1h=("rain_1h", "max"),
        rain_3h=("rain_3h", "max"),
        snow_3h=("snow_3h", "max"),
        clouds_all=("clouds_all", "mean"),
        weather_id=("weather_id", deterministic_mode),
        weather_main=("weather_main", deterministic_mode),
        weather_description=("weather_description", deterministic_mode),
        weather_icon=("weather_icon", deterministic_mode),
    ).reset_index()


def aggregate_weather(weather: pd.DataFrame) -> pd.DataFrame:
    keys = ["timestamp_utc", "city_name"]
    keep_cols = keys + list(WEATHER_VALUE_COLUMNS)
    sizes = weather.groupby(keys, sort=False).size()
    dup_index = sizes[sizes > 1].index
    if len(dup_index):
        key_index = pd.MultiIndex.from_frame(_group_key_frame(weather))
        is_dup_row = key_index.isin(dup_index)
    else:
        is_dup_row = np.zeros(len(weather), dtype=bool)

    unique_part = weather.loc[~is_dup_row, keep_cols].copy()
    if is_dup_row.any():
        aggregated_dups = _aggregate_duplicate_weather(weather.loc[is_dup_row, keep_cols])
        aggregated = pd.concat([unique_part, aggregated_dups], ignore_index=True)
    else:
        aggregated = unique_part

    aggregated = aggregated.sort_values(keys).reset_index(drop=True)
    if aggregated.duplicated(keys).any():
        raise DataPreparationError("Aggregation sonrası (timestamp_utc, city_name) unique değil.")
    if int(len(aggregated)) != EXPECTED_ROWS * len(CITIES_ORDER):
        raise DataPreparationError(
            f"Aggregation satır sayısı {len(aggregated)}, beklenen {EXPECTED_ROWS * len(CITIES_ORDER)}"
        )
    observed_cities = tuple(sorted(aggregated["city_name"].unique().tolist()))
    if observed_cities != CITIES_ORDER:
        raise DataPreparationError(f"Beklenmeyen şehir seti: {observed_cities}")
    return aggregated


def clean_weather_duplicates(weather: pd.DataFrame, report_path: Path | None = None) -> pd.DataFrame:
    summary = analyze_weather_duplicates(weather)
    write_weather_duplicate_report(summary, report_path or (REPORTS_DIR / "weather_duplicate_analysis.md"))
    return aggregate_weather(weather)


def pivot_weather(weather_agg: pd.DataFrame) -> pd.DataFrame:
    if weather_agg["city_name"].astype(str).str.contains(r"\s", regex=True).any():
        raise DataPreparationError("city_name içinde boşluk var; wide kolon adları geçersiz olur.")

    pieces = []
    for col in WEATHER_VALUE_COLUMNS:
        wide = weather_agg.pivot(index="timestamp_utc", columns="city_name", values=col)
        wide = wide.reindex(columns=list(CITIES_ORDER))
        wide.columns = [f"{col}_{city}" for city in wide.columns]
        pieces.append(wide)

    weather_wide = pd.concat(pieces, axis=1).sort_index()
    weather_wide.index.name = "timestamp_utc"
    weather_wide = weather_wide.reset_index()
    if weather_wide["timestamp_utc"].duplicated().any():
        raise DataPreparationError("Weather wide formatında duplicate timestamp var.")
    if len(weather_wide) != EXPECTED_ROWS:
        raise DataPreparationError(f"Weather wide satır {len(weather_wide)}, beklenen {EXPECTED_ROWS}")
    return weather_wide


def _nan_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i + 1
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def interpolate_short_gaps(series: pd.Series, max_gap: int = MAX_INTERPOLATION_GAP_HOURS) -> tuple[pd.Series, dict[str, Any]]:
    if series.index.tz is None:
        raise DataPreparationError("Interpolation naive DatetimeIndex üzerinde yapılamaz.")
    s = series.copy()
    original_na = s.isna()
    before = int(original_na.sum())
    if before == 0:
        return s, {"before": 0, "filled": 0, "remaining": 0, "filled_runs": [], "left_runs": []}

    interpolated = s.interpolate(method="time", limit_area="inside")
    mask = original_na.to_numpy()
    fill_mask = np.zeros(len(s), dtype=bool)
    filled_runs: list[dict[str, Any]] = []
    left_runs: list[dict[str, Any]] = []

    for start, end in _nan_runs(mask):
        length = end - start
        is_edge = start == 0 or end == len(s)
        record = {
            "start": _fmt_ts(s.index[start]),
            "end": _fmt_ts(s.index[end - 1]),
            "length_hours": length,
            "edge": is_edge,
        }
        if (not is_edge) and length <= max_gap:
            fill_mask[start:end] = True
            filled_runs.append(record)
        else:
            left_runs.append(record)

    result = s.copy()
    result.iloc[fill_mask] = interpolated.iloc[fill_mask]
    return result, {
        "before": before,
        "filled": int(fill_mask.sum()),
        "remaining": int(result.isna().sum()),
        "filled_runs": filled_runs,
        "left_runs": left_runs,
    }


def _missing_pattern(energy: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    subset = energy[columns]
    any_missing = subset.isna().any(axis=1)
    all_gen = [c for c in columns if c.startswith("generation ")]
    all_gen_missing = energy[all_gen].isna().all(axis=1) if all_gen else pd.Series(False, index=energy.index)
    hours = energy.loc[any_missing, ["timestamp_utc"] + columns]
    return {
        "rows_with_any_missing": int(any_missing.sum()),
        "rows_all_generation_missing": int(all_gen_missing.sum()),
        "missing_hours": [_fmt_ts(ts) for ts in energy.loc[any_missing, "timestamp_utc"]],
        "all_generation_missing_hours": [_fmt_ts(ts) for ts in energy.loc[all_gen_missing, "timestamp_utc"]],
        "per_column": {col: int(energy[col].isna().sum()) for col in columns},
        "sample_missing_table": hours.head(20),
    }


def clean_energy(energy: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = energy.copy()
    present_empty = [c for c in EMPTY_ENERGY_COLUMNS if c in out.columns]
    out = out.drop(columns=present_empty)

    if out["timestamp_utc"].dt.tz is None:
        raise DataPreparationError("clean_energy naive timestamp aldı.")
    out = out.sort_values("timestamp_utc").reset_index(drop=True)
    indexed = out.set_index("timestamp_utc", drop=False)
    if indexed.index.tz is None:
        raise DataPreparationError("DatetimeIndex naive.")

    imputable = [
        c
        for c in out.columns
        if (c.startswith("generation ") or c == "total load actual") and c not in (TARGET_COL, "price actual")
    ]
    pattern_before = _missing_pattern(out, imputable)

    imputation_stats: dict[str, Any] = {}
    for col in imputable:
        filled, stats = interpolate_short_gaps(indexed[col], max_gap=MAX_INTERPOLATION_GAP_HOURS)
        indexed[col] = filled
        imputation_stats[col] = stats

    cleaned = indexed.reset_index(drop=True)
    cols = list(cleaned.columns)
    if "time" in cols and "timestamp_utc" in cols:
        cols.remove("timestamp_utc")
        cols.insert(cols.index("time") + 1, "timestamp_utc")
        cleaned = cleaned[cols]
    if cleaned[TARGET_COL].isna().any():
        raise DataPreparationError("Target `price day ahead` üzerinde eksik değer oluştu; imputasyon yasak.")

    pattern_after = _missing_pattern(
        cleaned,
        [c for c in imputable if c in cleaned.columns],
    )
    report = {
        "dropped_columns": present_empty,
        "kept_zero_only_columns": [
            "generation fossil coal-derived gas",
            "generation fossil oil shale",
            "generation fossil peat",
            "generation geothermal",
            "generation marine",
            "generation wind offshore",
        ],
        "max_interpolation_gap_hours": MAX_INTERPOLATION_GAP_HOURS,
        "pattern_before": pattern_before,
        "pattern_after": pattern_after,
        "imputation_stats": imputation_stats,
        "target_imputed": False,
        "price_actual_used_in_imputation": False,
    }
    return cleaned, report


def write_data_cleaning_report(report: dict[str, Any], path: Path) -> None:
    drop_rows = [
        [
            "generation hydro pumped storage aggregated",
            "35064/35064 NaN (%100)",
            "Hiç gözlem yok; bilgi taşımıyor.",
        ],
        [
            "forecast wind offshore eday ahead",
            "35064/35064 NaN (%100)",
            "Hiç gözlem yok; kolon adında `eday` yazıyor. Kullanılamaz.",
        ],
    ]
    impute_rows = []
    remaining_runs = []
    filled_examples = []
    for col, stats in report["imputation_stats"].items():
        impute_rows.append([col, stats["before"], stats["filled"], stats["remaining"]])
        for run in stats["left_runs"]:
            remaining_runs.append(
                [col, run["start"], run["end"], run["length_hours"], "edge" if run["edge"] else "long gap"]
            )
        for run in stats["filled_runs"][:3]:
            filled_examples.append([col, run["start"], run["end"], run["length_hours"]])

    remaining_table = (
        _md_table(remaining_runs, ["kolon", "start_utc", "end_utc", "uzunluk (saat)", "neden"])
        if remaining_runs
        else "_Kalan uzun/edge gap yok._"
    )
    filled_table = (
        _md_table(filled_examples, ["kolon", "start_utc", "end_utc", "uzunluk (saat)"])
        if filled_examples
        else "_Kısa gap doldurulmadı._"
    )

    text = f"""# Data Cleaning

Ham CSV'ler değiştirilmedi. İşlem `data/raw/` kopyaları üzerindedir.

## Kaldırılan kolonlar

{_md_table(drop_rows, ["kolon", "eksiklik", "gerekçe"])}

Sıfır-only üretim kolonları **kaldırılmadı**. Feature selection sonraki aşamadadır:

- `generation fossil coal-derived gas`
- `generation fossil oil shale`
- `generation fossil peat`
- `generation geothermal`
- `generation marine`
- `generation wind offshore`

## Missing pattern (imputasyon öncesi)

- Herhangi bir generation / `total load actual` eksiği olan satır: **{report['pattern_before']['rows_with_any_missing']}**
- Tüm generation kolonlarının birlikte NaN olduğu satır: **{report['pattern_before']['rows_all_generation_missing']}**

{_md_table([[k, v] for k, v in report['pattern_before']['per_column'].items()], ["kolon", "NaN (önce)"])}

Tüm generation'ın boş olduğu UTC saatler:

{chr(10).join(f"- {h}" for h in report["pattern_before"]["all_generation_missing_hours"]) or "- yok"}

## Imputasyon stratejisi

Körü körüne 0 doldurma **yok**. NaN, fiziksel 0 üretim varsayılmadı.

Uygulanan kural:

1. Index: timezone-aware `timestamp_utc` (UTC).
2. Yalnızca `generation *` ve `total load actual`.
3. İçeride kalan (edge olmayan) NaN koşuları, uzunluk **≤ {report['max_interpolation_gap_hours']} saat** ise `method='time'` interpolasyon.
4. 4+ saatlik veya seri başı/sonu gap'ler NaN bırakılır.
5. `{TARGET_COL}` imputasyon yok (zaten eksik değil; kontrol edildi).
6. `price actual` ne target ne imputasyon kaynağıdır.

Gerekçe: 1–3 saatlik kopukluklar komşu saatlerden zaman ağırlıklı tahmin edilebilir. 6 saatlik blok (ör. 2015-01-05 öğleden sonra) kısa gap değildir; uydurma değer üretmek bias riski taşır.

## Imputasyon sonuçları

{_md_table(impute_rows, ["kolon", "NaN önce", "doldurulan", "NaN sonra"])}

### Doldurulan kısa gap örnekleri

{filled_table}

### Bilinçli olarak NaN bırakılan gap'ler

{remaining_table}

## Target ve leakage notu

- Target: `{TARGET_COL}` — imputasyon yok.
- `price actual` dataset içinde tutulur, feature listesine girmez.
"""
    path.write_text(text, encoding="utf-8")


def analyze_weather_outliers(weather: pd.DataFrame) -> dict[str, Any]:
    rules = {
        "pressure > 2000": weather["pressure"] > 2000,
        "pressure <= 0": weather["pressure"] <= 0,
        "wind_speed > 20": weather["wind_speed"] > 20,
        "humidity == 0": weather["humidity"] == 0,
    }
    n = len(weather)
    result: dict[str, Any] = {"total_rows": n, "anomalies": {}}
    for label, mask in rules.items():
        subset = weather.loc[mask, ["timestamp_utc", "city_name", "pressure", "wind_speed", "humidity"]]
        city_counts = subset["city_name"].value_counts().to_dict()
        result["anomalies"][label] = {
            "count": int(mask.sum()),
            "rate_pct": float(mask.sum() / n * 100) if n else 0.0,
            "city_counts": {k: int(v) for k, v in city_counts.items()},
            "min_ts": _fmt_ts(subset["timestamp_utc"].min()) if len(subset) else None,
            "max_ts": _fmt_ts(subset["timestamp_utc"].max()) if len(subset) else None,
        }
    return result


def write_weather_outliers_report(summary: dict[str, Any], path: Path) -> None:
    blocks = []
    for label, info in summary["anomalies"].items():
        city_rows = [[city, count] for city, count in sorted(info["city_counts"].items())] or [["—", 0]]
        blocks.append(
            f"""### `{label}`

| Ölçüt | Değer |
|---|---|
| Sayı | {info['count']} |
| Oran | {info['rate_pct']:.4f}% |
| Tarih aralığı (UTC) | {info['min_ts'] or '—'} → {info['max_ts'] or '—'} |

Şehir dağılımı:

{_md_table(city_rows, ["city_name", "adet"])}
"""
        )
    text = f"""# Weather Outliers

Kaynak: normalize edilmiş weather kopyası (aggregation öncesi, satır düzeyi).
Bu aşamada outlier **silinmedi** ve **düzeltilmedi**.

Toplam incelenen satır: {summary['total_rows']}

{chr(10).join(blocks)}

## Karar

Anomaliler belgelendi. Modelleme aşamasında clip, medyan yerine koyma veya satır/şehir özel kuralı ayrıca seçilecek.
"""
    path.write_text(text, encoding="utf-8")


def merge_datasets(energy: pd.DataFrame, weather_wide: pd.DataFrame) -> pd.DataFrame:
    if energy["timestamp_utc"].duplicated().any():
        raise DataPreparationError("Energy duplicate timestamp ile merge'e girdi.")
    if weather_wide["timestamp_utc"].duplicated().any():
        raise DataPreparationError("Weather wide duplicate timestamp ile merge'e girdi.")
    if energy["timestamp_utc"].dt.tz is None or weather_wide["timestamp_utc"].dt.tz is None:
        raise DataPreparationError("Merge naive datetime kabul etmez.")

    n_energy = len(energy)
    if n_energy != EXPECTED_ROWS:
        raise DataPreparationError(f"Energy satır {n_energy}, beklenen {EXPECTED_ROWS}")

    energy_ts = set(energy["timestamp_utc"])
    weather_ts = set(weather_wide["timestamp_utc"])
    if energy_ts != weather_ts:
        only_e = len(energy_ts - weather_ts)
        only_w = len(weather_ts - energy_ts)
        raise DataPreparationError(
            f"Timestamp kümeleri eşit değil. only_energy={only_e}, only_weather={only_w}"
        )

    merged = energy.merge(weather_wide, on="timestamp_utc", how="left", validate="one_to_one")
    if len(merged) != n_energy:
        raise DataPreparationError(f"Row multiplication: {n_energy} → {len(merged)}")
    if len(merged) != EXPECTED_ROWS:
        raise DataPreparationError(f"Final satır {len(merged)}, beklenen {EXPECTED_ROWS}")
    if merged["timestamp_utc"].duplicated().any():
        raise DataPreparationError("Merge sonrası duplicate timestamp.")
    if set(merged["timestamp_utc"]) != energy_ts:
        raise DataPreparationError("Energy timestamp kaybı.")

    weather_cols = [c for c in weather_wide.columns if c != "timestamp_utc"]
    if weather_cols:
        match_rate = float(merged[weather_cols].notna().all(axis=1).mean())
        if not np.isclose(match_rate, 1.0):
            raise DataPreparationError(f"Weather match rate {match_rate:.6f}, beklenen 1.0")
    return merged.sort_values("timestamp_utc").reset_index(drop=True)


def _weather_match_rate(df: pd.DataFrame) -> float:
    weather_cols = [c for c in df.columns if any(c.endswith(f"_{city}") for city in CITIES_ORDER)]
    if not weather_cols:
        return 0.0
    return float(df[weather_cols].notna().all(axis=1).mean())


def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    ts = df["timestamp_utc"]
    checks: dict[str, Any] = {}

    def add(name: str, ok: bool, detail: Any) -> None:
        checks[name] = {"ok": bool(ok), "detail": detail}

    add("row_count", len(df) == EXPECTED_ROWS, {"actual": int(len(df)), "expected": EXPECTED_ROWS})
    add("column_count", len(df.columns) > 0, {"actual": int(len(df.columns))})
    add("duplicate_timestamp", not ts.duplicated().any(), {"count": int(ts.duplicated().sum())})
    add("parseable_timestamp", not ts.isna().any(), {"unparsed": int(ts.isna().sum())})
    add("timezone_aware_utc", ts.dt.tz is not None and str(ts.dt.tz) == "UTC", {"tz": str(ts.dt.tz)})
    add("monotonic_timestamp", bool(ts.is_monotonic_increasing), True)
    expected = pd.date_range(ts.min(), ts.max(), freq="h", tz="UTC")
    missing = expected.difference(pd.DatetimeIndex(ts))
    add("missing_timestamp", len(missing) == 0, {"count": int(len(missing))})
    add("target_present", TARGET_COL in df.columns, TARGET_COL)
    add("target_missing_count", int(df[TARGET_COL].isna().sum()) == 0, {"count": int(df[TARGET_COL].isna().sum())})
    add(
        "target_min_max",
        True,
        {"min": float(df[TARGET_COL].min()), "max": float(df[TARGET_COL].max())},
    )
    match_rate = _weather_match_rate(df)
    add("weather_match_rate", np.isclose(match_rate, 1.0), {"rate": match_rate})
    add("price_actual_present_but_excluded_from_features", "price actual" in df.columns, True)
    add("original_time_kept", "time" in df.columns, True)

    numeric = df.select_dtypes(include=[np.number])
    inf_count = int(np.isinf(numeric.to_numpy(dtype=float, copy=False)).sum()) if len(numeric.columns) else 0
    add("infinite_values", inf_count == 0, {"count": inf_count})
    add("duplicate_rows", int(df.duplicated().sum()) == 0, {"count": int(df.duplicated().sum())})

    missing_by_col = (
        df.isna().sum().loc[lambda s: s > 0].sort_values(ascending=False)
    )
    checks["missing_values_by_column"] = {
        "ok": True,
        "detail": {k: int(v) for k, v in missing_by_col.to_dict().items()},
    }

    failed = [name for name, payload in checks.items() if name != "missing_values_by_column" and not payload["ok"]]
    if failed:
        raise DataPreparationError(f"Final quality checks failed: {failed}")
    return checks


def write_final_quality_report(df: pd.DataFrame, checks: dict[str, Any], path: Path) -> None:
    check_rows = []
    for name, payload in checks.items():
        if name == "missing_values_by_column":
            continue
        check_rows.append([name, "PASS" if payload["ok"] else "FAIL", json.dumps(payload["detail"], default=str)])

    missing_detail = checks["missing_values_by_column"]["detail"]
    missing_table = (
        _md_table([[k, v, f"{v / len(df) * 100:.4f}%"] for k, v in missing_detail.items()], ["kolon", "NaN", "oran"])
        if missing_detail
        else "_Hiç eksik değer yok._"
    )

    feature_exclude = list(LEAKAGE_EXCLUDED_FEATURES) + ["time", "timestamp_utc", TARGET_COL]
    feature_cols = [c for c in df.columns if c not in feature_exclude]
    text = f"""# Final Data Quality

Dataset: `data/processed/merged_energy_weather.csv` (+ parquet, üretildiyse).

## Checks

{_md_table(check_rows, ["check", "sonuç", "detay"])}

## Shape

- Satır: **{len(df)}**
- Kolon: **{len(df.columns)}**

## Timestamp

- Kolon: `timestamp_utc` (timezone-aware, UTC)
- Orijinal energy kolonu korundu: `time`
- min: `{_fmt_ts(df['timestamp_utc'].min())}`
- max: `{_fmt_ts(df['timestamp_utc'].max())}`
- duplicate: 0
- missing hour: 0
- monotonic: evet

## Target

- Kolon: `{TARGET_COL}`
- missing: {int(df[TARGET_COL].isna().sum())}
- min: {float(df[TARGET_COL].min())}
- max: {float(df[TARGET_COL].max())}
- `price actual` dataset içinde **var**, feature listesinde **yok**

## Feature exclusion (henüz model yok)

Feature olarak kullanılmayacak kolonlar:

{chr(10).join(f"- `{c}`" for c in feature_exclude)}

Aday feature sayısı (referans; feature engineering yapılmadı): {len(feature_cols)}

## Missing values by column (final)

{missing_table}

## Infinite values

- {checks['infinite_values']['detail']}

## Duplicate rows

- {checks['duplicate_rows']['detail']}

## Weather match rate

- {checks['weather_match_rate']['detail']}
"""
    path.write_text(text, encoding="utf-8")


def save_processed_data(df: pd.DataFrame) -> dict[str, str]:
    csv_path = MERGED_DIR / "merged_energy_weather.csv"
    parquet_path = MERGED_DIR / "merged_energy_weather.parquet"
    df.to_csv(csv_path, index=False)
    written = {"csv": str(csv_path)}
    try:
        df.to_parquet(parquet_path, index=False)
        written["parquet"] = str(parquet_path)
    except Exception as exc:  # pragma: no cover - optional engine
        written["parquet_error"] = str(exc)
    return written


def run_pipeline() -> dict[str, Any]:
    setup_directories()

    energy = load_energy_data()
    weather = load_weather_data()

    energy = normalize_timestamps(energy, "time")
    weather = normalize_timestamps(weather, "dt_iso")
    validate_hourly_utc_index(energy["timestamp_utc"], "energy")
    validate_hourly_utc_index(weather["timestamp_utc"].drop_duplicates(), "weather")

    outlier_summary = analyze_weather_outliers(weather)
    write_weather_outliers_report(outlier_summary, REPORTS_DIR / "weather_outliers.md")

    weather_agg = clean_weather_duplicates(weather, REPORTS_DIR / "weather_duplicate_analysis.md")
    weather_wide = pivot_weather(weather_agg)

    energy_clean, cleaning_report = clean_energy(energy)
    write_data_cleaning_report(cleaning_report, REPORTS_DIR / "data_cleaning.md")

    merged = merge_datasets(energy_clean, weather_wide)
    checks = validate_dataset(merged)
    write_final_quality_report(merged, checks, REPORTS_DIR / "final_data_quality.md")
    saved = save_processed_data(merged)

    return {
        "shape": [int(merged.shape[0]), int(merged.shape[1])],
        "columns": list(merged.columns),
        "saved": saved,
        "dropped_columns": cleaning_report["dropped_columns"],
        "target": TARGET_COL,
        "excluded_features": list(LEAKAGE_EXCLUDED_FEATURES),
        "weather_match_rate": checks["weather_match_rate"]["detail"]["rate"],
        "target_missing": checks["target_missing_count"]["detail"]["count"],
        "target_min": checks["target_min_max"]["detail"]["min"],
        "target_max": checks["target_min_max"]["detail"]["max"],
        "remaining_missing": checks["missing_values_by_column"]["detail"],
        "outlier_summary": outlier_summary,
        "imputation": {
            col: {k: stats[k] for k in ("before", "filled", "remaining")}
            for col, stats in cleaning_report["imputation_stats"].items()
        },
    }


if __name__ == "__main__":
    summary = run_pipeline()
    print(json.dumps({k: v for k, v in summary.items() if k != "outlier_summary"}, indent=2, default=str))
