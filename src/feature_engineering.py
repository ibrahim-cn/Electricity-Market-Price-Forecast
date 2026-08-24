"""Leakage-safe feature engineering for day-ahead price forecasting.

Reads data/processed/merged_energy_weather.parquet without modifying it.
Does not train, split, tune, or evaluate a model.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PARQUET = ROOT / "data" / "processed" / "merged" / "merged_energy_weather.parquet"
OUTPUT_PARQUET = ROOT / "data" / "processed" / "features" / "model_features.parquet"
REPORTS_DIR = ROOT / "reports" / "features"
MANIFEST_CSV = REPORTS_DIR / "feature_manifest.csv"
REPORT_MD = REPORTS_DIR / "feature_engineering.md"

EXPECTED_ROWS = 35064
LOCAL_TZ = "Europe/Madrid"
TARGET_COL = "price day ahead"
PRICE_ACTUAL_COL = "price actual"
ALLOWED_TARGET_LAGS = (24, 48, 168)
ALLOWED_ACTUAL_LAGS_PRIMARY = (24, 168)
LOAD_ACTUAL_LAGS = (24, 48, 168)

CITIES = ("Barcelona", "Bilbao", "Madrid", "Seville", "Valencia")

GENERATION_LAG_SOURCES = (
    "generation biomass",
    "generation fossil brown coal/lignite",
    "generation fossil gas",
    "generation fossil hard coal",
    "generation fossil oil",
    "generation hydro pumped storage consumption",
    "generation hydro run-of-river and poundage",
    "generation hydro water reservoir",
    "generation nuclear",
    "generation other",
    "generation other renewable",
    "generation solar",
    "generation waste",
    "generation wind onshore",
)
ZERO_ONLY_GENERATION = (
    "generation fossil coal-derived gas",
    "generation fossil oil shale",
    "generation fossil peat",
    "generation geothermal",
    "generation marine",
    "generation wind offshore",
)
RENEWABLE_GENERATION = (
    "generation biomass",
    "generation hydro run-of-river and poundage",
    "generation hydro water reservoir",
    "generation other renewable",
    "generation solar",
    "generation wind onshore",
)
FOSSIL_GENERATION = (
    "generation fossil brown coal/lignite",
    "generation fossil gas",
    "generation fossil hard coal",
    "generation fossil oil",
)
TOTAL_PRODUCTION_GENERATION = (
    "generation biomass",
    "generation fossil brown coal/lignite",
    "generation fossil gas",
    "generation fossil hard coal",
    "generation fossil oil",
    "generation hydro run-of-river and poundage",
    "generation hydro water reservoir",
    "generation nuclear",
    "generation other",
    "generation other renewable",
    "generation solar",
    "generation waste",
    "generation wind onshore",
)
WEATHER_LAG_FIELDS = (
    "temp",
    "humidity",
    "pressure",
    "wind_speed",
    "clouds_all",
    "rain_1h",
    "rain_3h",
    "snow_3h",
)
IDENTIFIER_COLS = ("timestamp_utc",)

FORBIDDEN_IN_X = {
    TARGET_COL,
    PRICE_ACTUAL_COL,
    "total load actual",
    "time",
}


class FeatureEngineeringError(ValueError):
    """Raised when a leakage or alignment check fails."""


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def sanitize(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = denominator.to_numpy(dtype=float)
    num = numerator.to_numpy(dtype=float)
    out = np.full(len(num), np.nan, dtype=float)
    ok = np.isfinite(den) & (den > 0) & np.isfinite(num)
    out[ok] = num[ok] / den[ok]
    return pd.Series(out, index=numerator.index)


def lag_hours(series: pd.Series, hours: int) -> pd.Series:
    if hours <= 0:
        raise FeatureEngineeringError(f"Only positive lags are allowed; got {hours}")
    return series.shift(hours)


class Manifest:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        feature_name: str,
        source_column: str,
        feature_group: str,
        lag: int | str,
        transformation: str,
        availability: str,
        description: str,
        leakage_status: str = "SAFE",
    ) -> None:
        if leakage_status != "SAFE":
            raise FeatureEngineeringError(f"{feature_name} is not SAFE")
        self.rows.append(
            {
                "feature_name": feature_name,
                "source_column": source_column,
                "feature_group": feature_group,
                "lag": lag,
                "transformation": transformation,
                "availability": availability,
                "leakage_status": leakage_status,
                "description": description,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def load_source() -> pd.DataFrame:
    if not SOURCE_PARQUET.exists():
        raise FeatureEngineeringError(f"Missing source parquet: {SOURCE_PARQUET}")
    df = pd.read_parquet(SOURCE_PARQUET)
    if "timestamp_utc" not in df.columns:
        raise FeatureEngineeringError("timestamp_utc is required")
    if df["timestamp_utc"].dt.tz is None:
        raise FeatureEngineeringError("timestamp_utc must be timezone-aware")
    if str(df["timestamp_utc"].dt.tz) != "UTC":
        raise FeatureEngineeringError(f"timestamp_utc must be UTC, got {df['timestamp_utc'].dt.tz}")
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    if len(df) != EXPECTED_ROWS:
        raise FeatureEngineeringError(f"Expected {EXPECTED_ROWS} rows, got {len(df)}")
    if df["timestamp_utc"].duplicated().any():
        raise FeatureEngineeringError("Source timestamps are not unique")
    diffs = df["timestamp_utc"].diff().dropna()
    if not bool((diffs == pd.Timedelta(hours=1)).all()):
        raise FeatureEngineeringError("Source is not strictly hourly UTC")
    return df


def add_calendar_features(ts_utc: pd.Series, manifest: Manifest) -> pd.DataFrame:
    local = ts_utc.dt.tz_convert(LOCAL_TZ)
    hour = local.dt.hour.astype(np.int16)
    dow = local.dt.dayofweek.astype(np.int16)
    month = local.dt.month.astype(np.int16)
    doy = local.dt.dayofyear.astype(np.int16)
    week = local.dt.isocalendar().week.astype(np.int16)

    out = pd.DataFrame(index=ts_utc.index)
    out["hour"] = hour
    out["day_of_week"] = dow
    out["day_of_month"] = local.dt.day.astype(np.int16)
    out["month"] = month
    out["quarter"] = local.dt.quarter.astype(np.int16)
    out["day_of_year"] = doy
    out["week_of_year"] = week
    out["is_weekend"] = (dow >= 5).astype(np.int8)
    out["is_month_start"] = local.dt.is_month_start.astype(np.int8)
    out["is_month_end"] = local.dt.is_month_end.astype(np.int8)
    out["is_year_start"] = local.dt.is_year_start.astype(np.int8)
    out["is_year_end"] = local.dt.is_year_end.astype(np.int8)

    two_pi = 2.0 * math.pi
    out["hour_sin"] = np.sin(two_pi * hour / 24.0)
    out["hour_cos"] = np.cos(two_pi * hour / 24.0)
    out["dow_sin"] = np.sin(two_pi * dow / 7.0)
    out["dow_cos"] = np.cos(two_pi * dow / 7.0)
    out["month_sin"] = np.sin(two_pi * month / 12.0)
    out["month_cos"] = np.cos(two_pi * month / 12.0)
    out["day_of_year_sin"] = np.sin(two_pi * doy / 365.25)
    out["day_of_year_cos"] = np.cos(two_pi * doy / 365.25)

    calendar_desc = {
        "hour": "Local hour in Europe/Madrid (DST-aware)",
        "day_of_week": "Monday=0 ... Sunday=6, Europe/Madrid",
        "day_of_month": "Calendar day of month, Europe/Madrid",
        "month": "Month 1-12, Europe/Madrid",
        "quarter": "Quarter 1-4, Europe/Madrid",
        "day_of_year": "Day of year 1-366, Europe/Madrid",
        "week_of_year": "ISO week number, Europe/Madrid",
        "is_weekend": "1 if Saturday or Sunday local",
        "is_month_start": "1 if first local calendar day of month",
        "is_month_end": "1 if last local calendar day of month",
        "is_year_start": "1 if 1 January local",
        "is_year_end": "1 if 31 December local",
        "hour_sin": "sin(2π hour/24)",
        "hour_cos": "cos(2π hour/24)",
        "dow_sin": "sin(2π day_of_week/7)",
        "dow_cos": "cos(2π day_of_week/7)",
        "month_sin": "sin(2π month/12)",
        "month_cos": "cos(2π month/12)",
        "day_of_year_sin": "sin(2π day_of_year/365.25)",
        "day_of_year_cos": "cos(2π day_of_year/365.25)",
    }
    for name, desc in calendar_desc.items():
        cyclical = name.endswith("_sin") or name.endswith("_cos")
        manifest.add(
            name,
            "timestamp_utc",
            "calendar",
            0,
            "cyclical_encoding" if cyclical else "calendar_extract_europe_madrid",
            "known_in_advance",
            desc,
        )
    return out


def add_forecast_features(df: pd.DataFrame, manifest: Manifest) -> pd.DataFrame:
    load_f = df["total load forecast"]
    solar_f = df["forecast solar day ahead"]
    wind_f = df["forecast wind onshore day ahead"]
    out = pd.DataFrame(index=df.index)
    out["total_load_forecast"] = load_f
    out["forecast_solar_day_ahead"] = solar_f
    out["forecast_wind_onshore_day_ahead"] = wind_f
    out["renewable_forecast_total"] = solar_f + wind_f
    out["forecast_wind_share_of_load"] = safe_divide(wind_f, load_f)
    out["forecast_solar_share_of_load"] = safe_divide(solar_f, load_f)

    specs = [
        ("total_load_forecast", "total load forecast", "identity", "Day-ahead national load forecast for hour t"),
        ("forecast_solar_day_ahead", "forecast solar day ahead", "identity", "Day-ahead solar forecast for hour t"),
        (
            "forecast_wind_onshore_day_ahead",
            "forecast wind onshore day ahead",
            "identity",
            "Day-ahead onshore wind forecast for hour t",
        ),
        (
            "renewable_forecast_total",
            "forecast solar day ahead + forecast wind onshore day ahead",
            "sum",
            "solar_forecast(t) + wind_forecast(t); no actuals",
        ),
        (
            "forecast_wind_share_of_load",
            "forecast wind onshore day ahead / total load forecast",
            "ratio_zero_safe",
            "wind_forecast(t) / load_forecast(t); NaN if load_forecast<=0",
        ),
        (
            "forecast_solar_share_of_load",
            "forecast solar day ahead / total load forecast",
            "ratio_zero_safe",
            "solar_forecast(t) / load_forecast(t); NaN if load_forecast<=0",
        ),
    ]
    for name, source, trans, desc in specs:
        manifest.add(name, source, "day_ahead_forecast", 0, trans, "named_day_ahead_forecast_for_t", desc)
    return out


def add_target_history(df: pd.DataFrame, manifest: Manifest) -> pd.DataFrame:
    price = df[TARGET_COL]
    out = pd.DataFrame(index=df.index)
    lag24 = lag_hours(price, 24)
    lag48 = lag_hours(price, 48)
    lag168 = lag_hours(price, 168)
    out["price_day_ahead_lag_24"] = lag24
    out["price_day_ahead_lag_48"] = lag48
    out["price_day_ahead_lag_168"] = lag168

    pair = pd.concat([lag24, lag48], axis=1)
    triple = pd.concat([lag24, lag48, lag168], axis=1)
    out["price_mean_lag24_lag48"] = pair.mean(axis=1, skipna=False)
    out["price_mean_lag24_lag48_lag168"] = triple.mean(axis=1, skipna=False)
    out["price_std_lag24_lag48_lag168"] = triple.std(axis=1, ddof=0, skipna=False)
    out["price_min_lag24_lag48_lag168"] = triple.min(axis=1, skipna=False)
    out["price_max_lag24_lag48_lag168"] = triple.max(axis=1, skipna=False)

    for hours, col in ((24, "price_day_ahead_lag_24"), (48, "price_day_ahead_lag_48"), (168, "price_day_ahead_lag_168")):
        manifest.add(
            col,
            TARGET_COL,
            "historical_target",
            hours,
            f"shift(+{hours})",
            "prior_auction_only",
            f"price day ahead at t-{hours}; previous auction, not same delivery day",
        )
    manifest.add(
        "price_mean_lag24_lag48",
        "price_day_ahead_lag_24, price_day_ahead_lag_48",
        "historical_target",
        "24+48",
        "elementwise_mean_of_allowed_lags",
        "prior_auction_only",
        "Mean of the two allowed target lags only; not a rolling window",
    )
    for name, trans, desc in (
        ("price_mean_lag24_lag48_lag168", "elementwise_mean_of_allowed_lags", "Mean of t-24, t-48, t-168 target lags"),
        ("price_std_lag24_lag48_lag168", "elementwise_std_ddof0_of_allowed_lags", "Population std of the three allowed target lags"),
        ("price_min_lag24_lag48_lag168", "elementwise_min_of_allowed_lags", "Min of the three allowed target lags"),
        ("price_max_lag24_lag48_lag168", "elementwise_max_of_allowed_lags", "Max of the three allowed target lags"),
    ):
        manifest.add(
            name,
            "price_day_ahead_lag_24, price_day_ahead_lag_48, price_day_ahead_lag_168",
            "historical_target",
            "24+48+168",
            trans,
            "prior_auction_only",
            desc,
        )
    return out


def add_load_history(df: pd.DataFrame, manifest: Manifest) -> pd.DataFrame:
    actual = df["total load actual"]
    forecast = df["total load forecast"]
    out = pd.DataFrame(index=df.index)
    for hours in LOAD_ACTUAL_LAGS:
        name = f"total_load_actual_lag_{hours}"
        out[name] = lag_hours(actual, hours)
        manifest.add(
            name,
            "total load actual",
            "historical_load",
            hours,
            f"shift(+{hours})",
            "historical_actual_only",
            f"Realized load at t-{hours}; current load actual is not used",
        )
    error = actual - forecast
    out["load_forecast_error_lag_24"] = lag_hours(error, 24)
    manifest.add(
        "load_forecast_error_lag_24",
        "total load actual - total load forecast",
        "historical_load",
        24,
        "shift(+24) of (actual-forecast) at same historical hour",
        "historical_actual_and_forecast",
        "load_actual(t-24) - load_forecast(t-24); both timestamps are historical",
    )
    return out


def add_generation_history(df: pd.DataFrame, manifest: Manifest) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in GENERATION_LAG_SOURCES:
        base = sanitize(col)
        for hours in ALLOWED_ACTUAL_LAGS_PRIMARY:
            name = f"{base}_lag_{hours}"
            out[name] = lag_hours(df[col], hours)
            manifest.add(
                name,
                col,
                "historical_generation",
                hours,
                f"shift(+{hours})",
                "historical_actual_only",
                f"{col} at t-{hours}; same-hour generation is excluded",
            )

    total = df[list(TOTAL_PRODUCTION_GENERATION)].sum(axis=1, min_count=1)
    renewable = df[list(RENEWABLE_GENERATION)].sum(axis=1, min_count=1)
    fossil = df[list(FOSSIL_GENERATION)].sum(axis=1, min_count=1)
    share = safe_divide(renewable, total)

    for hours in ALLOWED_ACTUAL_LAGS_PRIMARY:
        out[f"total_generation_lag_{hours}"] = lag_hours(total, hours)
        out[f"renewable_generation_lag_{hours}"] = lag_hours(renewable, hours)
        out[f"fossil_generation_lag_{hours}"] = lag_hours(fossil, hours)
        out[f"renewable_share_lag_{hours}"] = lag_hours(share, hours)
        manifest.add(
            f"total_generation_lag_{hours}",
            " + ".join(TOTAL_PRODUCTION_GENERATION),
            "historical_generation",
            hours,
            f"sum_then_shift(+{hours})",
            "historical_actual_only",
            "Sum of documented production columns at s, then s=t-k. Excludes pumped-storage consumption and zero-only columns.",
        )
        manifest.add(
            f"renewable_generation_lag_{hours}",
            " + ".join(RENEWABLE_GENERATION),
            "historical_generation",
            hours,
            f"sum_then_shift(+{hours})",
            "historical_actual_only",
            "Renewable = biomass + hydro ror + hydro reservoir + other renewable + solar + wind onshore. Waste and nuclear excluded.",
        )
        manifest.add(
            f"fossil_generation_lag_{hours}",
            " + ".join(FOSSIL_GENERATION),
            "historical_generation",
            hours,
            f"sum_then_shift(+{hours})",
            "historical_actual_only",
            "Fossil = lignite + fossil gas + hard coal + fossil oil. Zero-only fossil types omitted.",
        )
        manifest.add(
            f"renewable_share_lag_{hours}",
            "renewable_generation / total_generation",
            "historical_generation",
            hours,
            f"ratio_then_shift(+{hours})",
            "historical_actual_only",
            "renewable(s)/total(s) at historical s=t-k; NaN if total<=0. Aggregated before lag.",
        )
    return out


def add_weather_history(df: pd.DataFrame, manifest: Manifest) -> pd.DataFrame:
    cols: dict[str, pd.Series] = {}
    for city in CITIES:
        city_key = city.lower()
        rad = np.deg2rad(df[f"wind_deg_{city}"].to_numpy(dtype=float))
        wind_sin = pd.Series(np.sin(rad), index=df.index)
        wind_cos = pd.Series(np.cos(rad), index=df.index)
        for hours in ALLOWED_ACTUAL_LAGS_PRIMARY:
            sin_name = f"wind_deg_sin_{city_key}_lag_{hours}"
            cos_name = f"wind_deg_cos_{city_key}_lag_{hours}"
            cols[sin_name] = lag_hours(wind_sin, hours)
            cols[cos_name] = lag_hours(wind_cos, hours)
            manifest.add(
                sin_name,
                f"wind_deg_{city}",
                "historical_weather",
                hours,
                f"sin(deg2rad(wind_deg)) then shift(+{hours})",
                "historical_observation_only",
                f"{city} wind direction sine at t-{hours}; direction is not averaged in degrees",
            )
            manifest.add(
                cos_name,
                f"wind_deg_{city}",
                "historical_weather",
                hours,
                f"cos(deg2rad(wind_deg)) then shift(+{hours})",
                "historical_observation_only",
                f"{city} wind direction cosine at t-{hours}",
            )
        for field in WEATHER_LAG_FIELDS:
            source = f"{field}_{city}"
            for hours in ALLOWED_ACTUAL_LAGS_PRIMARY:
                name = f"{field}_{city_key}_lag_{hours}"
                cols[name] = lag_hours(df[source], hours)
                manifest.add(
                    name,
                    source,
                    "historical_weather",
                    hours,
                    f"shift(+{hours})",
                    "historical_observation_only",
                    f"{city} {field} observation at t-{hours}; same-hour weather excluded",
                )

    temp_mean = df[[f"temp_{c}" for c in CITIES]].mean(axis=1)
    hum_mean = df[[f"humidity_{c}" for c in CITIES]].mean(axis=1)
    wind_mean = df[[f"wind_speed_{c}" for c in CITIES]].mean(axis=1)
    cloud_mean = df[[f"clouds_all_{c}" for c in CITIES]].mean(axis=1)
    rain_max = df[[f"rain_1h_{c}" for c in CITIES]].max(axis=1)
    national = {
        "temp_national_mean": (temp_mean, "mean temp across 5 cities at s, then lag", "temp_*"),
        "humidity_national_mean": (hum_mean, "mean humidity across 5 cities at s, then lag", "humidity_*"),
        "wind_speed_national_mean": (wind_mean, "mean wind_speed across 5 cities at s, then lag", "wind_speed_*"),
        "clouds_all_national_mean": (cloud_mean, "mean clouds_all across 5 cities at s, then lag", "clouds_all_*"),
        "rain_1h_national_max": (rain_max, "max rain_1h across 5 cities at s, then lag", "rain_1h_*"),
    }
    for base, (series, desc, source) in national.items():
        for hours in ALLOWED_ACTUAL_LAGS_PRIMARY:
            name = f"{base}_lag_{hours}"
            cols[name] = lag_hours(series, hours)
            manifest.add(
                name,
                source,
                "weather_aggregate",
                hours,
                f"spatial_stat_then_shift(+{hours})",
                "historical_observation_only",
                f"{desc}; uses weather at t-{hours} only, never t",
            )
    return pd.DataFrame(cols, index=df.index)


def assemble_features(df: pd.DataFrame) -> tuple[pd.DataFrame, Manifest]:
    manifest = Manifest()
    parts = [
        pd.DataFrame({"timestamp_utc": df["timestamp_utc"]}),
        add_calendar_features(df["timestamp_utc"], manifest),
        add_forecast_features(df, manifest),
        add_target_history(df, manifest),
        add_load_history(df, manifest),
        add_generation_history(df, manifest),
        add_weather_history(df, manifest),
    ]
    features = pd.concat(parts, axis=1)
    if features.columns.duplicated().any():
        dups = features.columns[features.columns.duplicated()].tolist()
        raise FeatureEngineeringError(f"Duplicate feature names: {dups}")
    return features, manifest


def inspect_source_code(path: Path) -> None:
    """Fail if executable AST uses rolling, future shifts, or randomness."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "rolling":
            raise FeatureEngineeringError("AST contains rolling; forbidden for target-safe stats")
        if isinstance(node, ast.Attribute) and node.attr in {"random", "shuffle", "sample"}:
            raise FeatureEngineeringError(f"AST contains random operation: {node.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "shift":
            if not node.args:
                continue
            arg0 = node.args[0]
            if isinstance(arg0, ast.UnaryOp) and isinstance(arg0.op, ast.USub):
                raise FeatureEngineeringError("AST contains a negative shift")
            if isinstance(arg0, ast.Constant) and isinstance(arg0.value, (int, float)) and arg0.value < 0:
                raise FeatureEngineeringError("AST contains a negative shift")


def assert_leakage_safe(features: pd.DataFrame, source: pd.DataFrame, manifest: Manifest) -> dict[str, Any]:
    inspect_source_code(Path(__file__))

    x_cols = [c for c in features.columns if c not in IDENTIFIER_COLS]
    x = features[x_cols]

    if TARGET_COL in features.columns or TARGET_COL in x.columns:
        raise FeatureEngineeringError("Target is present in the feature dataset")
    if PRICE_ACTUAL_COL in features.columns:
        raise FeatureEngineeringError("price actual is present in the feature dataset")
    if "time" in features.columns:
        raise FeatureEngineeringError("raw time column is present in the feature dataset")

    normalized = {c.replace("_", " ").lower() for c in x_cols}
    if "price day ahead" in normalized:
        raise FeatureEngineeringError("Unlagged target name found in X")

    for col in x_cols:
        if col.startswith("generation ") or (col.startswith("generation_") and "_lag_" not in col):
            raise FeatureEngineeringError(f"Same-hour generation feature: {col}")
        if col in {"total load actual", "total_load_actual"}:
            raise FeatureEngineeringError("Same-hour total load actual is in X")
        if re.search(r"lag_1$", col) or col.endswith("_lag_1") or "lag_1_" in col:
            raise FeatureEngineeringError(f"Forbidden lag-1 feature: {col}")
        if re.search(r"(lead_|shift_minus|future_)", col):
            raise FeatureEngineeringError(f"Future-looking feature name: {col}")

    for city in CITIES:
        for field in WEATHER_LAG_FIELDS + ("wind_deg", "temp_min", "temp_max"):
            raw = f"{field}_{city}"
            raw_l = f"{field}_{city.lower()}"
            if raw in x_cols or raw_l in x_cols:
                raise FeatureEngineeringError(f"Same-hour weather in X: {raw}")

    for col in GENERATION_LAG_SOURCES + ZERO_ONLY_GENERATION:
        if col in x_cols:
            raise FeatureEngineeringError(f"Raw generation column in X: {col}")

    ts = features["timestamp_utc"]
    if ts.dt.tz is None or str(ts.dt.tz) != "UTC":
        raise FeatureEngineeringError("Output timestamp_utc is not UTC-aware")
    if not bool(ts.is_monotonic_increasing):
        raise FeatureEngineeringError("Timestamps are not sorted ascending")
    if ts.duplicated().any():
        raise FeatureEngineeringError("Output timestamps are not unique")
    if len(features) != len(source) or len(features) != EXPECTED_ROWS:
        raise FeatureEngineeringError("Row count changed; silent drop is forbidden")
    if not bool((features["timestamp_utc"] == source["timestamp_utc"]).all()):
        raise FeatureEngineeringError("Feature rows are not aligned with source/target timestamps")

    if int((manifest.to_frame()["leakage_status"] != "SAFE").sum()) != 0:
        raise FeatureEngineeringError("Manifest contains a non-SAFE feature")
    if set(manifest.to_frame()["feature_name"]) != set(x_cols):
        raise FeatureEngineeringError("Manifest features do not match X columns")

    forbidden_found = [c for c in features.columns if c in FORBIDDEN_IN_X]
    if forbidden_found:
        raise FeatureEngineeringError(f"Forbidden columns in output: {forbidden_found}")

    return {
        "x_columns": x_cols,
        "safe_features": len(x_cols),
        "forbidden_in_output": 0,
    }


def validation_stats(source: pd.DataFrame, features: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, Any]:
    ts = features["timestamp_utc"]
    diffs = ts.diff().dropna()
    x_cols = [c for c in features.columns if c not in IDENTIFIER_COLS]
    nan_by_group = {}
    for group, names in manifest.groupby("feature_group")["feature_name"]:
        nan_by_group[group] = int(features[list(names)].isna().sum().sum())
    return {
        "original_rows": int(len(source)),
        "final_rows": int(len(features)),
        "total_features": int(len(x_cols)),
        "safe_features": int((manifest["leakage_status"] == "SAFE").sum()),
        "forbidden_features": 0,
        "nans_per_feature_group": nan_by_group,
        "first_timestamp": ts.iloc[0].isoformat(),
        "last_timestamp": ts.iloc[-1].isoformat(),
        "min_timestamp_diff": str(diffs.min()),
        "max_timestamp_diff": str(diffs.max()),
        "identifier_columns": list(IDENTIFIER_COLS),
    }


def write_report(stats: dict[str, Any], manifest: pd.DataFrame) -> None:
    group_counts = manifest["feature_group"].value_counts().sort_index()
    group_table_rows = "\n".join(
        f"| {g} | {int(n)} |" for g, n in group_counts.items()
    )
    nan_rows = "\n".join(
        f"| {g} | {n} |" for g, n in stats["nans_per_feature_group"].items()
    )
    text = f"""# Feature Engineering Report

## 1. Feature engineering objective

Build a leakage-safe feature matrix for forecasting `price day ahead` at delivery hour t.
No model is trained. No split, tuning, or evaluation is performed.
Source `data/processed/merged_energy_weather.parquet` is read-only.

Output: `data/processed/model_features.parquet` (identifier `timestamp_utc` + SAFE features only).
The target stays in the source parquet and is joined later by `timestamp_utc`.

## 2. Information boundary

Predict `price_day_ahead(t)` before delivery day D.

Forbidden at t: `price actual`; current or future `price day ahead`; other hours of the same delivery-day auction; same-hour generation, `total load actual`, and weather; future-looking rolls; target-at-t stats; full-dataset target encoding; randomness.

Allowed target history: **t-24, t-48, t-168 only**. `price_day_ahead(t-1)` is not created.

Calendar fields use `timestamp_utc` converted to **Europe/Madrid** so hour-of-day follows the Spanish clock and DST. The UTC timestamp remains the index/alignment key and is not a numeric feature.

## 3. Calendar features

20 features: hour, day_of_week, day_of_month, month, quarter, day_of_year, week_of_year (ISO), weekend/month/year boundary flags, plus sin/cos for hour (period 24), weekday (7), month (12), day_of_year (365.25).

Raw `time` / `timestamp_utc` are not in X.

## 4. Day-ahead forecast features

Identity: `total_load_forecast`, `forecast_solar_day_ahead`, `forecast_wind_onshore_day_ahead`.

Derived (row t only, no future rows):

- `renewable_forecast_total` = solar_forecast(t) + wind_forecast(t)
- `forecast_wind_share_of_load` = wind_forecast(t) / load_forecast(t)
- `forecast_solar_share_of_load` = solar_forecast(t) / load_forecast(t)

Shares are NaN when load_forecast ≤ 0 or non-finite. No actual generation or load is used.

## 5. Historical target features

Created only from `price day ahead` via `shift(+k)`:

- `price_day_ahead_lag_24`
- `price_day_ahead_lag_48`
- `price_day_ahead_lag_168`

Statistics are element-wise over those three series (or the first two), **not** a rolling window on the raw target. `skipna=False`: if any required lag is NaN, the statistic is NaN.

- `price_mean_lag24_lag48`
- `price_mean_lag24_lag48_lag168`
- `price_std_lag24_lag48_lag168` (population std, ddof=0)
- `price_min_lag24_lag48_lag168`
- `price_max_lag24_lag48_lag168`

`price_day_ahead_lag_1` does not exist.

## 6. Historical load features

- `total_load_actual_lag_24`, `_48`, `_168`
- `load_forecast_error_lag_24` = actual(t-24) − forecast(t-24)

`total load actual` at t is not in the matrix.

## 7. Historical generation features

Individual `lag_24` and `lag_168` for meaningful generation columns (zero-only series omitted; 100% empty columns were already absent from the source).

**Renewable:** biomass, hydro run-of-river and poundage, hydro water reservoir, other renewable, solar, wind onshore.  
**Fossil:** lignite, fossil gas, hard coal, fossil oil.  
**Total production:** renewable + fossil + nuclear + other + waste.  
**Excluded from aggregates:** hydro pumped storage *consumption* (not production), zero-only columns, waste not in renewable.

Aggregates are computed at historical hour s, then lagged: `total_generation_lag_*`, `renewable_generation_lag_*`, `fossil_generation_lag_*`, `renewable_share_lag_*` (NaN if total≤0).

Hydro pumped storage consumption still has its own lags; it is not inside renewable/fossil/total production.

## 8. Historical weather features

Same-hour weather is excluded.

Per city, `lag_24` and `lag_168` for: temp, humidity, pressure, wind_speed, clouds_all, rain_1h, rain_3h, snow_3h.

Wind direction: `wind_deg` is converted to `sin`/`cos` **before** lagging. Degree values are never averaged. No t-1 weather lags.

## 9. Weather aggregation

At each historical hour s, then lagged to t-24 and t-168 only:

- `temp_national_mean_lag_*`
- `humidity_national_mean_lag_*`
- `wind_speed_national_mean_lag_*`
- `clouds_all_national_mean_lag_*`
- `rain_1h_national_max_lag_*` (max hourly rain across the five cities)

Current weather is not used. `wind_deg` is not spatially averaged.

## 10. Missing-value behavior

- No zero-fill unless a share denominator is invalid (then **NaN**, not 0).
- Target is not interpolated or forward-filled (target is not even written to this file).
- Leading NaNs on lags are expected: first 24 / 48 / 168 UTC hours have no history.
- Rows are **not** dropped. Final rows = {stats['final_rows']}.

## 11. Leakage prevention

Automated checks (script fails on violation):

1. target not in output
2. `price actual` not in output
3. no unsuffixed target name in X
4. no same-hour actual generation / load / weather
5. no negative `shift`
6. no lag-1 features
7. no random APIs in this source file
8. no `.rolling(` in this source file
9. timestamps sorted ascending
10. timestamps unique
11. 35064 rows, `timestamp_utc` identical to source

`price actual` is never lagged.

## 12. Final feature count

| Group | Count |
|---|---:|
{group_table_rows}
| **SAFE features (X)** | **{stats['total_features']}** |
| Identifier (`timestamp_utc`) | 1 |
| Target in this file | 0 |

## 13. Removed / forbidden features

Not copied into `model_features.parquet`:

- `price day ahead` (target; remains in source only)
- `price actual`
- `time`
- same-hour `total load actual` and all same-hour `generation *`
- same-hour weather (numeric and categorical)
- zero-only generation series (no lag features)
- `price_day_ahead_lag_1`

## 14. Validation results

| Check | Value |
|---|---|
| original rows | {stats['original_rows']} |
| final rows | {stats['final_rows']} |
| total features | {stats['total_features']} |
| SAFE features | {stats['safe_features']} |
| forbidden features in output | {stats['forbidden_features']} |
| first timestamp | {stats['first_timestamp']} |
| last timestamp | {stats['last_timestamp']} |
| min timestamp diff | {stats['min_timestamp_diff']} |
| max timestamp diff | {stats['max_timestamp_diff']} |

NaN cell counts by group (lag heads dominate):

| feature_group | NaN cells |
|---|---:|
{nan_rows}

Manifest: `reports/feature_manifest.csv`.
"""
    REPORT_MD.write_text(text, encoding="utf-8")


def run() -> dict[str, Any]:
    source_hash = md5(SOURCE_PARQUET)
    source = load_source()
    features, manifest_obj = assemble_features(source)
    leak = assert_leakage_safe(features, source, manifest_obj)
    manifest = manifest_obj.to_frame()
    stats = validation_stats(source, features, manifest)

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PARQUET, index=False)
    manifest.to_csv(MANIFEST_CSV, index=False)
    write_report(stats, manifest)

    if md5(SOURCE_PARQUET) != source_hash:
        raise FeatureEngineeringError("Source parquet was modified; this is forbidden")

    stats_print = {
        **stats,
        "output": str(OUTPUT_PARQUET),
        "manifest": str(MANIFEST_CSV),
        "report": str(REPORT_MD),
        "source_parquet_unchanged": True,
        "x_preview": leak["x_columns"][:12],
    }
    return stats_print


if __name__ == "__main__":
    result = run()
    print("=== FEATURE ENGINEERING VALIDATION ===")
    print(f"original rows: {result['original_rows']}")
    print(f"final rows: {result['final_rows']}")
    print(f"total features: {result['total_features']}")
    print(f"SAFE features: {result['safe_features']}")
    print(f"forbidden features: {result['forbidden_features']}")
    print("NaNs per feature group:")
    for group, n in result["nans_per_feature_group"].items():
        print(f"  {group}: {n}")
    print(f"first timestamp: {result['first_timestamp']}")
    print(f"last timestamp: {result['last_timestamp']}")
    print(f"min timestamp diff: {result['min_timestamp_diff']}")
    print(f"max timestamp diff: {result['max_timestamp_diff']}")
    print(f"source parquet unchanged: {result['source_parquet_unchanged']}")
    print(json.dumps({k: v for k, v in result.items() if k != "x_preview"}, indent=2, default=str))
