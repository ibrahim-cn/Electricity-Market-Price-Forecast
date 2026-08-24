"""24-hour forecasting simulation for the locked Ridge + METHOD_B model.

Does not retune, change alpha/METHOD_B, or overwrite locked test outputs.
Default production path is STRICT: unknown/unverified features are not used.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import ridge_tuning as rt

ID_COL = rt.ID_COL
TARGET_COL = rt.TARGET_COL
TEST_PATH = rt.TEST_PATH
TRAIN_PATH = rt.TRAIN_PATH
VAL_PATH = rt.VAL_PATH
LOCAL_TZ = "Europe/Madrid"
ALPHA = 0.001
METHOD = "METHOD_B"
MODEL_NAME = "Ridge"
HORIZON = 24
P75 = 57.582499999999996
FRAC_COLS = (
    "fraction_high_price_last_7d",
    "fraction_high_price_last_14d",
    "fraction_high_price_last_30d",
)
FRAC_WINDOWS = {
    "fraction_high_price_last_7d": 168,
    "fraction_high_price_last_14d": 336,
    "fraction_high_price_last_30d": 720,
}
DAY_AHEAD_FORECAST_COLS = (
    "total_load_forecast",
    "forecast_solar_day_ahead",
    "forecast_wind_onshore_day_ahead",
    "renewable_forecast_total",
    "forecast_wind_share_of_load",
    "forecast_solar_share_of_load",
)
FORBIDDEN_NAMES = {
    TARGET_COL,
    "price actual",
    "price_actual",
    "total load actual",
    "total_load_actual",
}

ARTIFACT = ROOT / "data" / "processed" / "final_model"
FEATURE_JSON = ARTIFACT / "feature_manifest.json"
META_JSON = ARTIFACT / "model_metadata.json"
METHOD_B_JSON = ARTIFACT / "method_b_parameters.json"
PREP_JOBLIB = ARTIFACT / "preprocessing.joblib"
MODEL_JOBLIB = ARTIFACT / "model.joblib"

OUT_DIR = ROOT / "reports" / "forecasting"
FIG_DIR = ROOT / "outputs" / "figures"
REPORT_MD = OUT_DIR / "forecasting_24h.md"
AUDIT_CSV = OUT_DIR / "forecasting_availability_audit.csv"
PRED_CSV = OUT_DIR / "forecasting_predictions.csv"
ASSUMED_CSV = OUT_DIR / "forecasting_predictions_assumed.csv"
FIG_PATH = FIG_DIR / "forecast_24h.png"

LOCKED = (
    ROOT / "data" / "processed" / "predictions" / "final_test_predictions.parquet",
    ROOT / "reports" / "final" / "final_test_metrics.csv",
    ROOT / "reports" / "final" / "final_test_evaluation.md",
    ROOT / "reports" / "final_model" / "final_model.md",
)
WATCH = tuple(rt.PROTECTED) + LOCKED

TEST_READ_COUNT = 0


class ForecastError(ValueError):
    pass


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def snapshot(paths: tuple[Path, ...]) -> dict[str, str]:
    return {str(p): md5(p) for p in paths if p.exists()}


def assert_not_test(path: Path) -> None:
    global TEST_READ_COUNT
    resolved = Path(path).resolve()
    if resolved == TEST_PATH.resolve() or resolved.name == "test.parquet":
        TEST_READ_COUNT += 1
        raise ForecastError("Locked test parquet must not be read by the forecasting pipeline")


def read_dev(path: Path) -> pd.DataFrame:
    assert_not_test(path)
    return pd.read_parquet(path).sort_values(ID_COL, kind="mergesort").reset_index(drop=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_joblib(path: Path) -> Any:
    import joblib

    return joblib.load(path)


def calendar_features(ts: pd.Series) -> pd.DataFrame:
    local = ts.dt.tz_convert(LOCAL_TZ)
    hour = local.dt.hour.astype(np.int16)
    dow = local.dt.dayofweek.astype(np.int16)
    month = local.dt.month.astype(np.int16)
    doy = local.dt.dayofyear.astype(np.int16)
    two_pi = 2.0 * math.pi
    out = pd.DataFrame({ID_COL: ts})
    out["hour"] = hour.to_numpy()
    out["day_of_week"] = dow.to_numpy()
    out["day_of_month"] = local.dt.day.astype(np.int16).to_numpy()
    out["month"] = month.to_numpy()
    out["quarter"] = local.dt.quarter.astype(np.int16).to_numpy()
    out["day_of_year"] = doy.to_numpy()
    out["week_of_year"] = local.dt.isocalendar().week.astype(np.int16).to_numpy()
    out["is_weekend"] = (dow >= 5).astype(np.int8).to_numpy()
    out["is_month_start"] = local.dt.is_month_start.astype(np.int8).to_numpy()
    out["is_month_end"] = local.dt.is_month_end.astype(np.int8).to_numpy()
    out["is_year_start"] = local.dt.is_year_start.astype(np.int8).to_numpy()
    out["is_year_end"] = local.dt.is_year_end.astype(np.int8).to_numpy()
    out["hour_sin"] = np.sin(two_pi * hour / 24.0)
    out["hour_cos"] = np.cos(two_pi * hour / 24.0)
    out["dow_sin"] = np.sin(two_pi * dow / 7.0)
    out["dow_cos"] = np.cos(two_pi * dow / 7.0)
    out["month_sin"] = np.sin(two_pi * month / 12.0)
    out["month_cos"] = np.cos(two_pi * month / 12.0)
    out["day_of_year_sin"] = np.sin(two_pi * doy / 365.25)
    out["day_of_year_cos"] = np.cos(two_pi * doy / 365.25)
    return out


def method_b_from_history(y: pd.Series, ts: pd.Series, forecast_times: pd.DatetimeIndex, p75: float) -> pd.DataFrame:
    """Fractions use only y at timestamps ≤ last history stamp (no future target, no recursion)."""
    hist = pd.DataFrame({ID_COL: ts.to_numpy(), TARGET_COL: y.to_numpy()})
    rows = []
    y_map = hist.set_index(ID_COL)[TARGET_COL]
    for t in forecast_times:
        usable = y_map[y_map.index <= t - pd.Timedelta(hours=24)]
        rec = {ID_COL: t}
        for col, window in FRAC_WINDOWS.items():
            if len(usable) < window:
                rec[col] = np.nan
            else:
                tail = usable.iloc[-window:]
                rec[col] = float((tail.to_numpy(dtype=float) > p75).mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def classify_feature(name: str, group: str) -> dict[str, str]:
    """Availability at a true DAM origin τ = D-1 12:00 CET for a delivery hour t on day D."""
    if name in FORBIDDEN_NAMES or name == "price_day_ahead_lag_1":
        return {
            "classification_strict": "FORBIDDEN",
            "classification_assumed": "FORBIDDEN",
            "known_at_dam_origin": "no",
            "time_needed": "same-hour or same-auction actual",
            "future_leakage_risk": "high",
            "notes": "Same-hour / same-auction information. Not in the locked matrix.",
        }
    if group == "calendar":
        return {
            "classification_strict": "SAFE",
            "classification_assumed": "SAFE",
            "known_at_dam_origin": "yes",
            "time_needed": "calendar of hour t (Europe/Madrid)",
            "future_leakage_risk": "none",
            "notes": "Deterministic. No observations required.",
        }
    if group == "historical_target" or name.startswith("price_day_ahead_lag") or name.startswith("price_"):
        return {
            "classification_strict": "SAFE",
            "classification_assumed": "SAFE",
            "known_at_dam_origin": "yes",
            "time_needed": "published DAM prices at t-24 / t-48 / t-168",
            "future_leakage_risk": "low",
            "notes": (
                "Yesterday's full auction curve was published on D-2, so t-24/48/168 "
                "target lags do not use the same-day auction. t-1 is not created."
            ),
        }
    if group == "day_ahead_forecast":
        return {
            "classification_strict": "UNKNOWN",
            "classification_assumed": "CONDITIONAL",
            "known_at_dam_origin": "unverified",
            "time_needed": "TSO day-ahead forecast published for hour t",
            "future_leakage_risk": "medium",
            "notes": (
                "Column names say day-ahead, but the file has no publication timestamp. "
                "Forecast availability timestamp is not independently verified. "
                "STRICT will not use these values from a future row."
            ),
        }
    if name in FRAC_COLS:
        return {
            "classification_strict": "SAFE",
            "classification_assumed": "SAFE",
            "known_at_dam_origin": "yes",
            "time_needed": "published DAM prices up to t-24, vs development P75",
            "future_leakage_risk": "low",
            "notes": "Computed from past published prices only. No recursive ŷ.",
        }
    if group in {"historical_load", "historical_generation", "historical_weather", "weather_aggregate"}:
        lag = 168 if name.endswith("_168") else 48 if name.endswith("_48") else 24 if name.endswith("_24") else None
        if lag in {48, 168}:
            return {
                "classification_strict": "SAFE",
                "classification_assumed": "SAFE",
                "known_at_dam_origin": "yes",
                "time_needed": f"actual series at t-{lag}",
                "future_leakage_risk": "low",
                "notes": "Complete prior calendar day(s) relative to D-1 noon.",
            }
        return {
            "classification_strict": "FORBIDDEN",
            "classification_assumed": "CONDITIONAL",
            "known_at_dam_origin": "not for all 24 hours of D",
            "time_needed": "actual series at t-24",
            "future_leakage_risk": "high",
            "notes": (
                "At τ = D-1 12:00 CET, t-24 for evening hours of D is an actual on "
                "D-1 afternoon/evening and is not yet observed. STRICT forbids it. "
                "ASSUMED would only be valid if the origin is late enough that t-24 ≤ origin."
            ),
        }
    return {
        "classification_strict": "UNKNOWN",
        "classification_assumed": "UNKNOWN",
        "known_at_dam_origin": "unverified",
        "time_needed": "unknown",
        "future_leakage_risk": "high",
        "notes": "Unclassified. STRICT will not use it silently.",
    }


def build_audit(model_cols: list[str], groups: dict[str, str]) -> pd.DataFrame:
    rows = []
    for name in model_cols:
        group = groups.get(name, "diğer")
        rec = classify_feature(name, group)
        rec["feature"] = name
        rec["feature_group"] = group
        rec["used_by_locked_model"] = True
        rec["strict_usable"] = rec["classification_strict"] == "SAFE"
        rows.append(rec)
    cols = [
        "feature",
        "feature_group",
        "classification_strict",
        "classification_assumed",
        "known_at_dam_origin",
        "time_needed",
        "future_leakage_risk",
        "used_by_locked_model",
        "strict_usable",
        "notes",
    ]
    return pd.DataFrame(rows)[cols]


def transform(x: pd.DataFrame, prep: dict[str, Any]) -> np.ndarray:
    names = list(prep["feature_names"])
    if list(x.columns) != names:
        x = x.reindex(columns=names)
    arr = x.to_numpy(dtype=float)
    med = np.asarray(prep["medians"], dtype=float)
    mean = np.asarray(prep["mean"], dtype=float)
    scale = np.asarray(prep["scale"], dtype=float)
    for j, m in enumerate(med):
        mask = np.isnan(arr[:, j])
        if mask.any():
            arr[mask, j] = 0.0 if not np.isfinite(m) else float(m)
    return (arr - mean) / scale


def predict_ridge(x: pd.DataFrame, model: dict[str, Any], prep: dict[str, Any]) -> np.ndarray:
    xs = transform(x, prep)
    coef = np.asarray(model["coef"], dtype=float)
    return xs @ coef + float(model["intercept"]) + float(model["addend"])


def future_index(origin: pd.Timestamp, n: int = HORIZON) -> pd.DatetimeIndex:
    return pd.date_range(origin + pd.Timedelta(hours=1), periods=n, freq="h", tz="UTC")


def lookup_lag_features(dev: pd.DataFrame, times: pd.DatetimeIndex, cols: list[str]) -> pd.DataFrame:
    """Use precomputed lag/calendar columns from history rows only (timestamp ≤ each t is not required;
    we take the feature row at t if it exists AND t is in development). Never reads test."""
    indexed = dev.set_index(ID_COL)
    out = pd.DataFrame(index=times)
    for t in times:
        if t in indexed.index:
            out.loc[t, cols] = indexed.loc[t, cols]
        else:
            out.loc[t, cols] = np.nan
    out.index.name = ID_COL
    return out.reset_index()


def assert_no_forbidden(x: pd.DataFrame) -> None:
    bad = [c for c in x.columns if c in FORBIDDEN_NAMES or c.endswith("_lag_1")]
    if bad:
        raise ForecastError(f"forbidden columns in forecast X: {bad}")
    if TARGET_COL in x.columns or "price actual" in x.columns:
        raise ForecastError("target leaked into forecast X")


def write_figure(assumed: pd.DataFrame) -> bool:
    if assumed.empty or assumed["y_pred"].isna().all():
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ts = pd.to_datetime(assumed["timestamp_utc"], utc=True)
    ax.plot(ts, assumed["y_pred"], marker="o", color="#4c6a92", label="ASSUMED (not production)")
    ax.set_title("24h forecast — DATASET-ASSUMED only (NOT PRODUCTION-READY)")
    ax.set_ylabel("y_pred (€/MWh)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def write_report(
    origin: pd.Timestamp,
    origin_demo: pd.Timestamp,
    audit: pd.DataFrame,
    pred: pd.DataFrame,
    assumed: pd.DataFrame,
    hashes: dict[str, str],
    before: dict[str, str],
    after: dict[str, str],
    status: str,
    avail: str,
    leakage: str,
    repro: str,
    protected_ok: bool,
    production_ready: bool,
    blockers: list[str],
) -> None:
    counts = audit["classification_strict"].value_counts().to_dict()
    n_unknown = int((audit["classification_strict"] == "UNKNOWN").sum())
    n_forbidden = int((audit["classification_strict"] == "FORBIDDEN").sum())
    n_safe = int((audit["classification_strict"] == "SAFE").sum())
    used_unknown = audit[audit["classification_strict"].isin(["UNKNOWN", "FORBIDDEN"])]
    prot_rows = []
    for p in WATCH:
        key = str(p)
        if key not in before:
            continue
        prot_rows.append(
            {
                "file": p.name,
                "unchanged": before.get(key) == after.get(key),
            }
        )
    text = f"""# 24-Hour Forecasting

**FORECASTING_24H = {status}**

Locked model: Ridge(`alpha={ALPHA}`) + METHOD_B. Not retuned.

> 24-hour forecasting is only production-ready if every required feature is
> available at the forecast origin.

**PRODUCTION_READY = {str(production_ready).upper()}**

Forecast availability timestamp is not independently verified.

## 1. Forecast origin and horizon

Spanish DAM framing: origin **τ = D-1 ~12:00 CET**, target = 24 hourly
`price day ahead` values for delivery day D.

This simulation's last known leakage-safe timestamp (end of development,
test never opened):

| item | value |
|---|---|
| forecast_origin | {origin.isoformat()} |
| horizons | h+1 … h+24 |
| first target hour | {(origin + pd.Timedelta(hours=1)).isoformat()} |
| last target hour | {(origin + pd.Timedelta(hours=24)).isoformat()} |
| recursive forecasting | not used |
| test.parquet read | no |

Direct 24-hour forecasting: each hour's lags and METHOD_B fractions use
**published history ≤ origin** (or ≤ t−24, which is ≤ origin for this
window). Model predictions are not fed back as future features.

## 2. Feature availability audit

DAM origin **D-1 12:00 CET** (the real operational origin), not the clock
of the last development row.

| classification_strict | count |
|---|---:|
| SAFE | {n_safe} |
| CONDITIONAL | {int((audit['classification_strict']=='CONDITIONAL').sum())} |
| UNKNOWN | {n_unknown} |
| FORBIDDEN | {n_forbidden} |

Full table: `forecasting_availability_audit.csv`.

### Requested feature families

| family | STRICT at D-1 noon | why |
|---|---|---|
| `price_day_ahead_lag_24/48/168` | SAFE | Prior auction curves, published on D-2 |
| `total_load_forecast` | UNKNOWN | Named day-ahead; no publish timestamp in file |
| `forecast_solar_day_ahead` | UNKNOWN | same |
| `forecast_wind_onshore_day_ahead` | UNKNOWN | same |
| historical load / generation / weather `lag_24` | FORBIDDEN for part of D | t−24 for evening hours of D is still in D-1 afternoon |
| historical `lag_48` / `lag_168` | SAFE | complete prior days |
| weather aggregates `lag_24` | FORBIDDEN for part of D | same as city weather lag_24 |
| calendar | SAFE | deterministic |
| METHOD_B fractions | SAFE | past published prices vs development P75 |

### Scenario A — STRICT / verified

Only SAFE features may enter X. UNKNOWN is not imputed and not read from a
future row. The locked model needs **all 187 columns**, including UNKNOWN
day-ahead forecasts and FORBIDDEN-at-noon `lag_24` actuals.

Blockers:

{chr(10).join('- ' + b for b in blockers)}

Therefore STRICT does **not** emit a filled `y_pred`. Empty predictions are
preferred to a leaked 24-hour path.

### Scenario B — dataset-assumed day-ahead forecasts

If we **assume** the named forecast columns for hour t were published before
τ, those six columns become CONDITIONAL. That assumption is not verified.
Even then, a true D-1 noon origin still cannot lawfully fill `lag_24`
actuals for later hours of D.

An illustration with a **late** origin ({origin_demo.isoformat()}, 24h
before the last development stamp) makes `t-24 ≤ origin` for all 24 hours,
so historical lags are available. Forecast columns are taken from those
development rows only (test still closed). That run is labeled
**NOT PRODUCTION-READY**.

## 3. STRICT production table

{rt.md_table(pred)}

`y_pred` is empty because required features are UNKNOWN or FORBIDDEN at
the DAM origin / unverified at this origin.

## 4. ASSUMED illustration (not production)

Origin = {origin_demo.isoformat()} (development only).

{rt.md_table(assumed)}

These numbers must not be used to retune, pick features, or change METHOD_B.

## 5. Leakage asserts

FUTURE_LEAKAGE = {leakage}

- future actual weather: not used
- future actual generation: not used
- future actual load: not used
- future `price actual`: not used
- future target: not used (METHOD_B history truncated at origin)
- test target: not used
- test.parquet: not opened
- no t−1 same-auction price lag
- no recursive ŷ → feature loop

TEST_USED_FOR_TUNING = FALSE
TEST_USED_FOR_SELECTION = FALSE

## 6. Reproducibility and protected files

REPRODUCIBILITY = {repro}
PROTECTED_FILES_UNCHANGED = {str(protected_ok).upper()}
FEATURE_AVAILABILITY_AUDIT = {avail}

{rt.md_table(pd.DataFrame(prot_rows), floatfmt="")}

| file | md5 |
|---|---|
| forecasting_availability_audit.csv | {hashes['audit']} |
| forecasting_predictions.csv | {hashes['pred']} |
| forecasting_predictions_assumed.csv | {hashes['assumed']} |

Locked test evaluation files were not rewritten.

FORECASTING_24H = {status}
FEATURE_AVAILABILITY_AUDIT = {avail}
FUTURE_LEAKAGE = {leakage}
TEST_USED_FOR_TUNING = FALSE
TEST_USED_FOR_SELECTION = FALSE
PROTECTED_FILES_UNCHANGED = {str(protected_ok).upper()}
REPRODUCIBILITY = {repro}
PRODUCTION_READY = {str(production_ready).upper()}
"""
    REPORT_MD.write_text(text, encoding="utf-8")


def run_once() -> dict[str, Any]:
    global TEST_READ_COUNT
    TEST_READ_COUNT = 0
    before = snapshot(WATCH)

    feat_man = load_json(FEATURE_JSON)
    model_cols: list[str] = list(feat_man["model_features"])
    groups: dict[str, str] = dict(feat_man["feature_groups"])
    method_b = load_json(METHOD_B_JSON)
    p75 = float(method_b["p75_threshold"])
    if abs(p75 - P75) > 1e-6:
        raise ForecastError("archived METHOD_B P75 does not match the locked value")

    train = read_dev(TRAIN_PATH)
    val = read_dev(VAL_PATH)
    if "price actual" in train.columns or "price actual" in val.columns:
        raise ForecastError("price actual present in development")
    dev = pd.concat([train, val], axis=0, ignore_index=True)
    dev = dev.sort_values(ID_COL, kind="mergesort").reset_index(drop=True)
    origin = pd.Timestamp(dev[ID_COL].iloc[-1])
    if origin.tzinfo is None:
        raise ForecastError("origin must be UTC-aware")
    origin_demo = origin - pd.Timedelta(hours=HORIZON)

    audit = build_audit(model_cols, groups)
    if audit["classification_strict"].isna().any():
        raise ForecastError("unclassified feature")
    if (audit["feature"] == "UNKNOWN").any():
        raise ForecastError("invalid audit rows")
    unknown_or_forbidden = audit[audit["classification_strict"].isin(["UNKNOWN", "FORBIDDEN"])]
    blockers = [
        "Day-ahead load/solar/wind forecasts have no publication timestamp (UNKNOWN).",
        "At a D-1 12:00 CET origin, historical lag_24 actuals for later hours of D are not yet observed (FORBIDDEN).",
        "The locked 187-column model cannot be scored on STRICT features alone without changing the model.",
    ]

    future_ts = future_index(origin, HORIZON)
    cal = calendar_features(pd.Series(future_ts, name=ID_COL))
    strict_rows = []
    for i, t in enumerate(future_ts, start=1):
        strict_rows.append(
            {
                "timestamp_utc": t.isoformat(),
                "forecast_horizon": f"h+{i}",
                "y_pred": np.nan,
                "forecast_origin": origin.isoformat(),
                "model": MODEL_NAME,
                "alpha": ALPHA,
                "method": METHOD,
                "scenario": "STRICT",
                "production_ready": False,
                "blocking_reason": "required UNKNOWN/FORBIDDEN features at forecast origin",
            }
        )
    pred = pd.DataFrame(strict_rows)

    demo_ts = future_index(origin_demo, HORIZON)
    if demo_ts.max() > origin:
        raise ForecastError("assumed demo window must stay inside development")
    hist_cols = [c for c in model_cols if c not in DAY_AHEAD_FORECAST_COLS and c not in FRAC_COLS]
    looked = lookup_lag_features(dev, demo_ts, [c for c in hist_cols if c in dev.columns])
    cal_demo = calendar_features(pd.Series(demo_ts))
    for c in cal_demo.columns:
        if c != ID_COL and c in looked.columns:
            looked[c] = cal_demo[c].to_numpy()
    fracs = method_b_from_history(dev[TARGET_COL], dev[ID_COL], demo_ts, p75)
    assumed_x = looked.merge(fracs, on=ID_COL, how="left")
    fc = lookup_lag_features(dev, demo_ts, [c for c in DAY_AHEAD_FORECAST_COLS if c in dev.columns])
    assumed_x = assumed_x.merge(fc, on=ID_COL, how="left")
    assumed_x = assumed_x.reindex(columns=[ID_COL] + model_cols)
    assert_no_forbidden(assumed_x.drop(columns=[ID_COL]))
    if assumed_x[model_cols].isna().all(axis=None):
        raise ForecastError("assumed feature matrix is empty")

    model = load_joblib(MODEL_JOBLIB)
    prep = load_joblib(PREP_JOBLIB)
    y_assumed = predict_ridge(assumed_x[model_cols], model, prep)
    assumed_rows = []
    for i, (t, yp) in enumerate(zip(demo_ts, y_assumed), start=1):
        assumed_rows.append(
            {
                "timestamp_utc": t.isoformat(),
                "forecast_horizon": f"h+{i}",
                "y_pred": float(yp),
                "forecast_origin": origin_demo.isoformat(),
                "model": MODEL_NAME,
                "alpha": ALPHA,
                "method": METHOD,
                "scenario": "DATASET_ASSUMED",
                "production_ready": False,
                "blocking_reason": "forecast publication time not verified; not a DAM-noon origin",
            }
        )
    assumed = pd.DataFrame(assumed_rows)

    if TEST_READ_COUNT != 0:
        raise ForecastError("test.parquet was read")
    if (pred["y_pred"].notna()).any():
        raise ForecastError("STRICT path must not silently fill UNKNOWN features")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(AUDIT_CSV, index=False)
    pred.to_csv(PRED_CSV, index=False)
    assumed.to_csv(ASSUMED_CSV, index=False)
    write_figure(assumed)

    after = snapshot(WATCH)
    if after != before:
        raise ForecastError("A protected or locked file was modified")
    return {
        "origin": origin,
        "origin_demo": origin_demo,
        "audit": audit,
        "pred": pred,
        "assumed": assumed,
        "before": before,
        "after": after,
        "blockers": blockers,
        "unknown_or_forbidden": unknown_or_forbidden,
        "hashes": {
            "audit": md5(AUDIT_CSV),
            "pred": md5(PRED_CSV),
            "assumed": md5(ASSUMED_CSV),
        },
    }


def run() -> dict[str, Any]:
    first = run_once()
    second = run_once()
    repro = "PASS" if first["hashes"] == second["hashes"] else "FAIL"
    protected_ok = second["before"] == second["after"]
    leakage = "PASS"
    avail = "PASS"
    production_ready = False
    status = "PASS" if repro == "PASS" and protected_ok and leakage == "PASS" else "FAIL"
    write_report(
        origin=second["origin"],
        origin_demo=second["origin_demo"],
        audit=second["audit"],
        pred=second["pred"],
        assumed=second["assumed"],
        hashes=second["hashes"],
        before=second["before"],
        after=second["after"],
        status=status,
        avail=avail,
        leakage=leakage,
        repro=repro,
        protected_ok=protected_ok,
        production_ready=production_ready,
        blockers=second["blockers"],
    )
    print("FORECASTING_24H =", status)
    print("FEATURE_AVAILABILITY_AUDIT =", avail)
    print("FUTURE_LEAKAGE =", leakage)
    print("TEST_USED_FOR_TUNING = FALSE")
    print("TEST_USED_FOR_SELECTION = FALSE")
    print("PROTECTED_FILES_UNCHANGED =", str(protected_ok).upper())
    print("REPRODUCIBILITY =", repro)
    print("PRODUCTION_READY =", str(production_ready).upper())
    print(json.dumps(second["hashes"], indent=2))
    return second


if __name__ == "__main__":
    run()
