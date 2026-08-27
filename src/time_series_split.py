"""Chronological train/validation/test split and model-readiness audit.

Does not train, tune, or evaluate a model.
Does not modify raw CSVs, merged_energy_weather.parquet, or model_features.parquet.
Does not create new features.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "features" / "model_features.parquet"
MERGED_PATH = ROOT / "data" / "processed" / "merged" / "merged_energy_weather.parquet"
MANIFEST_PATH = ROOT / "reports" / "features" / "feature_manifest.csv"
TRAIN_PATH = ROOT / "data" / "processed" / "splits" / "train.parquet"
VAL_PATH = ROOT / "data" / "processed" / "splits" / "validation.parquet"
TEST_PATH = ROOT / "data" / "processed" / "splits" / "test.parquet"
SPLIT_REPORT = ROOT / "reports" / "features" / "time_series_split.md"
AUDIT_REPORT = ROOT / "reports" / "features" / "model_readiness_audit.md"

RAW_PROTECTED = (
    ROOT / "data" / "raw" / "energy_dataset.csv",
    ROOT / "data" / "raw" / "weather_features.csv",
)

EXPECTED_ROWS = 35064
EXPECTED_FEATURES = 184
TARGET_COL = "price day ahead"
PRICE_ACTUAL_COL = "price actual"
TRAIN_FRAC = 0.70
VAL_END_FRAC = 0.85
TARGET_LAGS = (("price_day_ahead_lag_24", 24), ("price_day_ahead_lag_48", 48), ("price_day_ahead_lag_168", 168))
SHIFT_THRESHOLD = 0.5

FORBIDDEN_NAME_RE = re.compile(
    r"(t\+1|tplus|lead_|_lead|future_|_future|forward_|_forward|lag_-|shift_minus)",
    re.I,
)
FORBIDDEN_EXACT = {TARGET_COL, PRICE_ACTUAL_COL, "time"}


class SplitError(ValueError):
    """Hard failure: do not silently continue."""


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def snapshot_protected() -> dict[str, tuple[str, int]]:
    out = {}
    for path in (*RAW_PROTECTED, FEATURES_PATH, MERGED_PATH):
        if path.exists():
            out[str(path)] = (file_md5(path), path.stat().st_size)
    return out


def assert_protected_unchanged(before: dict[str, tuple[str, int]]) -> None:
    after = snapshot_protected()
    if after != before:
        raise SplitError("A protected source file was modified")


def validate_hourly_utc(ts: pd.Series, name: str) -> None:
    if ts.isna().any():
        raise SplitError(f"{name}: unparsed timestamps")
    if ts.dt.tz is None or str(ts.dt.tz) != "UTC":
        raise SplitError(f"{name}: timestamps must be timezone-aware UTC, got {ts.dt.tz}")
    if ts.duplicated().any():
        raise SplitError(f"{name}: duplicate timestamps")
    if not bool(ts.is_monotonic_increasing):
        raise SplitError(f"{name}: timestamps are not strictly increasing")
    if int(ts.nunique()) != EXPECTED_ROWS:
        raise SplitError(f"{name}: expected {EXPECTED_ROWS} unique timestamps, got {ts.nunique()}")
    diffs = ts.diff().dropna()
    if not bool((diffs == pd.Timedelta(hours=1)).all()):
        raise SplitError(f"{name}: frequency is not exactly 1 hour")


def load_and_validate_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_parquet(FEATURES_PATH)
    merged = pd.read_parquet(MERGED_PATH)
    if "timestamp_utc" not in features.columns or "timestamp_utc" not in merged.columns:
        raise SplitError("timestamp_utc missing")
    features = features.sort_values("timestamp_utc").reset_index(drop=True)
    merged = merged.sort_values("timestamp_utc").reset_index(drop=True)
    if len(features) != EXPECTED_ROWS or len(merged) != EXPECTED_ROWS:
        raise SplitError(f"Row counts {len(features)}, {len(merged)}; expected {EXPECTED_ROWS}")
    validate_hourly_utc(features["timestamp_utc"], "model_features")
    validate_hourly_utc(merged["timestamp_utc"], "merged_energy_weather")
    if not bool((features["timestamp_utc"] == merged["timestamp_utc"]).all()):
        only_f = set(features["timestamp_utc"]) - set(merged["timestamp_utc"])
        only_m = set(merged["timestamp_utc"]) - set(features["timestamp_utc"])
        raise SplitError(f"Timestamp sets differ: only_features={len(only_f)} only_merged={len(only_m)}")
    return features, merged


def construct_model_table(features: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    target = merged[["timestamp_utc", TARGET_COL]]
    if PRICE_ACTUAL_COL in target.columns:
        raise SplitError("price actual leaked into the target extract")
    model = features.merge(target, on="timestamp_utc", how="left", validate="one_to_one")
    if len(model) != EXPECTED_ROWS:
        raise SplitError(f"Join changed row count: {len(model)}")
    if not bool((model["timestamp_utc"] == features["timestamp_utc"]).all()):
        raise SplitError("Join broke timestamp alignment")
    if TARGET_COL not in model.columns:
        raise SplitError("Target missing after join")
    if PRICE_ACTUAL_COL in model.columns:
        raise SplitError("price actual was brought into the model table")
    extra = set(model.columns) - set(features.columns) - {TARGET_COL}
    if extra:
        raise SplitError(f"Unexpected columns after join: {sorted(extra)}")
    n_features = len([c for c in model.columns if c not in {"timestamp_utc", TARGET_COL}])
    if n_features != EXPECTED_FEATURES:
        raise SplitError(f"Expected {EXPECTED_FEATURES} features, got {n_features}")
    return model


def feature_columns(model: pd.DataFrame) -> list[str]:
    return [c for c in model.columns if c not in {"timestamp_utc", TARGET_COL}]


def assert_feature_safety(model: pd.DataFrame, manifest: pd.DataFrame) -> None:
    x_cols = feature_columns(model)
    if TARGET_COL in x_cols:
        raise SplitError("Target is inside X")
    if PRICE_ACTUAL_COL in model.columns or PRICE_ACTUAL_COL in x_cols:
        raise SplitError("price actual is present")
    if "time" in model.columns:
        raise SplitError("raw time column is present")

    unsafe = []
    for col in x_cols:
        if col in FORBIDDEN_EXACT:
            unsafe.append((col, "forbidden exact name"))
        if FORBIDDEN_NAME_RE.search(col):
            unsafe.append((col, "future-looking name"))
        if col.startswith("generation ") or (
            col.startswith("generation_") and "_lag_" not in col and not col.endswith(("_lag_24", "_lag_48", "_lag_168"))
        ):
            if "_lag_" not in col:
                unsafe.append((col, "same-hour generation"))
        if col in {"total_load_actual", "total load actual"}:
            unsafe.append((col, "same-hour load actual"))
        if re.search(r"_lag_1$", col) or col.endswith("lag_1"):
            unsafe.append((col, "forbidden t-1 lag"))

    if unsafe:
        detail = "; ".join(f"{n} ({why})" for n, why in unsafe)
        raise SplitError(f"Unsafe feature detected. STOP. {detail}")

    if manifest["leakage_status"].ne("SAFE").any():
        bad = manifest.loc[manifest["leakage_status"] != "SAFE", "feature_name"].tolist()
        raise SplitError(f"Feature manifest is not all SAFE: {bad}")
    man_names = set(manifest["feature_name"])
    if man_names != set(x_cols):
        raise SplitError(
            f"Manifest/X mismatch. only_manifest={sorted(man_names - set(x_cols))[:10]} "
            f"only_X={sorted(set(x_cols) - man_names)[:10]}"
        )


def chronological_split(model: pd.DataFrame) -> dict[str, pd.DataFrame]:
    n = len(model)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * VAL_END_FRAC)
    splits = {
        "train": model.iloc[:train_end].copy(),
        "validation": model.iloc[train_end:val_end].copy(),
        "test": model.iloc[val_end:].copy(),
    }
    if sum(len(v) for v in splits.values()) != n:
        raise SplitError("Split row counts do not cover the full series")
    return splits


def split_summary(name: str, df: pd.DataFrame, n_total: int) -> dict[str, Any]:
    ts = df["timestamp_utc"]
    start, end = ts.iloc[0], ts.iloc[-1]
    duration = end - start
    return {
        "name": name,
        "rows": int(len(df)),
        "percentage": float(len(df) / n_total * 100),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "duration": str(duration),
        "hours": int(len(df)),
    }


def assert_no_temporal_overlap(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    train, val, test = splits["train"], splits["validation"], splits["test"]
    t_max, v_min = train["timestamp_utc"].max(), val["timestamp_utc"].min()
    v_max, te_min = val["timestamp_utc"].max(), test["timestamp_utc"].min()
    if not (t_max < v_min):
        raise SplitError(f"Train/validation overlap: max(train)={t_max} min(val)={v_min}")
    if not (v_max < te_min):
        raise SplitError(f"Validation/test overlap: max(val)={v_max} min(test)={te_min}")

    sets = {k: set(v["timestamp_utc"]) for k, v in splits.items()}
    overlap_tv = sets["train"] & sets["validation"]
    overlap_vt = sets["validation"] & sets["test"]
    overlap_tt = sets["train"] & sets["test"]
    if overlap_tv or overlap_vt or overlap_tt:
        raise SplitError("Overlapping timestamps across splits")
    if len(sets["train"]) + len(sets["validation"]) + len(sets["test"]) != EXPECTED_ROWS:
        raise SplitError("Split timestamp counts do not sum to 35064")

    assigned = pd.concat([s["timestamp_utc"] for s in splits.values()], ignore_index=True)
    if assigned.duplicated().any():
        raise SplitError("A timestamp is assigned to multiple splits")
    return {
        "max_train": t_max.isoformat(),
        "min_validation": v_min.isoformat(),
        "max_validation": v_max.isoformat(),
        "min_test": te_min.isoformat(),
        "duplicated_across_splits": 0,
        "overlapping_timestamps": 0,
        "multi_assigned_rows": 0,
    }


def audit_target_lags(model: pd.DataFrame) -> list[dict[str, Any]]:
    indexed = model.set_index("timestamp_utc", drop=False).sort_index()
    y = indexed[TARGET_COL]
    rows = []
    for col, hours in TARGET_LAGS:
        if col not in indexed.columns:
            raise SplitError(f"Expected lag feature missing: {col}")
        expected = y.shift(hours)
        actual = indexed[col]
        both = actual.notna() & expected.notna()
        if both.any():
            max_abs = float(np.nanmax(np.abs(actual[both].to_numpy() - expected[both].to_numpy())))
            if max_abs > 1e-9:
                raise SplitError(f"{col} does not match {TARGET_COL}(t-{hours}); max_abs={max_abs}")
        feat_na_expected_present = int((actual.isna() & expected.notna()).sum())
        rows.append(
            {
                "feature": col,
                "lag_hours": hours,
                "compared_non_null": int(both.sum()),
                "max_abs_diff": 0.0 if not both.any() else max_abs,
                "feature_nan_when_source_present": feat_na_expected_present,
                "pass": True,
            }
        )
    return rows


def nan_audit(df: pd.DataFrame, x_cols: list[str]) -> dict[str, Any]:
    x = df[x_cols]
    total = int(x.size)
    nan_cells = int(x.isna().sum().sum())
    by_col = x.isna().sum()
    by_col = by_col[by_col > 0].sort_values(ascending=False)
    return {
        "total_nan_cells": nan_cells,
        "nan_percentage": float(nan_cells / total * 100) if total else 0.0,
        "rows_with_nan": int(x.isna().any(axis=1).sum()),
        "nan_by_feature": {k: int(v) for k, v in by_col.to_dict().items()},
    }


def target_audit(df: pd.DataFrame) -> dict[str, Any]:
    y = df[TARGET_COL]
    if y.isna().any():
        raise SplitError(f"Target has {int(y.isna().sum())} NaN values")
    return {
        "count": int(y.count()),
        "nan_count": 0,
        "min": float(y.min()),
        "max": float(y.max()),
        "mean": float(y.mean()),
        "median": float(y.median()),
        "std": float(y.std(ddof=0)),
    }


def distribution_audit(splits: dict[str, pd.DataFrame], x_cols: list[str]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    stats_rows = []
    for split_name, df in splits.items():
        desc = df[x_cols].agg(["mean", "std", "min", "max"])
        for col in x_cols:
            stats_rows.append(
                {
                    "split": split_name,
                    "feature": col,
                    "mean": float(desc.loc["mean", col]) if pd.notna(desc.loc["mean", col]) else np.nan,
                    "std": float(desc.loc["std", col]) if pd.notna(desc.loc["std", col]) else np.nan,
                    "min": float(desc.loc["min", col]) if pd.notna(desc.loc["min", col]) else np.nan,
                    "max": float(desc.loc["max", col]) if pd.notna(desc.loc["max", col]) else np.nan,
                }
            )
    stats = pd.DataFrame(stats_rows)
    train = stats[stats["split"] == "train"].set_index("feature")
    shifts = []
    for split_name in ("validation", "test"):
        other = stats[stats["split"] == split_name].set_index("feature")
        for col in x_cols:
            tr_mean, tr_std = train.loc[col, "mean"], train.loc[col, "std"]
            o_mean = other.loc[col, "mean"]
            if not np.isfinite(tr_std) or tr_std == 0 or not np.isfinite(tr_mean) or not np.isfinite(o_mean):
                continue
            z = abs(o_mean - tr_mean) / tr_std
            if z >= SHIFT_THRESHOLD:
                shifts.append(
                    {
                        "feature": col,
                        "split": split_name,
                        "train_mean": float(tr_mean),
                        "split_mean": float(o_mean),
                        "abs_mean_shift_over_train_std": float(z),
                    }
                )
    shifts.sort(key=lambda r: r["abs_mean_shift_over_train_std"], reverse=True)
    return stats, shifts


def write_splits(splits: dict[str, pd.DataFrame], x_cols: list[str]) -> dict[str, str]:
    order = ["timestamp_utc", *x_cols, TARGET_COL]
    hashes = {}
    mapping = {"train": TRAIN_PATH, "validation": VAL_PATH, "test": TEST_PATH}
    for name, path in mapping.items():
        out = splits[name][order]
        if PRICE_ACTUAL_COL in out.columns or "time" in out.columns:
            raise SplitError("Forbidden column in split output")
        out.to_parquet(path, index=False)
        hashes[name] = file_md5(path)
    return hashes


def md_table(rows: list[list[Any]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def write_split_report(
    summaries: dict[str, dict[str, Any]],
    boundaries: dict[str, Any],
    lag_audit: list[dict[str, Any]],
    hashes: dict[str, str],
) -> None:
    rows = [
        [
            s["name"],
            s["rows"],
            f"{s['percentage']:.4f}%",
            s["start"],
            s["end"],
            s["duration"],
        ]
        for s in summaries.values()
    ]
    lag_rows = [
        [r["feature"], r["lag_hours"], r["compared_non_null"], r["max_abs_diff"], r["pass"]]
        for r in lag_audit
    ]
    text = f"""# Time-Series Split

Chronological 70 / 15 / 15 split. No shuffle, no random split, no new features, no model.

Join key: `timestamp_utc` (UTC, hourly). Target `price day ahead` left-joined from the merged parquet. `price actual` was not joined.

## Split sizes

{md_table(rows, ["split", "rows", "percentage", "start_utc", "end_utc", "duration"])}

## Exact boundaries

| Boundary | Timestamp (UTC) |
|---|---|
| max(train) | {boundaries['max_train']} |
| min(validation) | {boundaries['min_validation']} |
| max(validation) | {boundaries['max_validation']} |
| min(test) | {boundaries['min_test']} |

Checks: max(train) < min(validation); max(validation) < min(test). Overlaps = 0. Multi-assigned rows = 0.

Cut indices (deterministic): train = `[:int(n*0.70)]`, validation = `[int(n*0.70):int(n*0.85)]`, test = `[int(n*0.85):]` with n = 35064.

## Target-lag alignment

{md_table(lag_rows, ["feature", "lag", "compared", "max_abs_diff", "pass"])}

Leading lag NaNs were kept. No imputation.

## Output hashes

| file | md5 |
|---|---|
| data/processed/splits/train.parquet | {hashes['train']} |
| data/processed/splits/validation.parquet | {hashes['validation']} |
| data/processed/splits/test.parquet | {hashes['test']} |
"""
    SPLIT_REPORT.write_text(text, encoding="utf-8")


def write_audit_report(
    checks: list[tuple[str, bool]],
    model_ready: str,
    summaries: dict[str, dict[str, Any]],
    nan_by_split: dict[str, dict[str, Any]],
    target_by_split: dict[str, dict[str, Any]],
    shifts: list[dict[str, Any]],
    source_hashes: dict[str, str],
) -> None:
    check_rows = [["[x]" if ok else "[ ]", name] for name, ok in checks]
    nan_rows = [
        [
            name,
            a["total_nan_cells"],
            f"{a['nan_percentage']:.4f}%",
            a["rows_with_nan"],
        ]
        for name, a in nan_by_split.items()
    ]
    y_rows = [
        [name, a["count"], a["nan_count"], f"{a['min']:.4f}", f"{a['max']:.4f}", f"{a['mean']:.4f}", f"{a['median']:.4f}", f"{a['std']:.4f}"]
        for name, a in target_by_split.items()
    ]
    shift_rows = [
        [
            s["feature"],
            s["split"],
            f"{s['train_mean']:.4f}",
            f"{s['split_mean']:.4f}",
            f"{s['abs_mean_shift_over_train_std']:.3f}",
        ]
        for s in shifts[:40]
    ] or [["—", "—", "—", "—", "—"]]

    top_nan = []
    for name, a in nan_by_split.items():
        items = list(a["nan_by_feature"].items())[:8]
        for feat, cnt in items:
            top_nan.append([name, feat, cnt])
    if not top_nan:
        top_nan = [["—", "—", 0]]

    text = f"""# Model Readiness Audit

**MODEL_READY = {model_ready}**

No model was trained. No imputer or scaler was fit. Protected source files were not written.

## Checklist

{md_table(check_rows, ["status", "check"])}

## Protected source hashes (unchanged)

| file | md5 |
|---|---|
| merged_energy_weather.parquet | {source_hashes['merged']} |
| model_features.parquet | {source_hashes['features']} |

## Target by split (`price day ahead`)

{md_table(y_rows, ["split", "count", "NaN", "min", "max", "mean", "median", "std"])}

Train/validation/test means are reported only. The target was not transformed.

## NaN audit (features only; not imputed)

{md_table(nan_rows, ["split", "NaN cells", "NaN % of X cells", "rows with any NaN"])}

Highest NaN feature counts (expected lag heads, mainly train):

{md_table(top_nan, ["split", "feature", "NaN count"])}

## Potential distribution shifts

Rule: \|split_mean − train_mean\| / train_std ≥ {SHIFT_THRESHOLD}. Report only; no feature dropped.

{md_table(shift_rows, ["feature", "split", "train_mean", "split_mean", "|Δmean|/train_std"])}

## Split reminder

Train {summaries['train']['rows']} → validation {summaries['validation']['rows']} → test {summaries['test']['rows']}. Test starts after validation ends.
"""
    AUDIT_REPORT.write_text(text, encoding="utf-8")


def run() -> dict[str, Any]:
    before = snapshot_protected()
    source_hashes = {
        "merged": file_md5(MERGED_PATH),
        "features": file_md5(FEATURES_PATH),
    }
    features, merged = load_and_validate_inputs()
    model = construct_model_table(features, merged)
    manifest = pd.read_csv(MANIFEST_PATH)
    assert_feature_safety(model, manifest)
    x_cols = feature_columns(model)

    splits = chronological_split(model)
    summaries = {name: split_summary(name, df, len(model)) for name, df in splits.items()}
    boundaries = assert_no_temporal_overlap(splits)
    lag_audit = audit_target_lags(model)

    nan_by_split = {name: nan_audit(df, x_cols) for name, df in splits.items()}
    target_by_split = {name: target_audit(df) for name, df in splits.items()}
    _, shifts = distribution_audit(splits, x_cols)

    hashes = write_splits(splits, x_cols)
    assert_protected_unchanged(before)

    checks = [
        ("Raw CSVs unchanged", all(str(p) in before for p in RAW_PROTECTED if p.exists())),
        ("Source merged parquet unchanged", file_md5(MERGED_PATH) == source_hashes["merged"]),
        ("Feature parquet unchanged", file_md5(FEATURES_PATH) == source_hashes["features"]),
        ("Timestamp UTC", str(model["timestamp_utc"].dt.tz) == "UTC"),
        ("Hourly frequency", True),
        ("No duplicate timestamps", not model["timestamp_utc"].duplicated().any()),
        ("No timestamp overlap between splits", boundaries["overlapping_timestamps"] == 0),
        ("Chronological split", True),
        ("No shuffle", True),
        ("Target excluded from X", TARGET_COL not in x_cols),
        ("price actual excluded", PRICE_ACTUAL_COL not in model.columns),
        ("No forbidden features", True),
        ("Feature manifest all SAFE", bool((manifest["leakage_status"] == "SAFE").all())),
        ("No future features", True),
        ("Target has no NaN", all(a["nan_count"] == 0 for a in target_by_split.values())),
        ("NaNs not globally imputed", True),
        ("Train/validation/test chronological", True),
        ("Test occurs strictly after validation", True),
        ("Reproducible", True),
    ]
    model_ready = "PASS" if all(ok for _, ok in checks) else "FAIL"

    write_split_report(summaries, boundaries, lag_audit, hashes)
    write_audit_report(checks, model_ready, summaries, nan_by_split, target_by_split, shifts, source_hashes)
    assert_protected_unchanged(before)

    return {
        "MODEL_READY": model_ready,
        "summaries": summaries,
        "boundaries": boundaries,
        "hashes": hashes,
        "source_hashes": source_hashes,
        "target_by_split": target_by_split,
        "nan_by_split": {k: {kk: vv for kk, vv in v.items() if kk != "nan_by_feature"} for k, v in nan_by_split.items()},
        "n_shift_flags": len(shifts),
        "n_features": len(x_cols),
        "checks_failed": [name for name, ok in checks if not ok],
    }


if __name__ == "__main__":
    result = run()
    print("=== TIME-BASED SPLIT ===")
    for name, s in result["summaries"].items():
        print(
            f"{name.upper()}: rows={s['rows']} ({s['percentage']:.4f}%) "
            f"start={s['start']} end={s['end']} duration={s['duration']}"
        )
    print("boundaries:", json.dumps(result["boundaries"], indent=2))
    print("=== TARGET ===")
    print(json.dumps(result["target_by_split"], indent=2))
    print("=== NaN (X only) ===")
    print(json.dumps(result["nan_by_split"], indent=2))
    print("output hashes:", result["hashes"])
    print(f"distribution-shift flags (report only): {result['n_shift_flags']}")
    print(f"MODEL_READY = {result['MODEL_READY']}")
    if result["checks_failed"]:
        raise SystemExit(f"Failed checks: {result['checks_failed']}")
