"""Leakage-safe baseline modeling. Test is not used for selection.

Does not modify split parquets, feature parquet, merged parquet, or raw CSVs.
Does not create features, tune with Optuna, or select models on test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "splits" / "train.parquet"
VAL_PATH = ROOT / "data" / "processed" / "splits" / "validation.parquet"
TEST_PATH = ROOT / "data" / "processed" / "splits" / "test.parquet"
MERGED_PATH = ROOT / "data" / "processed" / "merged" / "merged_energy_weather.parquet"
FEATURES_PATH = ROOT / "data" / "processed" / "features" / "model_features.parquet"

PRED_VAL_PATH = ROOT / "data" / "processed" / "predictions" / "baseline_validation_predictions.parquet"
PRED_TEST_PATH = ROOT / "data" / "processed" / "predictions" / "baseline_test_predictions.parquet"

REPORTS = ROOT / "reports" / "modeling"
CMP_CSV = REPORTS / "baseline_model_comparison.csv"
HOUR_CSV = REPORTS / "baseline_error_by_hour.csv"
MONTH_CSV = REPORTS / "baseline_error_by_month.csv"
WEEKDAY_CSV = REPORTS / "baseline_error_by_weekday.csv"
REPORT_MD = REPORTS / "baseline_modeling.md"

TARGET_COL = "price day ahead"
ID_COL = "timestamp_utc"
FORBIDDEN_IN_X = {TARGET_COL, "price actual", ID_COL, "time"}
RANDOM_STATE = 42
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
LAG24 = "price_day_ahead_lag_24"
LAG48 = "price_day_ahead_lag_48"
LAG168 = "price_day_ahead_lag_168"
MEAN24_48 = "price_mean_lag24_lag48"
MEAN24_48_168 = "price_mean_lag24_lag48_lag168"

PROTECTED = (
    ROOT / "data" / "raw" / "energy_dataset.csv",
    ROOT / "data" / "raw" / "weather_features.csv",
    MERGED_PATH,
    FEATURES_PATH,
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
)


class BaselineError(ValueError):
    pass


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def snapshot(paths: tuple[Path, ...]) -> dict[str, str]:
    return {str(p): md5(p) for p in paths if p.exists()}


def assert_x_safe(x: pd.DataFrame, name: str) -> None:
    overlap = FORBIDDEN_IN_X.intersection(x.columns)
    if overlap:
        raise BaselineError(f"{name} X contains forbidden columns: {sorted(overlap)}")
    if x.shape[1] != 184:
        raise BaselineError(f"{name} X has {x.shape[1]} columns, expected 184")


def load_splits() -> dict[str, pd.DataFrame]:
    frames = {
        "train": pd.read_parquet(TRAIN_PATH),
        "validation": pd.read_parquet(VAL_PATH),
        "test": pd.read_parquet(TEST_PATH),
    }
    for name, df in frames.items():
        if TARGET_COL not in df.columns:
            raise BaselineError(f"{name} missing target")
        if "price actual" in df.columns:
            raise BaselineError(f"{name} contains price actual")
    return frames


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    y = df[TARGET_COL]
    ts = df[ID_COL]
    x = df.drop(columns=[ID_COL, TARGET_COL])
    assert_x_safe(x, "split")
    return x, y, ts


def target_stats(y: pd.Series) -> dict[str, float]:
    return {
        "count": int(y.count()),
        "mean": float(y.mean()),
        "median": float(y.median()),
        "std": float(y.std(ddof=0)),
        "min": float(y.min()),
        "max": float(y.max()),
        "nan": int(y.isna().sum()),
    }


def nan_profile(x: pd.DataFrame) -> dict[str, Any]:
    counts = x.isna().sum()
    counts = counts[counts > 0].sort_values(ascending=False)
    return {
        "nan_cells": int(x.isna().sum().sum()),
        "rows_with_nan": int(x.isna().any(axis=1).sum()),
        "features_with_nan": int((x.isna().any()).sum()),
        "by_feature": {k: int(v) for k, v in counts.to_dict().items()},
    }


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if not mask.any():
        return float("nan")
    return float(100.0 * np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]))


def regression_metrics(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    yt = y_true.to_numpy(dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise BaselineError("Prediction length mismatch")
    return {
        "MAE": float(mean_absolute_error(yt, yp)),
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "R2": float(r2_score(yt, yp)),
        "sMAPE": smape(yt, yp),
        "bias": float(np.mean(yp - yt)),
    }


def row(model: str, split: str, mets: dict[str, float]) -> dict[str, Any]:
    return {"model": model, "split": split, **mets}


class TrainOnlyPreprocessor:
    """Median impute + standardize. Statistics from X_train only."""

    def __init__(self) -> None:
        self.medians_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x_train: pd.DataFrame) -> "TrainOnlyPreprocessor":
        arr = x_train.to_numpy(dtype=float)
        self.medians_ = np.nanmedian(arr, axis=0)
        filled = self._impute(arr)
        self.mean_ = filled.mean(axis=0)
        std = filled.std(axis=0, ddof=0)
        std[std == 0] = 1.0
        self.scale_ = std
        return self

    def _impute(self, arr: np.ndarray) -> np.ndarray:
        if self.medians_ is None:
            raise BaselineError("Preprocessor not fit")
        out = arr.copy()
        for j, med in enumerate(self.medians_):
            mask = np.isnan(out[:, j])
            if mask.any():
                fill = 0.0 if not np.isfinite(med) else float(med)
                out[mask, j] = fill
        return out

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise BaselineError("Preprocessor not fit")
        filled = self._impute(x.to_numpy(dtype=float))
        return (filled - self.mean_) / self.scale_


def ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_pred: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form Ridge: (X'X + αI)w = X'y, intercept via centered y. Deterministic NumPy."""
    y = y_train.astype(float)
    y_mean = y.mean()
    yc = y - y_mean
    xtx = x_train.T @ x_train
    p = xtx.shape[0]
    xtx = xtx + alpha * np.eye(p)
    w = np.linalg.solve(xtx, x_train.T @ yc)
    return x_pred @ w + y_mean


def ridge_select(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    prep: TrainOnlyPreprocessor,
) -> tuple[float, np.ndarray, list[dict[str, Any]], np.ndarray, float]:
    xtr = prep.transform(x_train)
    xva = prep.transform(x_val)
    ytr = y_train.to_numpy(dtype=float)
    grid_rows = []
    best_alpha = None
    best_mae = np.inf
    best_val = None
    for alpha in RIDGE_ALPHAS:
        print(f"Ridge alpha={alpha} (NumPy closed form)...", flush=True)
        pred = ridge_fit_predict(xtr, ytr, xva, alpha)
        mets = regression_metrics(y_val, pred)
        grid_rows.append({"alpha": alpha, **mets})
        if mets["MAE"] < best_mae:
            best_mae = mets["MAE"]
            best_alpha = alpha
            best_val = pred
    if best_alpha is None or best_val is None:
        raise BaselineError("Ridge grid failed")
    return float(best_alpha), best_val, grid_rows, xtr, ytr


def _md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [header, sep]
    for _, rec in df.iterrows():
        cells = []
        for c in cols:
            v = rec[c]
            if isinstance(v, (float, np.floating)):
                cells.append(format(float(v), floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    y_stats: dict[str, dict[str, float]],
    nan_train: dict[str, Any],
    comparison: pd.DataFrame,
    selected: str,
    ridge_alpha: float,
    ridge_grid: list[dict[str, Any]],
    val_mets: dict[str, float],
    test_mets: dict[str, float],
    naive_val: dict[str, float],
    naive_test: dict[str, float],
    hour_err: pd.DataFrame,
    month_err: pd.DataFrame,
    weekday_err: pd.DataFrame,
    hashes: dict[str, str],
    status: str,
) -> None:
    worst_hours = hour_err.sort_values("MAE", ascending=False).head(5)
    best_hours = hour_err.sort_values("MAE", ascending=True).head(5)
    worst_months = month_err.sort_values("MAE", ascending=False)
    worst_days = weekday_err.sort_values("MAE", ascending=False)
    ml_val = comparison[(comparison["split"] == "validation") & (comparison["model"].isin(["Ridge", "HistGradientBoosting"]))]
    naive_beats = naive_val["MAE"] <= val_mets["MAE"] + 1e-12
    text = f"""# Baseline Modeling

**BASELINE_MODELING = {status}**

No Optuna. No feature selection. Test was not used to choose a model, alpha, or preprocessor.
Protected split/feature/raw files were not modified.

## Target (`price day ahead`, untransformed)

| split | count | mean | median | std | min | max | NaN |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | {y_stats['train']['count']} | {y_stats['train']['mean']:.4f} | {y_stats['train']['median']:.4f} | {y_stats['train']['std']:.4f} | {y_stats['train']['min']:.4f} | {y_stats['train']['max']:.4f} | {y_stats['train']['nan']} |
| VALIDATION | {y_stats['validation']['count']} | {y_stats['validation']['mean']:.4f} | {y_stats['validation']['median']:.4f} | {y_stats['validation']['std']:.4f} | {y_stats['validation']['min']:.4f} | {y_stats['validation']['max']:.4f} | {y_stats['validation']['nan']} |
| TEST | {y_stats['test']['count']} | {y_stats['test']['mean']:.4f} | {y_stats['test']['median']:.4f} | {y_stats['test']['std']:.4f} | {y_stats['test']['min']:.4f} | {y_stats['test']['max']:.4f} | {y_stats['test']['nan']} |

## NaN handling

Train X: {nan_train['nan_cells']} NaN cells in {nan_train['rows_with_nan']} rows across {nan_train['features_with_nan']} features.
Validation/test X: 0 NaN.

These are almost entirely lag warm-up (`lag_24` / `lag_48` / `lag_168`) plus a few extra historical-load/generation NaNs that were already in the source series. Rows were **not** dropped.

**Strategy**

- Ridge: column-median imputation **fit on X_train only**, then standardize **fit on imputed X_train only**, then closed-form Ridge `(X'X + αI)w = X'y` (NumPy).
- HistGradientBoosting: native NaN support; no imputer (train NaNs stay missing).
- Naive lag predictors: use existing lag columns; validation/test have complete lags.

Validation/test were never used to compute imputer or scaler statistics.

## Comparison (validation for all; test only where allowed)

See `reports/baseline_model_comparison.csv`.

{_md_table(comparison)}

Ridge alpha grid (validation MAE only): {ridge_grid}. **Selected alpha = {ridge_alpha}**.

## Answers

1. **Naive 24-hour baseline (validation):** MAE={naive_val['MAE']:.4f}, RMSE={naive_val['RMSE']:.4f}, R2={naive_val['R2']:.4f}, sMAPE={naive_val['sMAPE']:.4f}, bias={naive_val['bias']:.4f}.  
   **Naive 24-hour baseline (test, pre-specified, not used for selection):** MAE={naive_test['MAE']:.4f}, RMSE={naive_test['RMSE']:.4f}, R2={naive_test['R2']:.4f}, sMAPE={naive_test['sMAPE']:.4f}, bias={naive_test['bias']:.4f}.
2. **Does ML beat naive lag-24 on validation MAE?** {"No" if naive_beats else "Yes"}. Winner MAE {val_mets['MAE']:.4f} vs naive {naive_val['MAE']:.4f}.
3. **Best on VALIDATION:** {selected}
4. **Validation MAE:** {val_mets['MAE']:.4f}
5. **Validation RMSE:** {val_mets['RMSE']:.4f}
6. **Validation R2:** {val_mets['R2']:.4f}
7. **Validation sMAPE:** {val_mets['sMAPE']:.4f}
8. **Validation bias (mean y_pred − y_true):** {val_mets['bias']:.4f} ({"overprediction" if val_mets['bias']>0 else "underprediction" if val_mets['bias']<0 else "none"})
9. **Hour/month variation:** worst hours {worst_hours['hour'].tolist()} (MAE {worst_hours['MAE'].round(3).tolist()}); best hours {best_hours['hour'].tolist()}; worst months {worst_months.head(3)['month'].tolist()}; worst weekdays {worst_days.head(3)['weekday'].tolist()}. Diagnostic only; model not changed.
10. **Selected baseline:** {selected} (lowest validation MAE).
11. **Untouched TEST ({selected}):** MAE={test_mets['MAE']:.4f}, RMSE={test_mets['RMSE']:.4f}, R2={test_mets['R2']:.4f}, sMAPE={test_mets['sMAPE']:.4f}, bias={test_mets['bias']:.4f}.

## Error analysis (selected model, validation)

Worst hours:

{_md_table(worst_hours, floatfmt=".4f")}

MAE by month:

{_md_table(month_err, floatfmt=".4f")}

MAE by weekday (Monday=0):

{_md_table(weekday_err, floatfmt=".4f")}

## Reproducibility

`random_state = {RANDOM_STATE}` on HistGradientBoosting. Ridge and naive are deterministic.

| file | md5 |
|---|---|
| baseline_validation_predictions.parquet | {hashes['val_pred']} |
| baseline_test_predictions.parquet | {hashes['test_pred']} |
| baseline_model_comparison.csv | {hashes['cmp']} |
"""
    REPORT_MD.write_text(text, encoding="utf-8")


def run() -> dict[str, Any]:
    before = snapshot(PROTECTED)
    frames = load_splits()
    x_train, y_train, _ = xy(frames["train"])
    x_val, y_val, ts_val = xy(frames["validation"])
    x_test, y_test, ts_test = xy(frames["test"])

    y_stats = {k: target_stats(frames[k][TARGET_COL]) for k in ("train", "validation", "test")}
    nan_train = nan_profile(x_train)
    nan_val = nan_profile(x_val)
    nan_test = nan_profile(x_test)
    if nan_val["nan_cells"] != 0 or nan_test["nan_cells"] != 0:
        raise BaselineError("Unexpected NaNs in validation/test features")
    if y_train.isna().any() or y_val.isna().any() or y_test.isna().any():
        raise BaselineError("Target contains NaN")

    print("=== TARGET DISTRIBUTION ===", flush=True)
    print(json.dumps(y_stats, indent=2), flush=True)
    print("=== TRAIN NaN PROFILE ===", flush=True)
    print(
        f"cells={nan_train['nan_cells']} rows={nan_train['rows_with_nan']} features={nan_train['features_with_nan']}",
        flush=True,
    )

    print("Fitting train-only median imputer and scaler...", flush=True)
    prep = TrainOnlyPreprocessor().fit(x_train)

    records: list[dict[str, Any]] = []
    val_preds: dict[str, np.ndarray] = {}

    naive_names = {
        "Naive Lag-24": LAG24,
        "Mean Lag-24/48": MEAN24_48,
        "Mean Lag-24/48/168": MEAN24_48_168,
    }
    for name, col in naive_names.items():
        pred = x_val[col].to_numpy(dtype=float)
        if np.isnan(pred).any():
            raise BaselineError(f"{name} has NaN on validation")
        mets = regression_metrics(y_val, pred)
        records.append(row(name, "validation", mets))
        val_preds[name] = pred

    naive_test_pred = x_test[LAG24].to_numpy(dtype=float)
    if np.isnan(naive_test_pred).any():
        raise BaselineError("Naive Lag-24 has NaN on test")
    naive_test_mets = regression_metrics(y_test, naive_test_pred)
    records.append(row("Naive Lag-24", "test", naive_test_mets))

    ridge_alpha, ridge_val, ridge_grid, xtr_scaled, ytr_np = ridge_select(
        x_train, y_train, x_val, y_val, prep
    )
    records.append(row("Ridge", "validation", regression_metrics(y_val, ridge_val)))
    val_preds["Ridge"] = ridge_val
    print("Ridge validation grid:", ridge_grid, "selected_alpha", ridge_alpha, flush=True)

    print("Fitting HistGradientBoosting on train only...", flush=True)
    hgb = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE,
        early_stopping=False,
    )
    hgb.fit(x_train, y_train)
    print("HistGradientBoosting fit complete.", flush=True)
    hgb_val = hgb.predict(x_val)
    records.append(row("HistGradientBoosting", "validation", regression_metrics(y_val, hgb_val)))
    val_preds["HistGradientBoosting"] = hgb_val

    val_table = pd.DataFrame([r for r in records if r["split"] == "validation"])
    best_idx = val_table["MAE"].idxmin()
    selected = str(val_table.loc[best_idx, "model"])
    val_mets = regression_metrics(y_val, val_preds[selected])

    if selected == "Naive Lag-24":
        test_pred = naive_test_pred
        test_mets = naive_test_mets
    elif selected == "Mean Lag-24/48":
        test_pred = x_test[MEAN24_48].to_numpy(dtype=float)
        test_mets = regression_metrics(y_test, test_pred)
        records.append(row(selected, "test", test_mets))
    elif selected == "Mean Lag-24/48/168":
        test_pred = x_test[MEAN24_48_168].to_numpy(dtype=float)
        test_mets = regression_metrics(y_test, test_pred)
        records.append(row(selected, "test", test_mets))
    elif selected == "Ridge":
        test_pred = ridge_fit_predict(xtr_scaled, ytr_np, prep.transform(x_test), ridge_alpha)
        test_mets = regression_metrics(y_test, test_pred)
        records.append(row("Ridge", "test", test_mets))
    elif selected == "HistGradientBoosting":
        test_pred = hgb.predict(x_test)
        test_mets = regression_metrics(y_test, test_pred)
        records.append(row("HistGradientBoosting", "test", test_mets))
    else:
        raise BaselineError(f"Unknown selected model: {selected}")

    comparison = pd.DataFrame(records)
    comparison = comparison[["model", "split", "MAE", "RMSE", "R2", "sMAPE", "bias"]]
    REPORTS.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(CMP_CSV, index=False)

    y_val_np = y_val.to_numpy(dtype=float)
    residual_val = val_preds[selected] - y_val_np
    val_pred_df = pd.DataFrame(
        {
            "timestamp_utc": ts_val.to_numpy(),
            "y_true": y_val_np,
            "y_pred": val_preds[selected],
            "residual": residual_val,
        }
    )
    val_pred_df.to_parquet(PRED_VAL_PATH, index=False)

    test_pred_df = pd.DataFrame(
        {
            "timestamp_utc": ts_test.to_numpy(),
            "y_true": y_test.to_numpy(dtype=float),
            "y_pred": np.asarray(test_pred, dtype=float),
            "residual": np.asarray(test_pred, dtype=float) - y_test.to_numpy(dtype=float),
        }
    )
    test_pred_df.to_parquet(PRED_TEST_PATH, index=False)

    err = frames["validation"][["hour", "day_of_week", "month"]].copy()
    err["abs_err"] = np.abs(residual_val)
    hour_err = err.groupby("hour", sort=True)["abs_err"].mean().rename("MAE").reset_index()
    month_err = err.groupby("month", sort=True)["abs_err"].mean().rename("MAE").reset_index()
    weekday_err = err.groupby("day_of_week", sort=True)["abs_err"].mean().rename("MAE").reset_index()
    weekday_err = weekday_err.rename(columns={"day_of_week": "weekday"})
    hour_err.to_csv(HOUR_CSV, index=False)
    month_err.to_csv(MONTH_CSV, index=False)
    weekday_err.to_csv(WEEKDAY_CSV, index=False)

    after = snapshot(PROTECTED)
    if after != before:
        raise BaselineError("A protected file was modified")

    hashes = {
        "val_pred": md5(PRED_VAL_PATH),
        "test_pred": md5(PRED_TEST_PATH),
        "cmp": md5(CMP_CSV),
    }
    status = "PASS"
    write_report(
        y_stats,
        nan_train,
        comparison,
        selected,
        ridge_alpha,
        ridge_grid,
        val_mets,
        test_mets,
        regression_metrics(y_val, val_preds["Naive Lag-24"]),
        naive_test_mets,
        hour_err,
        month_err,
        weekday_err,
        hashes,
        status,
    )
    if snapshot(PROTECTED) != before:
        raise BaselineError("A protected file was modified after reporting")

    return {
        "status": status,
        "selected": selected,
        "ridge_alpha": ridge_alpha,
        "y_stats": y_stats,
        "nan_train": {k: nan_train[k] for k in ("nan_cells", "rows_with_nan", "features_with_nan")},
        "validation": val_mets,
        "test_selected": test_mets,
        "naive_validation": regression_metrics(y_val, val_preds["Naive Lag-24"]),
        "naive_test": naive_test_mets,
        "hashes": hashes,
        "comparison": comparison.to_dict(orient="records"),
        "test_used_for_selection": False,
    }


if __name__ == "__main__":
    result = run()
    print("=== VALIDATION COMPARISON ===")
    print(pd.DataFrame(result["comparison"]).to_string(index=False))
    print("selected_on_validation:", result["selected"])
    print("ridge_alpha:", result["ridge_alpha"])
    print("validation_winner_metrics:", json.dumps(result["validation"], indent=2))
    print("test_winner_metrics:", json.dumps(result["test_selected"], indent=2))
    print("test_used_for_selection:", result["test_used_for_selection"])
    print(f"BASELINE_MODELING = {result['status']}")
