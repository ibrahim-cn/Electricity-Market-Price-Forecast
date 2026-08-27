"""Ridge alpha walk-forward tuning on TRAIN+VALIDATION only.

Does not load test.parquet. Does not overwrite prior walk-forward/baseline reports.
Does not modify source parquets or feature engineering.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "splits" / "train.parquet"
VAL_PATH = ROOT / "data" / "processed" / "splits" / "validation.parquet"
TEST_PATH = ROOT / "data" / "processed" / "splits" / "test.parquet"
MERGED_PATH = ROOT / "data" / "processed" / "merged" / "merged_energy_weather.parquet"
FEATURES_PATH = ROOT / "data" / "processed" / "features" / "model_features.parquet"
PRED_PATH = ROOT / "data" / "processed" / "predictions" / "ridge_walk_forward_validation_predictions.parquet"
REPORTS = ROOT / "reports" / "modeling"
REPORTS_FINAL = ROOT / "reports" / "final"

CMP_CSV = REPORTS / "ridge_alpha_comparison.csv"
FOLD_CSV = REPORTS / "ridge_tuning_fold_results.csv"
HOUR_CSV = REPORTS / "ridge_error_by_hour.csv"
MONTH_CSV = REPORTS / "ridge_error_by_month.csv"
WEEKDAY_CSV = REPORTS / "ridge_error_by_weekday.csv"
QUANTILE_CSV = REPORTS / "ridge_error_by_price_quantile.csv"
REPORT_MD = REPORTS / "ridge_tuning.md"

TARGET_COL = "price day ahead"
ID_COL = "timestamp_utc"
FORBIDDEN_IN_X = {TARGET_COL, "price actual", ID_COL, "time"}
N_FEATURES = 184
RANDOM_STATE = 42
CLOSE_MAE = 0.05
TRAIN_CUTS = (0.50, 0.60, 0.70, 0.80)
ALPHA_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
QUANTILE_LABELS = ("P25_below", "P25_P50", "P50_P75", "P75_above")

PROTECTED = (
    ROOT / "data" / "raw" / "energy_dataset.csv",
    ROOT / "data" / "raw" / "weather_features.csv",
    MERGED_PATH,
    FEATURES_PATH,
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
)

REQUIRED_PROTECTED = (MERGED_PATH, FEATURES_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH)

NEW_OUTPUTS = (CMP_CSV, FOLD_CSV, HOUR_CSV, MONTH_CSV, WEEKDAY_CSV, QUANTILE_CSV, REPORT_MD, PRED_PATH)


class RidgeTuningError(ValueError):
    pass


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def snapshot(paths: tuple[Path, ...]) -> dict[str, str]:
    return {str(p): md5(p) for p in paths if p.exists()}


def assert_not_test(path: Path) -> None:
    resolved = Path(path).resolve()
    if resolved == TEST_PATH.resolve() or resolved.name == "test.parquet":
        raise RidgeTuningError("TEST SET IS LOCKED and must not be loaded during Ridge tuning")


def read_parquet_locked(path: Path) -> pd.DataFrame:
    """Every parquet read in this script goes through the test lock."""
    assert_not_test(path)
    if "test" in Path(path).name.lower() and Path(path).resolve() == TEST_PATH.resolve():
        raise RidgeTuningError("TEST SET IS LOCKED")
    return pd.read_parquet(path)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if not mask.any():
        return float("nan")
    return float(100.0 * np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]))


def metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise RidgeTuningError("Prediction length mismatch")
    residual = yp - yt
    return {
        "MAE": float(mean_absolute_error(yt, yp)),
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "R2": float(r2_score(yt, yp)),
        "sMAPE": smape(yt, yp),
        "bias": float(np.mean(residual)),
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual, ddof=0)),
    }


class FoldPreprocessor:
    """Median impute + standardize. Statistics from the fold training block only."""

    def fit(self, x_train: pd.DataFrame) -> "FoldPreprocessor":
        arr = x_train.to_numpy(dtype=float)
        self.medians_ = np.nanmedian(arr, axis=0)
        filled = self._impute(arr)
        self.mean_ = filled.mean(axis=0)
        scale = filled.std(axis=0, ddof=0)
        scale[scale == 0] = 1.0
        self.scale_ = scale
        self.n_train_rows_ = int(len(x_train))
        return self

    def _impute(self, arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        for j, med in enumerate(self.medians_):
            mask = np.isnan(out[:, j])
            if mask.any():
                out[mask, j] = 0.0 if not np.isfinite(med) else float(med)
        return out

    def transform_linear(self, x: pd.DataFrame) -> np.ndarray:
        filled = self._impute(x.to_numpy(dtype=float))
        return (filled - self.mean_) / self.scale_


def assert_preproc_train_only(prep: FoldPreprocessor, x_train: pd.DataFrame, x_val: pd.DataFrame) -> None:
    tr = x_train.to_numpy(dtype=float)
    expected_med = np.nanmedian(tr, axis=0)
    if not np.allclose(prep.medians_, expected_med, equal_nan=True, rtol=0, atol=0):
        raise RidgeTuningError("Preprocessor medians were not computed from fold train only")
    filled_tr = prep._impute(tr)
    if not np.allclose(prep.mean_, filled_tr.mean(axis=0), rtol=0, atol=1e-12):
        raise RidgeTuningError("Preprocessor means were not computed from fold train only")
    concat = np.vstack([tr, x_val.to_numpy(dtype=float)])
    concat_med = np.nanmedian(concat, axis=0)
    if not np.array_equal(np.isnan(prep.medians_), np.isnan(expected_med)):
        raise RidgeTuningError("Preprocessor NaN-median mask leaked")
    # If train and concat medians differ, using concat would fail the first allclose.
    _ = concat_med
    if prep.n_train_rows_ != len(x_train):
        raise RidgeTuningError("Preprocessor row count does not match fold train")


def ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_pred: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form Ridge: (X'X + αI)w = X'y, intercept via centered y. Deterministic NumPy."""
    y_mean = y_train.mean()
    yc = y_train - y_mean
    xtx = x_train.T @ x_train
    xtx = xtx + float(alpha) * np.eye(xtx.shape[0])
    w = np.linalg.solve(xtx, x_train.T @ yc)
    return x_pred @ w + y_mean


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in {ID_COL, TARGET_COL}]
    leak = FORBIDDEN_IN_X.intersection(df.columns) - {ID_COL, TARGET_COL}
    if leak:
        raise RidgeTuningError(f"target leakage columns present: {sorted(leak)}")
    if "price actual" in df.columns:
        raise RidgeTuningError("price actual present")
    if len(cols) != N_FEATURES:
        raise RidgeTuningError(f"expected {N_FEATURES} features, got {len(cols)}")
    return cols


def load_dev_frame() -> tuple[pd.DataFrame, list[str]]:
    train = read_parquet_locked(TRAIN_PATH)
    val = read_parquet_locked(VAL_PATH)
    tr_cols = feature_columns(train)
    va_cols = feature_columns(val)
    if tr_cols != va_cols:
        raise RidgeTuningError("train and validation feature columns differ")
    df = pd.concat([train, val], axis=0, ignore_index=True)
    df = df.sort_values(ID_COL, kind="mergesort").reset_index(drop=True)
    if feature_columns(df) != tr_cols:
        raise RidgeTuningError("concatenated feature columns changed")
    ts = df[ID_COL]
    if ts.dt.tz is None or str(ts.dt.tz) != "UTC":
        raise RidgeTuningError("timestamps must be UTC-aware")
    if ts.duplicated().any():
        raise RidgeTuningError("duplicate timestamps in train+val")
    if not bool(ts.is_monotonic_increasing):
        raise RidgeTuningError("train+val is not chronological")
    diffs = ts.diff().dropna()
    if not bool((diffs == pd.Timedelta(hours=1)).all()):
        raise RidgeTuningError("train+val is not strictly hourly")
    if df[TARGET_COL].isna().any():
        raise RidgeTuningError("target has NaN in train+val")
    return df, tr_cols


def make_folds(df: pd.DataFrame) -> list[dict[str, Any]]:
    n = len(df)
    cuts = [int(n * f) for f in TRAIN_CUTS] + [n]
    folds: list[dict[str, Any]] = []
    seen_val: set[pd.Timestamp] = set()
    for i, (tr_end, va_end) in enumerate(zip(cuts[:-1], cuts[1:]), start=1):
        train = df.iloc[:tr_end]
        val = df.iloc[tr_end:va_end]
        if len(train) == 0 or len(val) == 0:
            raise RidgeTuningError(f"Fold {i} is empty")
        if train[ID_COL].max() >= val[ID_COL].min():
            raise RidgeTuningError(f"Fold {i}: train timestamp is not strictly before validation")
        overlap = set(train[ID_COL]) & set(val[ID_COL])
        if overlap:
            raise RidgeTuningError(f"Fold {i} train/validation timestamp overlap")
        val_ts = set(val[ID_COL])
        if seen_val & val_ts:
            raise RidgeTuningError(f"Fold {i} validation overlaps a previous validation block")
        seen_val |= val_ts
        folds.append(
            {
                "fold": i,
                "train": train,
                "val": val,
                "n_train": int(len(train)),
                "n_val": int(len(val)),
                "train_start": train[ID_COL].iloc[0].isoformat(),
                "train_end": train[ID_COL].iloc[-1].isoformat(),
                "val_start": val[ID_COL].iloc[0].isoformat(),
                "val_end": val[ID_COL].iloc[-1].isoformat(),
                "train_frac": round(tr_end / n, 4),
            }
        )
    for a, b in zip(folds, folds[1:]):
        if a["val"][ID_COL].max() >= b["val"][ID_COL].min():
            raise RidgeTuningError("validation blocks are not chronological")
        if a["n_train"] >= b["n_train"]:
            raise RidgeTuningError("expanding window did not grow")
    return folds


def select_alpha(summary: pd.DataFrame) -> float:
    ranked = summary.sort_values(["mean_MAE", "std_MAE", "abs_mean_bias", "alpha"]).reset_index(drop=True)
    best_mae = float(ranked.loc[0, "mean_MAE"])
    close = ranked[ranked["mean_MAE"] <= best_mae + CLOSE_MAE].copy()
    close = close.sort_values(["mean_MAE", "std_MAE", "abs_mean_bias", "alpha"]).reset_index(drop=True)
    return float(close.loc[0, "alpha"])


def md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, rec in df.iterrows():
        cells = []
        for c in cols:
            v = rec[c]
            if isinstance(v, (float, np.floating)):
                cells.append("nan" if not np.isfinite(v) else format(float(v), floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_outputs(
    comparison: pd.DataFrame,
    fold_df: pd.DataFrame,
    hour_err: pd.DataFrame,
    month_err: pd.DataFrame,
    weekday_err: pd.DataFrame,
    q_err: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    comparison.sort_values("alpha").to_csv(CMP_CSV, index=False)
    fold_df.sort_values(["alpha", "fold"]).to_csv(FOLD_CSV, index=False)
    hour_err.sort_values("hour").to_csv(HOUR_CSV, index=False)
    month_err.sort_values("month").to_csv(MONTH_CSV, index=False)
    weekday_err.sort_values("weekday").to_csv(WEEKDAY_CSV, index=False)
    q_err.to_csv(QUANTILE_CSV, index=False)
    pred_out = pred_df.sort_values(["fold", ID_COL], kind="mergesort").reset_index(drop=True)
    pred_out.to_parquet(PRED_PATH, index=False)


def output_hashes() -> dict[str, str]:
    return {
        "comparison": md5(CMP_CSV),
        "folds": md5(FOLD_CSV),
        "hour": md5(HOUR_CSV),
        "month": md5(MONTH_CSV),
        "weekday": md5(WEEKDAY_CSV),
        "quantile": md5(QUANTILE_CSV),
        "predictions": md5(PRED_PATH),
        "report": md5(REPORT_MD) if REPORT_MD.exists() else "",
    }


def write_report(
    folds: list[dict[str, Any]],
    comparison: pd.DataFrame,
    fold_df: pd.DataFrame,
    selected: float,
    hour_err: pd.DataFrame,
    month_err: pd.DataFrame,
    weekday_err: pd.DataFrame,
    q_err: pd.DataFrame,
    residual_mean: float,
    residual_std: float,
    hashes: dict[str, str],
    before: dict[str, str],
    after: dict[str, str],
    status: str,
    leakage: str,
    protected_ok: bool,
    repro: str,
    test_used: bool,
) -> None:
    sel = comparison[np.isclose(comparison["alpha"], selected)].iloc[0]
    sel_folds = fold_df[np.isclose(fold_df["alpha"], selected)].sort_values("fold")
    q_high = q_err[q_err["quantile"].astype(str) == "P75_above"].iloc[0]
    q_low = q_err[q_err["quantile"].astype(str) == "P25_below"].iloc[0]
    fold_meta = pd.DataFrame(
        [
            {k: f[k] for k in ("fold", "n_train", "n_val", "train_start", "train_end", "val_start", "val_end", "train_frac")}
            for f in folds
        ]
    )
    prot_rows = []
    for p in REQUIRED_PROTECTED:
        key = str(p)
        prot_rows.append(
            {
                "file": p.name,
                "md5_before": before.get(key, ""),
                "md5_after": after.get(key, ""),
                "unchanged": before.get(key) == after.get(key),
            }
        )
    prot_tbl = pd.DataFrame(prot_rows)

    text = f"""# Ridge Alpha Tuning

**RIDGE_TUNING = {status}**

TEST parquet was never loaded. Alpha was selected on TRAIN+VALIDATION expanding-window folds only.
No locked-test metric was computed. Feature engineering was not changed.

`random_state = {RANDOM_STATE}` is recorded for pipeline consistency. Ridge itself is a closed-form
NumPy solve and does not use a random number generator.

## 1. Alpha grid

{list(ALPHA_GRID)}

Grid was chosen before looking at any test metric. It densifies the previous walk-forward Ridge set
around the winning region (0.01) without using the locked test set.

## 2. Walk-forward methodology

Same expanding-window scheme as `src/walk_forward_validation.py`:

- Combine official TRAIN then VALIDATION, sort by `timestamp_utc` (UTC, hourly, no shuffle).
- Fold train fractions: {list(TRAIN_CUTS)}; each validation block is the next chronological remainder up to the next cut (fold 4 uses the final 20%).
- For every fold and every alpha:
  1. Fit median impute + standard scale on **fold train only**.
  2. Transform fold validation with those statistics.
  3. Fit Ridge `(X'X + αI)w = X'y` (centered y intercept).
  4. Score MAE, RMSE, R², sMAPE, bias on that fold's validation block.
- Primary selection: mean walk-forward MAE.
- If mean MAE values are within {CLOSE_MAE}: lower MAE std, then lower |mean bias|, then smaller alpha.

## 3. Fold timestamps

{md_table(fold_meta)}

Each fold satisfies `max(train timestamp) < min(validation timestamp)`. Validation blocks do not overlap.

## 4–6. Alpha comparison (mean ± std over 4 folds)

{md_table(comparison)}

## 7. Selected alpha

**BEST_ALPHA = {selected}**

- BEST_MEAN_MAE = {sel['mean_MAE']:.6f}
- BEST_MAE_STD = {sel['std_MAE']:.6f}
- mean RMSE = {sel['mean_RMSE']:.4f} (std {sel['std_RMSE']:.4f})
- mean R2 = {sel['mean_R2']:.4f}
- mean sMAPE = {sel['mean_sMAPE']:.4f}
- mean bias = {sel['mean_bias']:.4f}

Selected-alpha fold detail:

{md_table(sel_folds[['fold','val_start','val_end','y_mean','y_std','MAE','RMSE','R2','sMAPE','bias']])}

## 8. High-price regime (selected alpha, pooled walk-forward validation blocks)

Pooled residual mean = {residual_mean:.4f}  
Pooled residual std = {residual_std:.4f}

### Price quantiles (P25 / P50 / P75 of pooled y_true)

{md_table(q_err)}

HIGH_PRICE_BIAS (P75_above) = {float(q_high['bias']):.4f}  
P75_above MAE = {float(q_high['MAE']):.4f} vs P25_below MAE = {float(q_low['MAE']):.4f}

{"The selected Ridge still systematically underpredicts the upper price quartile." if float(q_high['bias']) < -1 else "Upper-quartile bias is not a large systematic underprediction."}

### Hour / weekday / month MAE

Worst hours: {hour_err.sort_values('MAE', ascending=False).head(5)['hour'].tolist()}  
Worst weekdays (Mon=0 … Sun=6): {weekday_err.sort_values('MAE', ascending=False).head(3)['weekday'].tolist()}  
Worst months: {month_err.sort_values('MAE', ascending=False).head(3)['month'].tolist()}

Hourly MAE range: {float(hour_err['MAE'].min()):.3f}–{float(hour_err['MAE'].max()):.3f}.

Weekday table is in `reports/ridge_error_by_weekday.csv` (extra diagnostic; not used for alpha selection).

## 9. Test lock

TEST_USED_FOR_TUNING = {str(test_used).upper()}

`read_parquet_locked` raises if the path is `data/processed/test.parquet`.
No test prediction, MAE, RMSE, R², sMAPE, or bias was computed.

## 10. Protected file hashes

LEAKAGE_CHECK = {leakage}  
PROTECTED_FILES_UNCHANGED = {str(protected_ok).upper()}

{md_table(prot_tbl, floatfmt="")}

## 11. Reproducibility

REPRODUCIBILITY = {repro}

The pipeline is executed twice in one process. Comparison / fold / prediction / error hashes must match.

| file | md5 |
|---|---|
| ridge_alpha_comparison.csv | {hashes['comparison']} |
| ridge_tuning_fold_results.csv | {hashes['folds']} |
| ridge_walk_forward_validation_predictions.parquet | {hashes['predictions']} |
| ridge_error_by_hour.csv | {hashes['hour']} |
| ridge_error_by_month.csv | {hashes['month']} |
| ridge_error_by_weekday.csv | {hashes['weekday']} |
| ridge_error_by_price_quantile.csv | {hashes['quantile']} |

## 12. Final status

RIDGE_TUNING = {status}
TEST_USED_FOR_TUNING = {str(test_used).upper()}
LEAKAGE_CHECK = {leakage}
PROTECTED_FILES_UNCHANGED = {str(protected_ok).upper()}
REPRODUCIBILITY = {repro}
BEST_ALPHA = {selected}
BEST_MEAN_MAE = {sel['mean_MAE']:.6f}
BEST_MAE_STD = {sel['std_MAE']:.6f}
HIGH_PRICE_BIAS = {float(q_high['bias']):.6f}

This stage does **not** evaluate a production/final model on the locked test set.
"""
    REPORT_MD.write_text(text, encoding="utf-8")


def run_once() -> dict[str, Any]:
    before = snapshot(PROTECTED)
    df, feat_cols = load_dev_frame()
    folds = make_folds(df)

    fold_records: list[dict[str, Any]] = []
    pred_parts: list[pd.DataFrame] = []

    for fold in folds:
        x_tr = fold["train"][feat_cols]
        y_tr = fold["train"][TARGET_COL].to_numpy(dtype=float)
        x_va = fold["val"][feat_cols]
        y_va = fold["val"][TARGET_COL]
        if list(x_tr.columns) != feat_cols or list(x_va.columns) != feat_cols:
            raise RidgeTuningError("feature columns changed inside a fold")
        prep = FoldPreprocessor().fit(x_tr)
        assert_preproc_train_only(prep, x_tr, x_va)
        xtr = prep.transform_linear(x_tr)
        xva = prep.transform_linear(x_va)
        y_mean = float(y_va.mean())
        y_std = float(y_va.std(ddof=0))
        print(f"=== Fold {fold['fold']} train={fold['n_train']} val={fold['n_val']} ===", flush=True)
        for alpha in ALPHA_GRID:
            pred = ridge_predict(xtr, y_tr, xva, alpha)
            mets = metrics(y_va, pred)
            fold_records.append(
                {
                    "fold": fold["fold"],
                    "alpha": float(alpha),
                    "n_train": fold["n_train"],
                    "n_val": fold["n_val"],
                    "train_start": fold["train_start"],
                    "train_end": fold["train_end"],
                    "val_start": fold["val_start"],
                    "val_end": fold["val_end"],
                    "y_mean": y_mean,
                    "y_std": y_std,
                    **mets,
                }
            )
            pred_parts.append(
                pd.DataFrame(
                    {
                        ID_COL: fold["val"][ID_COL].to_numpy(),
                        "y_true": y_va.to_numpy(dtype=float),
                        "y_pred": np.asarray(pred, dtype=float),
                        "residual": np.asarray(pred, dtype=float) - y_va.to_numpy(dtype=float),
                        "fold": fold["fold"],
                        "alpha": float(alpha),
                        "hour": fold["val"]["hour"].to_numpy(),
                        "weekday": fold["val"]["day_of_week"].to_numpy(),
                        "month": fold["val"]["month"].to_numpy(),
                    }
                )
            )

    fold_df = pd.DataFrame(fold_records).sort_values(["alpha", "fold"]).reset_index(drop=True)
    summary_rows = []
    for alpha, g in fold_df.groupby("alpha", sort=True):
        summary_rows.append(
            {
                "alpha": float(alpha),
                "mean_MAE": float(g["MAE"].mean()),
                "std_MAE": float(g["MAE"].std(ddof=0)),
                "mean_RMSE": float(g["RMSE"].mean()),
                "std_RMSE": float(g["RMSE"].std(ddof=0)),
                "mean_R2": float(g["R2"].mean()),
                "mean_sMAPE": float(g["sMAPE"].mean()),
                "mean_bias": float(g["bias"].mean()),
                "abs_mean_bias": float(abs(g["bias"].mean())),
                "n_folds": int(len(g)),
            }
        )
    comparison = pd.DataFrame(summary_rows).sort_values("alpha").reset_index(drop=True)
    selected = select_alpha(comparison)

    all_pred = pd.concat(pred_parts, axis=0, ignore_index=True)
    all_pred = all_pred.sort_values(["alpha", "fold", ID_COL], kind="mergesort").reset_index(drop=True)
    sel_pred = all_pred[np.isclose(all_pred["alpha"], selected)].copy()
    sel_pred = sel_pred.sort_values(["fold", ID_COL], kind="mergesort").reset_index(drop=True)
    y = sel_pred["y_true"].to_numpy(dtype=float)
    pred = sel_pred["y_pred"].to_numpy(dtype=float)
    residual = sel_pred["residual"].to_numpy(dtype=float)
    abs_err = np.abs(residual)
    hour_err = (
        pd.DataFrame({"hour": sel_pred["hour"], "MAE": abs_err})
        .groupby("hour", sort=True)["MAE"]
        .mean()
        .reset_index()
    )
    weekday_err = (
        pd.DataFrame({"weekday": sel_pred["weekday"], "MAE": abs_err})
        .groupby("weekday", sort=True)["MAE"]
        .mean()
        .reset_index()
    )
    month_err = (
        pd.DataFrame({"month": sel_pred["month"], "MAE": abs_err})
        .groupby("month", sort=True)["MAE"]
        .mean()
        .reset_index()
    )
    qs = pd.qcut(y, 4, labels=list(QUANTILE_LABELS))
    q_err = (
        pd.DataFrame({"quantile": qs, "MAE": abs_err, "bias": residual, "y_mean": y})
        .groupby("quantile", observed=True)
        .agg(
            MAE=("MAE", "mean"),
            bias=("bias", "mean"),
            residual_mean=("bias", "mean"),
            residual_std=("bias", lambda s: float(np.std(s.to_numpy(dtype=float), ddof=0))),
            y_mean=("y_mean", "mean"),
            n=("MAE", "size"),
        )
        .reset_index()
    )
    q_err["quantile"] = pd.Categorical(q_err["quantile"], categories=list(QUANTILE_LABELS), ordered=True)
    q_err = q_err.sort_values("quantile").reset_index(drop=True)
    q_err["residual_std"] = q_err["residual_std"].astype(float)

    pred_store = sel_pred[[ID_COL, "y_true", "y_pred", "residual", "fold", "alpha"]].copy()
    write_outputs(comparison, fold_df, hour_err, month_err, weekday_err, q_err, pred_store)

    after = snapshot(PROTECTED)
    if after != before:
        raise RidgeTuningError("A protected file was modified")
    leakage = "PASS"
    protected_ok = True
    test_used = False
    hashes = output_hashes()
    return {
        "comparison": comparison,
        "fold_df": fold_df,
        "hour_err": hour_err,
        "month_err": month_err,
        "weekday_err": weekday_err,
        "q_err": q_err,
        "pred": pred_store,
        "selected": selected,
        "folds": folds,
        "hashes": hashes,
        "before": before,
        "after": after,
        "leakage": leakage,
        "protected_ok": protected_ok,
        "test_used": test_used,
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual, ddof=0)),
        "n_dev_rows": int(len(df)),
        "feat_cols": feat_cols,
    }


def run() -> dict[str, Any]:
    first = run_once()
    hashes_first = {k: v for k, v in first["hashes"].items() if k != "report"}
    write_report(
        first["folds"],
        first["comparison"],
        first["fold_df"],
        first["selected"],
        first["hour_err"],
        first["month_err"],
        first["weekday_err"],
        first["q_err"],
        first["residual_mean"],
        first["residual_std"],
        first["hashes"],
        first["before"],
        first["after"],
        status="PASS",
        leakage=first["leakage"],
        protected_ok=first["protected_ok"],
        repro="PENDING",
        test_used=first["test_used"],
    )

    second = run_once()
    hashes_second = {k: v for k, v in second["hashes"].items() if k != "report"}
    repro = "PASS" if hashes_first == hashes_second else "FAIL"
    if second["before"] != second["after"] or first["before"] != first["after"]:
        raise RidgeTuningError("Protected files changed across runs")
    if second["selected"] != first["selected"]:
        raise RidgeTuningError("Selected alpha changed across runs")
    status = "PASS" if repro == "PASS" and second["leakage"] == "PASS" and second["protected_ok"] and not second["test_used"] else "FAIL"

    write_report(
        second["folds"],
        second["comparison"],
        second["fold_df"],
        second["selected"],
        second["hour_err"],
        second["month_err"],
        second["weekday_err"],
        second["q_err"],
        second["residual_mean"],
        second["residual_std"],
        {**second["hashes"], "report": ""},
        second["before"],
        second["after"],
        status=status,
        leakage=second["leakage"],
        protected_ok=second["protected_ok"],
        repro=repro,
        test_used=second["test_used"],
    )
    second["hashes"] = output_hashes()
    second["repro"] = repro
    second["status"] = status
    second["hashes_run1"] = hashes_first
    second["hashes_run2"] = hashes_second
    return second


if __name__ == "__main__":
    result = run()
    sel = result["comparison"][np.isclose(result["comparison"]["alpha"], result["selected"])].iloc[0]
    q_high = result["q_err"][result["q_err"]["quantile"].astype(str) == "P75_above"].iloc[0]
    print("=== RIDGE TUNING SUMMARY ===")
    print(result["comparison"].to_string(index=False))
    print(json.dumps(result["hashes_run1"], indent=2))
    print(json.dumps(result["hashes_run2"], indent=2))
    print(f"RIDGE_TUNING = {result['status']}")
    print(f"TEST_USED_FOR_TUNING = {str(result['test_used']).upper()}")
    print(f"LEAKAGE_CHECK = {result['leakage']}")
    print(f"PROTECTED_FILES_UNCHANGED = {str(result['protected_ok']).upper()}")
    print(f"REPRODUCIBILITY = {result['repro']}")
    print(f"BEST_ALPHA = {result['selected']}")
    print(f"BEST_MEAN_MAE = {sel['mean_MAE']:.6f}")
    print(f"BEST_MAE_STD = {sel['std_MAE']:.6f}")
    print(f"HIGH_PRICE_BIAS = {float(q_high['bias']):.6f}")
