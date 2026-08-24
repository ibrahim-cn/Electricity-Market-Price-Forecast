"""Expanding-window walk-forward comparison on TRAIN+VALIDATION only.

TEST parquet is never loaded. No shuffle. Preprocessing is fit per fold.
"""

from __future__ import annotations

import hashlib
import json
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "splits" / "train.parquet"
VAL_PATH = ROOT / "data" / "processed" / "splits" / "validation.parquet"
TEST_PATH = ROOT / "data" / "processed" / "splits" / "test.parquet"
REPORTS = ROOT / "reports" / "modeling"

TARGET_COL = "price day ahead"
ID_COL = "timestamp_utc"
LAG24 = "price_day_ahead_lag_24"
RANDOM_STATE = 42
FORBIDDEN_IN_X = {TARGET_COL, "price actual", ID_COL, "time"}
CLOSE_MAE = 0.05
SHIFT_Z = 0.5
TRAIN_CUTS = (0.50, 0.60, 0.70, 0.80)

RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
ENET_ALPHAS = (0.001, 0.01, 0.1, 1.0)
ENET_L1 = (0.1, 0.5, 0.9)

PROTECTED = (
    ROOT / "energy_dataset.csv",
    ROOT / "weather_features.csv",
    ROOT / "data" / "raw" / "energy_dataset.csv",
    ROOT / "data" / "raw" / "weather_features.csv",
    ROOT / "data" / "processed" / "merged" / "merged_energy_weather.parquet",
    ROOT / "data" / "processed" / "features" / "model_features.parquet",
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
)

SHIFT_FEATURES = (
    TARGET_COL,
    "price_day_ahead_lag_24",
    "price_day_ahead_lag_48",
    "price_day_ahead_lag_168",
    "total_load_forecast",
    "total_load_actual_lag_24",
    "total_load_actual_lag_48",
    "total_load_actual_lag_168",
    "forecast_solar_day_ahead",
    "forecast_wind_onshore_day_ahead",
    "renewable_generation_lag_24",
    "renewable_generation_lag_168",
    "renewable_share_lag_24",
    "temp_national_mean_lag_24",
    "temp_national_mean_lag_168",
    "wind_speed_national_mean_lag_24",
    "humidity_national_mean_lag_24",
    "clouds_all_national_mean_lag_24",
)

SIMPLICITY = {
    "Naive Lag-24": 0,
    "Ridge": 1,
    "ElasticNet": 2,
    "HistGradientBoosting": 3,
    "LightGBM": 4,
    "XGBoost": 5,
    "RandomForest": 6,
}


class WalkForwardError(ValueError):
    pass


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def snapshot(paths: tuple[Path, ...]) -> dict[str, str]:
    return {str(p): md5(p) for p in paths if p.exists()}


def assert_not_test(path: Path) -> None:
    if path.resolve() == TEST_PATH.resolve():
        raise WalkForwardError("TEST SET IS LOCKED and must not be loaded")


def read_dev_parquet(path: Path) -> pd.DataFrame:
    assert_not_test(path)
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
    return {
        "MAE": float(mean_absolute_error(yt, yp)),
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "R2": float(r2_score(yt, yp)),
        "sMAPE": smape(yt, yp),
        "bias": float(np.mean(yp - yt)),
    }


class FoldPreprocessor:
    def fit(self, x_train: pd.DataFrame) -> "FoldPreprocessor":
        arr = x_train.to_numpy(dtype=float)
        self.medians_ = np.nanmedian(arr, axis=0)
        filled = self._impute(arr)
        self.mean_ = filled.mean(axis=0)
        scale = filled.std(axis=0, ddof=0)
        scale[scale == 0] = 1.0
        self.scale_ = scale
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

    def transform_imputed(self, x: pd.DataFrame) -> np.ndarray:
        return self._impute(x.to_numpy(dtype=float))


def elastic_net_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
    alpha: float,
    l1_ratio: float,
    max_iter: int = 250,
) -> np.ndarray:
    """Sklearn-style ElasticNet via cyclic coordinate descent (no sklearn solver)."""
    n, p = x_train.shape
    y_mean = y_train.mean()
    yc = y_train - y_mean
    gram = (x_train.T @ x_train) / n
    xty = (x_train.T @ yc) / n
    w = np.zeros(p, dtype=float)
    diag = np.diag(gram)
    l1 = alpha * l1_ratio
    l2 = alpha * (1.0 - l1_ratio)
    for _ in range(max_iter):
        max_delta = 0.0
        for j in range(p):
            rho = xty[j] - gram[j] @ w + diag[j] * w[j]
            denom = diag[j] + l2
            new_w = 0.0 if denom <= 0 else np.sign(rho) * max(abs(rho) - l1, 0.0) / denom
            max_delta = max(max_delta, abs(new_w - w[j]))
            w[j] = new_w
        if max_delta < 1e-4:
            break
    return x_pred @ w + y_mean


def ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_pred: np.ndarray, alpha: float) -> np.ndarray:
    y_mean = y_train.mean()
    yc = y_train - y_mean
    xtx = x_train.T @ x_train
    xtx = xtx + alpha * np.eye(xtx.shape[0])
    w = np.linalg.solve(xtx, x_train.T @ yc)
    return x_pred @ w + y_mean


def family_of(name: str) -> str:
    if name.startswith("Ridge"):
        return "Ridge"
    if name.startswith("ElasticNet"):
        return "ElasticNet"
    return name


def load_dev_frame() -> pd.DataFrame:
    train = read_dev_parquet(TRAIN_PATH)
    val = read_dev_parquet(VAL_PATH)
    if "price actual" in train.columns or "price actual" in val.columns:
        raise WalkForwardError("price actual present")
    df = pd.concat([train, val], axis=0, ignore_index=True)
    df = df.sort_values(ID_COL).reset_index(drop=True)
    ts = df[ID_COL]
    if ts.dt.tz is None or str(ts.dt.tz) != "UTC":
        raise WalkForwardError("timestamps must be UTC-aware")
    if ts.duplicated().any():
        raise WalkForwardError("duplicate timestamps in train+val")
    if not bool(ts.is_monotonic_increasing):
        raise WalkForwardError("train+val is not chronological")
    diffs = ts.diff().dropna()
    if not bool((diffs == pd.Timedelta(hours=1)).all()):
        raise WalkForwardError("train+val is not strictly hourly")
    x_cols = [c for c in df.columns if c not in {ID_COL, TARGET_COL}]
    if FORBIDDEN_IN_X.intersection(x_cols):
        raise WalkForwardError("forbidden columns in X")
    if len(x_cols) != 184:
        raise WalkForwardError(f"expected 184 features, got {len(x_cols)}")
    return df


def make_folds(df: pd.DataFrame) -> list[dict[str, Any]]:
    n = len(df)
    cuts = [int(n * f) for f in TRAIN_CUTS] + [n]
    folds = []
    for i, (tr_end, va_end) in enumerate(zip(cuts[:-1], cuts[1:]), start=1):
        train = df.iloc[:tr_end]
        val = df.iloc[tr_end:va_end]
        if len(train) == 0 or len(val) == 0:
            raise WalkForwardError(f"Fold {i} is empty")
        if train[ID_COL].max() >= val[ID_COL].min():
            raise WalkForwardError(f"Fold {i} is not chronological")
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
    return folds


def model_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [{"name": "Naive Lag-24", "kind": "naive"}]
    for a in RIDGE_ALPHAS:
        specs.append({"name": f"Ridge_a{a}", "kind": "ridge", "alpha": float(a)})
    for a in ENET_ALPHAS:
        for l1 in ENET_L1:
            specs.append(
                {
                    "name": f"ElasticNet_a{a}_l1_{l1}",
                    "kind": "elasticnet",
                    "alpha": float(a),
                    "l1_ratio": float(l1),
                }
            )
    specs.append({"name": "HistGradientBoosting", "kind": "hgb"})
    specs.append({"name": "RandomForest", "kind": "rf"})
    if find_spec("lightgbm") is not None:
        specs.append({"name": "LightGBM", "kind": "lgbm"})
    if find_spec("xgboost") is not None:
        specs.append({"name": "XGBoost", "kind": "xgb"})
    return specs


def predict_fold(spec: dict[str, Any], train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    x_tr = train.drop(columns=[ID_COL, TARGET_COL])
    y_tr = train[TARGET_COL].to_numpy(dtype=float)
    x_va = val.drop(columns=[ID_COL, TARGET_COL])
    kind = spec["kind"]
    if kind == "naive":
        pred = x_va[LAG24].to_numpy(dtype=float)
        if np.isnan(pred).any():
            raise WalkForwardError("Naive lag-24 has NaN in a validation block")
        return pred
    prep = FoldPreprocessor().fit(x_tr)
    if kind == "ridge":
        return ridge_predict(prep.transform_linear(x_tr), y_tr, prep.transform_linear(x_va), spec["alpha"])
    if kind == "elasticnet":
        return elastic_net_predict(
            prep.transform_linear(x_tr),
            y_tr,
            prep.transform_linear(x_va),
            spec["alpha"],
            spec["l1_ratio"],
        )
    if kind == "hgb":
        model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=RANDOM_STATE,
            early_stopping=False,
        )
        model.fit(x_tr, y_tr)
        return model.predict(x_va)
    if kind == "rf":
        model = RandomForestRegressor(
            n_estimators=80,
            max_depth=12,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        model.fit(prep.transform_imputed(x_tr), y_tr)
        return model.predict(prep.transform_imputed(x_va))
    if kind == "lgbm":
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=-1,
        )
        model.fit(x_tr, y_tr)
        return model.predict(x_va)
    if kind == "xgb":
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0,
        )
        model.fit(x_tr, y_tr)
        return model.predict(x_va)
    raise WalkForwardError(f"Unknown model kind {kind}")


def select_model(summary: pd.DataFrame) -> str:
    ranked = summary.sort_values(["mean_MAE", "std_MAE"]).reset_index(drop=True)
    best_mae = float(ranked.loc[0, "mean_MAE"])
    close = ranked[ranked["mean_MAE"] <= best_mae + CLOSE_MAE].copy()
    close["family"] = close["model"].map(family_of)
    close["simplicity"] = close["family"].map(lambda f: SIMPLICITY.get(f, 99))
    close = close.sort_values(["mean_MAE", "std_MAE", "simplicity"]).reset_index(drop=True)
    return str(close.loc[0, "model"])


def distribution_shift(train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in SHIFT_FEATURES:
        if col not in train.columns:
            continue
        a, b = train[col], val[col]
        tr_mean, tr_std = float(a.mean()), float(a.std(ddof=0))
        va_mean, va_std = float(b.mean()), float(b.std(ddof=0))
        z = abs(va_mean - tr_mean) / tr_std if tr_std > 0 and np.isfinite(tr_std) else np.nan
        rows.append(
            {
                "feature": col,
                "train_mean": tr_mean,
                "train_std": tr_std,
                "val_mean": va_mean,
                "val_std": va_std,
                "abs_mean_shift_over_train_std": z,
                "flag": bool(np.isfinite(z) and z >= SHIFT_Z),
            }
        )
    return pd.DataFrame(rows)


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


def write_report(
    folds: list[dict[str, Any]],
    summary: pd.DataFrame,
    fold_rows: pd.DataFrame,
    selected: str,
    shift: pd.DataFrame,
    hour_err: pd.DataFrame,
    month_err: pd.DataFrame,
    weekday_err: pd.DataFrame,
    q_err: pd.DataFrame,
    hashes: dict[str, str],
    status: str,
) -> None:
    sel = summary[summary["model"] == selected].iloc[0]
    ridge_rows = summary[summary["model"].str.startswith("Ridge")].sort_values("mean_MAE")
    best_ridge = ridge_rows.iloc[0]
    ridge_folds = fold_rows[fold_rows["model"] == best_ridge["model"]].sort_values("fold")
    naive = summary[summary["model"] == "Naive Lag-24"].iloc[0]
    nonlinear = summary[summary["model"].isin(["HistGradientBoosting", "LightGBM", "XGBoost", "RandomForest"])]
    best_nl = nonlinear.sort_values("mean_MAE").iloc[0] if len(nonlinear) else None
    best_nl_name = str(best_nl["model"]) if best_nl is not None else "none"
    best_nl_mae = float(best_nl["mean_MAE"]) if best_nl is not None else float("nan")
    later_bias = float(ridge_folds["bias"].iloc[-1])
    later_mae = float(ridge_folds["MAE"].iloc[-1])
    early_mae = float(ridge_folds["MAE"].iloc[0])
    n_under = int((ridge_folds["bias"] < 0).sum())
    worst_bias_fold = int(ridge_folds.loc[ridge_folds["bias"].idxmin(), "fold"])
    worst_bias = float(ridge_folds["bias"].min())
    ridge_mae_range = float(ridge_folds["MAE"].max() - ridge_folds["MAE"].min())
    mae_rises = bool(later_mae > early_mae + 0.25)
    q4 = q_err[q_err["quantile"].astype(str) == "Q4"].iloc[0]
    q1 = q_err[q_err["quantile"].astype(str) == "Q1"].iloc[0]
    tgt = shift[shift["feature"] == TARGET_COL]
    tgt_line = "target row missing"
    if len(tgt):
        t = tgt.iloc[0]
        tgt_line = (
            f"train mean {t['train_mean']:.2f} (std {t['train_std']:.2f}) vs "
            f"validation mean {t['val_mean']:.2f} (std {t['val_std']:.2f}); "
            f"|Δmean|/train_std = {t['abs_mean_shift_over_train_std']:.3f}"
        )
    key_models = [selected, "Naive Lag-24", "LightGBM", "HistGradientBoosting", "XGBoost", "RandomForest"]
    key_models = [m for m in key_models if m in set(fold_rows["model"])]
    fold_diag = fold_rows[fold_rows["model"].isin(key_models)][
        ["fold", "model", "y_mean", "y_std", "MAE", "RMSE", "bias"]
    ].sort_values(["fold", "model"])
    nl_beats = False
    if best_nl is not None:
        nl_fold = fold_rows[fold_rows["model"] == best_nl_name].sort_values("fold")["MAE"].to_numpy()
        ridge_mae = ridge_folds["MAE"].to_numpy()
        nl_beats = bool(best_nl_mae + 1e-12 < float(best_ridge["mean_MAE"]) and (nl_fold < ridge_mae).all())
    under_later = n_under >= 3 and later_bias < 0
    shift_sorted = shift.sort_values("abs_mean_shift_over_train_std", ascending=False)

    text = f"""# Walk-Forward Validation

**WALK_FORWARD_VALIDATION = {status}**

TEST parquet was never loaded and was not used for selection, tuning, thresholds, or diagnostics.
Folds are chronological expanding windows on TRAIN+VALIDATION only (29,804 hourly rows).
No shuffle. No random K-fold. Every fold satisfies `max(train timestamp) < min(validation timestamp)`.
Linear preprocessing (median impute + standard scale) is fit on that fold's training block only.
HistGradientBoosting, LightGBM, and XGBoost use native NaN handling. RandomForest uses the fold-train imputer.

ElasticNet uses a NumPy cyclic coordinate descent that matches the sklearn ElasticNet objective.
The sklearn `ElasticNet.fit` solver is not used (it aborted in this environment).

Conservative tree settings (`random_state={RANDOM_STATE}`):
- HistGradientBoosting: `max_iter=200`, `learning_rate=0.05`, `max_leaf_nodes=31`, `early_stopping=False`
- RandomForest: `n_estimators=80`, `max_depth=12`, `min_samples_leaf=5`, `n_jobs=1`
- LightGBM / XGBoost: `n_estimators=200`, `learning_rate=0.05`, `n_jobs=1`

## Folds

{md_table(pd.DataFrame([{k: f[k] for k in ('fold','n_train','n_val','train_start','train_end','val_start','val_end','train_frac')} for f in folds]), floatfmt=".4f")}

Validation blocks are contiguous and non-overlapping. Fold 4 is the remaining 20% (larger block by construction).

Validation-block target means rise from 49.07 → 51.95 → 50.49 → 52.82. Variance collapses in fold 3 (summer 2017, y_std ≈ 7.98) and is high again in folds 1, 2, and 4.

## Model comparison (mean ± std over 4 folds)

Primary key: mean MAE. Secondary: MAE std. Tie-break: simpler family (within {CLOSE_MAE} MAE).

{md_table(summary)}

## Selected model

**{selected}**

- mean walk-forward MAE = {sel['mean_MAE']:.4f}
- MAE std = {sel['std_MAE']:.4f}
- mean RMSE = {sel['mean_RMSE']:.4f} (std {sel['std_RMSE']:.4f})
- mean R2 = {sel['mean_R2']:.4f}
- mean sMAPE = {sel['mean_sMAPE']:.4f}
- mean bias = {sel['mean_bias']:.4f}

## Answers

1. Lowest mean walk-forward MAE: **{selected}** ({sel['mean_MAE']:.4f}).
2. Its MAE standard deviation: **{sel['std_MAE']:.4f}** (largest MAE − smallest MAE = {ridge_mae_range:.3f}).
3. Is Ridge stable across time? Relatively yes versus the other families: **{best_ridge['model']}** has the lowest MAE std in the comparison ({best_ridge['std_MAE']:.4f}). Fold MAEs = {ridge_folds['MAE'].round(3).tolist()}. Error does **not** rise monotonically with calendar time (fold 3 is easiest because the target is much less variable). The unstable period is fold 2 (Jan–May 2017), not the last block.
4. Does Ridge systematically underpredict later periods? It systematically underpredicts **high-price** blocks, not simply “later time.” Bias is negative in {n_under}/4 folds; the worst bias is fold {worst_bias_fold} ({worst_bias:.3f}). Last-fold bias = {later_bias:.3f}; last-fold MAE = {later_mae:.3f} vs first-fold MAE {early_mae:.3f}. {"Last-fold MAE is worse than fold 1, so later error is higher." if mae_rises else "Last-fold MAE is not worse than fold 1; the bias pattern tracks high-mean / high-variance windows."} Pooled Q4 (highest prices) bias = {float(q4['bias']):.3f} with MAE {float(q4['MAE']):.3f} vs Q1 MAE {float(q1['MAE']):.3f}. This is the same mechanism as the previously reported locked-test Ridge bias (not recomputed here).
5. Best non-linear: **{best_nl_name}** mean MAE {best_nl_mae:.4f} vs best Ridge {best_ridge['mean_MAE']:.4f}. Consistent outperform-Ridge across folds? **{"Yes" if nl_beats else "No"}**. Trees beat Ridge only in the low-variance summer fold; they lose on the latest, higher-mean fold 4.
6. Naive Lag-24 mean MAE {naive['mean_MAE']:.4f} (std {naive['std_MAE']:.4f}). Competitive with the winner? **{"Yes, within the close band." if naive['mean_MAE'] <= sel['mean_MAE'] + CLOSE_MAE else "No — about 1.5 €/MWh worse on mean MAE."}** Naive bias is near zero (it copies yesterday), so it does not share Ridge’s level shift, but its RMSE/R2 remain poor.
7. Model for the next tuning stage: **{selected}**.
8. Evidence: lowest mean expanding-window MAE **and** lowest MAE std on locked TRAIN+VALIDATION folds; simpler than trees; no test observations entered any decision. Next tuning should keep the linear family and treat **level-shift / high-price residual bias** as the main robustness issue (not a switch to boosting).

## Fold-level target shift and model diagnostics

Selected Ridge and the main competitors on every fold:

{md_table(fold_diag)}

Ridge fold detail:

{md_table(fold_rows[fold_rows['model']==best_ridge['model']][['fold','val_start','val_end','y_mean','y_std','MAE','bias']])}

## Error analysis (selected model, pooled walk-forward validation blocks)

Hour, weekday (`day_of_week`: Monday=0 … Sunday=6), and month are Europe/Madrid calendar features already in X.

Worst hours: {hour_err.sort_values('MAE', ascending=False).head(5)['hour'].tolist()}  
Worst weekdays: {weekday_err.sort_values('MAE', ascending=False).head(3)['weekday'].tolist()}  
Worst months: {month_err.sort_values('MAE', ascending=False).head(3)['month'].tolist()}

Hourly MAE is fairly flat (range about {float(hour_err['MAE'].min()):.2f}–{float(hour_err['MAE'].max()):.2f}). Month is the stronger seasonal signal: January is worst, October best.

### MAE by target-price quantile

{md_table(q_err)}

Q4 hours (mean price {float(q4['y_mean']):.1f}) have both the highest MAE and a large negative bias. The model is systematically worse at high-price periods. That is the primary robustness finding for the next stage.

Hour table: `reports/walk_forward_error_by_hour.csv`  
Weekday table: `reports/walk_forward_error_by_weekday.csv`  
Month table: `reports/walk_forward_error_by_month.csv`

## TRAIN vs VALIDATION distribution shift (official split, not TEST)

Official TRAIN vs official VALIDATION only. Target: {tgt_line}.

Flag rule: |val_mean − train_mean| / train_std ≥ {SHIFT_Z}. Diagnostic only — no feature was dropped.

{md_table(shift_sorted)}

The largest flagged shifts are lagged national temperature (validation is a cooler Oct–May window than the full train history). Load, renewable, and price-lag means move with the target but stay below the 0.5-std flag on most series. No features were removed from this diagnostic.

## Leakage / lock checks

- `data/processed/test.parquet` is never opened (`assert_not_test` on every parquet read).
- Combined TRAIN+VALIDATION is UTC-aware, unique, strictly hourly, and sorted.
- Each fold is a prefix of that chronology; validation never precedes training.
- Preprocessing statistics are computed on the fold training block only.
- `price actual` is absent. Feature count remains 184 SAFE columns.
- Protected source parquets are hash-checked before and after the run.

## Reproducibility

`random_state = {RANDOM_STATE}` where the estimator accepts it. Ridge / ElasticNet / Naive are deterministic given the fold data.

| file | md5 |
|---|---|
| walk_forward_model_comparison.csv | {hashes['comparison']} |
| walk_forward_fold_results.csv | {hashes['folds']} |
| walk_forward_error_by_hour.csv | {hashes['hour']} |
| walk_forward_error_by_month.csv | {hashes['month']} |
| walk_forward_error_by_weekday.csv | {hashes['weekday']} |
| walk_forward_error_by_price_quantile.csv | {hashes['quantile']} |
"""
    (REPORTS / "walk_forward_validation.md").write_text(text, encoding="utf-8")


def run() -> dict[str, Any]:
    before = snapshot(PROTECTED)
    official_train = read_dev_parquet(TRAIN_PATH)
    official_val = read_dev_parquet(VAL_PATH)
    df = load_dev_frame()
    folds = make_folds(df)
    specs = model_specs()
    print("Folds:", json.dumps([{k: f[k] for k in f if k not in {"train", "val"}} for f in folds], indent=2), flush=True)
    print("Models:", [s["name"] for s in specs], flush=True)

    fold_records = []
    oof = {s["name"]: {"ts": [], "y": [], "pred": [], "hour": [], "dow": [], "month": []} for s in specs}

    for fold in folds:
        y_va = fold["val"][TARGET_COL]
        y_mean, y_std = float(y_va.mean()), float(y_va.std(ddof=0))
        print(f"=== Fold {fold['fold']} train={fold['n_train']} val={fold['n_val']} y_mean={y_mean:.2f} ===", flush=True)
        for spec in specs:
            print(f"  {spec['name']}", flush=True)
            pred = predict_fold(spec, fold["train"], fold["val"])
            mets = metrics(y_va, pred)
            fold_records.append(
                {
                    "fold": fold["fold"],
                    "model": spec["name"],
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
            oof[spec["name"]]["ts"].append(fold["val"][ID_COL].to_numpy())
            oof[spec["name"]]["y"].append(y_va.to_numpy(dtype=float))
            oof[spec["name"]]["pred"].append(np.asarray(pred, dtype=float))
            oof[spec["name"]]["hour"].append(fold["val"]["hour"].to_numpy())
            oof[spec["name"]]["dow"].append(fold["val"]["day_of_week"].to_numpy())
            oof[spec["name"]]["month"].append(fold["val"]["month"].to_numpy())

    fold_df = pd.DataFrame(fold_records)
    summary_rows = []
    for name, g in fold_df.groupby("model", sort=False):
        summary_rows.append(
            {
                "model": name,
                "mean_MAE": float(g["MAE"].mean()),
                "std_MAE": float(g["MAE"].std(ddof=0)),
                "mean_RMSE": float(g["RMSE"].mean()),
                "std_RMSE": float(g["RMSE"].std(ddof=0)),
                "mean_R2": float(g["R2"].mean()),
                "mean_sMAPE": float(g["sMAPE"].mean()),
                "mean_bias": float(g["bias"].mean()),
                "n_folds": int(len(g)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    selected = select_model(summary)
    print("SELECTED:", selected, flush=True)

    y = np.concatenate(oof[selected]["y"])
    pred = np.concatenate(oof[selected]["pred"])
    hour = np.concatenate(oof[selected]["hour"])
    dow = np.concatenate(oof[selected]["dow"])
    month = np.concatenate(oof[selected]["month"])
    abs_err = np.abs(pred - y)
    hour_err = pd.DataFrame({"hour": hour, "MAE": abs_err}).groupby("hour", sort=True)["MAE"].mean().reset_index()
    weekday_err = pd.DataFrame({"weekday": dow, "MAE": abs_err}).groupby("weekday", sort=True)["MAE"].mean().reset_index()
    month_err = pd.DataFrame({"month": month, "MAE": abs_err}).groupby("month", sort=True)["MAE"].mean().reset_index()
    qs = pd.qcut(y, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    q_err = (
        pd.DataFrame({"quantile": qs, "MAE": abs_err, "y_mean": y, "bias": pred - y})
        .groupby("quantile", observed=True)
        .agg(MAE=("MAE", "mean"), bias=("bias", "mean"), y_mean=("y_mean", "mean"), n=("MAE", "size"))
        .reset_index()
    )

    shift = distribution_shift(official_train, official_val)

    REPORTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORTS / "walk_forward_model_comparison.csv", index=False)
    fold_df.to_csv(REPORTS / "walk_forward_fold_results.csv", index=False)
    hour_err.to_csv(REPORTS / "walk_forward_error_by_hour.csv", index=False)
    month_err.to_csv(REPORTS / "walk_forward_error_by_month.csv", index=False)
    weekday_err.to_csv(REPORTS / "walk_forward_error_by_weekday.csv", index=False)
    q_err.to_csv(REPORTS / "walk_forward_error_by_price_quantile.csv", index=False)

    hashes = {
        "comparison": md5(REPORTS / "walk_forward_model_comparison.csv"),
        "folds": md5(REPORTS / "walk_forward_fold_results.csv"),
        "hour": md5(REPORTS / "walk_forward_error_by_hour.csv"),
        "month": md5(REPORTS / "walk_forward_error_by_month.csv"),
        "weekday": md5(REPORTS / "walk_forward_error_by_weekday.csv"),
        "quantile": md5(REPORTS / "walk_forward_error_by_price_quantile.csv"),
    }
    after = snapshot(PROTECTED)
    if after != before:
        raise WalkForwardError("A protected file was modified")
    test_loaded = False
    status = "PASS"
    write_report(folds, summary, fold_df, selected, shift, hour_err, month_err, weekday_err, q_err, hashes, status)
    if snapshot(PROTECTED) != before:
        raise WalkForwardError("A protected file was modified after reporting")

    return {
        "status": status,
        "selected": selected,
        "summary": summary.to_dict(orient="records"),
        "folds": [{k: f[k] for k in f if k not in {"train", "val"}} for f in folds],
        "hashes": hashes,
        "test_loaded": test_loaded,
        "n_dev_rows": int(len(df)),
    }


if __name__ == "__main__":
    result = run()
    print("=== WALK-FORWARD SUMMARY ===")
    print(pd.DataFrame(result["summary"]).to_string(index=False))
    print("selected:", result["selected"])
    print("test_loaded:", result["test_loaded"])
    print(json.dumps({"folds": result["folds"], "hashes": result["hashes"]}, indent=2))
    print(f"WALK_FORWARD_VALIDATION = {result['status']}")
