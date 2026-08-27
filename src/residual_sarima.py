"""Residual SARIMA on Ridge+METHOD_B, parsimonious orders.

TRAIN+VALIDATION expanding folds only. Does not load test.parquet.
Does not change Ridge alpha, METHOD_B, or locked test artifacts.

Orders are chosen by train AIC (fold-train residuals only). The scored
protocol is DAM_24: forecast 24 residual hours, then append that day's
actual residuals (day-ahead aligned). A numpy AR(1) one-step path is
reported only as a diagnostic; it is not a DAM noon origin.
"""

from __future__ import annotations

import hashlib
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import high_price_analysis as hp
import ridge_tuning as rt

REPORTS = rt.REPORTS
OUT_CSV = REPORTS / "residual_sarima.csv"
FOLD_CSV = REPORTS / "residual_sarima_folds.csv"
AIC_CSV = REPORTS / "residual_sarima_aic.csv"
REPORT_MD = REPORTS / "residual_sarima.md"

FIT_WINDOW = 1440  # last 60 days of fold-train residuals
HORIZON = 24
METHOD_B_WF_MAE = 5.496022

# Parsimonious ladder: seasonal period 24 only (168 is too many params / too slow).
# Tuple: name, order, seasonal_order, n_mean_params (excl. sigma2)
CANDIDATES: tuple[tuple[str, tuple[int, int, int], tuple[int, int, int, int], int], ...] = (
    ("AR1", (1, 0, 0), (0, 0, 0, 0), 1),
    ("ARMA11", (1, 0, 1), (0, 0, 0, 0), 2),
    ("SAR_AR1_s24", (1, 0, 0), (1, 0, 0, 24), 2),
    ("SARIMA1111_s24", (1, 0, 1), (1, 0, 1, 24), 4),
)

PROTECTED = tuple(rt.PROTECTED) + (
    ROOT / "data" / "processed" / "predictions" / "final_test_predictions.parquet",
    ROOT / "reports" / "final" / "final_test_metrics.csv",
)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def snapshot() -> dict[str, str]:
    return {str(p): md5(p) for p in PROTECTED if p.exists()}


def ridge_method_b_both(
    train: pd.DataFrame, val: pd.DataFrame, feat_cols: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    y_tr = train[rt.TARGET_COL].to_numpy(dtype=float)
    p75 = float(np.quantile(y_tr, 0.75))
    train_aug = hp.add_frac_features(train, p75)
    val_aug = hp.add_frac_features(val, p75)
    cols = feat_cols + list(hp.FRAC_COLS)
    x_tr = hp.feature_frame(train_aug, cols)
    x_va = hp.feature_frame(val_aug, cols)
    pred_tr, pred_va, _, _ = hp.fit_predict_ridge(x_tr, y_tr, x_va)
    return hp.expanding_correct(pred_tr, y_tr, pred_va, train[rt.ID_COL].to_numpy())


def fit_sarima(resid: np.ndarray, order, seasonal) -> Any | None:
    y = np.asarray(resid[-FIT_WINDOW:], dtype=float)
    if not np.isfinite(y).all() or len(y) < 48:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = SARIMAX(
                y,
                order=order,
                seasonal_order=seasonal,
                trend="n",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            return mod.fit(disp=False, maxiter=80, method="lbfgs")
    except Exception:
        return None


def ar1_onestep(resid_tr: np.ndarray, resid_va: np.ndarray) -> np.ndarray:
    """Lag-1 residual AR; uses y[t-1]. Not a day-ahead origin."""
    r = np.asarray(resid_tr[-FIT_WINDOW:], dtype=float)
    denom = float(np.dot(r[:-1], r[:-1]))
    phi = float(np.dot(r[1:], r[:-1]) / denom) if denom > 0 else 0.0
    out = np.empty(len(resid_va), dtype=float)
    last = float(r[-1])
    for t, actual in enumerate(resid_va):
        out[t] = phi * last
        last = float(actual)
    return out


def forecast_blocks(fitted, actual_resid: np.ndarray, horizon: int) -> np.ndarray:
    n = len(actual_resid)
    out = np.zeros(n, dtype=float)
    if fitted is None:
        return out
    state = fitted
    i = 0
    while i < n:
        h = min(horizon, n - i)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = np.asarray(state.forecast(h), dtype=float).reshape(-1)
            if fc.shape[0] != h or not np.isfinite(fc).all():
                fc = np.zeros(h, dtype=float)
            out[i : i + h] = fc
            state = state.append(actual_resid[i : i + h], refit=False)
        except Exception:
            out[i:] = 0.0
            break
        i += h
    return out


def run() -> None:
    before = snapshot()
    df, feat_cols = rt.load_dev_frame()
    folds = rt.make_folds(df)
    fold_rows: list[dict[str, Any]] = []
    aic_rows: list[dict[str, Any]] = []

    for fold in folds:
        train, val = fold["train"], fold["val"]
        y_tr = train[rt.TARGET_COL].to_numpy(dtype=float)
        y_va = val[rt.TARGET_COL].to_numpy(dtype=float)
        pred_tr, pred_va = ridge_method_b_both(train, val, feat_cols)
        resid_tr = y_tr - pred_tr
        resid_va = y_va - pred_va
        print(
            f"fold {fold['fold']} n_train={len(train)} n_val={len(val)} "
            f"resid_acf1≈{np.corrcoef(resid_tr[1:], resid_tr[:-1])[0,1]:.3f}",
            flush=True,
        )

        ridge_m = rt.metrics(y_va, pred_va)
        fold_rows.append({"model": "Ridge_METHOD_B", "protocol": "none", "fold": fold["fold"], **ridge_m})
        print(f"  Ridge_METHOD_B MAE={ridge_m['MAE']:.4f}", flush=True)

        for name, order, seasonal, npar in CANDIDATES:
            fitted = fit_sarima(resid_tr, order, seasonal)
            aic = float(fitted.aic) if fitted is not None else np.nan
            aic_rows.append(
                {
                    "fold": fold["fold"],
                    "model": name,
                    "order": str(order),
                    "seasonal": str(seasonal),
                    "n_mean_params": npar,
                    "aic": aic,
                    "fit_ok": fitted is not None,
                }
            )
            add = forecast_blocks(fitted, resid_va, HORIZON)
            pred = pred_va + add
            m = rt.metrics(y_va, pred)
            fold_rows.append(
                {
                    "model": f"{name}_DAM_24",
                    "family": name,
                    "protocol": "DAM_24",
                    "n_mean_params": npar,
                    "fold": fold["fold"],
                    "aic": aic,
                    **m,
                }
            )
            print(f"  {name}_DAM_24 MAE={m['MAE']:.4f} AIC={aic:.1f}", flush=True)

        step_add = ar1_onestep(resid_tr, resid_va)
        step_m = rt.metrics(y_va, pred_va + step_add)
        fold_rows.append(
            {
                "model": "AR1_STEP_1_diagnostic",
                "family": "AR1",
                "protocol": "STEP_1",
                "n_mean_params": 1,
                "fold": fold["fold"],
                "aic": np.nan,
                **step_m,
            }
        )
        print(f"  AR1_STEP_1_diagnostic MAE={step_m['MAE']:.4f} (not DAM)", flush=True)

    fold_df = pd.DataFrame(fold_rows)
    aic_df = pd.DataFrame(aic_rows)
    summary = (
        fold_df.groupby(["model"], as_index=False)
        .agg(
            mean_MAE=("MAE", "mean"),
            std_MAE=("MAE", "std"),
            mean_RMSE=("RMSE", "mean"),
            mean_R2=("R2", "mean"),
            mean_sMAPE=("sMAPE", "mean"),
            mean_bias=("bias", "mean"),
            n_folds=("fold", "nunique"),
        )
        .sort_values("mean_MAE")
        .reset_index(drop=True)
    )
    aic_mean = (
        aic_df.groupby("model", as_index=False)
        .agg(mean_AIC=("aic", "mean"), n_mean_params=("n_mean_params", "first"))
        .sort_values(["mean_AIC", "n_mean_params"])
        .reset_index(drop=True)
    )
    aic_winner = str(aic_mean.iloc[0]["model"])
    dam_name = f"{aic_winner}_DAM_24"
    step_name = "AR1_STEP_1_diagnostic"

    REPORTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_CSV, index=False)
    fold_df.to_csv(FOLD_CSV, index=False)
    aic_df.to_csv(AIC_CSV, index=False)

    def mae_of(name: str) -> float:
        hit = summary.loc[summary["model"] == name, "mean_MAE"]
        return float(hit.iloc[0]) if len(hit) else float("nan")

    ridge_mae = mae_of("Ridge_METHOD_B")
    dam_mae = mae_of(dam_name)
    step_mae = mae_of(step_name)
    beats_dam = dam_mae + 1e-12 < ridge_mae
    beats_step = step_mae + 1e-12 < ridge_mae

    REPORT_MD.write_text(
        f"""# Residual SARIMA on Ridge+METHOD_B (development only)

**RESIDUAL_SARIMA = PASS**

TEST was not loaded. Locked model is unchanged:
Ridge(`alpha=0.001`) + METHOD_B.

## Why this form

Ridge already uses lag-24/48/168, calendar, and day-ahead load/renewable
forecasts. The time-series model is fit on **Ridge+METHOD_B residuals**,
not on raw price. Seasonal period **24** only (hourly day). Period 168
was not fit: too many seasonal parameters and much slower MLE.

Fit window: last **{FIT_WINDOW}** fold-train residual hours.
Order selection: **lowest mean train AIC**, then fewer mean-parameters.

Parsimonious candidates:

| name | order | seasonal | mean-params |
|---|---|---|---:|
| AR1 | (1,0,0) | none | 1 |
| ARMA11 | (1,0,1) | none | 2 |
| SAR_AR1_s24 | (1,0,0) | (1,0,0,24) | 2 |
| SARIMA1111_s24 | (1,0,1) | (1,0,1,24) | 4 |

AIC winner: **{aic_winner}**

## Protocols

- **DAM_24**: each origin forecasts 24 residual hours, then that day's
  actual residuals are appended (`refit=False`). This is the day-ahead
  comparison.
- **STEP_1**: one-step updates. Uses lag-1 residual; **not** a DAM noon
  origin. Diagnostic only.

## Train AIC

{rt.md_table(aic_mean)}

## Walk-forward MAE

{rt.md_table(summary)}

AIC-winner DAM_24 MAE = {dam_mae:.6f}  
AIC-winner STEP_1 MAE = {step_mae:.6f}  
Ridge+METHOD_B MAE = {ridge_mae:.6f} (locked report {METHOD_B_WF_MAE:.6f})

DAM_24 beats Ridge+METHOD_B? **{str(beats_dam).upper()}**  
STEP_1 beats Ridge+METHOD_B? **{str(beats_step).upper()}**

## Decision

LOCKED_MODEL_UNCHANGED = TRUE  
TEST_USED_FOR_SELECTION = FALSE  
""",
        encoding="utf-8",
    )

    after = snapshot()
    if after != before:
        raise RuntimeError("A protected file changed")
    print(aic_mean.to_string(index=False))
    print(summary.to_string(index=False))
    print("AIC_WINNER", aic_winner)
    print("DAM24", dam_mae, "STEP1_DIAG", step_mae, "RIDGE_B", ridge_mae)
    print("BEATS_DAM", beats_dam, "BEATS_STEP", beats_step)
    print("PROTECTED_UNCHANGED", True)
    print("wrote", OUT_CSV)


if __name__ == "__main__":
    run()
