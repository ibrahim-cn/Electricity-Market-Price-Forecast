"""Final locked holdout evaluation.

Frozen pipeline: Ridge(alpha=0.001) + METHOD_B frequency features
+ the expanding_historical addend + AR(1) residual correction (DAM 24h blocks).

AR(1) was selected on TRAIN+VALIDATION walk-forward only. Test is read after
development fitting objects are frozen. Test y is never used for thresholds,
preprocessing, Ridge, the addend, or AR(1) phi. After each 24h forecast block,
published test residuals update the AR(1) state (same delay idea as lag-24).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import high_price_analysis as hpa
import residual_ar1 as ar1
import residual_correction as rc
import ridge_tuning as rt

TRAIN_PATH = rt.TRAIN_PATH
VAL_PATH = rt.VAL_PATH
TEST_PATH = rt.TEST_PATH
REPORTS = rt.REPORTS
ID_COL = rt.ID_COL
TARGET_COL = rt.TARGET_COL
LAG24 = "price_day_ahead_lag_24"
ALPHA = 0.001
RANDOM_STATE = 42
FRAC_COLS = hpa.FRAC_COLS

VAL_MAE = 4.529617
VAL_MAE_STD = 0.561220
VAL_BIAS = -0.71
VAL_P75_BIAS = -5.895626
VAL_P90_BIAS = -8.211142

PRED_PATH = ROOT / "data" / "processed" / "predictions" / "final_test_predictions.parquet"
REPORTS_FINAL = rt.REPORTS_FINAL
METRICS_CSV = REPORTS_FINAL / "final_test_metrics.csv"
HOUR_CSV = REPORTS_FINAL / "final_test_error_by_hour.csv"
MONTH_CSV = REPORTS_FINAL / "final_test_error_by_month.csv"
WEEKDAY_CSV = REPORTS_FINAL / "final_test_error_by_weekday.csv"
QUANTILE_CSV = REPORTS_FINAL / "final_test_error_by_price_quantile.csv"
REPORT_MD = REPORTS_FINAL / "final_test_evaluation.md"

REGIME_QS = (0.25, 0.50, 0.75, 0.90, 0.95)
REGIME_NAMES = {
    0.25: "P25+",
    0.50: "P50+",
    0.75: "P75+",
    0.90: "P90+",
    0.95: "P95+",
}


class FinalTestError(ValueError):
    pass


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def snapshot(paths: tuple[Path, ...]) -> dict[str, str]:
    return {str(p): md5(p) for p in paths if p.exists()}


def output_hashes() -> dict[str, str]:
    return {
        "predictions": md5(PRED_PATH),
        "metrics": md5(METRICS_CSV),
        "hour": md5(HOUR_CSV),
        "month": md5(MONTH_CSV),
        "weekday": md5(WEEKDAY_CSV),
        "quantile": md5(QUANTILE_CSV),
    }


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "price actual" in df.columns:
        raise FinalTestError(f"{path.name} contains price actual")
    if TARGET_COL not in df.columns or ID_COL not in df.columns:
        raise FinalTestError(f"{path.name} missing required columns")
    return df.sort_values(ID_COL, kind="mergesort").reset_index(drop=True)


def feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in {ID_COL, TARGET_COL}]
    if len(cols) != rt.N_FEATURES:
        raise FinalTestError(f"expected {rt.N_FEATURES} base features, got {len(cols)}")
    return cols


def regime_metrics(y: np.ndarray, pred: np.ndarray, threshold: float) -> dict[str, float]:
    mask = y >= threshold
    if not mask.any():
        return {"n": 0, "MAE": float("nan"), "bias": float("nan"), "y_mean": float("nan"), "y_pred_mean": float("nan")}
    yt, yp = y[mask], pred[mask]
    return {
        "n": int(mask.sum()),
        "MAE": float(np.mean(np.abs(yp - yt))),
        "bias": float(np.mean(yp - yt)),
        "y_mean": float(np.mean(yt)),
        "y_pred_mean": float(np.mean(yp)),
    }


def write_report(
    mets: dict[str, float],
    naive: dict[str, float],
    regimes: pd.DataFrame,
    hour_err: pd.DataFrame,
    month_err: pd.DataFrame,
    weekday_err: pd.DataFrame,
    hashes: dict[str, str],
    before: dict[str, str],
    after: dict[str, str],
    status: str,
    repro: str,
    p75_dev: float,
    n_test: int,
    n_dropped: int,
    dev_stats: dict[str, float],
    test_stats: dict[str, float],
    addend: float,
) -> None:
    prot_ok = before == after
    beats = mets["MAE"] < naive["MAE"]
    d_mae = mets["MAE"] - VAL_MAE
    d_bias = mets["bias"] - VAL_BIAS
    p75 = regimes[regimes["regime"] == "P75+"].iloc[0]
    p90 = regimes[regimes["regime"] == "P90+"].iloc[0]
    p95 = regimes[regimes["regime"] == "P95+"].iloc[0]
    prot_rows = []
    for p in rt.REQUIRED_PROTECTED:
        key = str(p)
        prot_rows.append(
            {
                "file": p.name,
                "md5_before": before.get(key, ""),
                "md5_after": after.get(key, ""),
                "unchanged": before.get(key) == after.get(key),
            }
        )
    text = f"""# Final Test Evaluation (locked holdout)

**FINAL_TEST_EVALUATION = {status}**

Test parquet was read **once the pipeline was frozen**. It was not used to
select the model, alpha, METHOD_B, AR(1) phi, thresholds, or residual addend.

## 1. Selected model

Ridge, closed-form NumPy, `alpha = {ALPHA}`, plus METHOD_B and AR(1) on
Ridge+METHOD_B residuals. `random_state = {RANDOM_STATE}` is recorded for
pipeline consistency; Ridge and AR(1) are deterministic.

## 2. Selected method

**METHOD_B + AR(1)** as selected on walk-forward:

- 184 SAFE features from `train`/`validation`/`test` parquets
- three causal high-price-frequency features (`y` shifted 24h, then 168/336/720-hour windows)
- high-price flag threshold = **development P75** = {p75_dev:.4f} (`quantile(y_train+y_val, 0.75)` only)
- frozen `expanding_historical` addend from development residuals only (addend = {addend:.6f})
- AR(1) on the last 1440 development residuals of Ridge+METHOD_B; 24-hour
  block forecasts, then that day's actual residuals update the state

No alpha, feature, threshold, AR(1) order, or correction search was run on test.

## 3. Walk-forward validation (frozen numbers)

| metric | value |
|---|---|
| mean MAE | {VAL_MAE:.6f} |
| MAE std | {VAL_MAE_STD:.6f} |
| mean bias | {VAL_BIAS:.2f} |
| P75+ bias (causal fold-train) | {VAL_P75_BIAS:.6f} |
| P90+ bias (causal fold-train) | {VAL_P90_BIAS:.6f} |

## 4. Final test performance

Test rows scored: **{n_test}**. Rows dropped: **{n_dropped}**.

| metric | test |
|---|---|
| MAE | {mets['MAE']:.6f} |
| RMSE | {mets['RMSE']:.6f} |
| R² | {mets['R2']:.6f} |
| sMAPE | {mets['sMAPE']:.6f} |
| bias | {mets['bias']:.6f} |

## 5. Naive Lag-24 (same test rows)

| | Ridge+METHOD_B+AR(1) | Naive Lag-24 |
|---|---:|---:|
| MAE | {mets['MAE']:.6f} | {naive['MAE']:.6f} |
| RMSE | {mets['RMSE']:.6f} | {naive['RMSE']:.6f} |
| bias | {mets['bias']:.6f} | {naive['bias']:.6f} |

MODEL_BEATS_NAIVE = {str(beats).upper()}

This comparison does **not** change the selected model.

## 6. High-price regimes (development quantiles applied to test y)

Thresholds come from train+validation only.

{rt.md_table(regimes)}

TEST_P75_BIAS = {float(p75['bias']):.6f}
TEST_P90_BIAS = {float(p90['bias']):.6f}
TEST_P95_BIAS = {float(p95['bias']):.6f}

## 7. Bias analysis

Overall test bias = {mets['bias']:.3f} (validation walk-forward mean bias {VAL_BIAS:.2f}).
The sign flipped: walk-forward underpredicted on average; the frozen holdout
**overpredicts**. P75+/P90+/P95+ biases are also positive on test.

## 8. Validation → test

| | walk-forward val | test | delta (test − val) |
|---|---:|---:|---:|
| MAE | {VAL_MAE:.6f} | {mets['MAE']:.6f} | {d_mae:.6f} |
| bias | {VAL_BIAS:.2f} | {mets['bias']:.6f} | {d_bias:.6f} |
| P75+ bias | {VAL_P75_BIAS:.6f} | {float(p75['bias']):.6f} | {float(p75['bias']) - VAL_P75_BIAS:.6f} |
| P90+ bias | {VAL_P90_BIAS:.6f} | {float(p90['bias']):.6f} | {float(p90['bias']) - VAL_P90_BIAS:.6f} |

{"Test MAE is higher than walk-forward MAE (degradation)." if d_mae > 0.05 else "Test MAE is not materially worse than walk-forward MAE."}

Do not retune on this gap.

## 9. Distribution-shift observations

Associated differences (not causal claims):

| | development (train+val) | test |
|---|---:|---:|
| target mean | {dev_stats['y_mean']:.3f} | {test_stats['y_mean']:.3f} |
| target std | {dev_stats['y_std']:.3f} | {test_stats['y_std']:.3f} |
| load forecast mean | {dev_stats['load']:.3f} | {test_stats['load']:.3f} |
| renewable forecast mean | {dev_stats['ren']:.3f} | {test_stats['ren']:.3f} |

Development mean {dev_stats['y_mean']:.2f} vs test mean {test_stats['y_mean']:.2f},
with **lower** test volatility ({test_stats['y_std']:.2f} vs {dev_stats['y_std']:.2f}).
Most test hours sit above the development P75 ({p75_dev:.2f}), so METHOD_B
frequency features are often saturated. Together with the positive development
addend, that is consistent with the observed **test overprediction**, not with
the walk-forward underprediction pattern. This is an association, not a cause.

## 10. Limitations

- High-price bias was never solved in walk-forward; the holdout is not a new search.
- METHOD_B frequency features use past prices vs a **development** P75. If the test
  level sits well above that P75, the features saturate and cannot express a new extreme.
- Expanding-historical addend is a single development statistic; it cannot track a
  further test-only level jump.
- Calendar MAE tables describe error concentration; they were not used for selection.

## 11. Test-use statement

TEST_USED_FOR_TUNING = FALSE
TEST_USED_FOR_SELECTION = FALSE

Test `y` entered the pipeline only to score already-frozen predictions (and, like the
precomputed `price_day_ahead_lag_24` already stored on test rows, as **past** values
inside causal 24h-shifted windows for later test hours). The P75 threshold, scaler,
Ridge weights, and residual addend used **no test target**.

Hour / weekday / month MAE:

Worst hours: {hour_err.sort_values('MAE', ascending=False).head(5)['hour'].tolist()}
Worst months: {month_err.sort_values('MAE', ascending=False).head(3)['month'].tolist()}
Worst weekdays (Mon=0): {weekday_err.sort_values('MAE', ascending=False).head(3)['weekday'].tolist()}

## Reproducibility and protected files

REPRODUCIBILITY = {repro}
PROTECTED_FILES_UNCHANGED = {str(prot_ok).upper()}

{rt.md_table(pd.DataFrame(prot_rows), floatfmt="")}

| file | md5 |
|---|---|
| final_test_predictions.parquet | {hashes['predictions']} |
| final_test_metrics.csv | {hashes['metrics']} |
| final_test_error_by_hour.csv | {hashes['hour']} |
| final_test_error_by_month.csv | {hashes['month']} |
| final_test_error_by_weekday.csv | {hashes['weekday']} |
| final_test_error_by_price_quantile.csv | {hashes['quantile']} |

## Machine-readable summary

FINAL_TEST_EVALUATION = {status}
SELECTED_MODEL = Ridge(alpha=0.001)+METHOD_B+AR(1)
SELECTED_METHOD = METHOD_B_AR1
VALIDATION_MAE = {VAL_MAE:.6f}
TEST_MAE = {mets['MAE']:.6f}
TEST_RMSE = {mets['RMSE']:.6f}
TEST_R2 = {mets['R2']:.6f}
TEST_SMAPE = {mets['sMAPE']:.6f}
TEST_BIAS = {mets['bias']:.6f}
TEST_P75_BIAS = {float(p75['bias']):.6f}
TEST_P90_BIAS = {float(p90['bias']):.6f}
TEST_P95_BIAS = {float(p95['bias']):.6f}
NAIVE_TEST_MAE = {naive['MAE']:.6f}
MODEL_BEATS_NAIVE = {str(beats).upper()}
TEST_USED_FOR_TUNING = FALSE
TEST_USED_FOR_SELECTION = FALSE
REPRODUCIBILITY = {repro}
PROTECTED_FILES_UNCHANGED = {str(prot_ok).upper()}
"""
    REPORT_MD.write_text(text, encoding="utf-8")


def run_once() -> dict[str, Any]:
    before = snapshot(rt.PROTECTED)
    train = load_split(TRAIN_PATH)
    val = load_split(VAL_PATH)
    base_cols = feature_cols(train)
    if feature_cols(val) != base_cols:
        raise FinalTestError("validation feature columns differ from train")

    dev = pd.concat([train, val], axis=0, ignore_index=True)
    dev = dev.sort_values(ID_COL, kind="mergesort").reset_index(drop=True)
    y_dev = dev[TARGET_COL].to_numpy(dtype=float)
    thresholds = {q: float(np.quantile(y_dev, q)) for q in REGIME_QS}
    p75_dev = thresholds[0.75]

    # Fit objects must be ready before test y is used for scoring.
    # Load test only after thresholds are frozen.
    test = load_split(TEST_PATH)
    if feature_cols(test) != base_cols:
        raise FinalTestError("test feature columns differ from train")

    full = pd.concat([dev, test], axis=0, ignore_index=True)
    full = full.sort_values(ID_COL, kind="mergesort").reset_index(drop=True)
    if not bool(full[ID_COL].is_monotonic_increasing):
        raise FinalTestError("combined series is not chronological")
    if full[ID_COL].duplicated().any():
        raise FinalTestError("duplicate timestamps across splits")
    full = hpa.add_frac_features(full, p75_dev)

    n_dev = len(dev)
    dev_f = full.iloc[:n_dev].copy()
    test_f = full.iloc[n_dev:].copy()
    if len(test_f) != len(test):
        raise FinalTestError("test row count changed after feature join")
    if test_f[ID_COL].iloc[0] != test[ID_COL].iloc[0]:
        raise FinalTestError("test block alignment failed")

    model_cols = base_cols + list(FRAC_COLS)
    x_dev = dev_f[model_cols]
    x_test = test_f[model_cols]
    prep = rt.FoldPreprocessor().fit(x_dev)
    rt.assert_preproc_train_only(prep, x_dev, x_test)
    xtr = prep.transform_linear(x_dev)
    xte = prep.transform_linear(x_test)
    pred_dev_raw = rt.ridge_predict(xtr, y_dev, xtr, ALPHA)
    pred_test_raw = rt.ridge_predict(xtr, y_dev, xte, ALPHA)
    addend = rc.expanding_addend(dev_f[ID_COL].to_numpy(), pred_dev_raw - y_dev)
    pred_dev = pred_dev_raw + addend
    pred_test_b = pred_test_raw + addend
    y_test = test_f[TARGET_COL].to_numpy(dtype=float)
    pred_test, ar1_phi, ar1_last = ar1.apply(pred_test_b, y_test, y_dev - pred_dev)
    naive = test_f[LAG24].to_numpy(dtype=float)
    n_dropped = 0
    if np.isnan(pred_test).any() or np.isnan(y_test).any():
        raise FinalTestError("NaN in test predictions or target; dropping rows is not part of the frozen pipeline")
    if np.isnan(naive).any():
        raise FinalTestError("Naive lag-24 has NaN on test")

    mets = rt.metrics(y_test, pred_test)
    naive_mets = rt.metrics(y_test, naive)
    residual = pred_test - y_test

    pred_df = pd.DataFrame(
        {
            ID_COL: test_f[ID_COL].to_numpy(),
            "y_true": y_test,
            "y_pred": pred_test,
            "residual": residual,
        }
    ).sort_values(ID_COL, kind="mergesort").reset_index(drop=True)

    regime_rows = []
    for q in REGIME_QS:
        rec = regime_metrics(y_test, pred_test, thresholds[q])
        regime_rows.append({"regime": REGIME_NAMES[q], "q": q, "threshold": thresholds[q], **rec})
    # also P25-style bins for the requested quantile file
    edges = [thresholds[0.25], thresholds[0.50], thresholds[0.75]]
    labels = ["P25_below", "P25_P50", "P50_P75", "P75_above"]
    bins = pd.cut(y_test, bins=[-np.inf, edges[0], edges[1], edges[2], np.inf], labels=labels, right=True)
    q_err = (
        pd.DataFrame({"quantile": bins, "MAE": np.abs(residual), "bias": residual, "y_mean": y_test})
        .groupby("quantile", observed=True)
        .agg(MAE=("MAE", "mean"), bias=("bias", "mean"), y_mean=("y_mean", "mean"), n=("MAE", "size"))
        .reset_index()
    )
    q_err["quantile"] = pd.Categorical(q_err["quantile"], categories=labels, ordered=True)
    q_err = q_err.sort_values("quantile").reset_index(drop=True)

    abs_err = np.abs(residual)
    hour_err = pd.DataFrame({"hour": test_f["hour"].to_numpy(), "MAE": abs_err}).groupby("hour", sort=True)["MAE"].mean().reset_index()
    weekday_err = pd.DataFrame({"weekday": test_f["day_of_week"].to_numpy(), "MAE": abs_err}).groupby("weekday", sort=True)["MAE"].mean().reset_index()
    month_err = pd.DataFrame({"month": test_f["month"].to_numpy(), "MAE": abs_err}).groupby("month", sort=True)["MAE"].mean().reset_index()

    metrics_tbl = pd.DataFrame(
        [
            {"split": "test", "model": "Ridge_a0.001_METHOD_B_AR1", **mets},
            {"split": "test", "model": "Naive Lag-24", **naive_mets},
        ]
    )
    regimes = pd.DataFrame(regime_rows)

    REPORTS_FINAL.mkdir(parents=True, exist_ok=True)
    metrics_tbl.to_csv(METRICS_CSV, index=False)
    hour_err.to_csv(HOUR_CSV, index=False)
    month_err.to_csv(MONTH_CSV, index=False)
    weekday_err.to_csv(WEEKDAY_CSV, index=False)
    q_err.to_csv(QUANTILE_CSV, index=False)
    pred_df.to_parquet(PRED_PATH, index=False)

    after = snapshot(rt.PROTECTED)
    if after != before:
        raise FinalTestError("A protected file was modified")

    hashes = output_hashes()
    load_col = "total_load_forecast"
    ren_col = "renewable_forecast_total"
    dev_stats = {
        "y_mean": float(y_dev.mean()),
        "y_std": float(y_dev.std(ddof=0)),
        "load": float(dev[load_col].mean()),
        "ren": float(dev[ren_col].mean()),
    }
    test_stats = {
        "y_mean": float(y_test.mean()),
        "y_std": float(y_test.std(ddof=0)),
        "load": float(test_f[load_col].mean()),
        "ren": float(test_f[ren_col].mean()),
    }
    return {
        "mets": mets,
        "naive": naive_mets,
        "regimes": regimes,
        "hour_err": hour_err,
        "month_err": month_err,
        "weekday_err": weekday_err,
        "hashes": hashes,
        "before": before,
        "after": after,
        "p75_dev": p75_dev,
        "n_test": int(len(pred_df)),
        "n_dropped": n_dropped,
        "dev_stats": dev_stats,
        "test_stats": test_stats,
        "addend": float(addend),
        "ar1_phi": float(ar1_phi),
        "ar1_last_resid": float(ar1_last),
        "thresholds": thresholds,
        "q_err": q_err,
    }


def run() -> dict[str, Any]:
    first = run_once()
    h1 = first["hashes"]
    write_report(
        first["mets"],
        first["naive"],
        first["regimes"],
        first["hour_err"],
        first["month_err"],
        first["weekday_err"],
        first["hashes"],
        first["before"],
        first["after"],
        status="PASS",
        repro="PENDING",
        p75_dev=first["p75_dev"],
        n_test=first["n_test"],
        n_dropped=first["n_dropped"],
        dev_stats=first["dev_stats"],
        test_stats=first["test_stats"],
        addend=first["addend"],
    )
    second = run_once()
    h2 = second["hashes"]
    repro = "PASS" if h1 == h2 else "FAIL"
    status = "PASS" if repro == "PASS" and second["before"] == second["after"] else "FAIL"
    write_report(
        second["mets"],
        second["naive"],
        second["regimes"],
        second["hour_err"],
        second["month_err"],
        second["weekday_err"],
        second["hashes"],
        second["before"],
        second["after"],
        status=status,
        repro=repro,
        p75_dev=second["p75_dev"],
        n_test=second["n_test"],
        n_dropped=second["n_dropped"],
        dev_stats=second["dev_stats"],
        test_stats=second["test_stats"],
        addend=second["addend"],
    )
    second["repro"] = repro
    second["status"] = status
    second["hashes_run1"] = h1
    second["hashes_run2"] = h2
    return second


if __name__ == "__main__":
    result = run()
    m = result["mets"]
    n = result["naive"]
    r = result["regimes"]
    p75 = r[r["regime"] == "P75+"].iloc[0]
    p90 = r[r["regime"] == "P90+"].iloc[0]
    p95 = r[r["regime"] == "P95+"].iloc[0]
    print("=== FINAL TEST EVALUATION ===")
    print(json.dumps(result["hashes_run1"], indent=2))
    print(json.dumps(result["hashes_run2"], indent=2))
    print(f"FINAL_TEST_EVALUATION = {result['status']}")
    print("SELECTED_MODEL = Ridge(alpha=0.001)+METHOD_B+AR(1)")
    print("SELECTED_METHOD = METHOD_B_AR1")
    print(f"VALIDATION_MAE = {VAL_MAE:.6f}")
    print(f"TEST_MAE = {m['MAE']:.6f}")
    print(f"TEST_RMSE = {m['RMSE']:.6f}")
    print(f"TEST_R2 = {m['R2']:.6f}")
    print(f"TEST_SMAPE = {m['sMAPE']:.6f}")
    print(f"TEST_BIAS = {m['bias']:.6f}")
    print(f"TEST_P75_BIAS = {float(p75['bias']):.6f}")
    print(f"TEST_P90_BIAS = {float(p90['bias']):.6f}")
    print(f"TEST_P95_BIAS = {float(p95['bias']):.6f}")
    print(f"NAIVE_TEST_MAE = {n['MAE']:.6f}")
    print(f"MODEL_BEATS_NAIVE = {str(m['MAE'] < n['MAE']).upper()}")
    print("TEST_USED_FOR_TUNING = FALSE")
    print("TEST_USED_FOR_SELECTION = FALSE")
    print(f"REPRODUCIBILITY = {result['repro']}")
    print("PROTECTED_FILES_UNCHANGED = TRUE")
