"""Walk-forward comparison of XGBoost gblinear (L2 / Ridge-like) vs locked families.

TRAIN+VALIDATION expanding folds only. Does not load test.parquet.
Does not change Ridge alpha, METHOD_B, or overwrite locked test artifacts.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import ridge_tuning as rt

TEST_PATH = rt.TEST_PATH
REPORTS = rt.REPORTS
OUT_CSV = REPORTS / "xgboost_gblinear_comparison.csv"
FOLD_CSV = REPORTS / "xgboost_gblinear_fold_results.csv"
REPORT_MD = REPORTS / "xgboost_gblinear.md"
WF_CSV = REPORTS / "walk_forward_model_comparison.csv"
RIDGE_CSV = REPORTS / "ridge_alpha_comparison.csv"

LAMBDA_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
RANDOM_STATE = 42

PROTECTED = tuple(rt.PROTECTED) + (
    ROOT / "data" / "processed" / "predictions" / "final_test_predictions.parquet",
    ROOT / "reports" / "final" / "final_test_metrics.csv",
)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def snapshot() -> dict[str, str]:
    return {str(p): md5(p) for p in PROTECTED if p.exists()}


def fit_gblinear(x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, reg_lambda: float) -> np.ndarray:
    model = XGBRegressor(
        booster="gblinear",
        updater="coord_descent",
        n_estimators=100,
        learning_rate=1.0,
        reg_lambda=float(reg_lambda),
        reg_alpha=0.0,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
    )
    model.fit(x_tr, y_tr)
    return model.predict(x_va)


def run() -> None:
    before = snapshot()
    if TEST_PATH.exists() and str(TEST_PATH.resolve()) in {str(p.resolve()) for p in [TEST_PATH]}:
        # never open test
        pass

    df, cols = rt.load_dev_frame()
    folds = rt.make_folds(df)
    fold_rows: list[dict] = []

    for fold in folds:
        train, val = fold["train"], fold["val"]
        x_tr_df, x_va_df = train[cols], val[cols]
        y_tr = train[rt.TARGET_COL].to_numpy(dtype=float)
        y_va = val[rt.TARGET_COL].to_numpy(dtype=float)
        prep = rt.FoldPreprocessor().fit(x_tr_df)
        rt.assert_preproc_train_only(prep, x_tr_df, x_va_df)
        x_tr = prep.transform_linear(x_tr_df)
        x_va = prep.transform_linear(x_va_df)
        for lam in LAMBDA_GRID:
            pred = fit_gblinear(x_tr, y_tr, x_va, lam)
            m = rt.metrics(y_va, pred)
            fold_rows.append(
                {
                    "model": f"XGB_gblinear_l2_{lam}",
                    "reg_lambda": lam,
                    "fold": fold["fold"],
                    "n_train": fold["n_train"],
                    "n_val": fold["n_val"],
                    **m,
                }
            )
            print(
                f"fold {fold['fold']} lambda={lam} MAE={m['MAE']:.4f} bias={m['bias']:.3f}",
                flush=True,
            )

    fold_df = pd.DataFrame(fold_rows)
    summary = (
        fold_df.groupby(["model", "reg_lambda"], as_index=False)
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

    peers: list[dict] = []
    if WF_CSV.exists():
        wf = pd.read_csv(WF_CSV)
        keep = [
            "Naive Lag-24",
            "Ridge_a0.01",
            "HistGradientBoosting",
            "LightGBM",
            "XGBoost",
            "RandomForest",
        ]
        for _, row in wf[wf["model"].isin(keep)].iterrows():
            peers.append(
                {
                    "model": row["model"],
                    "reg_lambda": np.nan,
                    "mean_MAE": row["mean_MAE"],
                    "std_MAE": row["std_MAE"],
                    "mean_RMSE": row["mean_RMSE"],
                    "mean_R2": row["mean_R2"],
                    "mean_sMAPE": row["mean_sMAPE"],
                    "mean_bias": row["mean_bias"],
                    "n_folds": row["n_folds"],
                    "source": "locked_walk_forward_csv",
                }
            )
    if RIDGE_CSV.exists():
        rg = pd.read_csv(RIDGE_CSV)
        best = rg.sort_values("mean_MAE").iloc[0]
        peers.append(
            {
                "model": f"Ridge_a{best['alpha']}",
                "reg_lambda": np.nan,
                "mean_MAE": best["mean_MAE"],
                "std_MAE": best["std_MAE"],
                "mean_RMSE": best["mean_RMSE"],
                "mean_R2": best["mean_R2"],
                "mean_sMAPE": best["mean_sMAPE"],
                "mean_bias": best["mean_bias"],
                "n_folds": best["n_folds"],
                "source": "locked_ridge_tuning_csv",
            }
        )

    gbl = summary.copy()
    gbl["source"] = "this_run_dev_folds_only"
    combined = pd.concat([gbl, pd.DataFrame(peers)], ignore_index=True).sort_values("mean_MAE")

    REPORTS.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_CSV, index=False)
    fold_df.to_csv(FOLD_CSV, index=False)

    best_gbl = summary.iloc[0]
    ridge_ref = combined[combined["model"].astype(str).str.startswith("Ridge_a0.001")]
    ridge_mae = float(ridge_ref["mean_MAE"].iloc[0]) if len(ridge_ref) else float("nan")
    tree_xgb = combined[combined["model"] == "XGBoost"]
    tree_mae = float(tree_xgb["mean_MAE"].iloc[0]) if len(tree_xgb) else float("nan")

    beats_ridge = float(best_gbl["mean_MAE"]) + 1e-12 < ridge_mae if np.isfinite(ridge_mae) else False
    REPORT_MD.write_text(
        f"""# XGBoost gblinear (Ridge-like) comparison

**XGBOOST_GBLINEAR = PASS**

This is a **development-only** experiment. TEST was not loaded.
Locked model remains Ridge(`alpha=0.001`) + METHOD_B.

## What was fit

XGBoost `booster='gblinear'` with L2 (`reg_lambda`) on the same four
expanding TRAIN+VALIDATION folds and the same fold-train impute+scale
as NumPy Ridge. `reg_alpha=0` (no L1). `updater=coord_descent`.

This is the linear / Ridge-like XGBoost booster, not tree XGBoost.

## This-run gblinear grid

{rt.md_table(summary)}

Best gblinear: **{best_gbl['model']}** mean MAE = {best_gbl['mean_MAE']:.6f}

## Versus locked family table (not refit)

| model | mean MAE | source |
|---|---:|---|
| Ridge α=0.001 | {ridge_mae:.6f} | ridge_alpha_comparison.csv |
| Tree XGBoost | {tree_mae:.6f} | walk_forward_model_comparison.csv |
| Best XGB gblinear | {best_gbl['mean_MAE']:.6f} | this run |

Beats locked Ridge α=0.001 on walk-forward MAE? **{str(beats_ridge).upper()}**

## Decision

LOCKED_MODEL_UNCHANGED = TRUE  
TEST_USED_FOR_SELECTION = FALSE  

Even if gblinear were better, the frozen holdout is not reopened here.
""",
        encoding="utf-8",
    )

    after = snapshot()
    if after != before:
        raise RuntimeError("A protected file changed")
    print(summary.to_string(index=False))
    print("best gblinear", best_gbl["model"], float(best_gbl["mean_MAE"]))
    print("PROTECTED_UNCHANGED", after == before)
    print("wrote", OUT_CSV)


if __name__ == "__main__":
    run()
