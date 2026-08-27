"""Tuned XGBoost gblinear (Ridge-like) walk-forward search.

TRAIN+VALIDATION expanding folds only. Early stopping uses an inner
chronological tail of fold-train, never the outer fold validation and
never the locked test set.

Does not change Ridge alpha, METHOD_B, or locked test artifacts.
Does not overwrite the earlier coarse gblinear comparison files.
"""

from __future__ import annotations

import hashlib
import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import ridge_tuning as rt

REPORTS = rt.REPORTS
OUT_CSV = REPORTS / "xgboost_gblinear_tuned.csv"
FOLD_CSV = REPORTS / "xgboost_gblinear_tuned_folds.csv"
REPORT_MD = REPORTS / "xgboost_gblinear_tuned.md"
WF_CSV = REPORTS / "walk_forward_model_comparison.csv"
RIDGE_CSV = REPORTS / "ridge_alpha_comparison.csv"
OLD_GBL_CSV = REPORTS / "xgboost_gblinear_comparison.csv"

RANDOM_STATE = 42
N_EST_MAX = 300
STOPPING = 40
INNER_FRAC = 0.80
RIDGE_WF_MAE = 5.796612
METHOD_B_WF_MAE = 5.496022
OLD_GBL_MAE = 6.013412

LAMBDA_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2)
ETA_GRID = (0.1, 0.3, 1.0)
SELECTORS = ("cyclic", "shuffle")

PROTECTED = tuple(rt.PROTECTED) + (
    ROOT / "data" / "processed" / "predictions" / "final_test_predictions.parquet",
    ROOT / "reports" / "final" / "final_test_metrics.csv",
)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def snapshot() -> dict[str, str]:
    return {str(p): md5(p) for p in PROTECTED if p.exists()}


def inner_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = max(int(len(train) * INNER_FRAC), 24)
    if len(train) - cut < 24:
        cut = max(len(train) - 24, 24)
    inner_tr = train.iloc[:cut]
    inner_va = train.iloc[cut:]
    if inner_tr[rt.ID_COL].max() >= inner_va[rt.ID_COL].min():
        raise RuntimeError("inner split is not chronological")
    return inner_tr, inner_va


def configs() -> list[dict[str, Any]]:
    out = []
    for lam, eta, sel in itertools.product(LAMBDA_GRID, ETA_GRID, SELECTORS):
        out.append(
            {
                "reg_lambda": float(lam),
                "learning_rate": float(eta),
                "feature_selector": sel,
            }
        )
    return out


def spec_name(spec: dict[str, Any]) -> str:
    lam = spec["reg_lambda"]
    lam_s = f"{lam:.0e}".replace("+", "")
    eta = spec["learning_rate"]
    eta_s = str(eta).replace(".", "p")
    return f"gblin_{lam_s}_eta{eta_s}_{spec['feature_selector']}"


def fit_gblinear(
    spec: dict[str, Any],
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_stop: np.ndarray,
    y_stop: np.ndarray,
    x_full: np.ndarray,
    y_full: np.ndarray,
    x_va: np.ndarray,
) -> tuple[np.ndarray, int]:
    def kw(base_score: float) -> dict[str, Any]:
        return dict(
            booster="gblinear",
            updater="coord_descent",
            feature_selector=spec["feature_selector"],
            learning_rate=float(spec["learning_rate"]),
            reg_lambda=float(spec["reg_lambda"]),
            reg_alpha=0.0,
            objective="reg:squarederror",
            base_score=float(base_score),
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0,
        )

    stopper = XGBRegressor(
        n_estimators=N_EST_MAX,
        early_stopping_rounds=STOPPING,
        eval_metric="mae",
        **kw(float(np.mean(y_tr))),
    )
    stopper.fit(x_tr, y_tr, eval_set=[(x_stop, y_stop)], verbose=False)
    best_iter = getattr(stopper, "best_iteration", None)
    if best_iter is None:
        n_rounds = N_EST_MAX
    else:
        n_rounds = max(int(best_iter) + 1, 10)
    final = XGBRegressor(n_estimators=n_rounds, **kw(float(np.mean(y_full))))
    final.fit(x_full, y_full)
    return final.predict(x_va), n_rounds


def peer_rows() -> list[dict[str, Any]]:
    peers: list[dict[str, Any]] = []
    if WF_CSV.exists():
        wf = pd.read_csv(WF_CSV)
        keep = ["Naive Lag-24", "Ridge_a0.01", "LightGBM", "XGBoost"]
        for _, row in wf[wf["model"].isin(keep)].iterrows():
            peers.append(
                {
                    "model": row["model"],
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
    if OLD_GBL_CSV.exists():
        old = pd.read_csv(OLD_GBL_CSV)
        gbl = old[old["model"].astype(str).str.startswith("XGB_gblinear")]
        if len(gbl):
            row = gbl.sort_values("mean_MAE").iloc[0]
            peers.append(
                {
                    "model": f"coarse_{row['model']}",
                    "mean_MAE": row["mean_MAE"],
                    "std_MAE": row["std_MAE"],
                    "mean_RMSE": row["mean_RMSE"],
                    "mean_R2": row["mean_R2"],
                    "mean_sMAPE": row["mean_sMAPE"],
                    "mean_bias": row["mean_bias"],
                    "n_folds": row["n_folds"],
                    "source": "coarse_gblinear_csv",
                }
            )
    return peers


def run() -> None:
    before = snapshot()
    df, cols = rt.load_dev_frame()
    folds = rt.make_folds(df)
    specs = configs()
    fold_rows: list[dict[str, Any]] = []

    for fold in folds:
        train, val = fold["train"], fold["val"]
        inner_tr, inner_va = inner_split(train)
        y_inner = inner_tr[rt.TARGET_COL].to_numpy(dtype=float)
        y_stop = inner_va[rt.TARGET_COL].to_numpy(dtype=float)
        y_full = train[rt.TARGET_COL].to_numpy(dtype=float)
        y_va = val[rt.TARGET_COL].to_numpy(dtype=float)

        prep_inner = rt.FoldPreprocessor().fit(inner_tr[cols])
        rt.assert_preproc_train_only(prep_inner, inner_tr[cols], inner_va[cols])
        x_tr = prep_inner.transform_linear(inner_tr[cols])
        x_stop = prep_inner.transform_linear(inner_va[cols])

        prep_full = rt.FoldPreprocessor().fit(train[cols])
        rt.assert_preproc_train_only(prep_full, train[cols], val[cols])
        x_full = prep_full.transform_linear(train[cols])
        x_va = prep_full.transform_linear(val[cols])

        print(
            f"fold {fold['fold']} n_train={len(train)} n_val={len(val)} configs={len(specs)}",
            flush=True,
        )
        for spec in specs:
            name = spec_name(spec)
            pred, n_rounds = fit_gblinear(
                spec, x_tr, y_inner, x_stop, y_stop, x_full, y_full, x_va
            )
            m = rt.metrics(y_va, pred)
            fold_rows.append(
                {
                    "model": name,
                    "fold": fold["fold"],
                    "n_rounds": n_rounds,
                    **spec,
                    **m,
                }
            )
            print(
                f"  {name} rounds={n_rounds} MAE={m['MAE']:.4f} bias={m['bias']:.3f}",
                flush=True,
            )

    fold_df = pd.DataFrame(fold_rows)
    summary = (
        fold_df.groupby(
            ["model", "reg_lambda", "learning_rate", "feature_selector"], as_index=False
        )
        .agg(
            mean_MAE=("MAE", "mean"),
            std_MAE=("MAE", "std"),
            mean_RMSE=("RMSE", "mean"),
            mean_R2=("R2", "mean"),
            mean_sMAPE=("sMAPE", "mean"),
            mean_bias=("bias", "mean"),
            mean_rounds=("n_rounds", "mean"),
            n_folds=("fold", "nunique"),
        )
        .sort_values(["mean_MAE", "std_MAE", "reg_lambda"])
        .reset_index(drop=True)
    )
    summary["source"] = "this_run_dev_folds_only"

    peers = pd.DataFrame(peer_rows())
    combined = pd.concat([summary, peers], ignore_index=True, sort=False)
    combined = combined.sort_values("mean_MAE").reset_index(drop=True)

    REPORTS.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_CSV, index=False)
    fold_df.to_csv(FOLD_CSV, index=False)

    best = summary.iloc[0]
    beats_ridge = float(best["mean_MAE"]) + 1e-12 < RIDGE_WF_MAE
    beats_method_b = float(best["mean_MAE"]) + 1e-12 < METHOD_B_WF_MAE
    beats_coarse = float(best["mean_MAE"]) + 1e-12 < OLD_GBL_MAE
    top = summary.head(15).drop(columns=["source"], errors="ignore")

    REPORT_MD.write_text(
        f"""# XGBoost gblinear tuned search (development only)

**XGBOOST_GBLINEAR_TUNED = PASS**

TEST was not loaded. Locked model is unchanged:
Ridge(`alpha=0.001`) + METHOD_B.

## What was tuned

XGBoost `booster='gblinear'` (linear / Ridge-like booster), not tree XGBoost.

Fixed Ridge-like choices:
- `updater=coord_descent`
- `reg_alpha=0` (L2 only)
- same fold-train median impute + scale as NumPy Ridge
- `base_score` = fold-inner y mean

Searched:
- `reg_lambda` ∈ {LAMBDA_GRID}
- `learning_rate` ∈ {ETA_GRID}
- `feature_selector` ∈ {SELECTORS}
- `n_estimators` via early stopping (cap {N_EST_MAX}, patience {STOPPING})
  on the last 20% of **fold-train** only, then refit on full fold-train.

Outer fold validation is scoring only. {len(specs)} configs × 4 folds.

## Best config

| field | value |
|---|---|
| model | `{best['model']}` |
| reg_lambda | {best['reg_lambda']} |
| learning_rate | {best['learning_rate']} |
| feature_selector | {best['feature_selector']} |
| mean rounds | {best['mean_rounds']:.1f} |
| mean MAE | {best['mean_MAE']:.6f} |
| MAE std | {best['std_MAE']:.6f} |
| mean bias | {best['mean_bias']:.6f} |

## Versus locked / prior numbers (not refit)

| competitor | mean MAE |
|---|---:|
| Best tuned gblinear | {best['mean_MAE']:.6f} |
| Coarse gblinear (λ grid only, 100 rounds) | {OLD_GBL_MAE:.6f} |
| Ridge α=0.001 (184 SAFE, no METHOD_B) | {RIDGE_WF_MAE:.6f} |
| Locked Ridge+METHOD_B | {METHOD_B_WF_MAE:.6f} |

Beats coarse gblinear? **{str(beats_coarse).upper()}**  
Beats Ridge α=0.001? **{str(beats_ridge).upper()}**  
Beats locked METHOD_B? **{str(beats_method_b).upper()}**

## Top 15 by mean MAE

{rt.md_table(top)}

## Decision

LOCKED_MODEL_UNCHANGED = TRUE  
TEST_USED_FOR_SELECTION = FALSE  

Even if tuned gblinear were better, the frozen holdout is not reopened here.
""",
        encoding="utf-8",
    )

    after = snapshot()
    if after != before:
        raise RuntimeError("A protected file changed")
    print(summary.head(15).to_string(index=False))
    print("BEST", best["model"], float(best["mean_MAE"]))
    print("BEATS_COARSE", beats_coarse, "BEATS_RIDGE", beats_ridge, "BEATS_METHOD_B", beats_method_b)
    print("PROTECTED_UNCHANGED", True)
    print("wrote", OUT_CSV)


if __name__ == "__main__":
    run()
