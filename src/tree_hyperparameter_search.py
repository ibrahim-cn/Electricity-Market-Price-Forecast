"""Walk-forward XGBoost / LightGBM hyperparameter search.

TRAIN+VALIDATION expanding folds only. Early stopping uses an inner
chronological tail of fold-train, never the outer fold validation and
never the locked test set.

Does not change Ridge alpha, METHOD_B, or locked test artifacts.
"""

from __future__ import annotations

import hashlib
import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import ridge_tuning as rt

REPORTS = rt.REPORTS
OUT_CSV = REPORTS / "tree_hyperparameter_search.csv"
FOLD_CSV = REPORTS / "tree_hyperparameter_search_folds.csv"
REPORT_MD = REPORTS / "tree_hyperparameter_search.md"

RANDOM_STATE = 42
N_EST_MAX = 500
STOPPING = 40
INNER_FRAC = 0.80
RIDGE_WF_MAE = 5.796612
METHOD_B_WF_MAE = 5.496022
OLD_XGB_MAE = 5.945953
OLD_LGBM_MAE = 5.916945

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


def xgb_grid() -> list[dict[str, Any]]:
    out = []
    for depth, lr, subsample in itertools.product((4, 6, 8), (0.05, 0.1), (0.8, 1.0)):
        out.append(
            {
                "family": "XGBoost",
                "max_depth": depth,
                "learning_rate": lr,
                "subsample": subsample,
                "colsample_bytree": 0.8,
                "min_child_weight": 1,
            }
        )
    return out


def lgb_grid() -> list[dict[str, Any]]:
    out = []
    for leaves, lr, subsample in itertools.product((15, 31, 63), (0.05, 0.1), (0.8, 1.0)):
        out.append(
            {
                "family": "LightGBM",
                "num_leaves": leaves,
                "learning_rate": lr,
                "subsample": subsample,
                "colsample_bytree": 0.8,
                "min_child_samples": 20,
            }
        )
    return out


def fit_xgb(spec: dict[str, Any], x_tr, y_tr, x_stop, y_stop, x_full, y_full, x_va) -> tuple[np.ndarray, int]:
    stopper = XGBRegressor(
        n_estimators=N_EST_MAX,
        max_depth=int(spec["max_depth"]),
        learning_rate=float(spec["learning_rate"]),
        subsample=float(spec["subsample"]),
        colsample_bytree=float(spec["colsample_bytree"]),
        min_child_weight=float(spec["min_child_weight"]),
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
        early_stopping_rounds=STOPPING,
        eval_metric="mae",
    )
    stopper.fit(x_tr, y_tr, eval_set=[(x_stop, y_stop)], verbose=False)
    best_iter = int(stopper.best_iteration) + 1
    best_iter = max(best_iter, 20)
    final = XGBRegressor(
        n_estimators=best_iter,
        max_depth=int(spec["max_depth"]),
        learning_rate=float(spec["learning_rate"]),
        subsample=float(spec["subsample"]),
        colsample_bytree=float(spec["colsample_bytree"]),
        min_child_weight=float(spec["min_child_weight"]),
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
    )
    final.fit(x_full, y_full, verbose=False)
    return final.predict(x_va), best_iter


def fit_lgb(spec: dict[str, Any], x_tr, y_tr, x_stop, y_stop, x_full, y_full, x_va) -> tuple[np.ndarray, int]:
    stopper = LGBMRegressor(
        n_estimators=N_EST_MAX,
        num_leaves=int(spec["num_leaves"]),
        learning_rate=float(spec["learning_rate"]),
        subsample=float(spec["subsample"]),
        subsample_freq=1 if float(spec["subsample"]) < 1.0 else 0,
        colsample_bytree=float(spec["colsample_bytree"]),
        min_child_samples=int(spec["min_child_samples"]),
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=-1,
    )
    stopper.fit(
        x_tr,
        y_tr,
        eval_set=[(x_stop, y_stop)],
        eval_metric="l1",
        callbacks=[early_stopping(STOPPING, verbose=False)],
    )
    best_iter = int(stopper.best_iteration_ or N_EST_MAX)
    best_iter = max(best_iter, 20)
    final = LGBMRegressor(
        n_estimators=best_iter,
        num_leaves=int(spec["num_leaves"]),
        learning_rate=float(spec["learning_rate"]),
        subsample=float(spec["subsample"]),
        subsample_freq=1 if float(spec["subsample"]) < 1.0 else 0,
        colsample_bytree=float(spec["colsample_bytree"]),
        min_child_samples=int(spec["min_child_samples"]),
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=-1,
    )
    final.fit(x_full, y_full)
    return final.predict(x_va), best_iter


def spec_name(spec: dict[str, Any]) -> str:
    if spec["family"] == "XGBoost":
        return (
            f"XGB_d{spec['max_depth']}_lr{spec['learning_rate']}_ss{spec['subsample']}"
        )
    return f"LGBM_nl{spec['num_leaves']}_lr{spec['learning_rate']}_ss{spec['subsample']}"


def run() -> None:
    before = snapshot()
    df, cols = rt.load_dev_frame()
    folds = rt.make_folds(df)
    specs = xgb_grid() + lgb_grid()
    fold_rows: list[dict[str, Any]] = []

    for fold in folds:
        train, val = fold["train"], fold["val"]
        inner_tr, inner_va = inner_split(train)
        x_tr = inner_tr[cols]
        y_tr = inner_tr[rt.TARGET_COL].to_numpy(dtype=float)
        x_stop = inner_va[cols]
        y_stop = inner_va[rt.TARGET_COL].to_numpy(dtype=float)
        x_full = train[cols]
        y_full = train[rt.TARGET_COL].to_numpy(dtype=float)
        x_va = val[cols]
        y_va = val[rt.TARGET_COL].to_numpy(dtype=float)
        print(f"fold {fold['fold']} n_train={len(train)} n_val={len(val)} configs={len(specs)}", flush=True)
        for spec in specs:
            name = spec_name(spec)
            if spec["family"] == "XGBoost":
                pred, n_trees = fit_xgb(spec, x_tr, y_tr, x_stop, y_stop, x_full, y_full, x_va)
            else:
                pred, n_trees = fit_lgb(spec, x_tr, y_tr, x_stop, y_stop, x_full, y_full, x_va)
            m = rt.metrics(y_va, pred)
            row = {
                "model": name,
                "family": spec["family"],
                "fold": fold["fold"],
                "n_trees": n_trees,
                **{k: spec[k] for k in spec if k != "family"},
                **m,
            }
            fold_rows.append(row)
            print(f"  {name} trees={n_trees} MAE={m['MAE']:.4f}", flush=True)

    fold_df = pd.DataFrame(fold_rows)
    group_cols = ["model", "family"]
    summary = (
        fold_df.groupby(group_cols, as_index=False)
        .agg(
            mean_MAE=("MAE", "mean"),
            std_MAE=("MAE", "std"),
            mean_RMSE=("RMSE", "mean"),
            mean_R2=("R2", "mean"),
            mean_sMAPE=("sMAPE", "mean"),
            mean_bias=("bias", "mean"),
            mean_trees=("n_trees", "mean"),
            n_folds=("fold", "nunique"),
        )
        .sort_values("mean_MAE")
        .reset_index(drop=True)
    )
    best = summary.iloc[0]
    best_xgb = summary[summary["family"] == "XGBoost"].iloc[0]
    best_lgb = summary[summary["family"] == "LightGBM"].iloc[0]
    beats_ridge = float(best["mean_MAE"]) + 1e-12 < RIDGE_WF_MAE
    beats_method_b = float(best["mean_MAE"]) + 1e-12 < METHOD_B_WF_MAE

    REPORTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_CSV, index=False)
    fold_df.to_csv(FOLD_CSV, index=False)
    top = summary.head(12)
    REPORT_MD.write_text(
        f"""# Tree hyperparameter search (development only)

**TREE_HYPERPARAMETER_SEARCH = PASS**

TEST was not loaded. Locked model is unchanged:
Ridge(`alpha=0.001`) + METHOD_B.

## Protocol

- Same 4 expanding TRAIN+VALIDATION folds as Ridge tuning.
- Early stopping on the last 20% of **fold-train** (chronological).
- Outer fold validation is used only for scoring, not for stopping.
- Trees see native NaNs (same as the original family comparison).
- Grid: XGBoost depth×lr×subsample (12) and LightGBM leaves×lr×subsample (12).
- `n_estimators` cap {N_EST_MAX}, stopping rounds {STOPPING}, then refit on full fold-train.

## Best configs

| competitor | mean MAE | MAE std | mean bias | mean trees |
|---|---:|---:|---:|---:|
| Best in this search (`{best['model']}`) | {best['mean_MAE']:.6f} | {best['std_MAE']:.6f} | {best['mean_bias']:.6f} | {best['mean_trees']:.1f} |
| Best XGBoost (`{best_xgb['model']}`) | {best_xgb['mean_MAE']:.6f} | {best_xgb['std_MAE']:.6f} | {best_xgb['mean_bias']:.6f} | {best_xgb['mean_trees']:.1f} |
| Best LightGBM (`{best_lgb['model']}`) | {best_lgb['mean_MAE']:.6f} | {best_lgb['std_MAE']:.6f} | {best_lgb['mean_bias']:.6f} | {best_lgb['mean_trees']:.1f} |
| Original tree XGBoost (untuned) | {OLD_XGB_MAE:.6f} | — | — | 200 |
| Original LightGBM (untuned) | {OLD_LGBM_MAE:.6f} | — | — | 200 |
| Ridge α=0.001 (184 SAFE, no METHOD_B) | {RIDGE_WF_MAE:.6f} | — | — | — |
| Locked Ridge+METHOD_B | {METHOD_B_WF_MAE:.6f} | — | — | — |

Beats Ridge α=0.001? **{str(beats_ridge).upper()}**  
Beats locked METHOD_B? **{str(beats_method_b).upper()}**

## Top 12 by mean MAE

{rt.md_table(top)}

## Decision

LOCKED_MODEL_UNCHANGED = TRUE  
TEST_USED_FOR_SELECTION = FALSE  
""",
        encoding="utf-8",
    )
    after = snapshot()
    if after != before:
        raise RuntimeError("A protected file changed")
    print(summary.head(15).to_string(index=False))
    print("BEST", best["model"], float(best["mean_MAE"]))
    print("BEATS_RIDGE", beats_ridge, "BEATS_METHOD_B", beats_method_b)
    print("PROTECTED_UNCHANGED", True)


if __name__ == "__main__":
    run()
