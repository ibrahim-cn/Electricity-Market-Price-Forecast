"""LightGBM with the same METHOD_B recipe as locked Ridge.

TRAIN+VALIDATION expanding folds only. Does not load test.parquet.
Does not change Ridge alpha, METHOD_B, or locked test artifacts.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import high_price_analysis as hp
import ridge_tuning as rt

REPORTS = rt.REPORTS
OUT_CSV = REPORTS / "lgbm_method_b.csv"
FOLD_CSV = REPORTS / "lgbm_method_b_folds.csv"
REPORT_MD = REPORTS / "lgbm_method_b.md"

RANDOM_STATE = 42
N_EST_MAX = 500
STOPPING = 40
INNER_FRAC = 0.80
RIDGE_WF_MAE = 5.796612
METHOD_B_WF_MAE = 5.496022
LGBM_184_MAE = 5.916945
LGBM_TUNED_MAE = 5.859459

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


def lgbm_orig() -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=-1,
    )


def fit_lgbm_orig(x_tr, y_tr, x_va) -> tuple[np.ndarray, np.ndarray, int]:
    model = lgbm_orig()
    model.fit(x_tr, y_tr)
    return model.predict(x_tr), model.predict(x_va), 200


def fit_lgbm_tuned(x_inner, y_inner, x_stop, y_stop, x_full, y_full, x_va) -> tuple[np.ndarray, np.ndarray, int]:
    stopper = LGBMRegressor(
        n_estimators=N_EST_MAX,
        num_leaves=63,
        learning_rate=0.1,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=-1,
    )
    stopper.fit(
        x_inner,
        y_inner,
        eval_set=[(x_stop, y_stop)],
        eval_metric="l1",
        callbacks=[early_stopping(STOPPING, verbose=False)],
    )
    n_trees = max(int(stopper.best_iteration_ or N_EST_MAX), 20)
    final = LGBMRegressor(
        n_estimators=n_trees,
        num_leaves=63,
        learning_rate=0.1,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=-1,
    )
    final.fit(x_full, y_full)
    return final.predict(x_full), final.predict(x_va), n_trees


def apply_addend(pred_tr: np.ndarray, y_tr: np.ndarray, pred_va: np.ndarray, train_ts) -> np.ndarray:
    _, pred_va_c = hp.expanding_correct(pred_tr, y_tr, pred_va, train_ts)
    return pred_va_c


def run() -> None:
    before = snapshot()
    df, feat_cols = rt.load_dev_frame()
    folds = rt.make_folds(df)
    fold_rows: list[dict[str, Any]] = []

    for fold in folds:
        train, val = fold["train"], fold["val"]
        y_tr = train[rt.TARGET_COL].to_numpy(dtype=float)
        y_va = val[rt.TARGET_COL].to_numpy(dtype=float)
        p75 = float(np.quantile(y_tr, 0.75))
        p90 = float(np.quantile(y_tr, 0.90))
        train_aug = hp.add_frac_features(train, p75)
        val_aug = hp.add_frac_features(val, p75)
        x184_tr = hp.feature_frame(train_aug, feat_cols)
        x184_va = hp.feature_frame(val_aug, feat_cols)
        cols_b = feat_cols + list(hp.FRAC_COLS)
        xb_tr = hp.feature_frame(train_aug, cols_b)
        xb_va = hp.feature_frame(val_aug, cols_b)
        ts = train[rt.ID_COL].to_numpy()

        inner_tr, inner_va = inner_split(train_aug)
        xb_inner = hp.feature_frame(inner_tr, cols_b)
        xb_stop = hp.feature_frame(inner_va, cols_b)
        y_inner = inner_tr[rt.TARGET_COL].to_numpy(dtype=float)
        y_stop = inner_va[rt.TARGET_COL].to_numpy(dtype=float)

        print(f"fold {fold['fold']} n_train={len(train)} n_val={len(val)} train_P75={p75:.2f}", flush=True)

        pred_tr, pred_va, _, _ = hp.fit_predict_ridge(xb_tr, y_tr, xb_va)
        ridge_b = apply_addend(pred_tr, y_tr, pred_va, ts)

        l184_tr, l184_va, n184 = fit_lgbm_orig(x184_tr, y_tr, x184_va)
        lgbm_b_tr, lgbm_b_va, nb = fit_lgbm_orig(xb_tr, y_tr, xb_va)
        lgbm_b_add = apply_addend(lgbm_b_tr, y_tr, lgbm_b_va, ts)
        tun_tr, tun_va, nt = fit_lgbm_tuned(
            xb_inner, y_inner, xb_stop, y_stop, xb_tr, y_tr, xb_va
        )
        tun_add = apply_addend(tun_tr, y_tr, tun_va, ts)

        specs = [
            ("Ridge_METHOD_B", ridge_b, np.nan),
            ("LGBM_184", l184_va, n184),
            ("LGBM_METHOD_B", lgbm_b_va, nb),
            ("LGBM_METHOD_B_addend", lgbm_b_add, nb),
            ("LGBM_tuned_METHOD_B_addend", tun_add, nt),
        ]
        for name, pred, n_trees in specs:
            m = rt.metrics(y_va, pred)
            p75m = hp.regime_metrics(y_va, pred, hp.regime_mask(y_va, p75))
            p90m = hp.regime_metrics(y_va, pred, hp.regime_mask(y_va, p90))
            fold_rows.append(
                {
                    "model": name,
                    "fold": fold["fold"],
                    "n_train": fold["n_train"],
                    "n_val": fold["n_val"],
                    "n_trees": n_trees,
                    "p75_bias": p75m["bias"],
                    "p90_bias": p90m["bias"],
                    **m,
                }
            )
            print(
                f"  {name} MAE={m['MAE']:.4f} bias={m['bias']:.3f} P75bias={p75m['bias']:.3f}",
                flush=True,
            )

    fold_df = pd.DataFrame(fold_rows)
    summary = (
        fold_df.groupby("model", as_index=False)
        .agg(
            mean_MAE=("MAE", "mean"),
            std_MAE=("MAE", "std"),
            mean_RMSE=("RMSE", "mean"),
            mean_R2=("R2", "mean"),
            mean_sMAPE=("sMAPE", "mean"),
            mean_bias=("bias", "mean"),
            mean_p75_bias=("p75_bias", "mean"),
            mean_p90_bias=("p90_bias", "mean"),
            mean_trees=("n_trees", "mean"),
            n_folds=("fold", "nunique"),
        )
        .sort_values("mean_MAE")
        .reset_index(drop=True)
    )

    REPORTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_CSV, index=False)
    fold_df.to_csv(FOLD_CSV, index=False)

    this_ridge = float(summary.loc[summary["model"] == "Ridge_METHOD_B", "mean_MAE"].iloc[0])
    best_lgbm = summary[summary["model"].str.startswith("LGBM")].iloc[0]
    beats_ridge_b = float(best_lgbm["mean_MAE"]) + 1e-12 < this_ridge
    beats_locked_b = float(best_lgbm["mean_MAE"]) + 1e-12 < METHOD_B_WF_MAE

    REPORT_MD.write_text(
        f"""# LightGBM + METHOD_B (development only)

**LGBM_METHOD_B = PASS**

TEST was not loaded. Locked model is unchanged:
Ridge(`alpha=0.001`) + METHOD_B.

## What was fit

Same four expanding TRAIN+VALIDATION folds. Fold-train P75 defines the
three causal high-price fractions (`y` shifted 24h, windows 168/336/720).
Expanding-historical addend is fit on fold-train residuals only.

| model | recipe |
|---|---|
| Ridge_METHOD_B | 184 SAFE + 3 fractions, Ridge α=0.001, expanding addend (reproduction) |
| LGBM_184 | original LightGBM (200, lr=0.05, leaves=31), 184 SAFE, no METHOD_B |
| LGBM_METHOD_B | same LightGBM, 184 + 3 fractions, no addend |
| LGBM_METHOD_B_addend | same LightGBM, 184 + 3 fractions, expanding addend |
| LGBM_tuned_METHOD_B_addend | tuned LightGBM (leaves=63, lr=0.1, subsample=0.8), fractions + addend; early stopping on last 20% of fold-train |

## This-run comparison

{rt.md_table(summary)}

Best LightGBM in this run: **{best_lgbm['model']}** mean MAE = {best_lgbm['mean_MAE']:.6f}

Ridge METHOD_B this run: {this_ridge:.6f} (locked report {METHOD_B_WF_MAE:.6f})

## Versus prior numbers (not refit)

| competitor | mean MAE |
|---|---:|
| Best LightGBM + METHOD_B (this run) | {best_lgbm['mean_MAE']:.6f} |
| Ridge + METHOD_B (this run) | {this_ridge:.6f} |
| Locked Ridge+METHOD_B | {METHOD_B_WF_MAE:.6f} |
| LightGBM 184 (original family table) | {LGBM_184_MAE:.6f} |
| LightGBM tuned, 184 only | {LGBM_TUNED_MAE:.6f} |
| Ridge α=0.001, 184 only | {RIDGE_WF_MAE:.6f} |

Beats this-run Ridge METHOD_B? **{str(beats_ridge_b).upper()}**  
Beats locked METHOD_B? **{str(beats_locked_b).upper()}**

## Decision

LOCKED_MODEL_UNCHANGED = TRUE  
TEST_USED_FOR_SELECTION = FALSE  
""",
        encoding="utf-8",
    )

    after = snapshot()
    if after != before:
        raise RuntimeError("A protected file changed")
    print(summary.to_string(index=False))
    print("BEST_LGBM", best_lgbm["model"], float(best_lgbm["mean_MAE"]))
    print("RIDGE_METHOD_B", this_ridge)
    print("BEATS_RIDGE_B", beats_ridge_b, "BEATS_LOCKED_B", beats_locked_b)
    print("PROTECTED_UNCHANGED", True)
    print("wrote", OUT_CSV)


if __name__ == "__main__":
    run()
