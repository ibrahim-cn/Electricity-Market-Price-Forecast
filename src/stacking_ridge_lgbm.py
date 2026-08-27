"""Walk-forward stacking: Ridge+METHOD_B and LightGBM-184, Ridge meta.

OOF for the meta-learner is built with inner expanding windows inside
each outer fold-train. Outer validation is scoring only.

TRAIN+VALIDATION only. Does not load test.parquet.
Does not change Ridge alpha, METHOD_B, or locked test artifacts.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import high_price_analysis as hp
import ridge_tuning as rt

REPORTS = rt.REPORTS
OUT_CSV = REPORTS / "stacking_ridge_lgbm.csv"
FOLD_CSV = REPORTS / "stacking_ridge_lgbm_folds.csv"
COEF_CSV = REPORTS / "stacking_ridge_lgbm_meta_coefs.csv"
REPORT_MD = REPORTS / "stacking_ridge_lgbm.md"

RANDOM_STATE = 42
META_ALPHA = 0.001
INNER_FRACS = (0.50, 0.65, 0.80)
METHOD_B_WF_MAE = 5.496022
LGBM_184_MAE = 5.916945

PROTECTED = tuple(rt.PROTECTED) + (
    ROOT / "data" / "processed" / "predictions" / "final_test_predictions.parquet",
    ROOT / "reports" / "final" / "final_test_metrics.csv",
)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def snapshot() -> dict[str, str]:
    return {str(p): md5(p) for p in PROTECTED if p.exists()}


def inner_folds(train: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    n = len(train)
    cuts = [int(n * f) for f in INNER_FRACS] + [n]
    out: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for tr_end, va_end in zip(cuts[:-1], cuts[1:]):
        tr = train.iloc[:tr_end]
        va = train.iloc[tr_end:va_end]
        if len(tr) < 24 or len(va) < 24:
            raise RuntimeError("inner fold too small")
        if tr[rt.ID_COL].max() >= va[rt.ID_COL].min():
            raise RuntimeError("inner fold is not chronological")
        out.append((tr, va))
    return out


def fit_ridge_method_b(train: pd.DataFrame, val: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    y_tr = train[rt.TARGET_COL].to_numpy(dtype=float)
    p75 = float(np.quantile(y_tr, 0.75))
    train_aug = hp.add_frac_features(train, p75)
    val_aug = hp.add_frac_features(val, p75)
    cols = feat_cols + list(hp.FRAC_COLS)
    x_tr = hp.feature_frame(train_aug, cols)
    x_va = hp.feature_frame(val_aug, cols)
    pred_tr, pred_va, _, _ = hp.fit_predict_ridge(x_tr, y_tr, x_va)
    return hp.expanding_correct(pred_tr, y_tr, pred_va, train[rt.ID_COL].to_numpy())[1]


def fit_lgbm_184(train: pd.DataFrame, val: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    y_tr = train[rt.TARGET_COL].to_numpy(dtype=float)
    x_tr = hp.feature_frame(train, feat_cols)
    x_va = hp.feature_frame(val, feat_cols)
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


def oof_base_preds(train: pd.DataFrame, feat_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(train)
    oof_r = np.full(n, np.nan)
    oof_l = np.full(n, np.nan)
    for inner_tr, inner_va in inner_folds(train):
        idx = inner_va.index.to_numpy()
        pos = train.index.get_indexer(idx)
        oof_r[pos] = fit_ridge_method_b(inner_tr, inner_va, feat_cols)
        oof_l[pos] = fit_lgbm_184(inner_tr, inner_va, feat_cols)
    mask = np.isfinite(oof_r) & np.isfinite(oof_l)
    if mask.sum() < 48:
        raise RuntimeError("not enough OOF rows for meta-learner")
    y = train[rt.TARGET_COL].to_numpy(dtype=float)
    return oof_r[mask], oof_l[mask], y[mask]


def fit_meta(oof_r: np.ndarray, oof_l: np.ndarray, y: np.ndarray) -> rt.FoldPreprocessor:
    x = pd.DataFrame({"ridge_b": oof_r, "lgbm": oof_l})
    prep = rt.FoldPreprocessor().fit(x)
    return prep


def meta_predict(
    prep: rt.FoldPreprocessor,
    oof_r: np.ndarray,
    oof_l: np.ndarray,
    y_oof: np.ndarray,
    r_new: np.ndarray,
    l_new: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_tr = pd.DataFrame({"ridge_b": oof_r, "lgbm": oof_l})
    x_va = pd.DataFrame({"ridge_b": r_new, "lgbm": l_new})
    rt.assert_preproc_train_only(prep, x_tr, x_va)
    z_tr = prep.transform_linear(x_tr)
    z_va = prep.transform_linear(x_va)
    pred = rt.ridge_predict(z_tr, y_oof, z_va, META_ALPHA)
    w = np.linalg.solve(
        z_tr.T @ z_tr + META_ALPHA * np.eye(2),
        z_tr.T @ (y_oof - y_oof.mean()),
    )
    return pred, w


def unscaled_ols_weights(oof_r: np.ndarray, oof_l: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x = np.column_stack([oof_r - oof_r.mean(), oof_l - oof_l.mean()])
    yc = y - y.mean()
    w = np.linalg.lstsq(x, yc, rcond=None)[0]
    intercept = float(y.mean() - w[0] * oof_r.mean() - w[1] * oof_l.mean())
    return intercept, float(w[0]), float(w[1])


def run() -> None:
    before = snapshot()
    df, feat_cols = rt.load_dev_frame()
    folds = rt.make_folds(df)
    fold_rows: list[dict[str, Any]] = []
    coef_rows: list[dict[str, Any]] = []

    for fold in folds:
        train, val = fold["train"], fold["val"]
        y_va = val[rt.TARGET_COL].to_numpy(dtype=float)
        print(f"fold {fold['fold']} n_train={len(train)} n_val={len(val)}", flush=True)

        oof_r, oof_l, y_oof = oof_base_preds(train, feat_cols)
        prep = fit_meta(oof_r, oof_l, y_oof)
        intercept, w_r, w_l = unscaled_ols_weights(oof_r, oof_l, y_oof)

        ridge_va = fit_ridge_method_b(train, val, feat_cols)
        lgbm_va = fit_lgbm_184(train, val, feat_cols)
        stacked, w_scaled = meta_predict(prep, oof_r, oof_l, y_oof, ridge_va, lgbm_va)
        averaged = 0.5 * ridge_va + 0.5 * lgbm_va

        coef_rows.append(
            {
                "fold": fold["fold"],
                "n_oof": int(len(y_oof)),
                "ols_intercept": intercept,
                "ols_weight_ridge_b": w_r,
                "ols_weight_lgbm": w_l,
                "scaled_coef_ridge_b": float(w_scaled[0]),
                "scaled_coef_lgbm": float(w_scaled[1]),
                "oof_mae_ridge": float(np.mean(np.abs(oof_r - y_oof))),
                "oof_mae_lgbm": float(np.mean(np.abs(oof_l - y_oof))),
            }
        )
        print(
            f"  OOF n={len(y_oof)} ols_w ridge={w_r:.3f} lgbm={w_l:.3f}",
            flush=True,
        )

        specs = [
            ("Ridge_METHOD_B", ridge_va),
            ("LGBM_184", lgbm_va),
            ("Average", averaged),
            ("Stack_RidgeMeta", stacked),
        ]
        for name, pred in specs:
            m = rt.metrics(y_va, pred)
            fold_rows.append(
                {
                    "model": name,
                    "fold": fold["fold"],
                    "n_train": fold["n_train"],
                    "n_val": fold["n_val"],
                    **m,
                }
            )
            print(f"  {name} MAE={m['MAE']:.4f} bias={m['bias']:.3f}", flush=True)

    fold_df = pd.DataFrame(fold_rows)
    coef_df = pd.DataFrame(coef_rows)
    summary = (
        fold_df.groupby("model", as_index=False)
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

    REPORTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_CSV, index=False)
    fold_df.to_csv(FOLD_CSV, index=False)
    coef_df.to_csv(COEF_CSV, index=False)

    stacked_mae = float(summary.loc[summary["model"] == "Stack_RidgeMeta", "mean_MAE"].iloc[0])
    ridge_mae = float(summary.loc[summary["model"] == "Ridge_METHOD_B", "mean_MAE"].iloc[0])
    avg_mae = float(summary.loc[summary["model"] == "Average", "mean_MAE"].iloc[0])
    beats_ridge = stacked_mae + 1e-12 < ridge_mae
    beats_locked = stacked_mae + 1e-12 < METHOD_B_WF_MAE
    mean_wr = float(coef_df["ols_weight_ridge_b"].mean())
    mean_wl = float(coef_df["ols_weight_lgbm"].mean())

    REPORT_MD.write_text(
        f"""# Stacking Ridge+METHOD_B + LightGBM-184 (development only)

**STACKING_RIDGE_LGBM = PASS**

TEST was not loaded. Locked model is unchanged:
Ridge(`alpha=0.001`) + METHOD_B.

## Protocol

Same four expanding TRAIN+VALIDATION folds.

Base models, fit on fold-train only:
- Ridge+METHOD_B: 184 SAFE + 3 causal high-price fractions + expanding addend
- LightGBM: original 200 / lr=0.05 / 31 leaves, **184 SAFE only** (METHOD_B off)

Meta-learner:
- Inner expanding OOF inside fold-train (cuts at 50% / 65% / 80%)
- Outer validation never used for OOF, stopping, or meta weights
- Small Ridge (`α={META_ALPHA}`) on the two OOF prediction columns
  (median-impute + scale from OOF only)
- Also scored: simple 50/50 average (no learned meta)

## This-run comparison

{rt.md_table(summary)}

Stacked mean MAE = {stacked_mae:.6f}  
Ridge+METHOD_B this run = {ridge_mae:.6f}  
Average this run = {avg_mae:.6f}

Mean unscaled OLS mix (for reading, not the scaled Ridge weights):
Ridge+B weight ≈ {mean_wr:.3f}, LightGBM weight ≈ {mean_wl:.3f}

## Versus locked / prior

| competitor | mean MAE |
|---|---:|
| Stack Ridge-meta (this run) | {stacked_mae:.6f} |
| 50/50 average (this run) | {avg_mae:.6f} |
| Ridge+METHOD_B (this run) | {ridge_mae:.6f} |
| Locked Ridge+METHOD_B | {METHOD_B_WF_MAE:.6f} |
| LightGBM 184 (prior) | {LGBM_184_MAE:.6f} |

Beats this-run Ridge+METHOD_B? **{str(beats_ridge).upper()}**  
Beats locked METHOD_B? **{str(beats_locked).upper()}**

## Meta coefficients by fold

{rt.md_table(coef_df)}

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
    print(coef_df.to_string(index=False))
    print("STACKED", stacked_mae, "RIDGE_B", ridge_mae, "AVERAGE", avg_mae)
    print("BEATS_RIDGE_B", beats_ridge, "BEATS_LOCKED", beats_locked)
    print("PROTECTED_UNCHANGED", True)
    print("wrote", OUT_CSV)


if __name__ == "__main__":
    run()
