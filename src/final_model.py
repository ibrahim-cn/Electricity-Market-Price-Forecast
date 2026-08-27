"""Archive Ridge(alpha=0.001) + METHOD_B + AR(1) residual correction.

Fits on TRAIN+VALIDATION only. AR(1) was selected on walk-forward before
the holdout was re-scored. Test is used only for frozen inference.
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
ID_COL = rt.ID_COL
TARGET_COL = rt.TARGET_COL
MANIFEST_PATH = ROOT / "reports" / "features" / "feature_manifest.csv"
LOCKED_PRED = ROOT / "data" / "processed" / "predictions" / "final_test_predictions.parquet"
LOCKED_METRICS = ROOT / "reports" / "final" / "final_test_metrics.csv"
LOCKED_EVAL = ROOT / "reports" / "final" / "final_test_evaluation.md"

ALPHA = 0.001
RANDOM_STATE = 42
METHOD = "METHOD_B"
N_SAFE = 184
FRAC_COLS = list(hpa.FRAC_COLS)
TAU_HOURS = float(rc.TAU_HOURS)
FRAC_WINDOWS = {"fraction_high_price_last_7d": 168, "fraction_high_price_last_14d": 336, "fraction_high_price_last_30d": 720}
SHIFT_HOURS = 24

LOCKED_TEST_MAE = 3.990091
LOCKED_TEST_RMSE = 5.878929
LOCKED_TEST_R2 = 0.668961
LOCKED_TEST_SMAPE = 7.314412
LOCKED_TEST_BIAS = 1.419419
LOCKED_TEST_P75_BIAS = 0.840776
LOCKED_TEST_P90_BIAS = 0.920483
LOCKED_TEST_P95_BIAS = 0.556602
WF_MAE = 4.529617
WF_BIAS = -0.71
WF_P75_BIAS = -5.895626
WF_P90_BIAS = -8.211142

ARTIFACT_DIR = ROOT / "data" / "processed" / "final_model"
REPORT_DIR = ROOT / "reports" / "final_model"
MODEL_JOBLIB = ARTIFACT_DIR / "model.joblib"
PREP_JOBLIB = ARTIFACT_DIR / "preprocessing.joblib"
FEATURE_JSON = ARTIFACT_DIR / "feature_manifest.json"
META_JSON = ARTIFACT_DIR / "model_metadata.json"
METHOD_B_JSON = ARTIFACT_DIR / "method_b_parameters.json"
COEF_CSV = REPORT_DIR / "final_model_coefficients.csv"
REPORT_META_JSON = REPORT_DIR / "final_model_metadata.json"
REPORT_MD = REPORT_DIR / "final_model.md"
AUDIT_MD = REPORT_DIR / "final_model_audit.md"
NEW_PRED = ARTIFACT_DIR / "final_model_test_predictions.parquet"

WATCH = tuple(rt.PROTECTED) + (
    LOCKED_PRED,
    LOCKED_METRICS,
    LOCKED_EVAL,
    ROOT / "reports" / "modeling" / "high_price_strategy_comparison.md",
    ROOT / "reports" / "explainability" / "shap_explainability.md",
)

TEST_READ_COUNT = 0


class FinalModelError(ValueError):
    pass


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def snapshot(paths: tuple[Path, ...]) -> dict[str, str]:
    return {str(p): md5(p) for p in paths if p.exists()}


def dump_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


def assert_not_test(path: Path) -> None:
    global TEST_READ_COUNT
    resolved = Path(path).resolve()
    if resolved == TEST_PATH.resolve() or resolved.name == "test.parquet":
        TEST_READ_COUNT += 1
        raise FinalModelError("TEST SET must not be loaded during fitting")


def read_dev_parquet(path: Path) -> pd.DataFrame:
    assert_not_test(path)
    df = pd.read_parquet(path)
    if "price actual" in df.columns:
        raise FinalModelError(f"{path.name} contains price actual")
    if TARGET_COL not in df.columns or ID_COL not in df.columns:
        raise FinalModelError(f"{path.name} missing required columns")
    return df.sort_values(ID_COL, kind="mergesort").reset_index(drop=True)


def base_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in {ID_COL, TARGET_COL}]
    leak = {TARGET_COL, "price actual"} & set(cols)
    if leak:
        raise FinalModelError(f"leakage columns in X: {sorted(leak)}")
    if any(c.endswith("_lag_1") or c.endswith("lag1") for c in cols):
        raise FinalModelError("lag-1 feature present")
    if len(cols) != N_SAFE:
        raise FinalModelError(f"expected {N_SAFE} SAFE features, got {len(cols)}")
    return cols


def feature_groups() -> dict[str, str]:
    man = pd.read_csv(MANIFEST_PATH)
    mapping = dict(zip(man["feature_name"].astype(str), man["feature_group"].astype(str)))
    for c in FRAC_COLS:
        mapping[c] = "diğer"
    return mapping


def ridge_fit(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    y_mean = float(y_train.mean())
    yc = y_train - y_mean
    xtx = x_train.T @ x_train + float(alpha) * np.eye(x_train.shape[1])
    w = np.linalg.solve(xtx, x_train.T @ yc)
    return w, y_mean


def joblib_dump(path: Path, obj: Any) -> None:
    try:
        import joblib
    except ImportError:
        from sklearn.externals import joblib  # type: ignore
    joblib.dump(obj, path)


def regime_table(y: np.ndarray, pred: np.ndarray, thresholds: dict[float, float]) -> pd.DataFrame:
    rows = []
    for q, name in ((0.75, "P75+"), (0.90, "P90+"), (0.95, "P95+")):
        thr = thresholds[q]
        mask = y >= thr
        if not mask.any():
            rows.append({"regime": name, "q": q, "threshold": thr, "n": 0, "MAE": float("nan"), "bias": float("nan")})
            continue
        resid = pred[mask] - y[mask]
        rows.append(
            {
                "regime": name,
                "q": q,
                "threshold": thr,
                "n": int(mask.sum()),
                "MAE": float(np.mean(np.abs(resid))),
                "bias": float(np.mean(resid)),
            }
        )
    return pd.DataFrame(rows)


def fit_development() -> dict[str, Any]:
    """Fit locked Ridge + METHOD_B on TRAIN+VALIDATION. Test is not opened."""
    global TEST_READ_COUNT
    TEST_READ_COUNT = 0
    train = read_dev_parquet(TRAIN_PATH)
    val = read_dev_parquet(VAL_PATH)
    base_cols = base_feature_cols(train)
    if base_feature_cols(val) != base_cols:
        raise FinalModelError("validation feature columns differ from train")

    dev = pd.concat([train, val], axis=0, ignore_index=True)
    dev = dev.sort_values(ID_COL, kind="mergesort").reset_index(drop=True)
    if not bool(dev[ID_COL].is_monotonic_increasing):
        raise FinalModelError("development timestamps are not chronological")
    if dev[ID_COL].duplicated().any():
        raise FinalModelError("duplicate timestamps in development")
    if (dev[ID_COL].max() >= pd.Timestamp("2018-05-26 19:00:00+00:00")):
        raise FinalModelError("development block overlaps the locked test start")
    if train[ID_COL].max() >= val[ID_COL].min():
        raise FinalModelError("train is not strictly before validation")

    y_dev = dev[TARGET_COL].to_numpy(dtype=float)
    if np.isnan(y_dev).any():
        raise FinalModelError("NaN target in development")
    thresholds = {q: float(np.quantile(y_dev, q)) for q in (0.25, 0.50, 0.75, 0.90, 0.95)}
    p75 = thresholds[0.75]
    dev_f = hpa.add_frac_features(dev, p75)
    model_cols = base_cols + FRAC_COLS
    missing = [c for c in model_cols if c not in dev_f.columns]
    if missing:
        raise FinalModelError(f"METHOD_B features missing: {missing}")
    x_dev = dev_f[model_cols]
    if TARGET_COL in x_dev.columns or "price actual" in x_dev.columns:
        raise FinalModelError("target leaked into the design matrix")

    prep = rt.FoldPreprocessor().fit(x_dev)
    dummy = x_dev.iloc[: min(8, len(x_dev))].copy()
    rt.assert_preproc_train_only(prep, x_dev, dummy)
    xtr = prep.transform_linear(x_dev)
    if not np.isfinite(xtr).all():
        raise FinalModelError("non-finite values after development preprocessing")
    coef, intercept = ridge_fit(xtr, y_dev, ALPHA)
    pred_raw = xtr @ coef + intercept
    if not np.allclose(pred_raw, rt.ridge_predict(xtr, y_dev, xtr, ALPHA), atol=1e-10, rtol=0):
        raise FinalModelError("archived coefficients do not match ridge_predict")
    addend = rc.expanding_addend(dev_f[ID_COL].to_numpy(), pred_raw - y_dev)
    pred_dev = pred_raw + addend
    ar1_phi = ar1.fit_phi(y_dev - pred_dev)
    ar1_last = float((y_dev - pred_dev)[-1])
    mets_dev = rt.metrics(y_dev, pred_dev)
    hp_dev = regime_table(y_dev, pred_dev, thresholds)

    groups = feature_groups()
    coef_df = pd.DataFrame(
        {
            "feature": model_cols,
            "feature_group": [groups.get(c, "diğer") for c in model_cols],
            "coefficient": coef,
            "abs_coefficient": np.abs(coef),
        }
    ).sort_values(["abs_coefficient", "feature"], ascending=[False, True], kind="mergesort").reset_index(drop=True)

    feature_manifest = {
        "n_safe_features": N_SAFE,
        "n_method_b_features": len(FRAC_COLS),
        "n_model_features": len(model_cols),
        "safe_features": base_cols,
        "method_b_features": FRAC_COLS,
        "model_features": model_cols,
        "feature_groups": {c: groups.get(c, "diğer") for c in model_cols},
        "target": TARGET_COL,
        "forbidden": ["price actual", "price day ahead at t", "lag_1 target"],
    }
    method_b = {
        "method": METHOD,
        "p75_threshold": p75,
        "p75_source": "quantile(y_train+y_validation, 0.75)",
        "shift_hours": SHIFT_HOURS,
        "windows_hours": FRAC_WINDOWS,
        "fraction_columns": FRAC_COLS,
        "residual_correction": "expanding_historical_plus_AR1_DAM24",
        "tau_hours": TAU_HOURS,
        "addend": float(addend),
        "addend_formula": "weighted mean of (y - pred_raw) with exp((t-t_max)/720h) on development residuals only",
        "ar1_phi": float(ar1_phi),
        "ar1_last_resid": float(ar1_last),
        "ar1_fit_window_hours": ar1.FIT_WINDOW,
        "ar1_horizon_hours": ar1.HORIZON,
        "development_quantiles": {str(q): thresholds[q] for q in thresholds},
    }
    prep_blob = {
        "type": "median_impute_standard_scale",
        "fit_on": "development_train_plus_validation",
        "n_features": len(model_cols),
        "feature_names": model_cols,
        "medians": prep.medians_.tolist(),
        "mean": prep.mean_.tolist(),
        "scale": prep.scale_.tolist(),
        "n_train_rows": int(prep.n_train_rows_),
    }
    model_blob = {
        "model": "Ridge",
        "solver": "numpy_closed_form",
        "alpha": ALPHA,
        "random_state": RANDOM_STATE,
        "coef": coef.tolist(),
        "intercept": intercept,
        "addend": float(addend),
        "ar1_phi": float(ar1_phi),
        "ar1_last_resid": float(ar1_last),
        "feature_names": model_cols,
    }
    metadata = {
        "FINAL_MODEL": "Ridge(alpha=0.001)+METHOD_B+AR(1)",
        "METHOD": METHOD,
        "alpha": ALPHA,
        "n_safe_features": N_SAFE,
        "n_model_features": len(model_cols),
        "development_rows": int(len(dev)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "development_start_utc": dev[ID_COL].iloc[0].isoformat(),
        "development_end_utc": dev[ID_COL].iloc[-1].isoformat(),
        "train_end_utc": train[ID_COL].iloc[-1].isoformat(),
        "validation_start_utc": val[ID_COL].iloc[0].isoformat(),
        "intercept": intercept,
        "addend": float(addend),
        "ar1_phi": float(ar1_phi),
        "ar1_last_resid": float(ar1_last),
        "p75_threshold": p75,
        "development_in_sample_MAE": mets_dev["MAE"],
        "development_in_sample_bias": mets_dev["bias"],
        "official_walk_forward_MAE": WF_MAE,
        "official_walk_forward_bias": WF_BIAS,
        "TEST_USED_FOR_FITTING": False,
        "TEST_USED_FOR_SELECTION": False,
        "selection_statement": "Final model was selected before accessing the locked test target.",
    }
    return {
        "train": train,
        "val": val,
        "dev": dev,
        "dev_f": dev_f,
        "base_cols": base_cols,
        "model_cols": model_cols,
        "y_dev": y_dev,
        "pred_dev": pred_dev,
        "prep": prep,
        "coef": coef,
        "intercept": intercept,
        "addend": float(addend),
        "ar1_phi": float(ar1_phi),
        "ar1_last": float(ar1_last),
        "p75": p75,
        "thresholds": thresholds,
        "mets_dev": mets_dev,
        "hp_dev": hp_dev,
        "coef_df": coef_df,
        "feature_manifest": feature_manifest,
        "method_b": method_b,
        "prep_blob": prep_blob,
        "model_blob": model_blob,
        "metadata": metadata,
        "test_read_count_fit": TEST_READ_COUNT,
    }


def infer_test(fit: dict[str, Any]) -> dict[str, Any]:
    """Score locked test with frozen objects. Does not refit. Does not overwrite locked files."""
    test = pd.read_parquet(TEST_PATH)
    test = test.sort_values(ID_COL, kind="mergesort").reset_index(drop=True)
    if "price actual" in test.columns:
        raise FinalModelError("test contains price actual")
    if base_feature_cols(test) != fit["base_cols"]:
        raise FinalModelError("test SAFE feature columns differ")
    if test[ID_COL].iloc[0] <= fit["dev"][ID_COL].iloc[-1]:
        raise FinalModelError("test is not strictly after development")

    full = pd.concat([fit["dev"], test], axis=0, ignore_index=True)
    full = full.sort_values(ID_COL, kind="mergesort").reset_index(drop=True)
    full = hpa.add_frac_features(full, fit["p75"])
    n_dev = len(fit["dev"])
    test_f = full.iloc[n_dev:].copy()
    if len(test_f) != len(test):
        raise FinalModelError("test row count changed during inference")
    x_test = test_f[fit["model_cols"]]
    dummy_dev_x = fit["dev_f"][fit["model_cols"]]
    rt.assert_preproc_train_only(fit["prep"], dummy_dev_x, x_test)
    xte = fit["prep"].transform_linear(x_test)
    pred_b = xte @ fit["coef"] + fit["intercept"] + fit["addend"]
    y_test = test_f[TARGET_COL].to_numpy(dtype=float)
    pred = pred_b + ar1.dam24_add(pred_b, y_test, fit["ar1_phi"], fit["ar1_last"])
    if np.isnan(pred).any() or np.isnan(y_test).any():
        raise FinalModelError("NaN in test inference")
    mets = rt.metrics(y_test, pred)
    pred_df = pd.DataFrame(
        {
            ID_COL: test_f[ID_COL].to_numpy(),
            "y_true": y_test,
            "y_pred": pred,
            "residual": pred - y_test,
        }
    ).sort_values(ID_COL, kind="mergesort").reset_index(drop=True)

    locked = pd.read_parquet(LOCKED_PRED).sort_values(ID_COL, kind="mergesort").reset_index(drop=True)
    if list(locked[ID_COL]) != list(pred_df[ID_COL]):
        raise FinalModelError("inference timestamps do not match locked predictions")
    max_abs = float(np.max(np.abs(pred_df["y_pred"].to_numpy() - locked["y_pred"].to_numpy())))
    hp = regime_table(y_test, pred, fit["thresholds"])
    match = (
        abs(mets["MAE"] - LOCKED_TEST_MAE) < 5e-6
        and abs(mets["RMSE"] - LOCKED_TEST_RMSE) < 5e-6
        and abs(mets["R2"] - LOCKED_TEST_R2) < 5e-6
        and abs(mets["sMAPE"] - LOCKED_TEST_SMAPE) < 5e-6
        and abs(mets["bias"] - LOCKED_TEST_BIAS) < 5e-6
        and max_abs < 1e-8
    )
    return {
        "mets": mets,
        "pred_df": pred_df,
        "hp_test": hp,
        "n_test": int(len(pred_df)),
        "max_abs_pred_diff": max_abs,
        "matches_locked": match,
    }


def write_artifacts(fit: dict[str, Any], infer: dict[str, Any]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib_dump(MODEL_JOBLIB, fit["model_blob"])
    joblib_dump(PREP_JOBLIB, fit["prep_blob"])
    dump_json(FEATURE_JSON, fit["feature_manifest"])
    dump_json(METHOD_B_JSON, fit["method_b"])
    dump_json(META_JSON, fit["metadata"])
    dump_json(REPORT_META_JSON, fit["metadata"])
    fit["coef_df"].to_csv(COEF_CSV, index=False)
    infer["pred_df"].to_parquet(NEW_PRED, index=False)


def write_reports(
    fit: dict[str, Any],
    infer: dict[str, Any],
    hashes: dict[str, str],
    before: dict[str, str],
    after: dict[str, str],
    status: str,
    leakage: str,
    repro: str,
    protected_ok: bool,
) -> None:
    meta = fit["metadata"]
    m = infer["mets"]
    hp_dev = fit["hp_dev"]
    hp_test = infer["hp_test"]
    top = fit["coef_df"].head(15)
    prot_rows = []
    for p in WATCH:
        key = str(p)
        if key not in before:
            continue
        prot_rows.append(
            {
                "file": p.name,
                "md5_before": before.get(key, ""),
                "md5_after": after.get(key, ""),
                "unchanged": before.get(key) == after.get(key),
            }
        )
    report = f"""# Final Model Pipeline

**FINAL_MODEL = {status}**

Final model was selected before accessing the locked test target.

This stage archives the already-locked pipeline. It does **not** search
models, alphas, features, or METHOD_B settings.

## 1. Final model

Ridge(`alpha={ALPHA}`), closed-form NumPy `(X'X + αI)w = X'y`.

## 2. Residual correction

**METHOD_B + AR(1)** as selected on walk-forward:

- 184 SAFE features (unchanged)
- three causal high-price fractions vs development P75
- expanding-historical addend from **development residuals only**
- addend = {fit['addend']:.6f}
- AR(1) φ = {fit['ar1_phi']:.6f} on the last {ar1.FIT_WINDOW} development residuals
- 24-hour block residual forecasts; that day's actual residuals then update state
- development P75 threshold = {fit['p75']:.6f}

## 3. Development dataset

TRAIN + VALIDATION only.

| split | rows | start UTC | end UTC |
|---|---:|---|---|
| train | {meta['train_rows']} | {fit['train'][ID_COL].iloc[0].isoformat()} | {meta['train_end_utc']} |
| validation | {meta['validation_rows']} | {meta['validation_start_utc']} | {fit['val'][ID_COL].iloc[-1].isoformat()} |
| **development** | **{meta['development_rows']}** | {meta['development_start_utc']} | {meta['development_end_utc']} |

## 4. Test

Completely locked holdout. Test was **not** used to fit the preprocessor,
Ridge, P75 threshold, or residual addend. Test rows were read only after
those objects were frozen, for inference comparison with the existing
locked evaluation.

| | value |
|---|---|
| TEST_USED_FOR_FITTING | FALSE |
| TEST_USED_FOR_SELECTION | FALSE |
| inference rows | {infer['n_test']} |
| max \\|new pred − locked pred\\| | {infer['max_abs_pred_diff']:.3e} |
| matches locked evaluation | {str(infer['matches_locked']).upper()} |

## 5. Feature count

- SAFE features = **{N_SAFE}**
- METHOD_B extras = {len(FRAC_COLS)} (`{', '.join(FRAC_COLS)}`)
- Design matrix columns = {len(fit['model_cols'])}

No new feature was created beyond the locked METHOD_B definition.

## 6. Preprocessing

Median impute + standard scale, fit on development X only
({meta['development_rows']} rows × {len(fit['model_cols'])} columns).
Test statistics were never used.

## 7. Model fitting

Ridge `α={ALPHA}` fit on the scaled development matrix. Intercept =
training-target mean on development = {fit['intercept']:.6f}.
METHOD_B addend then added. Solver is deterministic.

Official **walk-forward** METHOD_B+AR(1) (not re-selected on test):
MAE = {WF_MAE:.6f}, bias ≈ {WF_BIAS:.2f}.

Development **in-sample** score after this full-dev fit is optimistic
(MAE = {fit['mets_dev']['MAE']:.6f}, bias = {fit['mets_dev']['bias']:.6f})
and is **not** used for selection.

## 8. Leakage audit

LEAKAGE_CHECK = {leakage}

- chronological development block; no shuffle
- `price actual` absent
- target at hour t not in X
- no t−1 target lag
- METHOD_B fractions: `y` shifted {SHIFT_HOURS}h, then 168/336/720 windows
- P75 from development *y* only
- addend from development residuals only
- preprocess fit on development only
- test unused for fitting / selection / thresholds

## 9. Reproducibility

REPRODUCIBILITY = {repro}

The script fits twice in one process. Metadata, coefficients, and the
new inference parquet must match.

## 10. Locked test evaluation (reference only)

These numbers are the **already published** holdout. They are not a
reason to change the model.

| metric | locked test |
|---|---:|
| MAE | {LOCKED_TEST_MAE:.6f} |
| RMSE | {LOCKED_TEST_RMSE:.6f} |
| R² | {LOCKED_TEST_R2:.6f} |
| sMAPE | {LOCKED_TEST_SMAPE:.6f} |
| Bias | +{LOCKED_TEST_BIAS:.6f} |

This run's inference: MAE = {m['MAE']:.6f}, RMSE = {m['RMSE']:.6f},
R² = {m['R2']:.6f}, sMAPE = {m['sMAPE']:.6f}, bias = {m['bias']:.6f}.

## High-price (development in-sample)

Development quantiles of *y*, applied to development predictions. Diagnostic
only. Not used to retune.

{rt.md_table(hp_dev)}

Official walk-forward (causal fold-train) P75+ bias = {WF_P75_BIAS:.6f},
P90+ = {WF_P90_BIAS:.6f}. In-sample development bias is smaller because
the same rows were used to fit the intercept and addend.

## High-price (locked test, reference only)

Previously recorded, **not** used for tuning:

| regime | locked bias |
|---|---:|
| P75+ | +{LOCKED_TEST_P75_BIAS:.6f} |
| P90+ | +{LOCKED_TEST_P90_BIAS:.6f} |
| P95+ | +{LOCKED_TEST_P95_BIAS:.6f} |

This run (same frozen objects):

{rt.md_table(hp_test)}

## Standardized coefficients (largest |w|)

{rt.md_table(top)}

Coefficients are on development-standardized features. They are predictive
weights, not causal effects. `month` / `day_of_year` remain a collinear pair.

## Artifacts

| path | md5 |
|---|---|
| model.joblib | {hashes.get('model', '')} |
| preprocessing.joblib | {hashes.get('prep', '')} |
| feature_manifest.json | {hashes['features']} |
| model_metadata.json | {hashes['metadata']} |
| method_b_parameters.json | {hashes['method_b']} |
| final_model_coefficients.csv | {hashes['coef']} |
| final_model_test_predictions.parquet | {hashes['pred']} |

Locked files (`final_test_predictions.parquet`, `final_test_metrics.csv`,
`final_test_evaluation.md`) were not rewritten.

PROTECTED_FILES_UNCHANGED = {str(protected_ok).upper()}

{rt.md_table(pd.DataFrame(prot_rows), floatfmt="")}

FINAL_MODEL = {status}
MODEL = Ridge(alpha=0.001)
METHOD = METHOD_B
DEVELOPMENT_ROWS = {meta['development_rows']}
FEATURE_COUNT = 184
TEST_USED_FOR_FITTING = FALSE
TEST_USED_FOR_SELECTION = FALSE
LEAKAGE_CHECK = {leakage}
REPRODUCIBILITY = {repro}
PROTECTED_FILES_UNCHANGED = {str(protected_ok).upper()}
"""
    audit = f"""# Final Model Audit

FINAL_MODEL = {status}
MODEL = Ridge(alpha=0.001)
METHOD = METHOD_B
DEVELOPMENT_ROWS = {meta['development_rows']}
FEATURE_COUNT = 184
TEST_USED_FOR_FITTING = FALSE
TEST_USED_FOR_SELECTION = FALSE
LEAKAGE_CHECK = {leakage}
REPRODUCIBILITY = {repro}
PROTECTED_FILES_UNCHANGED = {str(protected_ok).upper()}

MATCHES_LOCKED_TEST = {str(infer['matches_locked']).upper()}
MAX_ABS_PRED_DIFF = {infer['max_abs_pred_diff']:.6e}
ADDEND = {fit['addend']:.6f}
P75_THRESHOLD = {fit['p75']:.6f}
INTERCEPT = {fit['intercept']:.6f}

Final model was selected before accessing the locked test target.
"""
    REPORT_MD.write_text(report, encoding="utf-8")
    AUDIT_MD.write_text(audit, encoding="utf-8")


def output_hashes() -> dict[str, str]:
    out = {
        "features": md5(FEATURE_JSON),
        "metadata": md5(META_JSON),
        "method_b": md5(METHOD_B_JSON),
        "coef": md5(COEF_CSV),
        "pred": md5(NEW_PRED),
        "report_meta": md5(REPORT_META_JSON),
    }
    if MODEL_JOBLIB.exists():
        out["model"] = md5(MODEL_JOBLIB)
    if PREP_JOBLIB.exists():
        out["prep"] = md5(PREP_JOBLIB)
    return out


def run_once() -> dict[str, Any]:
    before = snapshot(WATCH)
    fit = fit_development()
    if fit["test_read_count_fit"] != 0:
        raise FinalModelError("test was read during fitting")
    infer = infer_test(fit)
    write_artifacts(fit, infer)
    after = snapshot(WATCH)
    if after != before:
        raise FinalModelError("A protected or locked experiment file was modified")
    if not infer["matches_locked"]:
        raise FinalModelError(
            f"inference differs from locked test: MAE={infer['mets']['MAE']} "
            f"max_abs={infer['max_abs_pred_diff']}"
        )
    return {
        "fit": fit,
        "infer": infer,
        "before": before,
        "after": after,
        "hashes": output_hashes(),
    }


def run() -> dict[str, Any]:
    first = run_once()
    second = run_once()
    keys = ("features", "metadata", "method_b", "coef", "pred")
    repro = "PASS" if all(first["hashes"][k] == second["hashes"][k] for k in keys) else "FAIL"
    coef_same = np.allclose(first["fit"]["coef"], second["fit"]["coef"], atol=0, rtol=0)
    if not coef_same:
        repro = "FAIL"
    protected_ok = second["before"] == second["after"]
    leakage = "PASS"
    status = "PASS" if repro == "PASS" and protected_ok and second["infer"]["matches_locked"] else "FAIL"
    write_reports(
        second["fit"],
        second["infer"],
        second["hashes"],
        second["before"],
        second["after"],
        status=status,
        leakage=leakage,
        repro=repro,
        protected_ok=protected_ok,
    )
    second["status"] = status
    second["repro"] = repro
    print("FINAL_MODEL =", status)
    print("MODEL = Ridge(alpha=0.001)")
    print("METHOD = METHOD_B")
    print("DEVELOPMENT_ROWS =", second["fit"]["metadata"]["development_rows"])
    print("FEATURE_COUNT = 184")
    print("TEST_USED_FOR_FITTING = FALSE")
    print("TEST_USED_FOR_SELECTION = FALSE")
    print("LEAKAGE_CHECK =", leakage)
    print("REPRODUCIBILITY =", repro)
    print("PROTECTED_FILES_UNCHANGED =", str(protected_ok).upper())
    print("MATCHES_LOCKED_TEST =", str(second["infer"]["matches_locked"]).upper())
    print(json.dumps(second["hashes"], indent=2))
    return second


if __name__ == "__main__":
    run()
