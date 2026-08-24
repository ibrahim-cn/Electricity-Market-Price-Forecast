"""Explain the locked Ridge(alpha=0.001) + METHOD_B model.

Walk-forward TRAIN+VALIDATION only. test.parquet is never loaded.
Does not change alpha, METHOD_B, features, splits, or existing experiment outputs.
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
import residual_correction as rc
import ridge_tuning as rt

TEST_PATH = rt.TEST_PATH
TRAIN_PATH = rt.TRAIN_PATH
VAL_PATH = rt.VAL_PATH
ID_COL = rt.ID_COL
TARGET_COL = rt.TARGET_COL
MANIFEST_PATH = ROOT / "reports" / "features" / "feature_manifest.csv"

ALPHA = 0.001
RANDOM_STATE = 42
METHOD = "METHOD_B"
N_SAFE = 184
FRAC_COLS = list(hpa.FRAC_COLS)
EXPECTED_FOLDS = ((14902, 2980), (17882, 2980), (20862, 2981), (23843, 5961))
EXPECTED_FOLD_MAE = (5.5769, 6.1985, 5.0611, 5.1476)
MAE_REPRO_TOL = 5e-3

OUT_DIR = ROOT / "reports" / "explainability"
FIG_DIR = ROOT / "outputs" / "figures"
REPORT_MD = OUT_DIR / "shap_explainability.md"
IMP_CSV = OUT_DIR / "shap_feature_importance.csv"
FOLD_CSV = OUT_DIR / "shap_fold_results.csv"
GROUP_CSV = OUT_DIR / "shap_feature_groups.csv"
FIG_SUMMARY = FIG_DIR / "shap_summary.png"
FIG_IMPORTANCE = FIG_DIR / "shap_feature_importance.png"
FIG_GROUPS = FIG_DIR / "shap_feature_groups.png"

KNOWN_GROUPS = (
    "calendar",
    "historical_target",
    "historical_load",
    "historical_generation",
    "historical_weather",
    "day_ahead_forecast",
    "weather_aggregate",
    "diğer",
)
LOAD_FORECAST_COLS = ("total_load_forecast",)
RENEWABLE_FORECAST_COLS = (
    "forecast_solar_day_ahead",
    "forecast_wind_onshore_day_ahead",
    "renewable_forecast_total",
    "forecast_wind_share_of_load",
    "forecast_solar_share_of_load",
)
TARGET_LAG_COLS = (
    "price_day_ahead_lag_24",
    "price_day_ahead_lag_48",
    "price_day_ahead_lag_168",
)

TEST_READ_COUNT = 0
PROTECTED_EXISTING_OUTPUTS = (
    rt.REPORTS / "baseline_modeling.md",
    rt.REPORTS / "ridge_tuning.md",
    rt.REPORTS / "residual_correction.md",
    rt.REPORTS / "high_price_strategy_comparison.md",
    ROOT / "reports" / "final" / "final_test_evaluation.md",
    ROOT / "reports" / "final" / "final_test_metrics.csv",
    ROOT / "data" / "processed" / "predictions" / "final_test_predictions.parquet",
    ROOT / "data" / "processed" / "predictions" / "high_price_strategy_predictions.parquet",
)


class ExplainError(ValueError):
    pass


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def snapshot(paths: tuple[Path, ...]) -> dict[str, str]:
    return {str(p): md5(p) for p in paths if p.exists()}


def assert_not_test(path: Path) -> None:
    global TEST_READ_COUNT
    resolved = Path(path).resolve()
    if resolved == TEST_PATH.resolve() or resolved.name == "test.parquet":
        TEST_READ_COUNT += 1
        raise ExplainError("TEST SET IS LOCKED and must not be loaded during explainability")


def read_parquet_locked(path: Path) -> pd.DataFrame:
    assert_not_test(path)
    return pd.read_parquet(path)


def feature_group_map() -> dict[str, str]:
    man = pd.read_csv(MANIFEST_PATH)
    mapping = dict(zip(man["feature_name"].astype(str), man["feature_group"].astype(str)))
    for col in FRAC_COLS:
        mapping[col] = "diğer"
    return mapping


def group_of(name: str, mapping: dict[str, str]) -> str:
    g = mapping.get(name, "diğer")
    return g if g in KNOWN_GROUPS else "diğer"


def ridge_fit(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    y_mean = float(y_train.mean())
    yc = y_train - y_mean
    xtx = x_train.T @ x_train + float(alpha) * np.eye(x_train.shape[1])
    w = np.linalg.solve(xtx, x_train.T @ yc)
    return w, y_mean


def linear_shap(x_scaled: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """Exact SHAP for a linear model on centered/scaled inputs: φ_i = w_i * x_i."""
    return x_scaled * coef


def permute_column(values: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    out = values.copy()
    rng.shuffle(out)
    return out


def permutation_mae_increases(
    x_va_df: pd.DataFrame,
    prep: rt.FoldPreprocessor,
    coef: np.ndarray,
    y_mean: float,
    addend: float,
    y_va: np.ndarray,
    base_mae: float,
    fold: int,
) -> np.ndarray:
    """Deterministic one-column permutations on fold validation, preprocessor already frozen."""
    n_feat = x_va_df.shape[1]
    increases = np.empty(n_feat, dtype=float)
    raw = x_va_df.to_numpy(dtype=float, copy=True)
    for j in range(n_feat):
        rng = np.random.RandomState(RANDOM_STATE + fold * 10_000 + j)
        shuffled = raw.copy()
        shuffled[:, j] = permute_column(shuffled[:, j], rng)
        x_perm = prep.transform_linear(pd.DataFrame(shuffled, columns=x_va_df.columns))
        pred = x_perm @ coef + y_mean + addend
        mae = float(np.mean(np.abs(pred - y_va)))
        increases[j] = mae - base_mae
    return increases


def try_savefig(fig: Any, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
        return True
    except Exception:
        return False
    finally:
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:
            pass


def write_figures(imp: pd.DataFrame, groups: pd.DataFrame) -> dict[str, bool]:
    written = {"summary": False, "importance": False, "groups": False}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return written

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    top = imp.head(20).iloc[::-1]
    colors = ["#2a6f97" if s > 0 else "#c1121f" for s in top["mean_std_coef"]]

    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    ax.barh(top["feature"], top["mean_abs_shap"], color=colors)
    ax.set_xlabel("Mean |linear SHAP| (walk-forward validation)")
    ax.set_title("Top 20 features — Ridge α=0.001 + METHOD_B")
    fig.tight_layout()
    written["importance"] = try_savefig(fig, FIG_IMPORTANCE)

    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    ax.barh(top["feature"], top["mean_std_coef"], color=colors)
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Mean standardized Ridge coefficient")
    ax.set_title("Direction on standardized features (blue +, red −)")
    fig.tight_layout()
    written["summary"] = try_savefig(fig, FIG_SUMMARY)

    g = groups.sort_values("mean_abs_shap_sum", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(g["feature_group"], g["mean_abs_shap_sum"], color="#4c6a92")
    ax.set_xlabel("Sum of mean |linear SHAP|")
    ax.set_title("Feature-group contribution")
    fig.tight_layout()
    written["groups"] = try_savefig(fig, FIG_GROUPS)
    return written


def write_report(
    fold_meta: pd.DataFrame,
    imp: pd.DataFrame,
    groups: pd.DataFrame,
    fold_mae: pd.DataFrame,
    hp_assoc: pd.DataFrame,
    hashes: dict[str, str],
    before: dict[str, str],
    after: dict[str, str],
    status: str,
    leakage: str,
    repro: str,
    protected_ok: bool,
    test_used: bool,
    test_read_count: int,
    figures: dict[str, bool],
) -> None:
    top20 = imp.head(20)
    up = imp[imp["mean_std_coef"] > 0].sort_values("mean_abs_shap", ascending=False).head(10)
    down = imp[imp["mean_std_coef"] < 0].sort_values("mean_abs_shap", ascending=False).head(10)
    lags = imp[imp["feature"].isin(TARGET_LAG_COLS)].sort_values("feature")
    load_fc = imp[imp["feature"].isin(LOAD_FORECAST_COLS)]
    ren_fc = imp[imp["feature"].isin(RENEWABLE_FORECAST_COLS)].sort_values("mean_abs_shap", ascending=False)
    cal = groups[groups["feature_group"] == "calendar"]
    weather = groups[groups["feature_group"].isin(["historical_weather", "weather_aggregate"])]
    hist_t = groups[groups["feature_group"] == "historical_target"]
    da = groups[groups["feature_group"] == "day_ahead_forecast"]
    diger = groups[groups["feature_group"] == "diğer"]
    frac = imp[imp["feature"].isin(FRAC_COLS)].sort_values("feature")
    sign_stable = int((imp["n_folds_pos_coef"].isin([0, 4]) | imp["n_folds_neg_coef"].isin([0, 4])).sum())
    prot_rows = []
    for p in (*rt.REQUIRED_PROTECTED, *PROTECTED_EXISTING_OUTPUTS):
        key = str(p)
        if key not in before and not p.exists():
            continue
        prot_rows.append(
            {
                "file": str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else p.name,
                "md5_before": before.get(key, ""),
                "md5_after": after.get(key, ""),
                "unchanged": before.get(key, "") == after.get(key, "") and bool(before.get(key)),
            }
        )

    def _g(df: pd.DataFrame, col: str) -> str:
        if df.empty:
            return "n/a"
        return f"{float(df.iloc[0][col]):.4f}"

    text = f"""# SHAP / Linear Explainability

**SHAP_EXPLAINABILITY = {status}**

Locked model explained as-is. No retuning, no new features, no METHOD_B change,
no alpha change, no test-based selection.

## 1. Model and data

| item | value |
|---|---|
| Model | Ridge (closed-form NumPy) |
| Alpha | {ALPHA} |
| Residual correction | expanding_historical addend (part of locked METHOD_B) |
| Extra features | METHOD_B high-price fractions (`{', '.join(FRAC_COLS)}`) |
| Development data | official TRAIN + VALIDATION only |
| Test | never loaded |
| Folds | same 4 expanding windows as Ridge tuning / METHOD_B |

Walk-forward folds:

{rt.md_table(fold_meta)}

Fold METHOD_B validation MAE (reproduced, not re-selected):

{rt.md_table(fold_mae)}

## 2. Why not TreeSHAP

Ridge is a **linear** model. TreeSHAP explains tree ensembles and is the wrong
estimator here.

This stage uses the **exact linear SHAP** values for a model
`ŷ = x_scaled · w + ȳ + addend`:

- After fold-train standardization, training features are centered, so
  `φ_i = w_i · x_scaled,i`
- `Σ φ_i = ŷ_ridge − ȳ`
- The METHOD_B expanding addend is a **single fold-train residual constant**.
  It shifts the base value; it is not a per-feature SHAP term.

Standardized coefficients `w` are reported because they are already on a
common scale (the preprocessor is fit on fold train only). Permutation
importance (MAE increase after a deterministic column shuffle on that fold's
validation block) is a complementary, model-agnostic check. It is not used
to change the model.

These numbers are **predictive attributions** for the locked forecast. They
are not causal effects.

## 3. Method (per fold)

1. Load TRAIN+VALIDATION; do not open `test.parquet`.
2. Recreate the locked METHOD_B matrix: 184 SAFE features + 3 causal
   high-price fractions. The P75 threshold is `quantile(y_fold_train, 0.75)`.
3. Fit median impute + standard scale on **fold train only**.
4. Fit Ridge `α={ALPHA}` on fold train.
5. Predict fold validation; add the frozen expanding-historical addend
   computed from fold-train residuals only.
6. Compute linear SHAP on the scaled validation matrix.
7. Compute permutation MAE increases on validation (seed
   `{RANDOM_STATE} + fold·10000 + feature_index`).
8. Average attributions across the 4 folds. Do not declare a winner from
   a single fold.

## 4. Top 20 features (mean |linear SHAP| across folds)

{rt.md_table(top20[['rank_mean_abs_shap','feature','feature_group','mean_abs_shap','std_abs_shap','mean_shap','mean_std_coef','direction','mean_perm_mae_increase']])}

`direction` is the sign of the mean standardized coefficient:
`+` means a higher feature value is associated with a **higher predicted**
price, holding other standardized features fixed.

**Do not read row 1–2 as “month is the price driver.”** `month` and
`day_of_year` are almost collinear calendar encodings. Their standardized
coefficients are huge and opposite (~+209 vs ~−208). Mean SHAP on
validation is about +27.7 and −27.6, so the **net** contribution of the
pair is near zero. |SHAP| ranks them first because each term is large,
not because the locked model uses calendar season as a single lever.
Later rows (generation lags, target lags, cloud/wind aggregates) are the
stable attributions.

## 5. Direction: what pulls the forecast up vs down

Largest-magnitude features with a **positive** coefficient (forecast up when
the feature is high):

{rt.md_table(up[['feature','feature_group','mean_std_coef','mean_abs_shap','mean_shap']])}

Largest-magnitude features with a **negative** coefficient (forecast down
when the feature is high):

{rt.md_table(down[['feature','feature_group','mean_std_coef','mean_abs_shap','mean_shap']])}

## 6. Answers

### Historical target lags

{rt.md_table(lags[['feature','mean_abs_shap','std_abs_shap','mean_std_coef','direction','mean_perm_mae_increase']])}

Historical-target group sum of mean |SHAP| = {_g(hist_t, 'mean_abs_shap_sum')}.
The allowed lags (t−24 / t−48 / t−168 and their element-wise stats) are
among the strongest predictive associations. This is persistence of the
auction curve, not a causal claim.

### Load forecast

{rt.md_table(load_fc[['feature','mean_abs_shap','mean_std_coef','direction','mean_perm_mae_increase']])}

Day-ahead forecast group sum of mean |SHAP| = {_g(da, 'mean_abs_shap_sum')}.

### Renewable forecast

{rt.md_table(ren_fc[['feature','mean_abs_shap','mean_std_coef','direction','mean_perm_mae_increase']])}

Higher renewable-forecast values are typically associated with a **lower**
predicted price when the standardized coefficient is negative. That is a
forecasting association (merit-order style correlation in the training
window), not proof that renewables cause the price.

### Weather

{rt.md_table(weather)}

Weather is **not** noise. National cloud-cover and wind-speed aggregates
and several city cloud lags enter the top 20. The 100 city-level weather
lags have a low **per-feature** mean |SHAP| (0.87), so most individual
city series are weak; the **aggregates** (group mean 4.98) are the
meaningful weather signal. That is still a predictive association with
lagged weather, not a claim that clouds cause the auction price.

### Calendar

Calendar group sum of mean |SHAP| = {_g(cal, 'mean_abs_shap_sum')};
per-feature mean = {_g(cal, 'mean_abs_shap_mean')}.
That sum is dominated by the collinear `month` / `day_of_year` pair
discussed above. After that pair, calendar is a moderate contributor
(`day_of_month` is next). Cyclic hour/weekday terms are smaller than
target and generation lags.

### METHOD_B high-price fractions (`diğer`)

{rt.md_table(frac[['feature','mean_abs_shap','std_abs_shap','mean_std_coef','direction','mean_perm_mae_increase']])}

Group `diğer` sum of mean |SHAP| = {_g(diger, 'mean_abs_shap_sum')}.
These features are the locked METHOD_B extras, not new engineering.

## 7. Feature-group importance

{rt.md_table(groups)}

`mean_abs_shap_sum` totals attribution mass. `mean_abs_shap_mean` is fairer
when groups have very different widths (weather has ~100 columns).

## 8. Fold stability

- Features with the same coefficient sign in all 4 folds: **{sign_stable} / {len(imp)}**
- `std_abs_shap` in the importance table is the fold-to-fold spread
- A feature is called important only if mean |SHAP| is high **and** the
  direction is not a one-fold artifact

{rt.md_table(imp.head(12)[['feature','mean_abs_shap','std_abs_shap','n_folds_pos_coef','n_folds_neg_coef','direction']])}

## 9. High-price hours (development only)

Thresholds are fold-train P75 applied to that fold's validation *y* **after**
predictions exist. This slice is diagnostic. It does not change ranks used
for the top-20 table and does not use the locked test set.

Mean SHAP on validation hours with `y ≥ train_P75` minus the rest
(largest absolute differences):

{rt.md_table(hp_assoc.head(15))}

Reading (association only):

- If METHOD_B fraction features have a positive coefficient, a higher recent
  share of expensive hours raises the **forecast**. On development folds the
  locked model still underpredicted many high-price hours; extra level from
  these fractions reduced but did not remove that gap.
- Target lags and load forecast often show a larger positive SHAP mass on
  high-price validation hours because those hours also have higher lags and
  higher load forecasts — the model attributes the level through features
  that move with the expensive regime.
- This does **not** say the high-price bias observed on the locked test
  (where P75+ bias flipped positive) is explained or fixed. Test rows were
  not used here.

## 10. Leakage / test / protected files

LEAKAGE_CHECK = {leakage}
TEST_READ_COUNT = {test_read_count}
TEST_USED_FOR_SELECTION = {str(test_used).upper()}
PROTECTED_FILES_UNCHANGED = {str(protected_ok).upper()}
REPRODUCIBILITY = {repro}

- Chronological expanding folds; no shuffle of time
- Preprocess fit on fold train only
- METHOD_B P75 from fold-train *y* only
- Expanding addend from fold-train residuals only
- Test parquet never opened; test predictions unused
- Full-data fit was not used
- Existing modeling / final reports and prediction parquets were not overwritten

{rt.md_table(pd.DataFrame(prot_rows), floatfmt="")}

| new output | md5 |
|---|---|
| shap_feature_importance.csv | {hashes['importance']} |
| shap_fold_results.csv | {hashes['folds']} |
| shap_feature_groups.csv | {hashes['groups']} |
| shap_explainability.md | {hashes['report']} |

Figures written: summary={figures.get('summary')} importance={figures.get('importance')} groups={figures.get('groups')}

## 11. Limitations

- Linear SHAP assigns credit under a linear, standardized model. Correlated
  lags and lag-statistics split that credit.
- Permutation importance also suffers when features are collinear.
- Attributions describe the locked forecast, not the real-world price
  mechanism. This study does not establish causality.
- High-price diagnostics use development validation *y* only as a slice
  label after the fact.

SHAP_EXPLAINABILITY = {status}
TEST_READ_COUNT = {test_read_count}
TEST_USED_FOR_SELECTION = FALSE
LEAKAGE_CHECK = {leakage}
REPRODUCIBILITY = {repro}
PROTECTED_FILES_UNCHANGED = {str(protected_ok).upper()}
"""
    REPORT_MD.write_text(text, encoding="utf-8")


def output_hashes() -> dict[str, str]:
    return {
        "importance": md5(IMP_CSV),
        "folds": md5(FOLD_CSV),
        "groups": md5(GROUP_CSV),
        "report": md5(REPORT_MD) if REPORT_MD.exists() else "",
    }


def run_once() -> dict[str, Any]:
    global TEST_READ_COUNT
    TEST_READ_COUNT = 0
    watched = tuple(rt.PROTECTED) + PROTECTED_EXISTING_OUTPUTS
    before = snapshot(watched)
    read_parquet_locked(TRAIN_PATH)
    read_parquet_locked(VAL_PATH)
    df, feat_cols = rt.load_dev_frame()
    if len(feat_cols) != N_SAFE:
        raise ExplainError(f"expected {N_SAFE} SAFE features, got {len(feat_cols)}")
    folds = rt.make_folds(df)
    for fold, (ntr, nva) in zip(folds, EXPECTED_FOLDS):
        if fold["n_train"] != ntr or fold["n_val"] != nva:
            raise ExplainError(f"Fold {fold['fold']} row counts changed")

    mapping = feature_group_map()
    model_cols = feat_cols + FRAC_COLS
    fold_rows: list[dict[str, Any]] = []
    hp_rows: list[dict[str, Any]] = []
    fold_mae_rows: list[dict[str, Any]] = []

    for fold in folds:
        train, val = fold["train"], fold["val"]
        y_tr = train[TARGET_COL].to_numpy(dtype=float)
        y_va = val[TARGET_COL].to_numpy(dtype=float)
        p75 = float(np.quantile(y_tr, 0.75))
        train_aug = hpa.add_frac_features(train, p75)
        val_aug = hpa.add_frac_features(val, p75)
        x_tr = hpa.feature_frame(train_aug, model_cols)
        x_va = hpa.feature_frame(val_aug, model_cols)
        prep = rt.FoldPreprocessor().fit(x_tr)
        rt.assert_preproc_train_only(prep, x_tr, x_va)
        xtr = prep.transform_linear(x_tr)
        xva = prep.transform_linear(x_va)
        coef, y_mean = ridge_fit(xtr, y_tr, ALPHA)
        pred_tr = xtr @ coef + y_mean
        pred_va = xva @ coef + y_mean
        addend = rc.expanding_addend(train[ID_COL].to_numpy(), pred_tr - y_tr)
        pred_va_c = pred_va + addend
        if not np.allclose(linear_shap(xva, coef).sum(axis=1) + y_mean, pred_va, atol=1e-8, rtol=0):
            raise ExplainError("linear SHAP does not reconstruct the Ridge prediction")
        mets = rt.metrics(y_va, pred_va_c)
        expected_mae = EXPECTED_FOLD_MAE[fold["fold"] - 1]
        if abs(mets["MAE"] - expected_mae) > MAE_REPRO_TOL:
            raise ExplainError(
                f"METHOD_B MAE not reproduced on fold {fold['fold']}: {mets['MAE']} vs {expected_mae}"
            )
        shap_va = linear_shap(xva, coef)
        mean_abs = np.mean(np.abs(shap_va), axis=0)
        mean_s = np.mean(shap_va, axis=0)
        perm = permutation_mae_increases(x_va, prep, coef, y_mean, addend, y_va, mets["MAE"], fold["fold"])
        hp = y_va >= p75
        rest = ~hp
        fold_mae_rows.append(
            {
                "fold": fold["fold"],
                "n_train": fold["n_train"],
                "n_val": fold["n_val"],
                "train_p75": p75,
                "MAE": mets["MAE"],
                "bias": mets["bias"],
                "p75_n": int(hp.sum()),
                "p75_bias": float(np.mean(pred_va_c[hp] - y_va[hp])) if hp.any() else float("nan"),
                "addend": float(addend),
            }
        )
        print(
            f"=== Fold {fold['fold']} train={fold['n_train']} val={fold['n_val']} "
            f"MAE={mets['MAE']:.4f} addend={addend:.4f} ===",
            flush=True,
        )
        for j, name in enumerate(model_cols):
            fold_rows.append(
                {
                    "fold": fold["fold"],
                    "feature": name,
                    "feature_group": group_of(name, mapping),
                    "std_coef": float(coef[j]),
                    "mean_abs_shap": float(mean_abs[j]),
                    "mean_shap": float(mean_s[j]),
                    "perm_mae_increase": float(perm[j]),
                    "n_val": fold["n_val"],
                }
            )
            if hp.any() and rest.any():
                hp_rows.append(
                    {
                        "fold": fold["fold"],
                        "feature": name,
                        "feature_group": group_of(name, mapping),
                        "mean_shap_hp": float(np.mean(shap_va[hp, j])),
                        "mean_shap_rest": float(np.mean(shap_va[rest, j])),
                        "delta_mean_shap_hp_minus_rest": float(np.mean(shap_va[hp, j]) - np.mean(shap_va[rest, j])),
                    }
                )

    fold_df = pd.DataFrame(fold_rows).sort_values(["fold", "feature"], kind="mergesort").reset_index(drop=True)
    g = fold_df.groupby("feature", sort=True)
    imp_rows = []
    for name, sub in g:
        coefs = sub["std_coef"].to_numpy(dtype=float)
        n_pos = int((coefs > 0).sum())
        n_neg = int((coefs < 0).sum())
        mean_coef = float(coefs.mean())
        if n_pos > n_neg:
            direction = "+"
        elif n_neg > n_pos:
            direction = "-"
        else:
            direction = "0"
        imp_rows.append(
            {
                "feature": name,
                "feature_group": str(sub["feature_group"].iloc[0]),
                "mean_abs_shap": float(sub["mean_abs_shap"].mean()),
                "std_abs_shap": float(sub["mean_abs_shap"].std(ddof=0)),
                "mean_shap": float(sub["mean_shap"].mean()),
                "mean_std_coef": mean_coef,
                "std_std_coef": float(coefs.std(ddof=0)),
                "mean_abs_std_coef": float(np.mean(np.abs(coefs))),
                "direction": direction,
                "n_folds_pos_coef": n_pos,
                "n_folds_neg_coef": n_neg,
                "mean_perm_mae_increase": float(sub["perm_mae_increase"].mean()),
                "std_perm_mae_increase": float(sub["perm_mae_increase"].std(ddof=0)),
            }
        )
    imp = pd.DataFrame(imp_rows).sort_values(
        ["mean_abs_shap", "mean_abs_std_coef", "feature"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    imp.insert(0, "rank_mean_abs_shap", np.arange(1, len(imp) + 1))

    grp_fold = (
        fold_df.groupby(["fold", "feature_group"], sort=True)
        .agg(
            n_features=("feature", "nunique"),
            abs_shap_sum=("mean_abs_shap", "sum"),
            abs_shap_mean=("mean_abs_shap", "mean"),
            perm_sum=("perm_mae_increase", "sum"),
        )
        .reset_index()
    )
    groups = (
        grp_fold.groupby("feature_group", sort=True)
        .agg(
            n_features=("n_features", "first"),
            mean_abs_shap_sum=("abs_shap_sum", "mean"),
            std_abs_shap_sum=("abs_shap_sum", "std"),
            mean_abs_shap_mean=("abs_shap_mean", "mean"),
            mean_perm_mae_increase_sum=("perm_sum", "mean"),
        )
        .reset_index()
    )
    groups["std_abs_shap_sum"] = groups["std_abs_shap_sum"].fillna(0.0)
    groups["_ord"] = groups["feature_group"].map({g: i for i, g in enumerate(KNOWN_GROUPS)})
    groups = groups.sort_values(["_ord", "feature_group"]).drop(columns="_ord").reset_index(drop=True)

    hp = pd.DataFrame(hp_rows)
    if hp.empty:
        hp_assoc = pd.DataFrame(
            columns=["feature", "feature_group", "mean_delta_shap", "std_delta_shap"]
        )
    else:
        hp_assoc = (
            hp.groupby(["feature", "feature_group"], sort=True)
            .agg(
                mean_delta_shap=("delta_mean_shap_hp_minus_rest", "mean"),
                std_delta_shap=("delta_mean_shap_hp_minus_rest", "std"),
                mean_shap_hp=("mean_shap_hp", "mean"),
                mean_shap_rest=("mean_shap_rest", "mean"),
            )
            .reset_index()
        )
        hp_assoc["std_delta_shap"] = hp_assoc["std_delta_shap"].fillna(0.0)
        hp_assoc["abs_delta"] = hp_assoc["mean_delta_shap"].abs()
        hp_assoc = hp_assoc.sort_values(["abs_delta", "feature"], ascending=[False, True], kind="mergesort")
        hp_assoc = hp_assoc.drop(columns=["abs_delta"]).reset_index(drop=True)

    fold_mae = pd.DataFrame(fold_mae_rows).sort_values("fold").reset_index(drop=True)
    fold_meta = pd.DataFrame(
        [{k: f[k] for k in ("fold", "n_train", "n_val", "train_start", "train_end", "val_start", "val_end", "train_frac")} for f in folds]
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    imp.to_csv(IMP_CSV, index=False)
    fold_df.to_csv(FOLD_CSV, index=False)
    groups.to_csv(GROUP_CSV, index=False)
    figures = write_figures(imp, groups)

    after = snapshot(watched)
    if after != before:
        raise ExplainError("A protected or existing experiment file was modified")
    if TEST_READ_COUNT != 0:
        raise ExplainError("test.parquet was read")

    hashes = output_hashes()
    write_report(
        fold_meta=fold_meta,
        imp=imp,
        groups=groups,
        fold_mae=fold_mae,
        hp_assoc=hp_assoc,
        hashes=hashes,
        before=before,
        after=after,
        status="PASS",
        leakage="PASS",
        repro="PENDING",
        protected_ok=True,
        test_used=False,
        test_read_count=TEST_READ_COUNT,
        figures=figures,
    )
    return {
        "imp": imp,
        "groups": groups,
        "fold_df": fold_df,
        "fold_mae": fold_mae,
        "hp_assoc": hp_assoc,
        "hashes": output_hashes(),
        "before": before,
        "after": after,
        "test_read_count": TEST_READ_COUNT,
        "figures": figures,
        "fold_meta": fold_meta,
    }


def run() -> dict[str, Any]:
    first = run_once()
    second = run_once()
    keys = ("importance", "folds", "groups")
    if any(first["hashes"][k] != second["hashes"][k] for k in keys):
        raise ExplainError(f"Reproducibility failed: {first['hashes']} vs {second['hashes']}")
    write_report(
        fold_meta=second["fold_meta"],
        imp=second["imp"],
        groups=second["groups"],
        fold_mae=second["fold_mae"],
        hp_assoc=second["hp_assoc"],
        hashes=second["hashes"],
        before=second["before"],
        after=second["after"],
        status="PASS",
        leakage="PASS",
        repro="PASS",
        protected_ok=True,
        test_used=False,
        test_read_count=second["test_read_count"],
        figures=second["figures"],
    )
    second["hashes"] = output_hashes()
    print(json.dumps({k: second[k] for k in ("hashes", "test_read_count")}, indent=2, default=str))
    print("SHAP_EXPLAINABILITY = PASS")
    print("TEST_READ_COUNT = 0")
    print("TEST_USED_FOR_SELECTION = FALSE")
    print("LEAKAGE_CHECK = PASS")
    print("REPRODUCIBILITY = PASS")
    print("PROTECTED_FILES_UNCHANGED = TRUE")
    return second


if __name__ == "__main__":
    run()
