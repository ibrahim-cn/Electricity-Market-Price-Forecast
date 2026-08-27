"""AR(1) residual correction on Ridge+METHOD_B (DAM 24-hour blocks).

Same protocol as the development walk-forward winner:
fit AR(1) on the last FIT_WINDOW train residuals, forecast 24 hours,
then append that day's actual Ridge+METHOD_B residuals.
"""

from __future__ import annotations

import numpy as np

FIT_WINDOW = 1440
HORIZON = 24


def fit_phi(resid: np.ndarray, window: int = FIT_WINDOW) -> float:
    r = np.asarray(resid[-window:], dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return 0.0
    denom = float(np.dot(r[:-1], r[:-1]))
    if denom <= 0:
        return 0.0
    phi = float(np.dot(r[1:], r[:-1]) / denom)
    if not np.isfinite(phi):
        return 0.0
    return float(np.clip(phi, -0.999, 0.999))


def dam24_add(base_pred: np.ndarray, y: np.ndarray, phi: float, last_resid: float, horizon: int = HORIZON) -> np.ndarray:
    """Additive AR(1) correction. y is used only after each 24h block is forecasted."""
    n = len(base_pred)
    add = np.zeros(n, dtype=float)
    state = float(last_resid)
    i = 0
    while i < n:
        h = min(horizon, n - i)
        k = np.arange(1, h + 1, dtype=float)
        add[i : i + h] = (phi**k) * state
        actual = y[i : i + h] - base_pred[i : i + h]
        state = float(actual[-1])
        i += h
    return add


def apply(base_pred: np.ndarray, y: np.ndarray, resid_history: np.ndarray) -> tuple[np.ndarray, float, float]:
    phi = fit_phi(resid_history)
    last = float(resid_history[-1])
    add = dam24_add(base_pred, y, phi, last)
    return base_pred + add, phi, last


def horizon_add(phi: float, last_resid: float, n: int = HORIZON) -> np.ndarray:
    k = np.arange(1, n + 1, dtype=float)
    return (phi**k) * float(last_resid)
