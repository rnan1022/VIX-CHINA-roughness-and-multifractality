"""Lightweight complements to DFA/MF-DFA for a VIX-type series.

The functions here deliberately avoid extra dependencies.  They provide

* first-order autocorrelation of log-VIX changes;
* variance ratios at selected horizons;
* a change-of-frequency estimate of H from second differences;
* a two-stage fractional Ornstein-Uhlenbeck (fO-U) fit following
  Wang, Xiao and Yu (2023).

The fO-U estimates are descriptive: they are meant as a mean-reverting
parametric benchmark, not as a structural test of rough volatility.
"""

from __future__ import annotations

import math

import numpy as np


def acf1(x: np.ndarray) -> float:
    """First-order sample autocorrelation."""
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return float("nan")
    a = x[1:] - x[1:].mean()
    b = x[:-1] - x[:-1].mean()
    den = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
    return float(np.dot(a, b) / den) if den > 0 else float("nan")


def variance_ratio(x: np.ndarray, q: int) -> float:
    """Overlapping variance ratio Var(x_{t+q}-x_t) / (q Var(x_{t+1}-x_t))."""
    x = np.asarray(x, dtype=float)
    if q < 2 or x.size <= q + 2:
        return float("nan")
    d1 = np.diff(x)
    dq = x[q:] - x[:-q]
    var1 = float(np.var(d1, ddof=1))
    return float(np.var(dq, ddof=1) / (q * var1)) if var1 > 0 else float("nan")


def cof_h(x: np.ndarray) -> float:
    """Change-of-frequency H from second differences at lags 1 and 2.

    For a process with second-difference variance proportional to
    ``Delta**(2H)``, the ratio at lags 2 and 1 equals ``2**(2H)``.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 5:
        return float("nan")
    d2_1 = x[2:] - 2.0 * x[1:-1] + x[:-2]
    d2_2 = x[4:] - 2.0 * x[2:-2] + x[:-4]
    num = float(np.dot(d2_2, d2_2))
    den = float(np.dot(d2_1, d2_1))
    if num <= 0 or den <= 0:
        return float("nan")
    return float(0.5 * math.log(num / den, 2.0))


def fit_fou(x: np.ndarray) -> dict:
    """Two-stage fO-U fit for daily log-volatility observations.

    Model: dX_t = kappa (mu - X_t) dt + sigma dB^H_t.
    The time unit is one trading day.  ``half_life_days`` is ln(2)/kappa.
    """
    x = np.asarray(x, dtype=float)
    h = cof_h(x)
    if not np.isfinite(h) or h <= 0:
        return {"H": h, "kappa": np.nan, "half_life_days": np.nan,
                "sigma": np.nan, "mu": float(np.nanmean(x)), "n": int(x.size)}
    d2 = x[2:] - 2.0 * x[1:-1] + x[:-2]
    denom = max(len(d2) * (4.0 - 2.0 ** (2.0 * h)), 1e-12)
    sigma = math.sqrt(float(np.dot(d2, d2)) / denom)
    mu = float(np.mean(x))
    var = float(np.var(x, ddof=1))
    if sigma <= 0 or var <= 0:
        kappa = float("nan")
    else:
        kappa = (sigma**2 * h * math.gamma(2.0 * h) / var) ** (1.0 / (2.0 * h))
    half_life = math.log(2.0) / kappa if np.isfinite(kappa) and kappa > 0 else np.nan
    return {"H": h, "kappa": kappa, "half_life_days": half_life,
            "sigma": sigma, "mu": mu, "n": int(x.size)}


def diagnostics(x: np.ndarray) -> dict:
    """Convenience wrapper for one log-VIX series."""
    x = np.asarray(x, dtype=float)
    r = np.diff(x)
    fou = fit_fou(x)
    return {
        "n": int(x.size),
        "acf1_returns": acf1(r),
        "acf1_abs_returns": acf1(np.abs(r)),
        "vr2": variance_ratio(x, 2),
        "vr5": variance_ratio(x, 5),
        "vr22": variance_ratio(x, 22),
        **{f"fou_{k}": v for k, v in fou.items()},
    }
