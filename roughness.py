"""Gatheral-Jaisson-Rosenbaum (2018) roughness estimator and its bias check.

The estimator
-------------
For a volatility proxy X_t = log(sigma_t) observed daily, define the q-th
moment of increments over lag Delta

    m(q, Delta) = (1/N) * sum_t | X_{t+Delta} - X_t |^q ,

which scales as m(q, Delta) ~ Delta^{zeta_q} with zeta_q = H * q.  H is
obtained either from zeta_q / q or from the slope of zeta_q against q.

GJR report H ~ 0.1 for log realised volatility of major indices (S&P 0.142
and NASDAQ 0.139 when whole-day realised variance is used, which they show
biases H upwards).

Why we run a bias study
-----------------------
A volatility *index* is a forward-looking 30-day average of expected
volatility, i.e. an integrated (smoother) object than spot volatility.
Integration regularises the path, so H estimated from a VIX-type series is
biased upward relative to the H of the underlying volatility process.  We
quantify that bias by simulation so that the VIX-based estimate can be
compared with the realised-volatility-based estimates in the literature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dfa import fgn

DEFAULT_QS = (0.5, 1.0, 1.5, 2.0, 3.0)
DEFAULT_DELTAS = tuple(range(1, 31))


def m_q_delta(x: np.ndarray, q: float, delta: int) -> float:
    """q-th absolute moment of Delta-lag increments (overlapping pairs)."""
    x = np.asarray(x, dtype=float)
    d = np.abs(x[delta:] - x[:-delta])
    return float(np.mean(d**q))


def estimate_zeta(
    x: np.ndarray,
    qs=DEFAULT_QS,
    deltas=DEFAULT_DELTAS,
    delta_min: int = 1,
    delta_max: int = 30,
) -> dict:
    """Estimate zeta_q by regressing log m(q, Delta) on log Delta."""
    deltas = np.array([d for d in deltas if delta_min <= d <= delta_max], dtype=int)
    log_d = np.log(deltas.astype(float))
    zeta, r2 = {}, {}
    for q in qs:
        m = np.array([m_q_delta(x, q, int(d)) for d in deltas])
        log_m = np.log(m)
        coef = np.polyfit(log_d, log_m, 1)
        resid = log_m - np.polyval(coef, log_d)
        zeta[q] = float(coef[0])
        r2[q] = float(1 - resid.var() / log_m.var())
    return {"qs": np.array(qs, dtype=float), "deltas": deltas,
            "zeta": np.array([zeta[q] for q in qs]),
            "r2": r2}


def estimate_H(res: dict, q_min: float = 1.0) -> dict:
    """H from zeta_q / q (robust) and from the slope of zeta_q on q."""
    qs, zeta = res["qs"], res["zeta"]
    mask = qs >= q_min
    per_q = zeta[mask] / qs[mask]
    slope = float(np.polyfit(qs, zeta, 1)[0])
    return {
        "H_per_q": float(np.mean(per_q)),
        "H_per_q_sd": float(np.std(per_q, ddof=1)) if per_q.size > 1 else np.nan,
        "H_slope": slope,
    }


def roughness_report(
    x: np.ndarray,
    label: str,
    qs=DEFAULT_QS,
    delta_max: int = 30,
) -> dict:
    res = estimate_zeta(x, qs=qs, delta_max=delta_max)
    h = estimate_H(res)
    out = {"label": label, **h, "r2_mean": float(np.mean(list(res["r2"].values()))),
           "zeta_over_q": {float(q): float(z) / q for q, z in zip(res["qs"], res["zeta"])}}
    print(f"{label:<28} H(zeta_q/q)={h['H_per_q']:.3f} "
          f"(sd {h['H_per_q_sd']:.3f})  H(slope)={h['H_slope']:.3f}  "
          f"mean R2={out['r2_mean']:.3f}")
    return {"summary": out, "res": res}


def vix_like_from_spot(spot_logvol: np.ndarray, horizon: int = 30) -> np.ndarray:
    """Forward-looking 30-day average of an instantaneous log-volatility path.

    This is the idealised counterpart of a volatility index: it is an
    integrated (hence smoother) transformation of the spot volatility path.
    """
    spot = np.asarray(spot_logvol, dtype=float)
    n = len(spot)
    out = np.full(n, np.nan)
    if horizon <= 1:
        return spot.copy()
    csum = np.concatenate([[0.0], np.cumsum(spot)])
    for t in range(n - horizon):
        out[t] = (csum[t + horizon] - csum[t]) / horizon
    return out[: n - horizon]


def bias_study(
    H_true: float,
    n: int = 6000,
    horizon: int = 30,
    n_paths: int = 20,
    seed: int = 0,
    qs=DEFAULT_QS,
    delta_max: int = 30,
) -> pd.DataFrame:
    """How much does 30-day integration bias H upwards?"""
    rows = []
    for i in range(n_paths):
        spot = np.cumsum(fgn(n, H_true, seed=seed + i)) / np.sqrt(1.0)  # fBm path
        spot -= spot.mean()
        vix_like = vix_like_from_spot(spot, horizon=horizon)
        h_spot = estimate_H(estimate_zeta(spot, qs=qs, delta_max=delta_max))["H_per_q"]
        h_vix = estimate_H(estimate_zeta(vix_like, qs=qs, delta_max=delta_max))["H_per_q"]
        rows.append({"path": i, "H_true": H_true, "H_spot_est": h_spot,
                     "H_vix_like_est": h_vix})
    return pd.DataFrame(rows)
