"""Reproduce the trailing-VRP predictive regressions in Table 6.

This small script uses only numpy/pandas so that the ex ante VRP row can be
reproduced in the bundled Python environment.  The main economic table is
still produced by run_economic.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

YEARS = 365
TRADING_DAYS = 252


def constant_maturity_forward(df: pd.DataFrame, target_days: float = 30.0) -> np.ndarray:
    t1 = df["near_days"].to_numpy() / YEARS
    t2 = df["next_days"].to_numpy() / YEARS
    t = target_days / YEARS
    f1 = np.log(df["F"].to_numpy(dtype=float))
    f2 = np.log(df["F_next"].to_numpy(dtype=float))
    w = (t2 - t) / (t2 - t1)
    out = w * f1 + (1.0 - w) * f2
    only_near = ~np.isfinite(f2)
    out[only_near] = f1[only_near]
    return np.exp(out)


def ols_hac(y: np.ndarray, X: np.ndarray, maxlags: int):
    X = np.column_stack([np.ones(len(X)), X])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    meat = (X * resid[:, None]).T @ (X * resid[:, None])
    for lag in range(1, maxlags + 1):
        weight = 1.0 - lag / (maxlags + 1.0)
        cross = (X[lag:] * resid[lag:, None]).T @ (X[:-lag] * resid[:-lag, None])
        meat += weight * (cross + cross.T)
    cov = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov))
    r2 = 1.0 - np.sum(resid**2) / np.sum((y - y.mean()) ** 2)
    return beta, se, r2


def main() -> None:
    rows = []
    for label, path in (
        ("50ETF", "outputs/vix_china_daily_twoterm.csv"),
        ("CSI300", "outputs/vix_csi300_daily.csv"),
    ):
        df = pd.read_csv(path, parse_dates=["trade_date"]).dropna(subset=["vix"]).reset_index(drop=True)
        f30 = constant_maturity_forward(df)
        valid = np.isfinite(f30) & (f30 > 0)
        df = df[valid].reset_index(drop=True)
        f30 = f30[valid]
        rets = np.diff(np.log(f30))
        vix = df["vix"].to_numpy(dtype=float)

        for horizon in (5, 22):
            ys, xs = [], []
            for t in range(22, len(df) - horizon):
                past = rets[t - 22:t]
                rv_trailing = np.sqrt(TRADING_DAYS * np.mean(past**2)) * 100.0
                vrp_trailing = vix[t] - rv_trailing
                future_ret = (f30[t + horizon] / f30[t] - 1.0) * 100.0
                ys.append(future_ret)
                xs.append(vrp_trailing)
            beta, se, r2 = ols_hac(np.asarray(ys), np.asarray(xs), horizon)
            rows.append({
                "index": label,
                "horizon": horizon,
                "predictor": "VRP_trailing",
                "coef": beta[1],
                "t": beta[1] / se[1],
                "r2": r2,
                "n": len(ys),
            })
            print(f"{label:6s} h={horizon:2d}  VRP_trailing beta={beta[1]:+.4f} "
                  f"(t={beta[1]/se[1]:+.2f})  R2={r2:.4f}  n={len(ys)}")

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "economic_trailing_vrp.csv", index=False)


if __name__ == "__main__":
    main()
