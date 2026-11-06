"""Compute implied-minus-realised volatility and variance spreads."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_DAYS = 365
TRADING_DAYS = 252
HORIZON = 22
rows = []


def constant_maturity_forward(frame: pd.DataFrame) -> np.ndarray:
    t1 = frame["near_days"].to_numpy() / TARGET_DAYS
    t2 = frame["next_days"].to_numpy() / TARGET_DAYS
    target = 30 / TARGET_DAYS
    f1 = np.log(frame["F"].to_numpy(dtype=float))
    f2 = np.log(frame["F_next"].to_numpy(dtype=float))
    weight = (t2 - target) / (t2 - t1)
    output = weight * f1 + (1.0 - weight) * f2
    only_near = ~np.isfinite(f2)
    output[only_near] = f1[only_near]
    return np.exp(output)


def hac_mean(values: np.ndarray, maxlags: int = HORIZON) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    centred = values - values.mean()
    long_run = float(np.mean(centred**2))
    for lag in range(1, maxlags + 1):
        weight = 1.0 - lag / (maxlags + 1.0)
        covariance = float(np.mean(centred[lag:] * centred[:-lag]))
        long_run += 2.0 * weight * covariance
    standard_error = np.sqrt(max(long_run, 0.0) / len(values))
    return float(values.mean()), float(values.mean() / standard_error)


for label, path in (
    ("50ETF", "outputs/vix_china_daily_twoterm.csv"),
    ("CSI300", "outputs/vix_csi300_daily.csv"),
):
    frame = (
        pd.read_csv(path, parse_dates=["trade_date"])
        .dropna(subset=["vix"])
        .reset_index(drop=True)
    )
    forward = constant_maturity_forward(frame)
    valid = np.isfinite(forward) & (forward > 0)
    frame = frame.loc[valid].reset_index(drop=True)
    forward = forward[valid]

    returns = np.diff(np.log(forward))
    returns = np.where(np.abs(returns) > 0.105, np.nan, returns)
    realised = np.array(
        [
            np.sqrt(TRADING_DAYS * np.mean(returns[i : i + HORIZON] ** 2)) * 100
            for i in range(len(returns) - HORIZON)
        ]
    )
    implied = frame["vix"].to_numpy()[1 : 1 + len(realised)]
    volatility_spread = implied - realised
    variance_spread = implied**2 - realised**2
    future_returns = np.array(
        [
            (forward[1 + i + HORIZON] / forward[1 + i] - 1.0) * 100
            for i in range(len(realised))
        ]
    )
    down = future_returns < 0
    up = future_returns >= 0
    high_volatility = implied >= np.nanmedian(implied)
    spread_mean, spread_t = hac_mean(volatility_spread)

    print(
        label,
        "n=", len(implied),
        "vol-spread mean/median/positive=",
        round(float(np.nanmean(volatility_spread)), 3),
        round(float(np.nanmedian(volatility_spread)), 3),
        round(float(np.nanmean(volatility_spread > 0)), 3),
        "variance-spread mean/median/positive=",
        round(float(np.nanmean(variance_spread)), 3),
        round(float(np.nanmedian(variance_spread)), 3),
        round(float(np.nanmean(variance_spread > 0)), 3),
    )
    print(
        "  volatility spread: down/up=",
        round(float(np.nanmean(volatility_spread[down])), 3),
        round(float(np.nanmean(volatility_spread[up])), 3),
        "high/low VIX=",
        round(float(np.nanmean(volatility_spread[high_volatility])), 3),
        round(float(np.nanmean(volatility_spread[~high_volatility])), 3),
    )
    rows.append(
        {
            "index": label,
            "n": len(implied),
            "volatility_spread_mean": spread_mean,
            "volatility_spread_hac_t": spread_t,
            "volatility_spread_median": np.nanmedian(volatility_spread),
            "volatility_spread_positive_share": np.nanmean(volatility_spread > 0),
            "volatility_spread_future_down": np.nanmean(volatility_spread[down]),
            "volatility_spread_future_up": np.nanmean(volatility_spread[up]),
            "volatility_spread_high_vix": np.nanmean(volatility_spread[high_volatility]),
            "volatility_spread_low_vix": np.nanmean(
                volatility_spread[~high_volatility]
            ),
            "variance_spread_mean": np.nanmean(variance_spread),
            "variance_spread_median": np.nanmedian(variance_spread),
        }
    )

pd.DataFrame(rows).to_csv("outputs/submission_economic_checks.csv", index=False)
