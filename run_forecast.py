"""Out-of-sample forecasting test in the spirit of Gatheral et al. (2018).

The RFSV predictor of the log-volatility at horizon h is the Gaussian
projection of a fractional Brownian motion onto the observed history.  In
discrete time this is a linear combination of past increments with weights
solving Sigma w = c, where Sigma is the fGn autocovariance matrix of the
observed increments and c the covariance between the future h-day increment
and those past increments.

We treat log VIX as the volatility proxy (we do not have intraday data for a
realised-volatility target) and compare, out of sample:
    random walk (last value) | AR(1) | HAR | RFSV | HAR + RFSV signal

Outputs: outputs/forecast_results.csv, outputs/forecast_series.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from roughness import estimate_H, estimate_zeta

H_TARGETS = (5, 22)
WINDOW = 250


def fgn_autocov(k: np.ndarray | float, H: float) -> np.ndarray:
    k = np.abs(np.asarray(k, dtype=float))
    return 0.5 * ((k + 1) ** (2 * H) - 2 * k ** (2 * H) + np.abs(k - 1) ** (2 * H))


def rfsv_signal(x: np.ndarray, h: int, H: float, window: int = WINDOW) -> float:
    """Predicted h-day change of log-volatility from the fBm projection."""
    hist = np.asarray(x[-(window + 1):], dtype=float)
    d = np.diff(hist)                       # increments, oldest -> newest
    w = len(d)
    if w < 5:
        return 0.0
    wts = rfsv_weights(w, h, H)
    return float(wts @ d)


def rfsv_weights(w: int, h: int, H: float) -> np.ndarray:
    """Projection weights: data-independent given (w, h, H)."""
    idx = np.arange(w)
    Sigma = fgn_autocov(np.abs(idx[:, None] - idx[None, :]), H)
    # c[i] = sum_{j=1..h} Cov(future increment j, past increment at lag i)
    lags = np.arange(1, h + 1)[:, None] + idx[None, :] - 1
    c = fgn_autocov(lags, H).sum(axis=0)
    try:
        return np.linalg.solve(Sigma + 1e-10 * np.eye(w), c)
    except np.linalg.LinAlgError:
        return np.zeros(w)


def sliding_signals(x: np.ndarray, h: int, H: float, window: int = WINDOW) -> np.ndarray:
    """RFSV signal for every origin t (NaN where the window is not full)."""
    n = len(x)
    wts = rfsv_weights(window, h, H)
    out = np.full(n, np.nan)
    sw = np.lib.stride_tricks.sliding_window_view(x, window + 1)
    inc = np.diff(sw, axis=1)                     # (n - window, window)
    out[window:] = inc @ wts
    return out


def har_features(x: np.ndarray, t: int) -> tuple[float, float, float]:
    """HAR-style averages of log-volatility: 1 day, 1 week, 1 month."""
    daily = x[t]
    weekly = x[max(0, t - 4) : t + 1].mean()
    monthly = x[max(0, t - 21) : t + 1].mean()
    return daily, weekly, monthly


def main() -> None:
    v = pd.read_csv("outputs/vix_china_daily_twoterm.csv", parse_dates=["trade_date"])
    v = v.dropna(subset=["vix"]).reset_index(drop=True)
    x = np.log(v["vix"].to_numpy())
    n = len(x)
    dates = v["trade_date"].to_numpy()
    s = pd.Series(x)
    weekly = s.rolling(5).mean().to_numpy()
    monthly = s.rolling(22).mean().to_numpy()

    train_end = 1600
    refit_every = 22
    rows, series_rows = [], []
    for h in H_TARGETS:
        preds = {k: [] for k in ("rw", "ar1", "har", "rfsv", "har_rfsv")}
        actual, stamp = [], []
        coef = None
        for t in range(train_end, n - h):
            if (t - train_end) % refit_every == 0:
                # --- fit all models on data up to t (expanding window) ---
                tr = np.arange(WINDOW, t - h)      # origins with full window, known target
                y = x[tr + h]
                # AR(1)
                A = np.column_stack([np.ones_like(tr, dtype=float), x[tr]])
                beta_ar = np.linalg.lstsq(A, y, rcond=None)[0]
                # HAR
                Xh = np.column_stack(
                    [np.ones(len(tr)), x[tr], weekly[tr], monthly[tr]]
                )
                beta_har = np.linalg.lstsq(Xh, y, rcond=None)[0]
                # H on the training window (GJR estimator)
                H_hat = estimate_H(estimate_zeta(x[: t]))["H_per_q"]
                # RFSV signals on the training window for the combined model
                sig_all = sliding_signals(x, h, H_hat, WINDOW)
                sig_tr = sig_all[tr]
                Xc = np.column_stack([Xh, sig_tr])
                beta_comb = np.linalg.lstsq(Xc, y, rcond=None)[0]
                coef = {"ar": beta_ar, "har": beta_har, "comb": beta_comb,
                        "H": H_hat, "sig_all": sig_all}

            # --- one-step out-of-sample forecasts ---
            preds["rw"].append(x[t])
            preds["ar1"].append(coef["ar"][0] + coef["ar"][1] * x[t])
            f_t = np.array([x[t], weekly[t], monthly[t]])
            preds["har"].append(coef["har"][0] + coef["har"][1:] @ f_t)
            sig = float(coef["sig_all"][t]) if np.isfinite(coef["sig_all"][t]) \
                else rfsv_signal(x[: t + 1], h, coef["H"], WINDOW)
            preds["rfsv"].append(x[t] + sig)
            preds["har_rfsv"].append(
                coef["comb"][0] + coef["har"][1:] @ f_t + coef["comb"][-1] * sig
            )
            actual.append(x[t + h])
            stamp.append(dates[t + h])

        actual = np.asarray(actual)
        print(f"\n===== horizon h = {h} trading days (test n = {len(actual)}) =====")
        for k, p in preds.items():
            p = np.asarray(p)
            rmse = float(np.sqrt(np.mean((p - actual) ** 2)))
            mae = float(np.mean(np.abs(p - actual)))
            r2 = float(1 - np.sum((p - actual) ** 2) / np.sum((actual - actual.mean()) ** 2))
            rows.append({"horizon": h, "model": k, "rmse_log": rmse, "mae_log": mae, "r2": r2})
            print(f"  {k:<9} RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")
            series_rows.append(
                pd.DataFrame({"date": stamp, "horizon": h, "model": k,
                              "pred": p, "actual": actual})
            )

        # Diebold-Mariano: HAR+RFSV vs HAR (squared-error loss)
        e1 = (np.asarray(preds["har_rfsv"]) - actual) ** 2
        e2 = (np.asarray(preds["har"]) - actual) ** 2
        d = e2 - e1
        dm = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
        print(f"  DM statistic (HAR+RFSV better than HAR): {dm:.2f}")
        rows.append({"horizon": h, "model": "dm_har_rfsv_vs_har", "rmse_log": np.nan,
                     "mae_log": np.nan, "r2": np.nan, "dm": dm})

    pd.DataFrame(rows).to_csv("outputs/forecast_results.csv", index=False)
    pd.concat(series_rows).to_csv("outputs/forecast_series.csv", index=False)
    print("\nsaved outputs/forecast_results.csv and outputs/forecast_series.csv")


if __name__ == "__main__":
    main()
