"""Economic content: reconstruct the underlying from option-implied forwards,
then test the variance risk premium and return predictability of the index.

The T-quote file gives the put-call-parity forward for the near and next
expiries.  Interpolating log F linearly in maturity yields a constant 30-day
forward F30, which is (up to the small roll yield) a proxy for the underlying
level and lets us compute realised volatility without any additional data.

Tests
-----
* realised volatility of the following 22 trading days vs the index level
  (the "variance risk premium", VRP = VIX - RV);
* predictive regressions of future h-day returns on the VIX level, its
  innovation and the VRP, with Newey-West standard errors.

Run: python run_economic.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

YEARS = 365
TRADING_DAYS = 252


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["trade_date"])
    return df.dropna(subset=["vix"]).reset_index(drop=True)


def constant_maturity_forward(df: pd.DataFrame, target_days: float = 30.0) -> np.ndarray:
    """Interpolate log F_near and log F_next to a constant maturity."""
    t1 = df["near_days"].to_numpy() / YEARS
    t2 = df["next_days"].to_numpy() / YEARS
    t = target_days / YEARS
    f1 = np.log(df["F"].to_numpy(dtype=float))
    f2 = np.log(df["F_next"].to_numpy(dtype=float))
    w = (t2 - t) / (t2 - t1)
    out = w * f1 + (1 - w) * f2
    only_near = ~np.isfinite(f2)
    out[only_near] = f1[only_near]
    return np.exp(out)


def main() -> None:
    rows_summary, rows_reg = [], []
    for label, path in (
        ("50ETF", "outputs/vix_china_daily_twoterm.csv"),
        ("CSI300", "outputs/vix_csi300_daily.csv"),
    ):
        df = load(path)
        f30 = constant_maturity_forward(df)
        ok = np.isfinite(f30) & (f30 > 0)
        df, f30 = df[ok].reset_index(drop=True), f30[ok]
        r = np.diff(np.log(f30))
        print(f"\n===== {label}: {len(df)} days, {df['trade_date'].min().date()} "
              f"-> {df['trade_date'].max().date()} =====")
        print(f"  forward-based daily return: sd={r.std()*100:.2f}%  "
              f"skew={pd.Series(r).skew():.2f}  kurt={pd.Series(r).kurtosis():.1f}  "
              f"max|r|={np.abs(r).max()*100:.1f}%")
        big = np.abs(r) > 0.105          # exchange price limits make these impossible
        print(f"  implausible |r|>10.5%: {big.sum()} days ({big.mean():.2%})")
        if big.any():
            idx = np.argsort(-np.abs(r))[:5]
            for i in idx:
                print(f"    {df['trade_date'].iloc[i + 1].date()}  r={r[i]*100:+.1f}%  "
                      f"mode={df['mode'].iloc[i + 1]}  F={f30[i]:.4f}->{f30[i+1]:.4f}")
        r = np.where(big, np.nan, r)
        print(f"  after dropping those days: sd={np.nanstd(r)*100:.2f}%  "
              f"max|r|={np.nanmax(np.abs(r))*100:.1f}%")

        vix = df["vix"].to_numpy()
        # realised volatility over the next 22 trading days (annualised, %)
        h = 22
        rv = np.full(len(r) - h, np.nan)
        for i in range(len(rv)):
            rv[i] = np.sqrt(TRADING_DAYS * np.mean(r[i : i + h] ** 2)) * 100
        vix_t = vix[1 : 1 + len(rv)]
        fwd_ret = np.full(len(rv), np.nan)
        for i in range(len(rv)):
            fwd_ret[i] = (f30[1 + i + h] / f30[1 + i] - 1) * 100
        vrp = vix_t - rv
        dvix = np.diff(vix)[: len(rv)]

        print(f"  VIX mean={vix_t.mean():.2f}  RV(22d) mean={np.nanmean(rv):.2f}  "
              f"VRP mean={np.nanmean(vrp):.2f} vol points "
              f"(median {np.nanmedian(vrp):.2f})")
        print(f"  corr(VIX, RV)={np.corrcoef(vix_t, rv)[0,1]:.3f}")
        rows_summary.append({"index": label, "n": len(vix_t), "vix_mean": vix_t.mean(),
                             "rv_mean": np.nanmean(rv), "vrp_mean": np.nanmean(vrp),
                             "vrp_median": np.nanmedian(vrp),
                             "vrp_positive_share": np.nanmean(vrp > 0),
                             "corr_vix_rv": np.corrcoef(vix_t, rv)[0, 1]})

        # predictive regressions: future h-day return on VIX, dVIX and VRP
        for h_fwd in (5, 22):
            y = np.array([
                (f30[1 + i + h_fwd] / f30[1 + i] - 1) * 100 for i in range(len(rv))
            ])
            for name, X in (
                ("VIX", vix_t),
                ("dVIX", dvix),
                ("VRP", vrp),
            ):
                m = np.isfinite(y) & np.isfinite(X)
                Xm = sm.add_constant(X[m])
                res = sm.OLS(y[m], Xm).fit(cov_type="HAC", cov_kwds={"maxlags": h_fwd})
                rows_reg.append({
                    "index": label, "horizon": h_fwd, "predictor": name,
                    "coef": res.params[1], "t": res.tvalues[1], "r2": res.rsquared,
                    "n": int(m.sum()),
                })
                print(f"  predict {h_fwd:>2}d return on {name:<5}: "
                      f"beta={res.params[1]:+.4f} (t={res.tvalues[1]:+.2f})  "
                      f"R2={res.rsquared:.3f}")

    pd.DataFrame(rows_summary).to_csv("outputs/economic_summary.csv", index=False)
    pd.DataFrame(rows_reg).to_csv("outputs/economic_predictive.csv", index=False)
    print("\nsaved outputs/economic_summary.csv and outputs/economic_predictive.csv")


if __name__ == "__main__":
    main()
