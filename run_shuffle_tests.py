"""Shuffled-surrogate test of H = 0.5 for every index variant."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dfa import hurst_significance

SERIES = [
    ("50ETF (CBOE 8-day rule)", "outputs/vix_china_daily.csv"),
    ("50ETF (two-term)", "outputs/vix_china_daily_twoterm.csv"),
    ("CSI300 index options", "outputs/vix_csi300_daily.csv"),
]


def main() -> None:
    rows = []
    for label, path in SERIES:
        v = pd.read_csv(path, parse_dates=["trade_date"]).dropna(subset=["vix"])
        rets = np.diff(np.log(v["vix"].to_numpy()))
        r = hurst_significance(rets, n_surrogates=500, seed=7, order=1,
                               min_scale=10, n_scales=20)
        rows.append(
            {"series": label, "n": len(rets), "alpha": round(r["alpha_obs"], 4),
             "surrogate_mean": round(r["surrogate_mean"], 4),
             "z": round(r["z"], 2), "p_le": r["p_le"], "p_ge": r["p_ge"]}
        )
        print(f"{label:<26} alpha={r['alpha_obs']:.3f}  shuffle_mean="
              f"{r['surrogate_mean']:.3f}  z={r['z']:.2f}  p(>=0.5)={r['p_ge']:.4f}")
    pd.DataFrame(rows).to_csv("outputs/paper_table_shuffle.csv", index=False)


if __name__ == "__main__":
    main()
