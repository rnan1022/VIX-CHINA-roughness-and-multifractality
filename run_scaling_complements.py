"""Run ACF, variance-ratio and fO-U diagnostics for the constructed VIX series.

Run: python run_scaling_complements.py
Output: outputs/scaling_complements.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scaling_complements import diagnostics

SERIES = {
    "50ETF (two-term)": "outputs/vix_china_daily_twoterm.csv",
    "50ETF (CBOE rule)": "outputs/vix_china_daily.csv",
    "CSI300 options": "outputs/vix_csi300_daily.csv",
}


def load_log_vix(path: str):
    v = pd.read_csv(path, parse_dates=["trade_date"]).dropna(subset=["vix"])
    return v["vix"].to_numpy(dtype=float)


def main() -> None:
    rows = []
    for label, path in SERIES.items():
        vix = load_log_vix(path)
        x = np.log(vix)
        d = diagnostics(x)
        rows.append({"series": label, **d})
        print(
            f"{label:<20} ACF1(r)={d['acf1_returns']:.3f}  "
            f"VR2={d['vr2']:.3f}  VR5={d['vr5']:.3f}  VR22={d['vr22']:.3f}  "
            f"fO-U H={d['fou_H']:.3f}  kappa={d['fou_kappa']:.5f}  "
            f"half-life={d['fou_half_life_days']:.1f}d"
        )
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "scaling_complements.csv", index=False)
    print("\nsaved outputs/scaling_complements.csv")


if __name__ == "__main__":
    main()
