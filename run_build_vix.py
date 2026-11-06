"""Build a daily China VIX from the user's option T-quote data.

Run:  python run_build_vix.py --underlying 50etf   # SSE 50ETF options
      python run_build_vix.py --underlying io      # CFFEX CSI300 index options
Output: outputs/vix_<underlying>_daily.csv (+ diagnostics and a plot)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from vix_build import build_daily_vix
from vix_build.load_data import load_cached

LABELS = {"50etf": "SSE 50ETF", "io": "CFFEX CSI300 index"}
OUT_NAMES = {"50etf": "vix_china_daily", "io": "vix_csi300_daily"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--underlying", choices=["50etf", "io"], default="50etf")
    p.add_argument("--min-near-days", type=int, default=8,
                   help="near-term minimum days to expiry (8 = CBOE rule, 1 = always two terms)")
    p.add_argument("--suffix", default="", help="suffix for the output file name")
    args = p.parse_args()

    opt, shibor = load_cached(args.underlying)
    print(f"options rows: {len(opt):,}  dates: {opt['trade_date'].nunique():,}")
    print(f"shibor rows:  {len(shibor):,}")

    vix = build_daily_vix(opt, shibor, min_near_days=args.min_near_days)
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = OUT_NAMES[args.underlying] + args.suffix
    vix.to_csv(out_dir / f"{stem}.csv", index=False)

    good = vix.dropna(subset=["vix"])
    print(f"built {len(vix)} daily values, {len(good)} valid VIX levels")
    print(good["vix"].describe().round(2).to_string())
    print("mode counts:", good["mode"].value_counts().to_dict())
    print(f"extrapolated dates (near term > 30d): {good['extrapolated'].sum()} "
          f"({good['extrapolated'].mean():.1%})")

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(good["trade_date"], good["vix"], lw=0.8)
    ax.set_xlabel("date")
    ax.set_ylabel(f"constructed VIX ({args.underlying})")
    ax.set_title(f"Daily China VIX from {LABELS[args.underlying]} option settlement prices")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=150)
    plt.close(fig)
    print(f"saved outputs/{stem}.csv and .png")


if __name__ == "__main__":
    main()
