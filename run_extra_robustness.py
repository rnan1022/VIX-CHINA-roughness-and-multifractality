"""Extra DFA robustness variants needed for Section 5.4 of the draft."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dfa import dfa_fit


def main() -> None:
    v = pd.read_csv("outputs/vix_china_daily_twoterm.csv", parse_dates=["trade_date"])
    v = v.dropna(subset=["vix"]).reset_index(drop=True)
    rets = np.diff(np.log(v["vix"].to_numpy()))
    rows = []

    variants = [
        ("baseline (min_scale=10, order=1)", dict(order=1, min_scale=10, n_scales=20)),
        ("min_scale=8", dict(order=1, min_scale=8, n_scales=20)),
        ("min_scale=20", dict(order=1, min_scale=20, n_scales=20)),
        ("quadratic detrending (order=2)", dict(order=2, min_scale=10, n_scales=20)),
        ("cubic detrending (order=3)", dict(order=3, min_scale=10, n_scales=20)),
        ("scales capped at N/8", dict(order=1, min_scale=10, n_scales=20, max_scale_frac=0.125)),
    ]
    for name, kw in variants:
        f = dfa_fit(rets, **kw)
        rows.append({"variant": name, "alpha": f["alpha"], "se": f["se"], "r2": f["r2"]})
        print(f"{name:<34} alpha={f['alpha']:.3f} (se {f['se']:.3f}, R2 {f['r2']:.3f})")

    # drop the days whose VIX used term-structure extrapolation, keeping the
    # remaining consecutive stretches and pooling their DFA estimates
    ex = v["extrapolated"].fillna(False).to_numpy()
    keep = ~ex[1:]  # returns align with the second observation
    r_kept = rets[keep]
    f = dfa_fit(r_kept, order=1, min_scale=10, n_scales=20)
    rows.append({"variant": "excluding extrapolated days", "alpha": f["alpha"],
                 "se": f["se"], "r2": f["r2"]})
    print(f"{'excluding extrapolated days':<34} alpha={f['alpha']:.3f} "
          f"(se {f['se']:.3f}, n={len(r_kept)})")

    for label, mask in (("excl. near-expiry rollover days", ~np.isin(
            v["mode"].to_numpy()[:-1], ["next_only", "near_only"])),):
        r = rets[mask]
        f = dfa_fit(r, order=1, min_scale=10, n_scales=20)
        rows.append({"variant": label, "alpha": f["alpha"], "se": f["se"], "r2": f["r2"]})
        print(f"{label:<34} alpha={f['alpha']:.3f} (se {f['se']:.3f}, n={len(r)})")

    pd.DataFrame(rows).to_csv("outputs/paper_table_extra_robustness.csv", index=False)


if __name__ == "__main__":
    main()
