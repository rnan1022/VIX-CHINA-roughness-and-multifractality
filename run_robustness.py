"""Paper-readiness robustness checks for the constructed China VIX.

Addresses the two objections a referee is most likely to raise:
  1. "Anti-persistence is mechanical mean reversion of a bounded index."
     -> AR(1)-in-log benchmark: simulate mean-reverting paths with the same
        persistence, compare their DFA alpha with the observed one.
  2. "Multifractality is only fat tails / only correlations."
     -> MF-DFA source decomposition (shuffled vs phase-randomised surrogates).

Also reports subsample stability and a block-bootstrap CI for alpha.

Run: python run_robustness.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dfa import dfa_fit, mf_decomposition


def load_returns(path: str = "outputs/vix_china_daily.csv"):
    v = pd.read_csv(path, parse_dates=["trade_date"]).dropna(subset=["vix"])
    v = v[v["vix"] > 0].reset_index(drop=True)
    return v, np.diff(np.log(v["vix"].to_numpy()))


def block_bootstrap_alpha(
    x: np.ndarray, n_boot: int = 1000, block: int = 20, seed: int = 0, **kw
) -> dict:
    """Circular block bootstrap CI for the DFA exponent (keeps short memory)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    n_blocks = int(np.ceil(n / block))
    alphas = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        sample = np.concatenate(
            [np.take(x, np.arange(s, s + block), mode="wrap") for s in starts]
        )[:n]
        alphas[b] = dfa_fit(sample, **kw)["alpha"]
    return {
        "mean": float(alphas.mean()),
        "std": float(alphas.std(ddof=1)),
        "ci_low": float(np.percentile(alphas, 2.5)),
        "ci_high": float(np.percentile(alphas, 97.5)),
        # share of bootstrap samples with alpha >= 0.5 -> evidence of roughness
        "p_ge_half": float(np.mean(alphas >= 0.5)),
    }


def ar1_benchmark(log_vix: np.ndarray, n_sim: int = 500, seed: int = 0, **kw) -> dict:
    """Simulate AR(1) mean-reverting paths matching the index's own dynamics.

    If the observed alpha is well below the simulated distribution, the
    anti-persistence is not just mechanical mean reversion of a bounded index.
    """
    rng = np.random.default_rng(seed)
    y = log_vix - log_vix.mean()
    ylag, ycur = y[:-1], y[1:]
    phi = float(np.dot(ylag, ycur) / np.dot(ylag, ylag))
    resid = ycur - phi * ylag
    sigma = float(resid.std(ddof=1))
    n = len(y)

    alphas = np.empty(n_sim)
    for i in range(n_sim):
        eps = rng.normal(0.0, sigma, n)
        sim = np.empty(n)
        sim[0] = y[0]
        for t in range(1, n):
            sim[t] = phi * sim[t - 1] + eps[t]
        alphas[i] = dfa_fit(np.diff(sim), **kw)["alpha"]
    return {
        "phi": phi,
        "sigma": sigma,
        "bench_mean": float(alphas.mean()),
        "bench_std": float(alphas.std(ddof=1)),
        "bench_ci_low": float(np.percentile(alphas, 2.5)),
        "bench_ci_high": float(np.percentile(alphas, 97.5)),
        "p_le": float(np.mean(alphas <= 0.0)) if False else None,
        "alphas": alphas,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="outputs/vix_china_daily.csv")
    p.add_argument("--label", default="50ETF China VIX")
    p.add_argument("--out", default="outputs/paper_robustness")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kw = dict(order=1, min_scale=10, n_scales=20)

    v, rets = load_returns(args.input)
    print(f"{args.label}: {len(v)} obs {v['trade_date'].min().date()} -> {v['trade_date'].max().date()}")
    obs = dfa_fit(rets, **kw)
    print(f"observed alpha(returns) = {obs['alpha']:.3f} (se {obs['se']:.3f})")

    # 1) mean-reversion benchmark
    print("\n[1] AR(1) mean-reversion benchmark")
    bench = ar1_benchmark(v["vix"].to_numpy(), n_sim=300, seed=1, **kw)
    p_le = float(np.mean(bench["alphas"] <= obs["alpha"]))
    print(f"  fitted AR(1) phi = {bench['phi']:.4f}")
    print(f"  benchmark alpha = {bench['bench_mean']:.3f} "
          f"(2.5%..97.5%: {bench['bench_ci_low']:.3f}..{bench['bench_ci_high']:.3f})")
    print(f"  share of benchmark paths with alpha <= observed: {p_le:.4f}")
    pd.DataFrame({"alpha": bench["alphas"]}).to_csv(out / "ar1_benchmark_alphas.csv", index=False)

    # 2) multifractality source decomposition
    print("\n[2] Multifractality sources (shuffled vs phase-randomised)")
    qs = np.arange(-5.0, 5.01, 0.5)
    dec = mf_decomposition(rets, qs=qs, n_surrogates=30, seed=0, **kw)
    for k, val in dec["widths"].items():
        print(f"  width_{k:<9} = {val:.3f}")
    pd.DataFrame(
        {"q": qs, "h_orig": dec["orig"]["hq"],
         "h_shuffled": dec["hq_shuffled"], "h_phase": dec["hq_phase"]}
    ).to_csv(out / "mf_decomposition_hq.csv", index=False)

    # 3) subsample stability
    print("\n[3] Subsample stability of alpha(returns)")
    eras = [
        ("2015-2018 (pre/post-crash)", "2015-01-01", "2018-12-31"),
        ("2019-2022 (incl. COVID)", "2019-01-01", "2022-12-31"),
        ("2023-2026", "2023-01-01", "2026-12-31"),
    ]
    rows = []
    for name, a, b in eras:
        m = (v["trade_date"] >= a) & (v["trade_date"] <= b)
        r = np.diff(np.log(v.loc[m, "vix"].to_numpy()))
        if len(r) < 250:
            print(f"  {name:<28} skipped (only {len(r)} obs)")
            continue
        f = dfa_fit(r, **kw)
        rows.append({"era": name, "n": len(r), "alpha": f["alpha"], "se": f["se"]})
        print(f"  {name:<28} n={len(r):5d}  alpha={f['alpha']:.3f} (se {f['se']:.3f})")

    # 4) block bootstrap CI for the full sample
    print("\n[4] Block-bootstrap CI (block=20, 1000 reps)")
    bs = block_bootstrap_alpha(rets, n_boot=1000, block=20, seed=2, **kw)
    print(f"  alpha {bs['mean']:.3f}, 95% CI [{bs['ci_low']:.3f}, {bs['ci_high']:.3f}], "
          f"share >= 0.5: {bs['p_ge_half']:.4f}")

    # 5) how sensitive is the bootstrap to the block length?
    print("\n[5] Bootstrap sensitivity to block length")
    for blk in (20, 50, 100, 200, 400):
        b = block_bootstrap_alpha(rets, n_boot=400, block=blk, seed=3, **kw)
        print(f"  block={blk:4d}  mean={b['mean']:.3f}  95% CI "
              f"[{b['ci_low']:.3f}, {b['ci_high']:.3f}]")

    # 6) scale-window sensitivity: is there a single power law?
    print("\n[6] Scale-window sensitivity of alpha (full sample)")
    n = len(rets)
    windows = [(10, 60), (10, 120), (10, 250), (10, 400), (10, 10), (10, n // 4)]
    seen = []
    for lo, hi in windows:
        if hi in seen or hi <= lo:
            continue
        seen.append(hi)
        scales = np.unique(
            np.round(np.geomspace(lo, hi, 20)).astype(int)
        )
        scales = scales[scales >= lo]
        f = dfa_fit(rets, scales=scales, order=kw["order"])
        print(f"  scales {lo}-{hi:<4d} ({len(scales)} pts): alpha={f['alpha']:.3f} "
              f"(se {f['se']:.3f}, R2 {f['r2']:.3f})")

    print("\n[7] Matched-scale subsample comparison (scales 10-150)")
    matched = np.unique(np.round(np.geomspace(10, 150, 20)).astype(int))
    rows_matched = []
    for name, a, b in eras:
        m = (v["trade_date"] >= a) & (v["trade_date"] <= b)
        r = np.diff(np.log(v.loc[m, "vix"].to_numpy()))
        if len(r) < 250:
            continue
        f = dfa_fit(r, scales=matched, order=kw["order"])
        rows_matched.append({"era": name, "n": len(r), "alpha": f["alpha"], "se": f["se"]})
        print(f"  {name:<28} alpha={f['alpha']:.3f} (se {f['se']:.3f})")
    pd.DataFrame(rows_matched).to_csv(out / "subsample_matched_scales.csv", index=False)

    summary = pd.DataFrame(
        [
            {"check": "observed_alpha_returns", "value": obs["alpha"], "se": obs["se"]},
            {"check": "observed_alpha_returns_se", "value": obs["se"], "se": np.nan},
            {"check": "ar1_phi", "value": bench["phi"], "se": np.nan},
            {"check": "ar1_benchmark_mean", "value": bench["bench_mean"], "se": bench["bench_std"]},
            {"check": "ar1_benchmark_lo", "value": bench["bench_ci_low"], "se": np.nan},
            {"check": "ar1_benchmark_hi", "value": bench["bench_ci_high"], "se": np.nan},
            {"check": "ar1_share_below_observed", "value": p_le, "se": np.nan},
            {"check": "width_orig", "value": dec["widths"]["orig"], "se": np.nan},
            {"check": "width_shuffled", "value": dec["widths"]["shuffled"], "se": np.nan},
            {"check": "width_phase", "value": dec["widths"]["phase"], "se": np.nan},
            {"check": "boot_ci_low", "value": bs["ci_low"], "se": np.nan},
            {"check": "boot_ci_high", "value": bs["ci_high"], "se": np.nan},
            {"check": "boot_share_ge_half", "value": bs["p_ge_half"], "se": np.nan},
        ]
    )
    pd.concat([summary, pd.DataFrame(rows)]).to_csv(out / "summary_robustness.csv", index=False)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
