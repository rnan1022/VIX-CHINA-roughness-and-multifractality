"""Generate paper-ready figures and a numbers dump for the manuscript."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dfa import dfa_fit, hurst_significance, mf_dfa

FIG = Path("figures")
FIG.mkdir(parents=True, exist_ok=True)

SERIES = {
    "50ETF (two-term)": "outputs/vix_china_daily_twoterm.csv",
    "50ETF (CBOE rule)": "outputs/vix_china_daily.csv",
    "CSI300 options": "outputs/vix_csi300_daily.csv",
}
COLORS = {"50ETF (two-term)": "#1f4e79", "50ETF (CBOE rule)": "#b0b7c3", "CSI300 options": "#c0504d"}
EVENTS = [
    ("2015-06-15", "2015-09-30", "2015 crash"),
    ("2016-01-04", "2016-02-29", "2016 circuit breaker"),
    ("2018-02-01", "2018-03-30", "2018 vol shock"),
    ("2020-02-20", "2020-04-15", "COVID-19"),
    ("2024-09-20", "2024-10-31", "2024 policy rally"),
]


def load(path: str) -> pd.Series:
    v = pd.read_csv(path, parse_dates=["trade_date"]).dropna(subset=["vix"])
    return v.set_index("trade_date")["vix"].sort_index()


def save(fig, name: str) -> None:
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  saved figures/{name}.pdf/.png")


def fig1_index(series: dict[str, pd.Series]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6))
    for ax, key in zip(axes, ["50ETF (two-term)", "CSI300 options"]):
        s = series[key]
        ax.plot(s.index, s.values, lw=0.7, color=COLORS[key])
        for start, end, label in EVENTS:
            if s.index.min() <= pd.Timestamp(end) and s.index.max() >= pd.Timestamp(start):
                ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="0.85", zorder=0)
                ax.annotate(
                    label, xy=(pd.Timestamp(start), s.max() * 0.97),
                    fontsize=6.5, rotation=90, ha="right", va="top", color="0.35",
                )
        ax.set_ylabel("VIX level")
        ax.set_ylim(0, s.max() * 1.08)
    axes[0].set_title("Panel A: China VIX from SSE 50ETF options (2015-02 to 2026-05)")
    axes[1].set_title("Panel B: China VIX from CFFEX CSI 300 index options (2019-12 to 2026-05)")
    save(fig, "fig1_index")


def fig2_scale_dependence(series: dict[str, pd.Series]) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    rows = []
    for key, s in series.items():
        rets = np.diff(np.log(s.values))
        n = len(rets)
        grid = [60, 90, 120, 180, 250, 350, 500, n // 4]
        xs, ys = [], []
        for hi in sorted({g for g in grid if 20 <= g <= n // 4}):
            scales = np.unique(np.round(np.geomspace(10, hi, 20)).astype(int))
            fit = dfa_fit(rets, scales=scales, order=1)
            xs.append(hi)
            ys.append(fit["alpha"])
            rows.append({"series": key, "max_scale": hi, "alpha": fit["alpha"], "se": fit["se"]})
        ax.plot(xs, ys, "o-", ms=4, lw=1.2, color=COLORS[key], label=key)
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel("upper scale bound $s_{max}$ (trading days)")
    ax.set_ylabel(r"DFA exponent $\alpha$")
    ax.set_title("Scale-window sensitivity of $\\alpha$ (log-returns)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig2_scale_dependence")
    return pd.DataFrame(rows)


def fig3_multifractal(series: dict[str, pd.Series]) -> pd.DataFrame:
    qs = np.arange(-5.0, 5.01, 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    rows = []
    for key in ["50ETF (two-term)", "CSI300 options"]:
        for label, x in (
            ("returns", np.diff(np.log(series[key].values))),
            ("|returns|", np.abs(np.diff(np.log(series[key].values)))),
        ):
            res = mf_dfa(x, qs=qs, order=1, min_scale=10, n_scales=20)
            ls = "-" if label == "returns" else "--"
            axes[0].plot(res["qs"], res["hq"], ls, lw=1.4, color=COLORS[key],
                         label=f"{key}, {label}")
            axes[1].plot(res["alpha"], res["f"], ls, lw=1.4, color=COLORS[key],
                         label=f"{key}, {label}")
            for q, h in zip(res["qs"], res["hq"]):
                rows.append({"series": key, "transform": label, "q": q, "hq": h})
    axes[0].axhline(0.5, color="k", ls=":", lw=1)
    axes[0].set_xlabel("$q$")
    axes[0].set_ylabel("$h(q)$")
    axes[0].set_title("Generalized Hurst exponent")
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel(r"$f(\alpha)$")
    axes[1].set_title("Singularity spectrum")
    axes[0].legend(frameon=False, fontsize=6.5)
    save(fig, "fig3_multifractal")
    return pd.DataFrame(rows)


def fig4_validation(series: dict[str, pd.Series]) -> dict:
    old = pd.read_csv("outputs/vix_old_pipeline.csv", parse_dates=["date"]).set_index("date")
    old = old["vix_old"] * 100.0
    out = {}
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.plot(
        old.index,
        old.values,
        lw=2.2,
        color="#D55E00",
        linestyle="-",
        label="prior pipeline (JoinQuant data)",
    )
    for key in ["50ETF (two-term)", "50ETF (CBOE rule)"]:
        s = series[key]
        both = pd.concat([old, s], axis=1).dropna()
        out[key] = {
            "n_overlap": int(len(both)),
            "corr": float(both.iloc[:, 0].corr(both.iloc[:, 1])),
            "mad": float((both.iloc[:, 1] - both.iloc[:, 0]).abs().mean()),
            "mean_diff": float((both.iloc[:, 1] - both.iloc[:, 0]).mean()),
        }
    s = series["50ETF (two-term)"]
    ax.plot(
        s.index,
        s.values,
        lw=1.2,
        color=COLORS["50ETF (two-term)"],
        linestyle="-",
        label="this paper",
    )
    ax.set_ylabel("VIX level")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Overlap validation, 2021-01 to 2023-09")
    save(fig, "fig4_validation")
    return out


def main() -> None:
    series = {k: load(p) for k, p in SERIES.items()}
    print("figures:")
    fig1_index(series)
    scale = fig2_scale_dependence(series)
    hq = fig3_multifractal(series)
    val = fig4_validation(series)
    scale.to_csv("outputs/table_scale_dependence.csv", index=False)
    hq.to_csv("outputs/table_hq.csv", index=False)

    # numbers used in the manuscript text
    numbers = {"validation_overlap": val, "series_summary": {}, "alpha": {}, "shuffle": {}}
    for key, s in series.items():
        rets = np.diff(np.log(s.values))
        numbers["series_summary"][key] = {
            "start": str(s.index.min().date()),
            "end": str(s.index.max().date()),
            "n": int(len(s)),
            "mean": float(s.mean()),
            "sd": float(s.std()),
            "min": float(s.min()),
            "median": float(np.median(s)),
            "max": float(s.max()),
        }
        numbers["alpha"][key] = {
            "level": dfa_fit(s.values, order=1, min_scale=10, n_scales=20)["alpha"],
            "returns": dfa_fit(rets, order=1, min_scale=10, n_scales=20)["alpha"],
            "abs_returns": dfa_fit(np.abs(rets), order=1, min_scale=10, n_scales=20)["alpha"],
        }
        sh = hurst_significance(rets, n_surrogates=500, seed=11, order=1,
                                min_scale=10, n_scales=20)
        numbers["shuffle"][key] = {k: sh[k] for k in ("alpha_obs", "surrogate_mean", "z", "p_ge")}
    # year means for the main series
    s0 = series["50ETF (two-term)"]
    numbers["year_means"] = {str(y): float(g.mean()) for y, g in s0.groupby(s0.index.year)}
    with open("outputs/manuscript_numbers.json", "w", encoding="utf-8") as f:
        json.dump(numbers, f, indent=2, ensure_ascii=False)
    print("\nnumbers written to outputs/manuscript_numbers.json")
    print(json.dumps(numbers["shuffle"], indent=2))
    print(json.dumps(numbers["validation_overlap"], indent=2))
    print("year means:", {k: round(v, 1) for k, v in numbers["year_means"].items()})


if __name__ == "__main__":
    main()
