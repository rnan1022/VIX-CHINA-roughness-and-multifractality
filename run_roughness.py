"""Apply the Gatheral-Jaisson-Rosenbaum (2018) roughness estimator to our data.

Run: python run_roughness.py
Outputs: outputs/roughness_*.csv and figures/fig5_roughness.pdf/.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from roughness import (
    DEFAULT_QS,
    bias_study,
    estimate_H,
    estimate_zeta,
    m_q_delta,
    roughness_report,
)

SERIES = {
    "50ETF (two-term)": "outputs/vix_china_daily_twoterm.csv",
    "50ETF (CBOE rule)": "outputs/vix_china_daily.csv",
    "CSI300 options": "outputs/vix_csi300_daily.csv",
}


def load_log_vix(path: str) -> np.ndarray:
    v = pd.read_csv(path, parse_dates=["trade_date"]).dropna(subset=["vix"])
    return np.log(v["vix"].to_numpy())


def main() -> None:
    out = Path("outputs")
    fig_dir = Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("GJR roughness estimator on the log of each constructed China VIX")
    print("(Delta = 1..30 trading days, q = 0.5, 1, 1.5, 2, 3)")
    print("=" * 90)
    reports, summaries = {}, []
    for label, path in SERIES.items():
        x = load_log_vix(path)
        rep = roughness_report(x, label)
        reports[label] = rep
        summaries.append({"series": label, **rep["summary"]})

    df_sum = pd.DataFrame(summaries)
    df_sum.to_csv(out / "roughness_summary.csv", index=False)
    print()
    for row in summaries:
        zq = row["zeta_over_q"]
        print(f"  {row['series']:<18} zeta_q/q: " +
              ", ".join(f"q={q}: {v:.3f}" for q, v in zq.items()))

    print()
    print("=" * 90)
    print("Bias study: how much does a 30-day forward-looking VIX-type index")
    print("inflate H relative to the underlying spot log-volatility?")
    print("=" * 90)
    rows = []
    for H_true in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4):
        sim = bias_study(H_true, n=6000, horizon=30, n_paths=15, seed=100)
        rows.append({
            "H_true": H_true,
            "H_spot_est_mean": sim["H_spot_est"].mean(),
            "H_spot_est_sd": sim["H_spot_est"].std(ddof=1),
            "H_vix_like_est_mean": sim["H_vix_like_est"].mean(),
            "H_vix_like_est_sd": sim["H_vix_like_est"].std(ddof=1),
        })
        r = rows[-1]
        print(f"  H_true={H_true:.2f}  ->  H(spot)={r['H_spot_est_mean']:.3f} "
              f"(sd {r['H_spot_est_sd']:.3f})   H(VIX-like)={r['H_vix_like_est_mean']:.3f} "
              f"(sd {r['H_vix_like_est_sd']:.3f})")
    df_bias = pd.DataFrame(rows)
    df_bias.to_csv(out / "roughness_bias_study.csv", index=False)

    # invert the bias map for our observed VIX-based estimate
    observed = float(df_sum.loc[df_sum["series"] == "50ETF (two-term)", "H_per_q"].iloc[0])
    implied = float(np.interp(observed, df_bias["H_vix_like_est_mean"], df_bias["H_true"]))
    print()
    print(f"Observed H from our China VIX (50ETF two-term): {observed:.3f}")
    print(f"Implied H of the underlying spot log-volatility, "
          f"using the simulated bias map: {implied:.3f}")

    # ---- finite-sample concavity benchmark (GJR attribute it to sample size) ----
    print()
    print("=" * 90)
    print("Is the concave zeta_q/q curve a genuine multifractal signal or a")
    print("finite-sample artefact?  (simulated fBm paths of the same length)")
    print("=" * 90)
    from roughness import estimate_zeta
    from dfa import fgn

    observed_decline = float(
        df_sum.loc[df_sum["series"] == "50ETF (two-term)", "zeta_over_q"]
        .iloc[0][0.5] - df_sum.loc[df_sum["series"] == "50ETF (two-term)", "zeta_over_q"].iloc[0][3.0]
    )
    print(f"observed decline in zeta_q/q (q=0.5 -> q=3): {observed_decline:.3f}")
    conc = []
    for H_true in (0.2, 0.3, 0.4, 0.5):
        declines = []
        for i in range(15):
            path = np.cumsum(fgn(len(load_log_vix(SERIES["50ETF (two-term)"])), H_true, seed=500 + i))
            r = estimate_zeta(path, qs=DEFAULT_QS)
            zq = r["zeta"] / r["qs"]
            declines.append(float(zq[0] - zq[-1]))
        conc.append({"H_true": H_true, "decline_mean": float(np.mean(declines)),
                     "decline_sd": float(np.std(declines, ddof=1))})
        c = conc[-1]
        print(f"  fBm H={H_true:.2f}: decline {c['decline_mean']:.3f} "
              f"(sd {c['decline_sd']:.3f})")
    pd.DataFrame(conc).to_csv(out / "roughness_concavity_benchmark.csv", index=False)

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    rep = reports["50ETF (two-term)"]
    res = rep["res"]
    log_d = np.log(res["deltas"])
    for q in DEFAULT_QS:
        m = np.array([m_q_delta(load_log_vix(SERIES["50ETF (two-term)"]), q, int(d))
                      for d in res["deltas"]])
        axes[0].plot(log_d, np.log(m), "o", ms=3, label=f"q={q}")
        coef = np.polyfit(log_d, np.log(m), 1)
        axes[0].plot(log_d, np.polyval(coef, log_d), "-", lw=0.9)
    axes[0].set_xlabel(r"$\log \Delta$")
    axes[0].set_ylabel(r"$\log m(q,\Delta)$")
    axes[0].set_title("moment scaling (China VIX)")
    axes[0].legend(frameon=False, fontsize=6)

    for label in SERIES:
        r = reports[label]["res"]
        axes[1].plot(r["qs"], r["zeta"], "o-", ms=4, lw=1, label=label)
    axes[1].set_xlabel("q")
    axes[1].set_ylabel(r"$\zeta_q$")
    axes[1].set_title(r"$\zeta_q$ vs $q$")
    axes[1].legend(frameon=False, fontsize=6)

    axes[2].plot(df_bias["H_true"], df_bias["H_spot_est_mean"], "o-", label="spot log-vol (check)")
    axes[2].plot(df_bias["H_true"], df_bias["H_vix_like_est_mean"], "s-", label="30-day averaged (VIX-like)")
    axes[2].plot([0, 0.45], [0, 0.45], "k:", lw=1)
    axes[2].axhline(observed, color="C3", ls="--", lw=1)
    axes[2].set_xlabel(r"true $H$")
    axes[2].set_ylabel(r"estimated $H$")
    axes[2].set_title("integration bias of a volatility index")
    axes[2].legend(frameon=False, fontsize=6)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"fig5_roughness.{ext}", dpi=300)
    plt.close(fig)
    print("\nsaved outputs/roughness_*.csv and figures/fig5_roughness.pdf/.png")


if __name__ == "__main__":
    main()
