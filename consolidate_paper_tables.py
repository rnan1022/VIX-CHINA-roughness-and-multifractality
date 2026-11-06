"""Consolidate the robustness runs into paper-ready tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

RUNS = [
    ("50ETF (CBOE 8-day rule)", "outputs/paper_robustness/summary_robustness.csv"),
    ("50ETF (two-term)", "outputs/paper_robustness_twoterm/summary_robustness.csv"),
    ("CSI300 index options", "outputs/paper_robustness_csi300/summary_robustness.csv"),
]
BOOT_LABELS = {
    "50ETF (CBOE 8-day rule)": "50ETF CBOE rule",
    "50ETF (two-term)": "50ETF two-term",
    "CSI300 index options": "CSI300",
}


def main() -> None:
    rows = []
    bootstrap = pd.read_csv("outputs/submission_bootstrap_results.csv").set_index("index")
    for label, path in RUNS:
        s = pd.read_csv(path)
        got = dict(zip(s["check"], s["value"]))
        matched = pd.read_csv(
            path.replace("summary_robustness.csv", "subsample_matched_scales.csv")
        ) if "twoterm" in path or "csi300" in path or True else None
        sub = " / ".join(
            f"{r.alpha:.3f}" for r in matched.itertuples()
        ) if matched is not None and len(matched) else ""
        rows.append(
            {
                "index_variant": label,
                "alpha_returns": round(got["observed_alpha_returns"], 4),
                "fit_se": round(got["observed_alpha_returns_se"], 4)
                if "observed_alpha_returns_se" in got
                else np.nan,
                "ar1_phi": round(got["ar1_phi"], 4) if "ar1_phi" in got else np.nan,
                "ar1_benchmark": round(got["ar1_benchmark_mean"], 4),
                "ar1_lo": round(got["ar1_benchmark_lo"], 4),
                "ar1_hi": round(got["ar1_benchmark_hi"], 4),
                "p_ar1_le_obs": round(got["ar1_share_below_observed"], 4),
                "width_orig": round(got["width_orig"], 4),
                "width_shuffled": round(got["width_shuffled"], 4),
                "width_phase": round(got["width_phase"], 4),
                "boot_ci_low": round(bootstrap.loc[BOOT_LABELS[label], "low"], 4),
                "boot_ci_high": round(bootstrap.loc[BOOT_LABELS[label], "high"], 4),
                "subsample_matched_alphas": sub,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv("outputs/paper_table_alpha.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
