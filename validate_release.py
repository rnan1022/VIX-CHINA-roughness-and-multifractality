"""Validate the public replication package."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent

REQUIRED_FILES = [
    "README.md",
    "CITATION.cff",
    "DATA_DICTIONARY.md",
    "requirements.txt",
    "outputs/vix_china_daily.csv",
    "outputs/vix_china_daily_twoterm.csv",
    "outputs/vix_csi300_daily.csv",
    "outputs/paper_table_shuffle.csv",
    "outputs/paper_table_alpha.csv",
    "outputs/roughness_summary.csv",
    "outputs/scaling_complements.csv",
    "outputs/paper_table_extra_robustness.csv",
    "outputs/submission_bootstrap_results.csv",
    "outputs/economic_summary.csv",
    "outputs/economic_predictive.csv",
    "outputs/economic_trailing_vrp.csv",
    "outputs/submission_economic_checks.csv",
    "outputs/forecast_results.csv",
    "figures/fig1_index.pdf",
    "figures/fig2_multifractal_sources.pdf",
    "figures/figA1_scale_dependence.pdf",
    "figures/figA2_independent_validation.pdf",
    "figures/figA3_roughness_diagnostics.pdf",
]

REQUIRED_COLUMNS = {
    "trade_date",
    "vix",
    "mode",
    "adj",
    "weight_near",
    "extrapolated",
    "near_days",
    "next_days",
}


def main() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    series_specs = (
        ("50ETF baseline", "outputs/vix_china_daily_twoterm.csv", 2739),
        ("50ETF CBOE rule", "outputs/vix_china_daily.csv", 2739),
        ("CSI 300", "outputs/vix_csi300_daily.csv", 1558),
    )
    for label, path, expected_n in series_specs:
        frame = pd.read_csv(ROOT / path, parse_dates=["trade_date"])
        if not REQUIRED_COLUMNS.issubset(frame.columns):
            raise ValueError(f"{label}: missing required columns")
        valid = frame.dropna(subset=["vix"])
        if len(valid) != expected_n:
            raise ValueError(
                f"{label}: expected {expected_n} valid observations, got {len(valid)}"
            )
        if valid["vix"].le(0).any():
            raise ValueError(f"{label}: non-positive VIX values found")
        print(
            f"{label}: {len(valid):,} valid observations, "
            f"{valid['trade_date'].min().date()} to "
            f"{valid['trade_date'].max().date()}"
        )

    for name in (
        "vix_china_daily.csv",
        "vix_china_daily_twoterm.csv",
        "vix_csi300_daily.csv",
    ):
        output = (ROOT / "outputs" / name).read_bytes()
        processed = (ROOT / "data" / "processed" / name).read_bytes()
        if output != processed:
            raise ValueError(f"Dataset copy differs from outputs/{name}")

    unsupported_raw = [
        path.name
        for path in (ROOT / "data" / "raw").iterdir()
        if path.is_file() and path.name != "README.md"
    ]
    if unsupported_raw:
        raise ValueError(
            "Public package contains raw vendor files that require a license "
            f"review: {unsupported_raw}"
        )

    bootstrap = pd.read_csv(ROOT / "outputs/submission_bootstrap_results.csv")
    bootstrap = bootstrap.set_index("index")
    consolidated = pd.read_csv(ROOT / "outputs/paper_table_alpha.csv")
    consolidated = consolidated.set_index("index_variant")
    for table_label, bootstrap_label in (
        ("50ETF (two-term)", "50ETF two-term"),
        ("50ETF (CBOE 8-day rule)", "50ETF CBOE rule"),
        ("CSI300 index options", "CSI300"),
    ):
        observed = consolidated.loc[table_label, ["boot_ci_low", "boot_ci_high"]]
        expected = bootstrap.loc[bootstrap_label, ["low", "high"]].to_numpy()
        if not (observed.to_numpy() == expected.round(4)).all():
            raise ValueError(f"Inconsistent bootstrap values for {table_label}")

    print("Release package validation passed.")


if __name__ == "__main__":
    main()
