"""Recompute full-sample DFA block-bootstrap intervals."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dfa import dfa_fit


def circular_block_bootstrap(
    values: np.ndarray,
    n_boot: int = 400,
    block: int = 400,
    seed: int = 2,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    n_blocks = int(np.ceil(n / block))
    alphas = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        sample = np.concatenate(
            [np.take(values, np.arange(s, s + block), mode="wrap") for s in starts]
        )[:n]
        alphas[b] = dfa_fit(sample, order=1, min_scale=10, n_scales=20)["alpha"]
    return {
        "mean": float(alphas.mean()),
        "low": float(np.percentile(alphas, 2.5)),
        "high": float(np.percentile(alphas, 97.5)),
        "share_ge_half": float(np.mean(alphas >= 0.5)),
    }


rows = []
for label, path in (
    ("50ETF two-term", "outputs/vix_china_daily_twoterm.csv"),
    ("50ETF CBOE rule", "outputs/vix_china_daily.csv"),
    ("CSI300", "outputs/vix_csi300_daily.csv"),
):
    vix = pd.read_csv(path, parse_dates=["trade_date"]).dropna(subset=["vix"])
    returns = np.diff(np.log(vix["vix"].to_numpy()))
    result = circular_block_bootstrap(returns)
    print(label, result)
    rows.append({"index": label, **result})

pd.DataFrame(rows).to_csv("outputs/submission_bootstrap_results.csv", index=False)
