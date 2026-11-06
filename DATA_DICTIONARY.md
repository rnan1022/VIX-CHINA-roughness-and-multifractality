# Data dictionary

## Processed daily index files

The files in `data/processed/` and the corresponding `outputs/vix_*.csv` files
have one row per trading day unless stated otherwise.

| Variable | Description |
|---|---|
| `trade_date` | Trading date in `YYYY-MM-DD` format. |
| `vix` | Annualised constant-30-day model-free volatility index in percentage points. |
| `mode` | Maturity configuration used: `both`, `near_only` or `next_only`. |
| `adj` | Dividend-adjustment convention used: `0` for unadjusted, `1` for adjusted. |
| `weight_near` | Interpolation weight assigned to the near maturity in the 30-day variance calculation. |
| `extrapolated` | `True` when the 30-day target lies outside the available maturity interval. |
| `near_days` | Calendar days to the near expiry. |
| `next_days` | Calendar days to the next expiry. |
| `F` | Forward price implied by put-call parity for the near expiry. |
| `F_next` | Forward price implied by put-call parity for the next expiry. |
| `K0` | Largest listed near-maturity strike not exceeding the near-term forward. |
| `n_strikes_near` | Number of selected strikes in the near-term calculation. |
| `n_strikes_next` | Number of selected strikes in the next-term calculation. |
| `sigma2_near` | Near-term annualised risk-neutral variance. |
| `sigma2_next` | Next-term annualised risk-neutral variance. |
| `fail_near` | Diagnostic message when the near-term variance cannot be computed. |
| `fail_next` | Diagnostic message when the next-term variance cannot be computed. |

`outputs/vix_old_pipeline.csv` contains `date` and `vix_old`. The latter is an
independently constructed validation series expressed as a decimal rather than
a percentage.

## Result files

- `paper_table_shuffle.csv`: shuffled-surrogate tests for the DFA exponent.
- `roughness_summary.csv`: moment-scaling roughness estimates.
- `scaling_complements.csv`: autocorrelations, variance ratios and fO-U fits.
- `paper_table_extra_robustness.csv`: alternative DFA specifications.
- `submission_bootstrap_results.csv`: 400-day moving-block bootstrap intervals.
- `economic_summary.csv`: index, realised volatility and spread averages.
- `economic_predictive.csv`: return-predictability regressions.
- `economic_trailing_vrp.csv`: regressions using the trailing spread.
- `submission_economic_checks.csv`: spread diagnostics reported in the paper.
- `forecast_results.csv`: out-of-sample forecast errors.
- `paper_robustness*/`: surrogate and subsample distributions.

Column names in the research-output files follow the scripts that generate
them. The manuscript describes all retained quantities in Sections 3 and 4 and
the appendix.
