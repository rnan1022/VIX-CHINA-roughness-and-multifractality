# Roughness and multifractality of China's implied volatility

This repository contains the code, processed data and reported results for:

> Rao, Nan (2026). *Roughness and multifractality of China's implied
> volatility: evidence from a newly constructed daily VIX.* Manuscript
> prepared for *Finance Research Letters*.

The project constructs daily China VIX indices from Shanghai Stock Exchange
50ETF options and CFFEX CSI 300 options. It then studies anti-persistence,
multifractality and the source of the multifractal spectrum, and evaluates the
implied-minus-realised volatility spread.

## Repository contents

- `outputs/vix_china_daily_twoterm.csv`: baseline 50ETF VIX using the two
  nearest maturities.
- `outputs/vix_china_daily.csv`: 50ETF VIX under the strict CBOE eight-day
  maturity rule.
- `outputs/vix_csi300_daily.csv`: CSI 300 VIX constructed from CFFEX index
  options.
- `data/processed/`: convenience copies of the processed daily indices.
- `dfa/`: DFA and MF-DFA estimators.
- `vix_build/`: model-free VIX construction from option chains.
- `run_*.py`: construction, scaling, robustness, forecasting and economic
  analysis scripts.
- `outputs/`: reported tables, surrogate draws and supplementary results.
- `figures/`: vector figures produced from the processed series.

## Data availability

The processed daily indices and empirical outputs are included in this
repository. Raw T-format option quotations are supplied by a commercial data
vendor and are **not** redistributed here.

Users with licensed access can place the original files in `data/raw/` using
the filenames documented in [`data/raw/README.md`](data/raw/README.md), then
rebuild the indices with the commands below.

See [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) for variable definitions and
[`data/raw/README.md`](data/raw/README.md) for the raw-file layout.

## Environment

Python 3.11 or later is recommended.

```bash
python -m venv .venv
```

On Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the empirical results

All commands should be run from the repository root.

```bash
python run_shuffle_tests.py
python run_roughness.py
python run_scaling_complements.py
python run_economic.py
python run_economic_trailing.py
python run_forecast.py
python run_extra_robustness.py
python recompute_bootstrap.py
python calc_volatility_premium.py
python consolidate_paper_tables.py
python make_paper_figures.py
```

Randomised procedures use fixed seeds. Small last-digit differences may arise
across Python or NumPy versions.

## Rebuild the VIX series from licensed raw data

Place the files described in `data/raw/README.md` in `data/raw/`, then run:

```bash
python run_build_vix.py --underlying 50etf --min-near-days 8
python run_build_vix.py --underlying 50etf --min-near-days 1 --suffix _twoterm
python run_build_vix.py --underlying io --min-near-days 8
```

The first 50ETF command reproduces the strict CBOE-rule series, the second
reproduces the two-term baseline, and the third creates the CSI 300 series.

## Validation

Run the repository checks with:

```bash
python validate_release.py
python -m compileall -q .
```

The validation script checks file presence, processed-data coverage, required
columns and primary paper numbers.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Please
update the manuscript status and add the journal DOI after publication.

## License

A license has not yet been assigned to this repository. The authors should
choose the code and data licenses before making the repository public.
