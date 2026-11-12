# bfportf

**Python 3.10+ | MIT | no PyPI release — install the wheel from the GitHub releases**

Tool to find equity portfolios by a target metric (Sharpe ratio, drawdown, profit factor, ...) with
greedy, beam or exhaustive search.

-----

**Table of Contents**

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Metrics](#metrics)
- [Search strategies](#search-strategies)
- [Weights and normalization](#weights-and-normalization)
- [Examples](#examples)
- [Development](#development)
- [License](#license)

## Installation

The package is not published on PyPI. Download the `bfportf-<version>-py3-none-any.whl` artifact from
the [releases page](https://github.com/answering007/bfportf/releases) and install it:

```console
pip install "path/to/bfportf-0.3.0-py3-none-any.whl"
```

The examples additionally need `jupyter`, `quantstats` and `pyarrow` (the optional `examples`
extra, `pyproject.toml`). `pyarrow` is required only to read the parquet example data, so the
library itself does not depend on it.

## Quickstart

```python
import pandas as pd
import bfportf as bf

prices = pd.read_parquet("assets/prices.parquet")  # local copy of the example data
returns = bf.load_returns(prices)  # pct_change + drop incomplete periods
bf.validate_returns(returns)  # NaN / inf / duplicates / index order

# Equalize the average absolute return of every symbol (inverse volatility weights)
scaled, weights = bf.scale_return_data(returns)

found = bf.find_portfolios(
    scaled,
    metric="sharpe",  # any name from bf.METRICS, a Metric or a callable
    custom_weights=weights,
    max_number=3,
)
best = found[0]  # PortfolioResult(symbols, weights, value, metric_name)
print(best.symbols, round(best.value, 3))

bf.results_to_frame(found).head()  # legacy dataframe view

report = bf.create_report(
    returns=scaled,
    portfolio_symbols=best.symbols,
    return_weights=weights,
    benchmark_symbols=["SPY"],
    custom_formatting={"PnL": "{:.2%}", "Max drawdown": "{:.2%}"},
)
```

`find_portfolios` returns a `list[PortfolioResult]` ordered from the best to the worst portfolio.
`PortfolioResult` carries the symbols, the (normalized) weights, the metric value and the metric
name; `to_dict()` and `to_frame()` are available for logging and tabular output.

## Metrics

The metric direction is part of the registry, so `find_portfolios` never "optimizes" a drawdown in
the wrong direction:

```python
bf.METRICS["sharpe"]  # Metric(name='sharpe', maximize=True, ...)
bf.METRICS["max_drawdown"]  # negative percentage, maximize=True -> the smallest drawdown wins
```

| Name | Description |
|---|---|
| `pnl` | total compounded return |
| `cagr` | compound annual growth rate |
| `max_drawdown` | max drawdown as a negative percentage |
| `profit_factor` | gross profit / gross loss |
| `recovery_factor` | compounded PnL / absolute max drawdown |
| `sharpe` | annualized Sharpe ratio |
| `sortino` | annualized Sortino ratio |
| `skew` | skewness |
| `kurtosis` | kurtosis |
| `autocorrelation` | absolute autocorrelation, lag 1 |

Undefined values are reported as `NaN` (for example the profit factor of a series without losing
periods) and such candidates are never selected. A plain callable is also accepted and defaults to
maximization (`maximize=False` overrides the direction).

## Search strategies

| Strategy | Guarantee | Cost | Use it when |
|---|---|---|---|
| `GreedySearch()` (default) | local optimum | `O(k * n)` | large universes, speed matters |
| `BeamSearch(width=5)` | better than greedy, no guarantee | `O(width * k * n)` | a few times more time is acceptable |
| `ExhaustiveSearch()` | global optimum for `max_number` | `O(C(n, k))` | small universes (n <= 15) and tests |

```python
beam = bf.find_portfolios(scaled, metric="sharpe", max_number=3, strategy=bf.BeamSearch(width=5))
exhaustive = bf.find_portfolios(scaled, metric="sharpe", max_number=3, strategy=bf.ExhaustiveSearch())
```

`ExhaustiveSearch` refuses to run above `max_combinations` (2 000 000 by default) to protect from an
accidental launch on a 700 symbol universe (~57 million combinations).

## Weights and normalization

`scale_return_data` returns inverse volatility shares that sum up to one and the scaled returns
(`symbol return * share`). The weights are therefore *allocation shares applied to the input
returns*, not weights applied internally by `get_portfolio_return` / `create_report`. Those
functions compute

```
portfolio return = sum(returns[symbols]) / sum(weights[symbols])
```

so the weights only normalize the aggregate. The practical consequences:

- to weight arbitrary symbols, scale the returns first (`scaled = returns * weights`) and pass the
  same weights along, exactly like `scale_return_data` does;
- passing raw returns together with unequal weights does **not** produce a weighted average, it
  only changes the divisor: `sum(r) / sum(w)`;
- for a single symbol portfolio the normalization cancels out and the raw symbol returns are
  returned;
- a portfolio is implicitly levered according to the sum of its weights (`normalize="none"` keeps
  the original scale, i.e. `sum(w * r)`).

Keep this in mind when comparing a found portfolio with a benchmark.

## Examples

See the [examples folder](https://github.com/answering007/bfportf/tree/master/examples):

- `BfPortf 01 Simple example.ipynb` — Sharpe ratio search with an in-sample / out-of-sample split, the
  reports and the three search strategies;
- `BfPortf 02 Custom target function.ipynb` — a custom `Metric`, its direction and a QuantStats report;
- `BfPortf 03 Metrics and strategies.ipynb` — the registry, the optimization direction, `tolerance`, the
  cost of the strategies, the errors and short legs;
- `BfPortf 04 Sequential time series splitting.ipynb` — walk-forward validation: a local
  `split_sequential` helper cuts the history into consecutive train/test segments, the out-of-sample
  returns are stitched into a single track record and compared with the SPY benchmark.

The example data (`prices.parquet`, 4.5 MB, 500 symbols) is not tracked in git: it lives in `assets/` and is
published as an asset of the latest GitHub release. The notebooks look for `assets/prices.parquet` (or a copy
next to the notebook) and download the asset when the file is missing:

```console
curl -L -o assets/prices.parquet https://github.com/answering007/bfportf/releases/latest/download/prices.parquet
```

Reading parquet needs `pyarrow`, which comes with the `examples` extra. Run the notebooks with it:

```console
uv sync --extra examples
uv run jupyter lab examples
```

## Development

The project uses [uv](https://docs.astral.sh/uv/) as the single toolchain (`hatchling` is only the
build backend):

```console
uv sync --group dev
uv run pytest -q --cov=bfportf
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pre-commit install      # optional: ruff, ruff-format, nbstripout, uv-lock on commit
```

The CI workflow (`.github/workflows/ci.yml`) runs the same checks on Python 3.10-3.13 with
`uv sync --locked`, then builds the wheel and installs it into a clean environment.

## License

`bfportf` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
