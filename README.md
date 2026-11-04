# bfportf

[![PyPI - Version](https://img.shields.io/pypi/v/bfportf.svg)](https://pypi.org/project/bfportf)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/bfportf.svg)](https://pypi.org/project/bfportf)

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

```console
pip install bfportf
```

## Quickstart

```python
import pandas as pd
import bfportf as bf

prices = pd.read_csv("prices.csv", index_col=0, parse_dates=["Date"])
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

`scale_return_data` returns inverse volatility shares that sum up to one. Weights passed to
`get_portfolio_return` / `create_report` are *allocation shares*: with `normalize="sum"` (the
default) a sub portfolio is rescaled to a fully invested position, which means

- for a single symbol portfolio the scaling cancels out and the raw symbol returns are returned;
- a portfolio is implicitly levered according to the sum of its weights (`normalize="none"` keeps
  the original scale).

Keep this in mind when comparing a found portfolio with a benchmark.

## Examples

See the [examples folder](https://github.com/answering007/bfportf/tree/master/examples):

- `BfPortf 01 Simple example.ipynb` — Sharpe ratio search with a train/test split;
- `BfPortf 02 Custom target function.ipynb` — custom callable metric and a QuantStats report.

## Development

The project uses [uv](https://docs.astral.sh/uv/) as the single toolchain (`hatchling` is only the
build backend):

```console
uv sync --group dev
uv run pytest -q --cov=bfportf
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

## License

`bfportf` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
