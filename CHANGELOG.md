# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). For `0.x` releases a minor
version bump may contain breaking changes.

## [0.3.0] - 2026-11-13

### Added

- `BfPortf 03 Metrics and strategies.ipynb`: the metric registry, the optimization direction, `tolerance`,
  the cost of the three strategies, the validation errors and short legs.
- CI workflow (`.github/workflows/ci.yml`): `ruff`, `ruff format`, `mypy` and `pytest` on Python
  3.10-3.13 with `uv sync --locked`, a build job that installs the wheel into a clean environment and
  an `examples` extra resolution job.
- `.pre-commit-config.yaml`: `uv-lock`, `ruff`, `ruff-format`, `nbstripout` and whitespace/EOF checks.
- `examples` optional dependency group (`jupyter`, `quantstats`, `pyarrow`).
- Tests that every optional argument is keyword-only (`find_portfolios`, `load_returns`,
  `sanitize_returns`, `validate_returns`, `resolve_metric`, `autocorrelation_target`).

### Changed

- Distribution without PyPI: wheels and sdists are attached to GitHub releases (see ADR-5 in
  `plan.md`), the PyPI badges were removed from the README.
- Package metadata: PEP 639 `license-files`, real project URLs, author, keywords,
  `hatchling>=1.27`.
- Examples are ported to the 0.2 API: a plain time series slice instead of `train_test_split`, the inverse
  volatility shares are estimated in-sample only (the test set is scaled with the same shares), the custom
  metric is a `Metric` with a name and a direction, the strategies are compared on the same universe, and
  `tolerance` and the validation errors are demonstrated.
- The example data is `assets/prices.parquet` (4.5 MB, 500 symbols) instead of `examples/prices.csv` (13.9 MB):
  the notebooks read the parquet file from `assets/` or a copy next to the notebook and download the release
  asset when the file is missing.
- Documentation of the weights contract: `get_portfolio_return`, `find_portfolios` and `create_report`
  compute `sum(returns) / sum(weights)`; weights are applied to the returns by the caller (for example by
  `scale_return_data`), see the README section «Weights and normalization».
- Type annotations use PEP 604 syntax (`X | None`, `X | Y`).

### Breaking changes

- Python 3.9 is no longer supported: `requires-python = ">=3.10"`.
- Optional arguments are keyword-only (the ruff `FBT001`/`FBT002` ignores were removed):
  - `find_portfolios(returns, metric, *, maximize, target_symbols, custom_weights, max_number, verbose,
    tolerance, validate, strategy)`;
  - `load_returns(prices, *, drop_nan)`, `sanitize_returns(returns, *, drop_nan_rows)`,
    `validate_returns(returns, *, allow_nan)`;
  - `resolve_metric(metric, *, maximize)`, `autocorrelation_target(returns, *, absolute, periods)`.
- `pyarrow` moved from the runtime dependencies to the `examples` extra: the library no longer installs it.
  Code that reads parquet files must install `pyarrow` itself.

### Removed

- `scikit-learn` from the `examples` extra: the notebooks split the time series with a slice.
- `BfPortf 04 Sequential time series splitting.ipynb` was dropped: the walk-forward example left the
  examples, the single in-sample / out-of-sample split of `01` and `02` stays the out-of-sample check.
- Python 3.9 classifier and `target-version = "py39"`.

## [Unreleased]

## [0.2.0] - 2026-11-04

### Added

- `metrics` package: `Metric(name, func, maximize, vectorized)` and the `METRICS` registry with ten
  built in metrics; the direction of every metric is part of the registry.
- `search` package: `GreedySearch` (default), `BeamSearch(width=5)` and `ExhaustiveSearch`
  (guarded by `max_combinations`), driven by the `SearchStrategy` protocol and `SearchConfig`.
- `PortfolioResult` with `symbols`, `weights`, `value`, `metric_name`, `to_dict()` and `to_frame()`;
  `results_to_frame()` for the tabular view.
- `data` module: `load_returns`, `sanitize_returns`, `validate_returns`, `validate_symbols`.
- `scaling` module: `scale_return_data`, `add_inverted_return`.
- `portfolio` module: `get_portfolio_return`, `normalized_weights`, `resolve_weights`,
  `calculate_metric`.
- Public API in `bfportf/__init__.py` (`__all__`, `__version__`) and the `py.typed` marker.
- Vectorized metric evaluation (`Metric.vectorized`): about 2.7x faster search on 200 symbols x 1000
  periods compared to the scalar path.
- `pandas-stubs` and `types-tqdm` in the dev group; `mypy` is configured and clean.

### Changed

- `find_portfolios(returns, metric, ...)` accepts a registry name, a `Metric` or a callable;
  `maximize=None` takes the direction from the metric.
- `create_report` passes 1d numpy arrays to the metrics (built in and custom ones alike).
- `bfportf.targets` is now a thin alias of `bfportf.metrics.basic`.

### Breaking changes

- `find_portfolios` returns `list[PortfolioResult]` instead of a `pd.DataFrame`
  (use `bfportf.results_to_frame`).
- `scale_return_data`, `add_inverted_return`, `get_portfolio_return`, `calculate_metric`,
  `validate_returns` moved to `bfportf.scaling`, `bfportf.portfolio` and `bfportf.data`;
  `bfportf.utils` keeps only the price/return conversions and `drawdown_pct`.
- `find_next_symbol` and `find_portfolio_for_symbol` are no longer public.
- Example notebooks and the README use the new API.
- `max_drawdown` is registered with `maximize=True`: the metric is a negative percentage, so the
  portfolio with the smallest drawdown has the largest value.

## [0.1.3] - 2026-11-04

### Fixed

- CAGR used one period too many (`periods = len(returns) / days_per_year`).
- Metrics return `NaN` for undefined values instead of `inf`; candidates with non finite values are
  skipped by the search instead of being selected accidentally.
- `get_portfolio_return` gained an explicit `normalize` option (`"sum"` keeps the previous
  behaviour) and reports unknown symbols/normalizations with typed errors.
- `create_report` accepts `benchmark_weights`; the benchmark evaluation is documented.
- `recovery_factor` uses the compounded PnL numerator; `sharpe`/`sortino` share `ddof=1` and accept
  a `risk_free` rate.
- `results.groupby(...)` keeps `NaN` rows in view and sorts them last.

### Added

- `exceptions` module: `BfPortfError`, `DataValidationError`, `UnknownSymbolError`.
- `tolerance` parameter of the search, `validate_returns`, `calculate_metric` accepts arrays.
- Test suite (85 tests, 95% coverage) and `pytest-cov` in the dev group.

## [0.1.2] - 2026-11-04

### Fixed

- `np.in1d` (removed in NumPy 2.0) replaced with `np.isin`.
- `create_report` formats metrics on an object frame, fixing `TypeError` on pandas 2.2+.
- `reset_index(inplace=True)` replaced to avoid the pandas 4 warning.

### Added

- `scipy` is declared as a dependency; lower bounds for `numpy`, `pandas`, `tqdm`.
- Smoke tests, `plan.md` with the architecture decisions.

### Changed

- Single toolchain: `uv` (`[dependency-groups]`, `uv.lock` in git), hatch environments removed.
- `ruff format` replaces `black`; ruff configuration moved to the `[tool.ruff.lint]` sections.
- `requires-python = ">=3.9"`, classifiers updated to 3.9-3.13.
