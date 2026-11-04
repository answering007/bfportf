"""Smoke tests for the public bfportf flow.

Stage 0: the package must import and the main entry points must run
end-to-end on synthetic data (regression for P0-1, P0-2, P0-3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bfportf.__about__ import __version__
from bfportf.optimization import find_portfolios
from bfportf.report import create_report
from bfportf.targets import sharpe_target


def make_returns(n_symbols: int = 12, n_rows: int = 250, seed: int = 42) -> pd.DataFrame:
    """Build deterministic synthetic daily returns."""
    rng = np.random.default_rng(seed)
    values = rng.normal(loc=0.0005, scale=0.01, size=(n_rows, n_symbols))
    index = pd.date_range("2020-01-02", periods=n_rows, freq="B")
    columns = [f"SYM{i:02d}" for i in range(n_symbols)]
    return pd.DataFrame(values, index=index, columns=columns)


def test_version_is_available() -> None:
    assert isinstance(__version__, str)
    assert __version__.count(".") == 2


def test_find_portfolios_returns_finite_values() -> None:
    returns = make_returns()
    found = find_portfolios(
        returns=returns,
        target_function=sharpe_target,
        maximize=True,
        max_number=2,
        verbose=False,
    )

    assert not found.empty
    assert list(found.columns[:2]) == ["symbols", "target_function_value"]
    assert np.isfinite(found["target_function_value"]).all()


def test_create_report_numeric() -> None:
    returns = make_returns()
    symbols = list(returns.columns[:3])
    report = create_report(
        returns=returns,
        portfolio_symbols=symbols,
        benchmark_symbols=["SYM00"],
    )

    assert "Portfolio" in report.columns
    assert "Benchmark" in report.columns
    assert "Diff" in report.columns
    assert isinstance(report.loc["PnL", "Portfolio"], (float, np.floating))


def test_create_report_with_custom_formatting() -> None:
    returns = make_returns()
    symbols = list(returns.columns[:3])
    report = create_report(
        returns=returns,
        portfolio_symbols=symbols,
        benchmark_symbols=["SYM00"],
        custom_formatting={"PnL": "{:.2%}", "Sharpe ratio": "{:.2f}"},
    )

    assert isinstance(report.loc["PnL", "Portfolio"], str)
    assert report.loc["PnL", "Portfolio"].endswith("%")
    assert isinstance(report.loc["Sharpe ratio", "Portfolio"], str)
