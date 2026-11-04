"""Report functions."""

from collections.abc import Mapping, Sequence
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from bfportf.metrics.basic import (
    cagr_target,
    kurtosis_target,
    max_drawdown_target,
    pnl_target,
    profit_factor_target,
    recovery_factor_target,
    sharpe_target,
    skew_target,
    sortino_target,
)
from bfportf.portfolio import WeightLike, get_portfolio_return


def _to_array(series: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """Convert a portfolio series into a 1d numpy array of returns."""
    if isinstance(series, pd.Series):
        return series.to_numpy()
    return np.asarray(series, dtype=float)


def create_report(
    returns: pd.DataFrame,
    portfolio_symbols: Sequence[str],
    return_weights: WeightLike = None,
    benchmark_symbols: Optional[Sequence[str]] = None,
    custom_benchmark: Optional[pd.Series] = None,
    custom_portfolio_metrics: Optional[Mapping[str, Callable[[np.ndarray], float]]] = None,
    custom_formatting: Optional[Mapping[str, str]] = None,
    benchmark_weights: WeightLike = None,
) -> pd.DataFrame:
    """Create the performance report.

    Args:
        returns (pd.DataFrame): Dataframe with returns.
        portfolio_symbols (Sequence[str]): Portfolio symbols.
        return_weights (WeightLike, optional): Symbols weights. Defaults to None (equal weights).
        benchmark_symbols (Sequence[str], optional): Symbols to use as a benchmark portfolio.
            Defaults to None.
        custom_benchmark (pd.Series, optional): Custom benchmark series. Defaults to None and is
            ignored if ``benchmark_symbols`` is not None.
        custom_portfolio_metrics (Mapping[str, Callable[[np.ndarray], float]], optional): Custom
            metrics to add. Every metric receives the 1d numpy array of the portfolio (or
            benchmark) returns, exactly like the built in metrics. Defaults to None.
        custom_formatting (Mapping[str, str], optional): Custom formatting as
            ``{metric_name: format_string}``. Defaults to None.
        benchmark_weights (WeightLike, optional): Weights used for the benchmark portfolio.
            Defaults to None, which means ``return_weights`` is reused (0.1.x behaviour).

    Returns:
        pd.DataFrame: Report dataframe with the ``Portfolio``, optional ``Benchmark`` and
        ``Diff`` columns.

    Notes:
        With ``benchmark_weights=None`` a single symbol benchmark is rescaled back to its raw
        returns by ``normalize="sum"`` (see :func:`bfportf.utils.get_portfolio_return`). Pass
        ``benchmark_weights`` explicitly to evaluate the benchmark with its own weights.
    """
    portfolio_metrics: dict = {
        "PnL": pnl_target,
        "CAGR": cagr_target,
        "Max drawdown": max_drawdown_target,
        "Profit factor": profit_factor_target,
        "Recovery factor": recovery_factor_target,
        "Sharpe ratio": sharpe_target,
        "Sortino ratio": sortino_target,
        "Skew": skew_target,
        "Kurtosis": kurtosis_target,
    }
    if custom_portfolio_metrics is not None:
        portfolio_metrics.update(custom_portfolio_metrics)

    columns: dict = {
        "Portfolio": get_portfolio_return(returns, portfolio_symbols, return_weights),
    }
    if benchmark_symbols is not None:
        weights = return_weights if benchmark_weights is None else benchmark_weights
        columns["Benchmark"] = get_portfolio_return(returns, benchmark_symbols, weights)
    elif custom_benchmark is not None:
        columns["Benchmark"] = custom_benchmark

    report_data = pd.DataFrame(
        {
            column: {name: func(_to_array(series)) for name, func in portfolio_metrics.items()}
            for column, series in columns.items()
        },
    )

    if "Benchmark" in report_data.columns:
        report_data["Diff"] = report_data["Portfolio"] - report_data["Benchmark"]

    if custom_formatting is not None:
        report_data = report_data.astype(object)
        for metric_name, formatting in custom_formatting.items():
            if metric_name in report_data.index:
                report_data.loc[metric_name] = report_data.loc[metric_name].apply(formatting.format)

    return report_data
