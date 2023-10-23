"""Report functions"""
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from bfportf.targets import (
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
from bfportf.utils import get_portfolio_return


def create_report(
        returns: pd.DataFrame,
        portfolio_symbols: List[str],
        return_weights: pd.Series = None,
        benchmark_symbols: Optional[List[str]] = None,
        custom_benchmark: pd.Series = None,
        custom_portfolio_metrics: Optional[Dict[str, Callable[[np.ndarray], np.float64]]] = None,
        custom_formatting: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Create report

    Args:
        returns (pd.DataFrame): Dataframe with returns
        portfolio_symbols (List[str]): Portfolio symbols
        return_weights (pd.Series, optional): Symbols weights. Defaults to None. If None, equal weights will be used.
        benchmark_symbols (List[str], optional): Symbols to use as a benchmark portfolio. Defaults to None.
        custom_benchmark (pd.Series, optional): Custom benchmark series. Defaults to None. Ignored if benchmark_symbols is not None.
        custom_portfolio_metrics (Dict[str, Callable[[np.ndarray], np.float64]], optional): Custom portfolio metrics to add. Defaults to None.
        custom_formatting (Dict[str, str], optional): Custom formatting [metric_name, format]. Defaults to None.

    Returns:
        pd.DataFrame: Report dataframe
    """  # noqa: E501
    # Portfolio metrics
    portfolio_metrics = {"PnL": lambda x: pnl_target(x.values),
                    "CAGR": lambda x: cagr_target(x.values),
                    "Max drawdown": lambda x: max_drawdown_target(x.values),
                    "Profit factor": lambda x: profit_factor_target(x.values),
                    "Recovery factor": lambda x: recovery_factor_target(x.values),
                    "Sharpe ratio": lambda x: sharpe_target(x.values),
                    "Sortino ratio": lambda x: sortino_target(x.values),
                    "Skew": lambda x: skew_target(x.values),
                    "Kurtosis": lambda x: kurtosis_target(x.values), }
    if custom_portfolio_metrics is not None:
        portfolio_metrics.update(custom_portfolio_metrics)

    # Portfolio return
    portfolio_series = get_portfolio_return(
        returns, portfolio_symbols, return_weights)
    report_data = pd.DataFrame(data={"Portfolio": portfolio_series})

    # Benchmark return
    if benchmark_symbols is not None:
        benchmark_series = get_portfolio_return(
            returns, benchmark_symbols, return_weights)
        report_data["Benchmark"] = benchmark_series
    elif custom_benchmark is not None:
        report_data["Benchmark"] = custom_benchmark

    # Compute portfolio metrics
    agg_parameters = [pair[1] for pair in portfolio_metrics.items()]
    report_data = report_data.agg(agg_parameters, axis=0)
    report_data.index = [pair[0] for pair in portfolio_metrics.items()]

    # Add difference
    if "Benchmark" in report_data.columns:
        report_data["Diff"] = report_data["Portfolio"] - \
            report_data["Benchmark"]

    # Custom formatting
    if custom_formatting is not None:
        for metric_name, formatting in custom_formatting.items():
            if metric_name in report_data.index:
                report_data.loc[metric_name] = report_data.loc[metric_name].apply(formatting.format)

    return report_data
