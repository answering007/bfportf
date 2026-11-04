"""Metrics registry: name -> Metric(name, func, maximize, vectorized)."""

from dataclasses import replace
from typing import Optional, Union

from bfportf.exceptions import DataValidationError
from bfportf.metrics.base import Metric, MetricFunction
from bfportf.metrics.basic import (
    autocorrelation_target,
    cagr_target,
    cagr_target_matrix,
    kurtosis_target,
    kurtosis_target_matrix,
    max_drawdown_target,
    max_drawdown_target_matrix,
    pnl_target,
    pnl_target_matrix,
    profit_factor_target,
    profit_factor_target_matrix,
    recovery_factor_target,
    recovery_factor_target_matrix,
    sharpe_target,
    sharpe_target_matrix,
    skew_target,
    skew_target_matrix,
    sortino_target,
    sortino_target_matrix,
)

MetricLike = Union[str, Metric, MetricFunction]

DEFAULT_METRIC = "sharpe"

METRICS: dict = {
    "pnl": Metric("pnl", pnl_target, True, pnl_target_matrix, "Total compounded return"),
    "cagr": Metric("cagr", cagr_target, True, cagr_target_matrix, "Compound annual growth rate"),
    "max_drawdown": Metric(
        "max_drawdown",
        max_drawdown_target,
        True,
        max_drawdown_target_matrix,
        "Max drawdown as a negative percentage: the best portfolio has the largest value",
    ),
    "profit_factor": Metric(
        "profit_factor",
        profit_factor_target,
        True,
        profit_factor_target_matrix,
        "Gross profit divided by gross loss",
    ),
    "recovery_factor": Metric(
        "recovery_factor",
        recovery_factor_target,
        True,
        recovery_factor_target_matrix,
        "Compounded PnL divided by the absolute max drawdown",
    ),
    "sharpe": Metric("sharpe", sharpe_target, True, sharpe_target_matrix, "Annualized Sharpe ratio"),
    "sortino": Metric(
        "sortino",
        sortino_target,
        True,
        sortino_target_matrix,
        "Annualized Sortino ratio",
    ),
    "skew": Metric("skew", skew_target, True, skew_target_matrix, "Return skewness"),
    "kurtosis": Metric("kurtosis", kurtosis_target, True, kurtosis_target_matrix, "Return kurtosis"),
    "autocorrelation": Metric(
        "autocorrelation",
        autocorrelation_target,
        True,
        None,
        "Absolute autocorrelation of lag 1 (scalar only)",
    ),
}


def resolve_metric(metric: MetricLike, maximize: Optional[bool] = None) -> Metric:
    """Resolve a metric reference into a :class:`bfportf.metrics.base.Metric`.

    Args:
        metric (MetricLike): Registry name, :class:`Metric` instance or a plain callable that
            accepts a 1d array of returns.
        maximize (bool, optional): Overrides the optimization direction of the resolved metric.
            Defaults to None (direction comes from the metric; plain callables default to True).

    Returns:
        Metric: Resolved metric.

    Raises:
        DataValidationError: If the metric name is unknown or the type is not supported.
    """
    if isinstance(metric, Metric):
        resolved = metric
    elif isinstance(metric, str):
        if metric not in METRICS:
            available = ", ".join(sorted(METRICS))
            msg = f"unknown metric {metric!r}, available: {available}"
            raise DataValidationError(msg)
        resolved = METRICS[metric]
    elif callable(metric):
        name = getattr(metric, "__name__", "custom")
        resolved = Metric(name, metric, True)
    else:
        msg = f"metric must be a name, a Metric or a callable, got {type(metric).__name__}"
        raise DataValidationError(msg)

    if maximize is not None and maximize != resolved.maximize:
        return replace(resolved, maximize=maximize)
    return resolved
