"""Metrics: definitions, implementations and the name registry."""

from bfportf.metrics.base import Metric, MetricFunction, VectorizedMetricFunction
from bfportf.metrics.basic import (
    autocorrelation_target,
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
from bfportf.metrics.registry import DEFAULT_METRIC, METRICS, MetricLike, resolve_metric

__all__ = [
    "DEFAULT_METRIC",
    "METRICS",
    "Metric",
    "MetricFunction",
    "MetricLike",
    "VectorizedMetricFunction",
    "autocorrelation_target",
    "cagr_target",
    "kurtosis_target",
    "max_drawdown_target",
    "pnl_target",
    "profit_factor_target",
    "recovery_factor_target",
    "resolve_metric",
    "sharpe_target",
    "skew_target",
    "sortino_target",
]
