"""Backward compatible alias of :mod:`bfportf.metrics.basic`.

The implementations live in :mod:`bfportf.metrics.basic`; this module keeps the historical
``bfportf.targets`` import path working.
"""

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

__all__ = [
    "autocorrelation_target",
    "cagr_target",
    "kurtosis_target",
    "max_drawdown_target",
    "pnl_target",
    "profit_factor_target",
    "recovery_factor_target",
    "sharpe_target",
    "skew_target",
    "sortino_target",
]
