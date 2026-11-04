"""bfportf - find portfolios by a target metric.

Public API: everything listed in ``__all__`` is stable, anything else is internal.
"""

from bfportf.__about__ import __version__
from bfportf.data import load_returns, sanitize_returns, validate_returns
from bfportf.exceptions import BfPortfError, DataValidationError, UnknownSymbolError
from bfportf.metrics import DEFAULT_METRIC, METRICS, Metric, resolve_metric
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
from bfportf.optimization import find_portfolios
from bfportf.portfolio import calculate_metric, get_portfolio_return, normalized_weights
from bfportf.report import create_report
from bfportf.scaling import add_inverted_return, scale_return_data
from bfportf.search import (
    BeamSearch,
    ExhaustiveSearch,
    GreedySearch,
    PortfolioResult,
    SearchConfig,
    SearchStrategy,
    results_to_frame,
)

__all__ = [
    "DEFAULT_METRIC",
    "METRICS",
    "BeamSearch",
    "BfPortfError",
    "DataValidationError",
    "ExhaustiveSearch",
    "GreedySearch",
    "Metric",
    "PortfolioResult",
    "SearchConfig",
    "SearchStrategy",
    "UnknownSymbolError",
    "__version__",
    "add_inverted_return",
    "autocorrelation_target",
    "cagr_target",
    "calculate_metric",
    "create_report",
    "find_portfolios",
    "get_portfolio_return",
    "kurtosis_target",
    "load_returns",
    "max_drawdown_target",
    "normalized_weights",
    "pnl_target",
    "profit_factor_target",
    "recovery_factor_target",
    "resolve_metric",
    "results_to_frame",
    "sanitize_returns",
    "scale_return_data",
    "sharpe_target",
    "skew_target",
    "sortino_target",
    "validate_returns",
]
