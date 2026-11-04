"""Portfolio search facade.

Kept as a stable import path (``from bfportf.optimization import find_portfolios``) on top of the
:mod:`bfportf.search` strategies.
"""

from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import pandas as pd

from bfportf.metrics.registry import MetricLike, resolve_metric
from bfportf.portfolio import resolve_weights
from bfportf.search.base import SearchConfig, SearchStrategy, run_search
from bfportf.search.greedy import GreedySearch


def find_portfolios(
    returns: pd.DataFrame,
    metric: MetricLike,
    maximize: Optional[bool] = None,
    target_symbols: Optional[Sequence[str]] = None,
    custom_weights: Optional[Union[pd.Series, np.ndarray]] = None,
    max_number: Optional[int] = None,
    verbose: bool = True,
    tolerance: float = 0.0,
    validate: bool = True,
    strategy: Optional[SearchStrategy] = None,
) -> list:
    """Find the best portfolio for every symbol with the given metric.

    Args:
        returns (pd.DataFrame): Returns dataframe.
        metric (MetricLike): Metric to optimize: a registry name (``"sharpe"``,
            ``"max_drawdown"``, ..., see :data:`bfportf.metrics.METRICS`), a
            :class:`bfportf.metrics.Metric` instance or a callable that accepts a 1d array of
            returns.
        maximize (bool, optional): Overrides the optimization direction. Defaults to None, which
            means the direction comes from the metric (a plain callable defaults to maximization).
        target_symbols (Sequence[str], optional): Symbols to use as targets. Defaults to None
            (all columns).
        custom_weights (pd.Series | np.ndarray, optional): Weight of each symbol. Defaults to None
            (equal weights).
        max_number (int, optional): Maximum number of symbols in a portfolio. Defaults to None.
        verbose (bool, optional): Show a progress bar. Defaults to True.
        tolerance (float, optional): Minimal improvement required to add the next symbol.
            Defaults to 0.0.
        validate (bool, optional): Validate the returns with
            :func:`bfportf.data.validate_returns`. Defaults to True.
        strategy (SearchStrategy, optional): Search strategy. Defaults to None
            (:class:`bfportf.search.GreedySearch`).

    Returns:
        list: ``list[PortfolioResult]`` ordered from the best to the worst. Use
        :func:`bfportf.search.results_to_frame` to get a dataframe.

    Raises:
        UnknownSymbolError: If a target symbol or a weights entry is missing from the columns.
        DataValidationError: If the returns, the weights or the configuration are invalid.
    """
    resolved_metric = resolve_metric(metric, maximize=maximize)
    weights = resolve_weights(returns, custom_weights).to_numpy(dtype=float)
    config = SearchConfig(
        max_number=max_number,
        tolerance=tolerance,
        validate=validate,
        verbose=verbose,
    )
    active_strategy = GreedySearch() if strategy is None else strategy
    return run_search(active_strategy, returns, weights, resolved_metric, config, target_symbols)
