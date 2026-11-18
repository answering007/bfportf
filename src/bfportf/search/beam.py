"""Beam search strategy."""

import numpy as np

from bfportf.exceptions import DataValidationError, UnknownSymbolError
from bfportf.metrics.base import Metric
from bfportf.search.base import (
    SearchConfig,
    candidate_values,
    improves,
    portfolio_metric,
    sort_key,
)


class BeamSearch:
    """Beam search over partial portfolios.

    Instead of a single best partial portfolio (greedy), the ``width`` best ones are kept and
    extended on every step, which helps to escape some local optima at the cost of roughly
    ``width`` times more metric evaluations.

    Attributes:
        width (int): Number of partial portfolios kept on every step. Defaults to 5.
    """

    name = "beam"

    def __init__(self, width: int = 5) -> None:
        """Initialize the strategy.

        Args:
            width (int, optional): Number of partial portfolios kept on every step. Defaults to 5.

        Raises:
            DataValidationError: If ``width`` is lower than 1.
        """
        if width < 1:
            msg = f"width must be >= 1, got {width}"
            raise DataValidationError(msg)
        self.width = width

    def validate(self, n_symbols: int, config: SearchConfig) -> None:
        """Nothing to check: the cost is bounded by ``width`` times the greedy cost."""

    def select(
        self,
        returns: np.ndarray,
        symbols: np.ndarray,
        weights: np.ndarray,
        metric: Metric,
        target_symbol: str,
        config: SearchConfig,
    ) -> tuple[tuple[str, ...], float]:
        """Select a portfolio for the target symbol.

        Args:
            returns (np.ndarray): 2d array of returns (periods x symbols).
            symbols (np.ndarray): 1d array of symbols.
            weights (np.ndarray): 1d array of weights.
            metric (Metric): Metric to optimize.
            target_symbol (str): Target symbol.
            config (SearchConfig): Search configuration.

        Returns:
            tuple[tuple[str, ...], float]: ``(portfolio_symbols, metric value)``.

        Raises:
            UnknownSymbolError: If ``target_symbol`` is missing from ``symbols``.
        """
        available = symbols.tolist()
        if target_symbol not in available:
            raise UnknownSymbolError([target_symbol], available)
        max_number = config.max_number if config.max_number is not None else len(available)

        start = (available.index(target_symbol),)
        beam = [start]
        best_symbols = start
        best_value = portfolio_metric(returns, weights, list(start), metric)

        while len(best_symbols) < max_number:
            pool: dict = {}
            for member in beam:
                candidates = np.array([index for index in range(len(available)) if index not in member])
                if candidates.size == 0:
                    continue
                values = candidate_values(returns, weights, list(member), candidates, metric)
                for position, candidate in enumerate(candidates):
                    value = float(values[position])
                    if not np.isfinite(value):
                        continue
                    portfolio = tuple(sorted((*member, int(candidate))))
                    previous = pool.get(portfolio)
                    if previous is None or improves(value, previous, metric):
                        pool[portfolio] = value

            if not pool:
                break

            ranked = sorted(pool.items(), key=lambda item: sort_key(item[1], metric))[: self.width]
            top_symbols, top_value = ranked[0]
            if not improves(top_value, best_value, metric, config.tolerance):
                break

            beam = [portfolio for portfolio, _ in ranked]
            best_symbols, best_value = top_symbols, top_value

        return tuple(symbols[list(best_symbols)].tolist()), best_value
