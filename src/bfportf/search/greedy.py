"""Greedy forward selection strategy."""

import numpy as np

from bfportf.exceptions import UnknownSymbolError
from bfportf.metrics.base import Metric
from bfportf.search.base import (
    SearchConfig,
    best_candidate,
    candidate_values,
    portfolio_metric,
)


class GreedySearch:
    """Greedy forward selection.

    The portfolio starts from the target symbol and, on every step, the symbol that improves the
    metric the most is added. The search stops when no candidate improves the metric by more than
    ``tolerance`` or when ``max_number`` is reached.
    """

    name = "greedy"

    def validate(self, n_symbols: int, config: SearchConfig) -> None:
        """Nothing to check: greedy always runs in ``O(k * n)`` metric evaluations."""

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

        selected = [available.index(target_symbol)]
        best_value = portfolio_metric(returns, weights, selected, metric)
        max_number = config.max_number if config.max_number is not None else len(available)

        while len(selected) < max_number:
            candidates = np.array([index for index in range(len(available)) if index not in selected])
            if candidates.size == 0:
                break
            values = candidate_values(returns, weights, selected, candidates, metric)
            result = best_candidate(values, metric, best_value, config.tolerance)
            if result is None:
                break
            best_value, position = result
            selected.append(int(candidates[position]))
            selected.sort()

        return tuple(symbols[selected].tolist()), best_value
