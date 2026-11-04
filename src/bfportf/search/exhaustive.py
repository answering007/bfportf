"""Exhaustive (full enumeration) search strategy."""

import itertools
from math import comb

import numpy as np

from bfportf.exceptions import DataValidationError, UnknownSymbolError
from bfportf.metrics.base import Metric
from bfportf.search.base import SearchConfig, improves, portfolio_metric


class ExhaustiveSearch:
    """Enumerate every portfolio up to ``max_number`` symbols.

    This is the reference implementation: on small universes it returns the true optimum for the
    given metric and ``max_number``. Because the number of combinations grows as ``C(n, k)``, the
    strategy refuses to run when the budget is exceeded.

    Attributes:
        max_combinations (int): Upper bound of evaluated combinations. Defaults to 2_000_000.
    """

    name = "exhaustive"

    def __init__(self, max_combinations: int = 2_000_000) -> None:
        """Initialize the strategy.

        Args:
            max_combinations (int, optional): Upper bound of evaluated combinations.
                Defaults to 2_000_000.

        Raises:
            DataValidationError: If ``max_combinations`` is lower than 1.
        """
        if max_combinations < 1:
            msg = f"max_combinations must be >= 1, got {max_combinations}"
            raise DataValidationError(msg)
        self.max_combinations = max_combinations

    def validate(self, n_symbols: int, config: SearchConfig) -> None:
        """Raise when the number of combinations exceeds ``max_combinations``.

        Args:
            n_symbols (int): Size of the universe.
            config (SearchConfig): Search configuration.

        Raises:
            DataValidationError: If the exhaustive search would be too expensive.
        """
        max_number = config.max_number if config.max_number is not None else n_symbols
        total = sum(comb(n_symbols, size) for size in range(1, min(max_number, n_symbols) + 1))
        if total > self.max_combinations:
            msg = (
                f"exhaustive search needs {total:,} combinations, which exceeds "
                f"max_combinations={self.max_combinations:,}; reduce the universe or use "
                f"GreedySearch / BeamSearch"
            )
            raise DataValidationError(msg)

    def select(
        self,
        returns: np.ndarray,
        symbols: np.ndarray,
        weights: np.ndarray,
        metric: Metric,
        target_symbol: str,
        config: SearchConfig,
    ) -> tuple:
        """Select the best portfolio that contains the target symbol.

        Args:
            returns (np.ndarray): 2d array of returns (periods x symbols).
            symbols (np.ndarray): 1d array of symbols.
            weights (np.ndarray): 1d array of weights.
            metric (Metric): Metric to optimize.
            target_symbol (str): Target symbol.
            config (SearchConfig): Search configuration.

        Returns:
            tuple: ``(portfolio_symbols, metric value)``.

        Raises:
            UnknownSymbolError: If ``target_symbol`` is missing from ``symbols``.
        """
        available = symbols.tolist()
        if target_symbol not in available:
            raise UnknownSymbolError([target_symbol], available)

        target_index = available.index(target_symbol)
        others = [index for index in range(len(available)) if index != target_index]
        max_number = config.max_number if config.max_number is not None else len(available)

        best_symbols = (target_index,)
        best_value = portfolio_metric(returns, weights, [target_index], metric)

        for size in range(2, min(max_number, len(available)) + 1):
            for combination in itertools.combinations(others, size - 1):
                indices = [target_index, *combination]
                value = portfolio_metric(returns, weights, indices, metric)
                if improves(value, best_value, metric, config.tolerance):
                    best_value = value
                    best_symbols = tuple(sorted(indices))

        return tuple(symbols[list(best_symbols)].tolist()), best_value
