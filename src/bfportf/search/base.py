"""Search primitives: configuration, results, shared helpers and the strategy protocol."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from tqdm import tqdm

from bfportf.data import validate_returns
from bfportf.exceptions import DataValidationError, UnknownSymbolError
from bfportf.metrics.base import Metric
from bfportf.portfolio import normalized_weights


@dataclass(frozen=True)
class SearchConfig:
    """Configuration of a portfolio search.

    Attributes:
        max_number (int, optional): Maximum number of symbols in a portfolio (None means the whole
            universe). Defaults to None.
        tolerance (float): Minimal metric improvement required to accept the next symbol.
            Defaults to 0.0.
        validate (bool): Validate the input returns dataframe. Defaults to True.
        verbose (bool): Show a progress bar. Defaults to True.
    """

    max_number: int | None = None
    tolerance: float = 0.0
    validate: bool = True
    verbose: bool = True

    def __post_init__(self) -> None:
        """Validate the configuration values."""
        if self.max_number is not None and self.max_number < 1:
            msg = f"max_number must be >= 1, got {self.max_number}"
            raise DataValidationError(msg)
        if self.tolerance < 0:
            msg = f"tolerance must be >= 0, got {self.tolerance}"
            raise DataValidationError(msg)


@dataclass(frozen=True)
class PortfolioResult:
    """A portfolio found by a search strategy.

    Attributes:
        symbols (tuple): Portfolio symbols in the order of the returns columns.
        weights (pd.Series): Weights of the portfolio symbols, rescaled to sum up to one.
        value (float): Metric value of the portfolio.
        metric_name (str): Name of the optimized metric.
    """

    symbols: tuple
    weights: pd.Series
    value: float
    metric_name: str

    @property
    def portfolio(self) -> str:
        """Symbols joined by a semicolon (the legacy ``Portfolio`` column)."""
        return ";".join(self.symbols)

    def to_dict(self) -> dict:
        """Plain python representation, convenient for logging or JSON."""
        return {
            "symbols": list(self.symbols),
            "weights": self.weights.to_dict(),
            "value": self.value,
            "metric_name": self.metric_name,
        }

    def to_frame(self) -> pd.DataFrame:
        """Single row dataframe in the legacy layout."""
        return results_to_frame([self])


@runtime_checkable
class SearchStrategy(Protocol):
    """Portfolio search strategy.

    A strategy picks a portfolio for a single target symbol. The shared driver
    (:func:`run_search`) loops over the target symbols, removes duplicates and builds the results.
    """

    name: str

    def validate(self, n_symbols: int, config: SearchConfig) -> None:
        """Check that the strategy can run on the given universe (for example the budget)."""

    def select(
        self,
        returns: np.ndarray,
        symbols: np.ndarray,
        weights: np.ndarray,
        metric: Metric,
        target_symbol: str,
        config: SearchConfig,
    ) -> tuple:
        """Select a portfolio for the target symbol and return ``(symbols, metric value)``."""


def portfolio_metric(returns: np.ndarray, weights: np.ndarray, indices: Sequence[int], metric: Metric) -> float:
    """Evaluate the metric of the portfolio built from the given column indices.

    The portfolio series is ``sum(returns[:, indices]) / sum(weights[indices])``: the weights
    normalize the aggregate and are expected to be applied to the returns by the caller.

    Args:
        returns (np.ndarray): 2d array of returns (periods x symbols).
        weights (np.ndarray): 1d array of weights aligned with the columns.
        indices (Sequence[int]): Column indices of the portfolio.
        metric (Metric): Metric to evaluate.

    Returns:
        float: Metric value, ``nan`` when the total weight is zero.
    """
    index_list = list(indices)
    total_weight = float(weights[index_list].sum())
    if total_weight == 0:
        return float("nan")
    values = returns[:, index_list].sum(axis=1) / total_weight
    return float(metric.func(values))


def candidate_values(
    returns: np.ndarray,
    weights: np.ndarray,
    selected: Sequence[int],
    candidates: np.ndarray,
    metric: Metric,
) -> np.ndarray:
    """Evaluate the metric of the current portfolio extended with every candidate.

    Args:
        returns (np.ndarray): 2d array of returns (periods x symbols).
        weights (np.ndarray): 1d array of weights aligned with the columns.
        selected (Sequence[int]): Column indices of the current portfolio.
        candidates (np.ndarray): Column indices of the candidates.
        metric (Metric): Metric to evaluate.

    Returns:
        np.ndarray: One metric value per candidate.
    """
    initial_returns = returns[:, list(selected)].sum(axis=1)
    initial_weight = float(weights[list(selected)].sum())
    matrix = (returns[:, candidates] + initial_returns[:, None]) / (weights[candidates] + initial_weight)
    return metric.evaluate_matrix(matrix)


def sort_key(value: float, metric: Metric) -> tuple:
    """Sort key that puts better values first and ``nan`` values last."""
    if not np.isfinite(value):
        return (1, 0.0)
    return (0, -value if metric.maximize else value)


def improves(value: float, current: float, metric: Metric, tolerance: float = 0.0) -> bool:
    """Return True when ``value`` is better than ``current`` by more than ``tolerance``."""
    if not np.isfinite(value):
        return False
    if not np.isfinite(current):
        return True
    delta = value - current if metric.maximize else current - value
    return bool(np.isfinite(delta) and delta > tolerance)


def best_candidate(values: np.ndarray, metric: Metric, current: float, tolerance: float = 0.0) -> tuple | None:
    """Return ``(value, position)`` of the best candidate that improves ``current``.

    Candidates with non finite values are ignored, so degenerate portfolios are never selected.
    """
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return None
    if metric.maximize:
        position = int(np.argmax(np.where(finite, values, -np.inf)))
    else:
        position = int(np.argmin(np.where(finite, values, np.inf)))
    value = float(values[position])
    if not improves(value, current, metric, tolerance):
        return None
    return value, position


def results_to_frame(results: Sequence[PortfolioResult]) -> pd.DataFrame:
    """Convert search results into the legacy dataframe layout.

    Args:
        results (Sequence[PortfolioResult]): Results returned by :func:`run_search`.

    Returns:
        pd.DataFrame: Frame with ``symbols``, ``target_function_value`` and ``Portfolio`` columns.
    """
    return pd.DataFrame(
        {
            "symbols": [np.array(result.symbols) for result in results],
            "target_function_value": [result.value for result in results],
            "Portfolio": [result.portfolio for result in results],
        },
        columns=["symbols", "target_function_value", "Portfolio"],
    )


def run_search(
    strategy: SearchStrategy,
    returns: pd.DataFrame,
    weights: np.ndarray,
    metric: Metric,
    config: SearchConfig,
    target_symbols: Sequence[str] | None = None,
) -> list:
    """Run the strategy for every target symbol and build the ordered results.

    Args:
        strategy (SearchStrategy): Search strategy.
        returns (pd.DataFrame): Returns dataframe.
        weights (np.ndarray): 1d array of weights aligned with the returns columns.
        metric (Metric): Metric to optimize.
        config (SearchConfig): Search configuration.
        target_symbols (Sequence[str], optional): Symbols to use as targets. Defaults to None
            (all columns).

    Returns:
        list: ``list[PortfolioResult]`` ordered from the best to the worst.

    Raises:
        UnknownSymbolError: If a target symbol is missing from the returns columns.
        DataValidationError: If the returns dataframe does not pass validation.
    """
    if config.validate:
        validate_returns(returns)

    np_returns = returns.to_numpy(dtype=float)
    np_symbols = np.array(returns.columns)
    available = np_symbols.tolist()

    targets = available if target_symbols is None else list(target_symbols)
    missing = [symbol for symbol in targets if symbol not in available]
    if missing:
        raise UnknownSymbolError(missing, available)

    strategy.validate(len(available), config)

    iterator = tqdm(targets) if config.verbose else targets
    best_values: dict = {}
    for target_symbol in iterator:
        selected, value = strategy.select(np_returns, np_symbols, weights, metric, target_symbol, config)
        found_symbols = tuple(selected)
        previous = best_values.get(found_symbols)
        if previous is None or improves(value, previous, metric):
            best_values[found_symbols] = float(value)

    ordered = sorted(best_values.items(), key=lambda item: sort_key(item[1], metric))

    results = []
    for found_symbols, value in ordered:
        results.append(
            PortfolioResult(
                symbols=found_symbols,
                weights=normalized_weights(returns, list(found_symbols), weights),
                value=value,
                metric_name=metric.name,
            ),
        )
    return results
