"""Metric definition."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import numpy as np

MetricFunction = Callable[[np.ndarray], float]
VectorizedMetricFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Metric:
    """A scalar metric together with its optimization direction.

    Attributes:
        name (str): Human readable name, also the key in :data:`bfportf.metrics.METRICS`.
        func (MetricFunction): Callable that accepts a 1d array of returns and returns a float.
            ``np.nan`` means "undefined" and such candidates are never selected.
        maximize (bool): True when a bigger value is better (Sharpe ratio), False for
            drawdown-like metrics. Defaults to True.
        vectorized (VectorizedMetricFunction, optional): Fast path that accepts a 2d array
            (periods x candidates) and returns one metric value per column. Defaults to None.
        description (str): One line description. Defaults to "".
    """

    name: str
    func: MetricFunction
    maximize: bool = True
    vectorized: Optional[VectorizedMetricFunction] = None
    description: str = ""

    def __call__(self, values: np.ndarray) -> float:
        """Evaluate the metric for a 1d array of returns."""
        return float(self.func(np.asarray(values, dtype=float)))

    def evaluate_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Evaluate the metric for every column of a 2d array (periods x candidates).

        Args:
            matrix (np.ndarray): 2d array of per-candidate returns.

        Returns:
            np.ndarray: 1d array of metric values, one per column.
        """
        matrix = np.asarray(matrix, dtype=float)
        if self.vectorized is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.asarray(self.vectorized(matrix), dtype=float)
        return np.apply_along_axis(self.func, 0, matrix).astype(float)
