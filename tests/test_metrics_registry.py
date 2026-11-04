"""Tests for the metric registry and the vectorized fast path."""

import numpy as np
import pandas as pd
import pytest

from bfportf import METRICS
from bfportf.exceptions import DataValidationError
from bfportf.metrics import DEFAULT_METRIC, resolve_metric

EXPECTED_METRICS = {
    "autocorrelation",
    "cagr",
    "kurtosis",
    "max_drawdown",
    "pnl",
    "profit_factor",
    "recovery_factor",
    "sharpe",
    "skew",
    "sortino",
}

VECTORIZED_METRICS = [name for name, metric in METRICS.items() if metric.vectorized is not None]


def test_registry_contains_all_metrics() -> None:
    assert EXPECTED_METRICS == set(METRICS)
    assert DEFAULT_METRIC in METRICS


def test_all_metrics_are_maximized() -> None:
    # max_drawdown is reported as a negative percentage, so its best value is the largest one
    minimized = {name for name, metric in METRICS.items() if not metric.maximize}
    assert minimized == set()


def test_metric_names_match_registry_keys() -> None:
    for name, metric in METRICS.items():
        assert metric.name == name
        assert callable(metric.func)
        assert metric.description


def test_resolve_metric_by_name_returns_registry_entry() -> None:
    assert resolve_metric("sharpe") is METRICS["sharpe"]


def test_resolve_metric_unknown_name() -> None:
    with pytest.raises(DataValidationError, match="unknown metric"):
        resolve_metric("nope")


def test_resolve_metric_unsupported_type() -> None:
    with pytest.raises(DataValidationError, match="metric must be"):
        resolve_metric(42)


def test_resolve_metric_callable_defaults_to_maximize() -> None:
    def custom(values: np.ndarray) -> float:
        return float(np.mean(values))

    metric = resolve_metric(custom)

    assert metric.name == "custom"
    assert metric.maximize is True
    assert metric.vectorized is None


def test_resolve_metric_maximize_override_does_not_mutate_registry() -> None:
    overridden = resolve_metric("sharpe", maximize=False)

    assert overridden.maximize is False
    assert METRICS["sharpe"].maximize is True


@pytest.mark.parametrize("name", VECTORIZED_METRICS)
def test_vectorized_matches_scalar(name: str, returns: pd.DataFrame) -> None:
    metric = METRICS[name]
    matrix = returns.to_numpy(dtype=float)
    expected = np.array([metric.func(column) for column in matrix.T])

    np.testing.assert_allclose(metric.evaluate_matrix(matrix), expected, rtol=1e-10, equal_nan=True)


def test_evaluate_matrix_falls_back_for_scalar_only_metric(returns: pd.DataFrame) -> None:
    metric = METRICS["autocorrelation"]
    matrix = returns.to_numpy(dtype=float)
    expected = np.array([metric.func(column) for column in matrix.T])

    assert metric.vectorized is None
    np.testing.assert_allclose(metric.evaluate_matrix(matrix), expected, equal_nan=True)


def test_metric_call_returns_float(returns: pd.DataFrame) -> None:
    assert isinstance(METRICS["sharpe"](returns["SYM00"].to_numpy()), float)


def test_vectorized_and_scalar_agree_on_constant_column(returns: pd.DataFrame) -> None:
    matrix = returns.to_numpy(dtype=float).copy()
    matrix[:, 0] = 0.01

    for name in VECTORIZED_METRICS:
        metric = METRICS[name]
        expected = np.array([metric.func(column) for column in matrix.T])
        np.testing.assert_allclose(metric.evaluate_matrix(matrix), expected, rtol=1e-10, equal_nan=True)

    # dispersion based metrics are undefined for a constant series
    assert np.isnan(METRICS["sharpe"].evaluate_matrix(matrix)[0])
    assert np.isnan(METRICS["sortino"].evaluate_matrix(matrix)[0])
    assert np.isnan(METRICS["skew"].evaluate_matrix(matrix)[0])
