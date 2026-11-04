"""Tests for the search facade and the search strategies."""

import numpy as np
import pandas as pd
import pytest

from bfportf.exceptions import DataValidationError, UnknownSymbolError
from bfportf.metrics import METRICS, Metric
from bfportf.optimization import find_portfolios
from bfportf.portfolio import get_portfolio_return
from bfportf.search import PortfolioResult, results_to_frame


def _value(returns: pd.DataFrame, symbols, metric_name: str = "sharpe") -> float:
    series = get_portfolio_return(returns, list(symbols))
    return float(METRICS[metric_name].func(series.to_numpy()))


def test_find_portfolios_returns_sorted_results(small_returns: pd.DataFrame) -> None:
    found = find_portfolios(small_returns, "sharpe", max_number=2, verbose=False)

    assert found
    assert all(isinstance(item, PortfolioResult) for item in found)
    values = [item.value for item in found]
    assert values == sorted(values, reverse=True)


def test_find_portfolios_values_and_weights(small_returns: pd.DataFrame) -> None:
    found = find_portfolios(small_returns, "sharpe", max_number=2, verbose=False)

    for item in found:
        assert item.metric_name == "sharpe"
        assert item.weights.sum() == pytest.approx(1.0)
        assert list(item.weights.index) == list(item.symbols)
        assert item.value == pytest.approx(_value(small_returns, item.symbols))


def test_find_portfolios_respects_max_number(small_returns: pd.DataFrame) -> None:
    found = find_portfolios(small_returns, "sharpe", max_number=1, verbose=False)
    assert all(len(item.symbols) == 1 for item in found)


def test_find_portfolios_accepts_metric_objects_and_callables(small_returns: pd.DataFrame) -> None:
    by_name = find_portfolios(small_returns, "sharpe", max_number=1, verbose=False)
    by_metric = find_portfolios(small_returns, METRICS["sharpe"], max_number=1, verbose=False)
    by_callable = find_portfolios(small_returns, METRICS["sharpe"].func, max_number=1, verbose=False)

    assert [item.symbols for item in by_name] == [item.symbols for item in by_metric]
    assert [item.symbols for item in by_name] == [item.symbols for item in by_callable]


def test_find_portfolios_direction_comes_from_metric(returns: pd.DataFrame) -> None:
    # max_drawdown is negative, so its best value is the largest one
    found = find_portfolios(returns, "max_drawdown", max_number=1, verbose=False)
    values = [item.value for item in found]
    assert found[0].value == pytest.approx(max(values))

    flipped = find_portfolios(returns, "max_drawdown", maximize=False, max_number=1, verbose=False)
    flipped_values = [item.value for item in flipped]
    assert flipped[0].value == pytest.approx(min(flipped_values))


def test_find_portfolios_unknown_metric(small_returns: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="unknown metric"):
        find_portfolios(small_returns, "no_such_metric", verbose=False)


def test_find_portfolios_single_step_is_best_single_symbol(small_returns: pd.DataFrame) -> None:
    found = find_portfolios(small_returns, "sharpe", max_number=1, verbose=False)
    singles = {column: _value(small_returns, [column]) for column in small_returns.columns}
    assert found[0].value == pytest.approx(max(singles.values()))


def test_find_portfolios_is_deterministic(small_returns: pd.DataFrame) -> None:
    first = results_to_frame(find_portfolios(small_returns, "sharpe", max_number=2, verbose=False))
    second = results_to_frame(find_portfolios(small_returns, "sharpe", max_number=2, verbose=False))
    pd.testing.assert_frame_equal(first, second)


def test_find_portfolios_huge_tolerance_stops_after_first_symbol(returns: pd.DataFrame) -> None:
    found = find_portfolios(returns, "sharpe", max_number=5, tolerance=1e9, verbose=False)
    assert all(len(item.symbols) == 1 for item in found)


def test_find_portfolios_rejects_nan_when_validating(returns: pd.DataFrame) -> None:
    broken = returns.copy()
    broken.iloc[0, 0] = np.nan
    with pytest.raises(DataValidationError):
        find_portfolios(broken, "sharpe", verbose=False)


def test_find_portfolios_validation_can_be_disabled(returns: pd.DataFrame) -> None:
    broken = returns.copy()
    broken.iloc[0, 0] = np.nan
    found = find_portfolios(broken, "sharpe", max_number=1, verbose=False, validate=False)
    assert found


def test_find_portfolios_unknown_target_symbol(small_returns: pd.DataFrame) -> None:
    with pytest.raises(UnknownSymbolError):
        find_portfolios(small_returns, "sharpe", target_symbols=["MISSING"], verbose=False)


def test_find_portfolios_weights_validation(small_returns: pd.DataFrame) -> None:
    accepted = find_portfolios(
        small_returns,
        "sharpe",
        custom_weights=np.ones(len(small_returns.columns)),
        max_number=1,
        verbose=False,
    )
    assert accepted

    with pytest.raises(DataValidationError):
        find_portfolios(small_returns, "sharpe", custom_weights=np.ones(3), verbose=False)


def test_find_portfolios_rejects_max_number_below_one(small_returns: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError):
        find_portfolios(small_returns, "sharpe", max_number=0, verbose=False)


def test_custom_metric_minimize_direction(small_returns: pd.DataFrame) -> None:
    def spread(values: np.ndarray) -> np.float64:
        return np.float64(values.max() - values.min())

    metric = Metric(name="spread", func=spread, maximize=False)
    found = find_portfolios(small_returns, metric, max_number=1, verbose=False)

    values = [item.value for item in found]
    assert found[0].value == pytest.approx(min(values))
    assert found[0].metric_name == "spread"

    found = find_portfolios(small_returns, "sharpe", max_number=2, verbose=False)
    frame = results_to_frame(found)

    assert list(frame.columns) == ["symbols", "target_function_value", "Portfolio"]
    assert len(frame) == len(found)
    assert list(frame["symbols"].iloc[0]) == list(found[0].symbols)
    assert frame["Portfolio"].iloc[0] == ";".join(found[0].symbols)
