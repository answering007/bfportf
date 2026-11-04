"""Tests for portfolio aggregation helpers."""

import numpy as np
import pandas as pd
import pytest

from bfportf.exceptions import DataValidationError, UnknownSymbolError
from bfportf.portfolio import (
    calculate_metric,
    get_portfolio_return,
    normalized_weights,
    resolve_weights,
)
from bfportf.scaling import scale_return_data


def test_get_portfolio_return_equal_weights(returns: pd.DataFrame) -> None:
    symbols = ["SYM00", "SYM01"]
    result = get_portfolio_return(returns, symbols)
    expected = returns[symbols].sum(axis=1) / 2
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_get_portfolio_return_normalize_none(returns: pd.DataFrame) -> None:
    symbols = ["SYM00", "SYM01"]
    result = get_portfolio_return(returns, symbols, normalize="none")
    expected = returns[symbols].sum(axis=1)
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_get_portfolio_return_single_symbol_cancels_scaling(returns: pd.DataFrame) -> None:
    scaled, shares = scale_return_data(returns)
    result = get_portfolio_return(scaled, ["SYM00"], shares)
    pd.testing.assert_series_equal(result, returns["SYM00"], check_names=False)


def test_get_portfolio_return_array_weights(returns: pd.DataFrame) -> None:
    weights = np.full(len(returns.columns), 1.0)
    result = get_portfolio_return(returns, ["SYM00"], weights)
    pd.testing.assert_series_equal(result, returns["SYM00"], check_names=False)


def test_get_portfolio_return_weights_series_order_is_irrelevant(returns: pd.DataFrame) -> None:
    ordered = pd.Series(np.arange(1, len(returns.columns) + 1, dtype=float), index=returns.columns)
    reversed_weights = ordered.iloc[::-1]
    expected = get_portfolio_return(returns, ["SYM00", "SYM01"], ordered)
    result = get_portfolio_return(returns, ["SYM00", "SYM01"], reversed_weights)
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_get_portfolio_return_unknown_symbol(returns: pd.DataFrame) -> None:
    with pytest.raises(UnknownSymbolError):
        get_portfolio_return(returns, ["MISSING"])


def test_get_portfolio_return_empty_symbols(returns: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError):
        get_portfolio_return(returns, [])


def test_get_portfolio_return_unknown_normalization(returns: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="normalize"):
        get_portfolio_return(returns, ["SYM00"], normalize="zscore")


def test_get_portfolio_return_zero_total_weight(returns: pd.DataFrame) -> None:
    weights = pd.Series(0.0, index=returns.columns)
    with pytest.raises(DataValidationError, match="total weight"):
        get_portfolio_return(returns, ["SYM00"], weights)


def test_resolve_weights_missing_column(returns: pd.DataFrame) -> None:
    weights = pd.Series(1.0, index=returns.columns[:-1])
    with pytest.raises(UnknownSymbolError):
        resolve_weights(returns, weights)


def test_resolve_weights_wrong_shape(returns: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="shape"):
        resolve_weights(returns, np.ones(3))


def test_resolve_weights_defaults_to_equal_weights(returns: pd.DataFrame) -> None:
    weights = resolve_weights(returns, None)
    assert (weights.to_numpy() == 1.0).all()
    assert weights.index.equals(returns.columns)


def test_normalized_weights_sum_to_one(returns: pd.DataFrame) -> None:
    weights = normalized_weights(returns, ["SYM00", "SYM01"], np.arange(1.0, len(returns.columns) + 1))
    assert weights.sum() == pytest.approx(1.0)
    assert list(weights.index) == ["SYM00", "SYM01"]


def test_normalized_weights_zero_total(returns: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError):
        normalized_weights(returns, ["SYM00"], pd.Series(0.0, index=returns.columns))


def test_calculate_metric_with_array_weights(returns: pd.DataFrame) -> None:
    results = pd.DataFrame({"symbols": [["SYM00"], ["SYM01", "SYM02"]]})
    weights = np.ones(len(returns.columns))

    calculated = calculate_metric(
        results,
        returns,
        weights,
        lambda values: np.float64(values.sum()),
    )

    expected = pd.Series(
        [
            returns["SYM00"].sum(),
            returns[["SYM01", "SYM02"]].sum(axis=1).sum() / 2,
        ],
    )
    pd.testing.assert_series_equal(calculated, expected, check_names=False)
