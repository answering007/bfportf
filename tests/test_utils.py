"""Tests for bfportf.utils."""

import numpy as np
import pandas as pd
import pytest

from bfportf.exceptions import DataValidationError, UnknownSymbolError
from bfportf.utils import (
    add_inverted_return,
    drawdown_pct,
    price_to_return,
    return_to_price,
    scale_return_data,
    validate_returns,
)


def test_return_to_price_roundtrip() -> None:
    returns = np.array([0.01, -0.02, 0.03, 0.005])
    prices = return_to_price(returns)

    assert prices.shape == (returns.size + 1,)
    assert prices[0] == pytest.approx(1.0)
    np.testing.assert_allclose(price_to_return(prices), returns)


def test_return_to_price_first_value() -> None:
    prices = return_to_price(np.array([0.1, 0.1]), first_value=np.array(10.0))
    np.testing.assert_allclose(prices, [10.0, 11.0, 12.1])


def test_drawdown_pct() -> None:
    prices = np.array([1.0, 1.1, 0.99, 1.21])
    np.testing.assert_allclose(drawdown_pct(prices), [0.0, 0.0, 0.99 / 1.1 - 1, 0.0])


def test_scale_return_data_shares(returns: pd.DataFrame) -> None:
    scaled, shares = scale_return_data(returns)

    assert shares.sum() == pytest.approx(1.0)
    assert shares.index.equals(returns.columns)
    mean_abs = returns.abs().mean()
    assert shares[mean_abs.sort_values().index].is_monotonic_decreasing
    np.testing.assert_allclose(scaled.sum(axis=1), returns.mul(shares, axis=1).sum(axis=1))


def test_scale_return_data_target_return(returns: pd.DataFrame) -> None:
    _, shares = scale_return_data(returns, target_return=0.02)
    assert shares.sum() == pytest.approx(1.0)


def test_scale_return_data_unknown_target_symbol(returns: pd.DataFrame) -> None:
    with pytest.raises(UnknownSymbolError):
        scale_return_data(returns, target_symbol="MISSING")


def test_scale_return_data_zero_mean_column(returns: pd.DataFrame) -> None:
    broken = returns.copy()
    broken["FLAT"] = 0.0
    with pytest.raises(DataValidationError):
        scale_return_data(broken)


def test_validate_returns_accepts_valid_data(returns: pd.DataFrame) -> None:
    validate_returns(returns)


def test_validate_returns_rejects_nan(returns: pd.DataFrame) -> None:
    broken = returns.copy()
    broken.iloc[0, 0] = np.nan
    with pytest.raises(DataValidationError, match="NaN values"):
        validate_returns(broken)


def test_validate_returns_can_allow_nan(returns: pd.DataFrame) -> None:
    broken = returns.copy()
    broken.iloc[0, 0] = np.nan
    validate_returns(broken, allow_nan=True)


def test_validate_returns_rejects_inf(returns: pd.DataFrame) -> None:
    broken = returns.copy()
    broken.iloc[0, 0] = np.inf
    with pytest.raises(DataValidationError, match="infinite values"):
        validate_returns(broken)


def test_validate_returns_rejects_duplicated_columns(returns: pd.DataFrame) -> None:
    duplicated = pd.concat([returns, returns.iloc[:, [0]]], axis=1)
    with pytest.raises(DataValidationError, match="duplicated columns"):
        validate_returns(duplicated)


def test_validate_returns_rejects_unsorted_index(returns: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="monotonically"):
        validate_returns(returns.iloc[::-1])


def test_validate_returns_rejects_empty_frame() -> None:
    empty = pd.DataFrame(index=pd.DatetimeIndex([]), columns=["A"], dtype=float)
    with pytest.raises(DataValidationError):
        validate_returns(empty)


def test_validate_returns_warns_about_constant_columns(returns: pd.DataFrame, caplog: pytest.LogCaptureFixture) -> None:
    broken = returns.copy()
    broken["FLAT"] = 0.0
    with caplog.at_level("WARNING"):
        validate_returns(broken)
    assert "FLAT" in caplog.text


def test_add_inverted_return(returns: pd.DataFrame) -> None:
    extended, names = add_inverted_return(returns, ["SYM00"])

    assert names == ["SYM00-1"]
    np.testing.assert_allclose(extended["SYM00-1"].to_numpy(), -returns["SYM00"].to_numpy())


def test_add_inverted_return_unknown_symbol(returns: pd.DataFrame) -> None:
    with pytest.raises(UnknownSymbolError):
        add_inverted_return(returns, ["MISSING"])


def test_add_inverted_return_duplicated_name(returns: pd.DataFrame) -> None:
    broken = returns.copy()
    broken["SYM00-1"] = 0.0
    with pytest.raises(DataValidationError, match="already exist"):
        add_inverted_return(broken, ["SYM00"])
