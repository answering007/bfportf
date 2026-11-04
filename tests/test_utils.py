"""Tests for the price/return conversions."""

import numpy as np
import pytest

from bfportf.utils import drawdown_pct, price_to_return, return_to_price


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
