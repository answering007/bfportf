"""Tests for bfportf.targets."""

import numpy as np
import pytest
from scipy.stats import skew as scipy_skew

from bfportf.targets import (
    autocorrelation_target,
    cagr_target,
    kurtosis_target,
    max_drawdown_target,
    pnl_target,
    profit_factor_target,
    recovery_factor_target,
    sharpe_target,
    skew_target,
    sortino_target,
)


def test_pnl_target() -> None:
    assert pnl_target(np.array([0.1, 0.1])) == pytest.approx(0.21)


def test_pnl_target_empty_is_nan() -> None:
    assert np.isnan(pnl_target(np.array([])))


def test_max_drawdown_target() -> None:
    # prices: 1.0 -> 1.1 -> 0.99
    assert max_drawdown_target(np.array([0.1, -0.1])) == pytest.approx(0.99 / 1.1 - 1)


def test_max_drawdown_target_monotonic() -> None:
    assert max_drawdown_target(np.array([0.01, 0.02, 0.03])) == pytest.approx(0.0)


def test_profit_factor_target() -> None:
    assert profit_factor_target(np.array([0.02, -0.01, 0.03, -0.02])) == pytest.approx(0.05 / 0.03)


def test_profit_factor_without_losses_is_nan() -> None:
    assert np.isnan(profit_factor_target(np.array([0.01, 0.02])))


def test_profit_factor_without_gains_is_nan() -> None:
    assert np.isnan(profit_factor_target(np.array([-0.01, -0.02])))


def test_recovery_factor_target() -> None:
    returns = np.array([0.1, -0.1])
    expected = pnl_target(returns) / abs(max_drawdown_target(returns))
    assert recovery_factor_target(returns) == pytest.approx(expected)


def test_recovery_factor_without_drawdown_is_nan() -> None:
    assert np.isnan(recovery_factor_target(np.array([0.01, 0.02])))


def test_sharpe_matches_manual_calculation() -> None:
    returns = np.array([0.01, -0.02, 0.03, 0.005, -0.01])
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert sharpe_target(returns) == pytest.approx(expected)


def test_sharpe_without_annualization() -> None:
    returns = np.array([0.01, -0.02, 0.03])
    expected = returns.mean() / returns.std(ddof=1)
    assert sharpe_target(returns, periods=None) == pytest.approx(expected)


def test_sharpe_risk_free_reduces_value() -> None:
    returns = np.array([0.01, -0.02, 0.03, 0.005])
    assert sharpe_target(returns, risk_free=0.05) < sharpe_target(returns)


def test_sharpe_constant_is_nan() -> None:
    assert np.isnan(sharpe_target(np.full(10, 0.01)))


def test_sharpe_too_short_is_nan() -> None:
    assert np.isnan(sharpe_target(np.array([0.01])))


def test_sortino_matches_manual_calculation() -> None:
    returns = np.array([0.01, -0.02, 0.03, 0.005, -0.01])
    downside = returns[returns < 0]
    deviation = np.sqrt(np.square(downside).sum() / (returns.size - 1))
    expected = returns.mean() / deviation * np.sqrt(252)
    assert sortino_target(returns) == pytest.approx(expected)


def test_sortino_without_negative_is_nan() -> None:
    assert np.isnan(sortino_target(np.array([0.01, 0.02])))


def test_skew_and_kurtosis_constant_is_nan() -> None:
    constant = np.full(20, 0.01)
    assert np.isnan(skew_target(constant))
    assert np.isnan(kurtosis_target(constant))


def test_skew_matches_scipy() -> None:
    returns = np.array([0.01, -0.02, 0.03, 0.005, -0.01, 0.02])
    assert skew_target(returns) == pytest.approx(scipy_skew(returns))


def test_autocorrelation_ar1() -> None:
    rng = np.random.default_rng(3)
    phi = 0.6
    values = np.zeros(500)
    shocks = rng.normal(scale=0.01, size=values.size)
    for index in range(1, values.size):
        values[index] = phi * values[index - 1] + shocks[index]
    assert autocorrelation_target(values, absolute=False) == pytest.approx(phi, abs=0.05)


def test_autocorrelation_absolute_is_non_negative() -> None:
    values = np.array([0.01, -0.02, 0.03, 0.005, -0.01, 0.02])
    assert autocorrelation_target(values) >= 0


def test_autocorrelation_constant_is_nan() -> None:
    assert np.isnan(autocorrelation_target(np.full(10, 0.01)))


def test_autocorrelation_periods_validation() -> None:
    with pytest.raises(ValueError):
        autocorrelation_target(np.array([0.01, 0.02]), periods=0)


def test_cagr_matches_compound_growth() -> None:
    rate = 0.01
    assert cagr_target(np.full(252, rate)) == pytest.approx((1 + rate) ** 252 - 1)


def test_cagr_empty_is_nan() -> None:
    assert np.isnan(cagr_target(np.array([])))
