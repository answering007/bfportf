"""Tests for bfportf.optimization."""

import numpy as np
import pandas as pd
import pytest

from bfportf.exceptions import DataValidationError, UnknownSymbolError
from bfportf.optimization import find_portfolios
from bfportf.targets import profit_factor_target, sharpe_target
from bfportf.utils import get_portfolio_return


def _portfolio_value(returns: pd.DataFrame, symbols: list) -> float:
    """Metric of the equal weight portfolio built from the symbols."""
    return float(sharpe_target(get_portfolio_return(returns, symbols).to_numpy()))


def test_find_portfolios_respects_max_number(small_returns: pd.DataFrame) -> None:
    found = find_portfolios(small_returns, sharpe_target, max_number=1, verbose=False)
    assert all(len(symbols) == 1 for symbols in found["symbols"])


def test_find_portfolios_value_matches_direct_calculation(small_returns: pd.DataFrame) -> None:
    found = find_portfolios(small_returns, sharpe_target, max_number=2, verbose=False)

    for _, row in found.iterrows():
        symbols = list(row["symbols"])
        assert row["target_function_value"] == pytest.approx(_portfolio_value(small_returns, symbols))


def test_find_portfolios_single_step_is_best_single_symbol(small_returns: pd.DataFrame) -> None:
    found = find_portfolios(small_returns, sharpe_target, max_number=1, verbose=False)
    singles = {column: _portfolio_value(small_returns, [column]) for column in small_returns.columns}
    assert found["target_function_value"].iloc[0] == pytest.approx(max(singles.values()))


def test_find_portfolios_is_not_worse_than_single_step(small_returns: pd.DataFrame) -> None:
    singles_best = max(_portfolio_value(small_returns, [column]) for column in small_returns.columns)
    found = find_portfolios(small_returns, sharpe_target, max_number=3, verbose=False)
    assert found["target_function_value"].iloc[0] >= singles_best - 1e-12


def test_find_portfolios_is_deterministic(small_returns: pd.DataFrame) -> None:
    first = find_portfolios(small_returns, sharpe_target, max_number=2, verbose=False)
    second = find_portfolios(small_returns, sharpe_target, max_number=2, verbose=False)
    pd.testing.assert_frame_equal(first, second)


def test_find_portfolios_single_symbol_universe(returns: pd.DataFrame) -> None:
    single = returns.iloc[:, [0]]
    found = find_portfolios(single, sharpe_target, max_number=3, verbose=False)

    assert len(found) == 1
    assert list(found["symbols"].iloc[0]) == [single.columns[0]]


def test_find_portfolios_huge_tolerance_stops_after_first_symbol(returns: pd.DataFrame) -> None:
    found = find_portfolios(returns, sharpe_target, max_number=5, tolerance=1e9, verbose=False)
    assert all(len(symbols) == 1 for symbols in found["symbols"])


def test_find_portfolios_sorted_best_first(small_returns: pd.DataFrame) -> None:
    found = find_portfolios(small_returns, sharpe_target, max_number=2, verbose=False)
    values = found["target_function_value"].to_numpy(dtype=float)
    assert np.all(np.diff(values) <= 1e-12)


def test_find_portfolios_minimize(returns: pd.DataFrame) -> None:
    found = find_portfolios(returns, profit_factor_target, maximize=False, max_number=1, verbose=False)
    values = found["target_function_value"].to_numpy(dtype=float)
    assert np.all(np.diff(values) >= -1e-12)


def test_find_portfolios_skips_non_finite_candidates(returns: pd.DataFrame) -> None:
    biased = returns.copy()
    biased["ONLYUP"] = np.abs(returns["SYM00"])

    found = find_portfolios(biased, profit_factor_target, max_number=1, verbose=False)

    values = found["target_function_value"].to_numpy(dtype=float)
    finite = np.isfinite(values)
    assert finite.any()
    assert finite[: int(finite.sum())].all()
    assert np.isfinite(values[0])


def test_find_portfolios_rejects_nan_when_validating(returns: pd.DataFrame) -> None:
    broken = returns.copy()
    broken.iloc[0, 0] = np.nan
    with pytest.raises(DataValidationError):
        find_portfolios(broken, sharpe_target, verbose=False)


def test_find_portfolios_validation_can_be_disabled(returns: pd.DataFrame) -> None:
    broken = returns.copy()
    broken.iloc[0, 0] = np.nan
    found = find_portfolios(broken, sharpe_target, max_number=1, verbose=False, validate=False)
    assert not found.empty


def test_find_portfolios_unknown_target_symbol(small_returns: pd.DataFrame) -> None:
    with pytest.raises(UnknownSymbolError):
        find_portfolios(small_returns, sharpe_target, target_symbols=["MISSING"], verbose=False)


def test_find_portfolios_target_symbols_subset(small_returns: pd.DataFrame) -> None:
    subset = list(small_returns.columns[:2])
    found = find_portfolios(small_returns, sharpe_target, target_symbols=subset, max_number=1, verbose=False)
    assert len(found) == 2


def test_find_portfolios_accepts_array_weights(small_returns: pd.DataFrame) -> None:
    weights = np.ones(len(small_returns.columns))
    found = find_portfolios(small_returns, sharpe_target, custom_weights=weights, max_number=1, verbose=False)
    assert not found.empty


def test_find_portfolios_rejects_wrong_weights_shape(small_returns: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError):
        find_portfolios(small_returns, sharpe_target, custom_weights=np.ones(3), verbose=False)


def test_find_portfolios_rejects_max_number_below_one(small_returns: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError):
        find_portfolios(small_returns, sharpe_target, max_number=0, verbose=False)
