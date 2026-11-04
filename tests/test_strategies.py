"""Tests for the search strategies (greedy, beam, exhaustive)."""

import itertools

import pandas as pd
import pytest

from bfportf.exceptions import DataValidationError
from bfportf.metrics import METRICS
from bfportf.optimization import find_portfolios
from bfportf.portfolio import get_portfolio_return
from bfportf.search import BeamSearch, ExhaustiveSearch, GreedySearch, results_to_frame


def _value(returns: pd.DataFrame, symbols) -> float:
    series = get_portfolio_return(returns, list(symbols))
    return float(METRICS["sharpe"].func(series.to_numpy()))


def test_greedy_strategy_is_the_default(small_returns: pd.DataFrame) -> None:
    default = find_portfolios(small_returns, "sharpe", max_number=2, verbose=False)
    explicit = find_portfolios(small_returns, "sharpe", max_number=2, verbose=False, strategy=GreedySearch())

    assert [item.symbols for item in default] == [item.symbols for item in explicit]


def test_beam_width_one_matches_greedy(small_returns: pd.DataFrame) -> None:
    greedy = results_to_frame(find_portfolios(small_returns, "sharpe", max_number=2, verbose=False))
    beam = results_to_frame(
        find_portfolios(
            small_returns,
            "sharpe",
            max_number=2,
            verbose=False,
            strategy=BeamSearch(width=1),
        ),
    )

    pd.testing.assert_frame_equal(greedy, beam)


def test_beam_is_not_worse_than_greedy(small_returns: pd.DataFrame) -> None:
    greedy = find_portfolios(small_returns, "sharpe", max_number=3, verbose=False, strategy=GreedySearch())
    beam = find_portfolios(small_returns, "sharpe", max_number=3, verbose=False, strategy=BeamSearch(width=4))

    assert beam[0].value >= greedy[0].value - 1e-12


def test_beam_width_validation() -> None:
    with pytest.raises(DataValidationError):
        BeamSearch(width=0)


def test_beam_respects_max_number(small_returns: pd.DataFrame) -> None:
    found = find_portfolios(
        small_returns,
        "sharpe",
        max_number=2,
        verbose=False,
        strategy=BeamSearch(width=3),
    )

    assert all(len(item.symbols) <= 2 for item in found)


def test_exhaustive_matches_brute_force(small_returns: pd.DataFrame) -> None:
    for max_number in (1, 2, 3):
        found = find_portfolios(
            small_returns,
            "sharpe",
            max_number=max_number,
            verbose=False,
            strategy=ExhaustiveSearch(),
        )
        best = max(
            _value(small_returns, combination)
            for size in range(1, max_number + 1)
            for combination in itertools.combinations(small_returns.columns, size)
        )

        assert found[0].value == pytest.approx(best)


def test_exhaustive_not_worse_than_greedy(small_returns: pd.DataFrame) -> None:
    greedy = find_portfolios(small_returns, "sharpe", max_number=2, verbose=False, strategy=GreedySearch())
    exhaustive = find_portfolios(small_returns, "sharpe", max_number=2, verbose=False, strategy=ExhaustiveSearch())

    assert exhaustive[0].value >= greedy[0].value - 1e-12


def test_exhaustive_budget_is_enforced(returns: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="max_combinations"):
        find_portfolios(
            returns,
            "sharpe",
            max_number=2,
            verbose=False,
            strategy=ExhaustiveSearch(max_combinations=10),
        )


def test_exhaustive_budget_validation() -> None:
    with pytest.raises(DataValidationError):
        ExhaustiveSearch(max_combinations=0)


def test_strategies_agree_on_single_symbol_universe(returns: pd.DataFrame) -> None:
    single = returns.iloc[:, [0]]

    greedy = find_portfolios(single, "sharpe", max_number=3, verbose=False, strategy=GreedySearch())
    beam = find_portfolios(single, "sharpe", max_number=3, verbose=False, strategy=BeamSearch(width=3))
    exhaustive = find_portfolios(single, "sharpe", max_number=3, verbose=False, strategy=ExhaustiveSearch())

    assert [item.symbols for item in greedy] == [item.symbols for item in beam]
    assert [item.symbols for item in greedy] == [item.symbols for item in exhaustive]
