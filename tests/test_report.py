"""Tests for bfportf.report."""

import numpy as np
import pandas as pd
import pytest

from bfportf.exceptions import UnknownSymbolError
from bfportf.metrics.basic import pnl_target, sharpe_target
from bfportf.report import create_report
from bfportf.scaling import scale_return_data


def test_report_values_match_direct_calculation(returns: pd.DataFrame) -> None:
    symbols = ["SYM00", "SYM01"]
    report = create_report(returns, symbols)
    # Direct formula: equal weights, so the aggregate is sum(returns) / len(symbols)
    portfolio = returns[symbols].sum(axis=1).to_numpy() / len(symbols)

    assert report.loc["PnL", "Portfolio"] == pytest.approx(pnl_target(portfolio))
    assert report.loc["Sharpe ratio", "Portfolio"] == pytest.approx(sharpe_target(portfolio))


def test_report_benchmark_and_diff(returns: pd.DataFrame) -> None:
    report = create_report(returns, ["SYM00", "SYM01"], benchmark_symbols=["SYM02"])
    portfolio = returns[["SYM00", "SYM01"]].sum(axis=1).to_numpy() / 2
    benchmark = returns["SYM02"].to_numpy()

    assert {"Portfolio", "Benchmark", "Diff"}.issubset(report.columns)
    assert report.loc["PnL", "Benchmark"] == pytest.approx(pnl_target(benchmark))
    assert report.loc["PnL", "Diff"] == pytest.approx(pnl_target(portfolio) - pnl_target(benchmark))


def test_report_without_benchmark_has_no_diff(returns: pd.DataFrame) -> None:
    report = create_report(returns, ["SYM00"])
    assert list(report.columns) == ["Portfolio"]
    assert "Diff" not in report.columns


def test_report_custom_benchmark(returns: pd.DataFrame) -> None:
    custom = returns["SYM05"]
    report = create_report(returns, ["SYM00"], custom_benchmark=custom)
    assert report.loc["PnL", "Benchmark"] == pytest.approx(pnl_target(custom.to_numpy()))


def test_report_benchmark_weights_are_used(returns: pd.DataFrame) -> None:
    weights = pd.Series(np.linspace(0.5, 1.5, num=len(returns.columns)), index=returns.columns)
    report = create_report(
        returns,
        ["SYM00"],
        benchmark_symbols=["SYM01"],
        benchmark_weights=weights,
    )

    # Direct formula: the weights only normalize the aggregate, they are not applied to the returns
    expected = returns["SYM01"].to_numpy() / weights["SYM01"]

    assert report.loc["Sharpe ratio", "Benchmark"] == pytest.approx(sharpe_target(expected))


def test_report_custom_metric_receives_numpy_array(returns: pd.DataFrame) -> None:
    received: dict[str, object] = {}

    def metric(values: np.ndarray) -> np.float64:
        received["type"] = type(values)
        received["size"] = values.size
        return np.float64(1.5)

    report = create_report(returns, ["SYM00"], custom_portfolio_metrics={"Custom": metric})

    assert received["type"] is np.ndarray
    assert received["size"] == len(returns)
    assert report.loc["Custom", "Portfolio"] == pytest.approx(1.5)


def test_report_custom_formatting_returns_strings(returns: pd.DataFrame) -> None:
    report = create_report(returns, ["SYM00"], custom_formatting={"PnL": "{:.2%}"})
    raw = pnl_target(returns["SYM00"].to_numpy())

    formatted = report.loc["PnL", "Portfolio"]
    assert isinstance(formatted, str)
    assert formatted == f"{raw:.2%}"


def test_report_custom_metric_and_formatting_apply_to_benchmark_and_diff(returns: pd.DataFrame) -> None:
    def length(values: np.ndarray) -> np.float64:
        return np.float64(values.size)

    report = create_report(
        returns,
        ["SYM00"],
        benchmark_symbols=["SYM01"],
        custom_portfolio_metrics={"Length": length},
        custom_formatting={"Length": "{:.1f}"},
    )

    # The custom metric sees the benchmark returns too and Diff is computed before formatting
    expected_length = f"{len(returns):.1f}"
    assert list(report.loc["Length"]) == [expected_length, expected_length, "0.0"]


def test_report_benchmark_symbols_take_precedence_over_custom_benchmark(returns: pd.DataFrame) -> None:
    report = create_report(
        returns,
        ["SYM00"],
        benchmark_symbols=["SYM02"],
        custom_benchmark=returns["SYM05"],
    )

    assert report.loc["PnL", "Benchmark"] == pytest.approx(pnl_target(returns["SYM02"].to_numpy()))


def test_report_unknown_formatting_metric_is_ignored(returns: pd.DataFrame) -> None:
    report = create_report(returns, ["SYM00"], custom_formatting={"MISSING": "{:.2%}"})
    assert "MISSING" not in report.index


def test_report_unknown_symbol_raises(returns: pd.DataFrame) -> None:
    with pytest.raises(UnknownSymbolError):
        create_report(returns, ["MISSING"])


def test_report_with_scaled_weights(returns: pd.DataFrame) -> None:
    scaled, shares = scale_return_data(returns)
    report = create_report(scaled, ["SYM00", "SYM01"], shares, benchmark_symbols=["SYM02"])

    symbols = ["SYM00", "SYM01"]
    expected = (returns[symbols] * shares[symbols]).sum(axis=1) / shares[symbols].sum()
    assert report.loc["PnL", "Portfolio"] == pytest.approx(pnl_target(expected.to_numpy()))
