"""End to end smoke test of the public API."""

import numpy as np
import pandas as pd
import pytest

import bfportf as bf


def make_prices(n_symbols: int = 12, n_rows: int = 250, seed: int = 42) -> pd.DataFrame:
    """Build deterministic synthetic prices."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0005, scale=0.01, size=(n_rows, n_symbols))
    index = pd.date_range("2020-01-02", periods=n_rows, freq="B")
    columns = [f"SYM{i:02d}" for i in range(n_symbols)]
    return pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=index, columns=columns)


def test_full_workflow() -> None:
    returns = bf.load_returns(make_prices())
    bf.validate_returns(returns)

    scaled, weights = bf.scale_return_data(returns)
    found = bf.find_portfolios(scaled, metric="sharpe", custom_weights=weights, max_number=3, verbose=False)

    assert found
    best = found[0]
    assert best.weights.sum() == pytest.approx(1.0)

    report = bf.create_report(
        returns=scaled,
        portfolio_symbols=best.symbols,
        return_weights=weights,
        benchmark_symbols=[returns.columns[0]],
        custom_formatting={"PnL": "{:.2%}"},
    )

    assert report.loc["PnL", "Portfolio"].endswith("%")
    assert "Diff" in report.columns


def test_frame_view_matches_results() -> None:
    returns = bf.load_returns(make_prices(n_symbols=6, n_rows=120))
    found = bf.find_portfolios(returns, metric="sharpe", max_number=2, verbose=False)
    frame = bf.results_to_frame(found)

    assert frame["Portfolio"].iloc[0] == ";".join(found[0].symbols)
    assert frame["target_function_value"].iloc[0] == pytest.approx(found[0].value)
