"""Shared pytest helpers and fixtures."""

import numpy as np
import pandas as pd
import pytest


def make_returns(n_symbols: int = 8, n_rows: int = 250, seed: int = 7, scale: float = 0.01) -> pd.DataFrame:
    """Build deterministic synthetic daily returns.

    Args:
        n_symbols (int, optional): Number of symbols. Defaults to 8.
        n_rows (int, optional): Number of periods. Defaults to 250.
        seed (int, optional): Random seed. Defaults to 7.
        scale (float, optional): Daily return standard deviation. Defaults to 0.01.

    Returns:
        pd.DataFrame: Synthetic returns with a business day index.
    """
    rng = np.random.default_rng(seed)
    values = rng.normal(loc=0.0004, scale=scale, size=(n_rows, n_symbols))
    index = pd.date_range("2021-01-04", periods=n_rows, freq="B")
    columns = [f"SYM{i:02d}" for i in range(n_symbols)]
    return pd.DataFrame(values, index=index, columns=columns)


@pytest.fixture()
def returns() -> pd.DataFrame:
    """Synthetic returns (8 symbols, 250 periods)."""
    return make_returns()


@pytest.fixture()
def small_returns() -> pd.DataFrame:
    """Synthetic returns (5 symbols, 60 periods)."""
    return make_returns(n_symbols=5, n_rows=60, seed=11)
