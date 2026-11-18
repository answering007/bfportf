"""Data loading, sanitizing and validation."""

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from bfportf.exceptions import DataValidationError, UnknownSymbolError

logger = logging.getLogger(__name__)


def load_returns(prices: pd.DataFrame, *, drop_nan: bool = True) -> pd.DataFrame:
    """Convert a price dataframe into a dataframe of simple returns.

    Args:
        prices (pd.DataFrame): Prices, one column per symbol, indexed by date.
        drop_nan (bool, optional): Drop periods with any missing return. Defaults to True.

    Returns:
        pd.DataFrame: Returns dataframe.

    Raises:
        DataValidationError: If the prices dataframe is empty.
    """
    if prices.empty:
        msg = "prices dataframe is empty"
        raise DataValidationError(msg)
    returns = prices.pct_change()
    if drop_nan:
        returns = returns.dropna(how="any")
    return returns


def sanitize_returns(returns: pd.DataFrame, *, drop_nan_rows: bool = True) -> pd.DataFrame:
    """Replace infinite values with ``NaN`` and optionally drop incomplete periods.

    Args:
        returns (pd.DataFrame): Returns dataframe.
        drop_nan_rows (bool, optional): Drop rows that contain ``NaN``. Defaults to True.

    Returns:
        pd.DataFrame: A cleaned copy of the returns dataframe.
    """
    cleaned = returns.replace([np.inf, -np.inf], np.nan)
    if drop_nan_rows:
        cleaned = cleaned.dropna(how="any")
    return cleaned


def validate_returns(returns: pd.DataFrame, *, allow_nan: bool = False) -> None:
    """Validate a returns dataframe.

    Args:
        returns (pd.DataFrame): Returns dataframe, one column per symbol.
        allow_nan (bool, optional): Allow ``NaN`` values. Defaults to False.

    Raises:
        DataValidationError: Collected description of all found problems.

    Notes:
        Columns with zero dispersion are reported through the ``bfportf.data`` logger instead of
        failing: the data is valid, but the inverse volatility weight is undefined.
    """
    problems = []
    if returns.empty:
        problems.append("returns dataframe is empty")
    if returns.columns.has_duplicates:
        duplicated = returns.columns[returns.columns.duplicated()].unique().tolist()
        problems.append(f"duplicated columns: {duplicated}")
    if returns.index.has_duplicates:
        problems.append("index contains duplicated timestamps")
    if not returns.index.is_monotonic_increasing:
        problems.append("index is not monotonically increasing")

    values = returns.to_numpy(dtype=float)
    infinite = np.isinf(values)
    if infinite.any():
        problems.append(f"infinite values in columns: {returns.columns[infinite.any(axis=0)].tolist()}")
    if not allow_nan:
        not_a_number = np.isnan(values)
        if not_a_number.any():
            problems.append(f"NaN values in columns: {returns.columns[not_a_number.any(axis=0)].tolist()}")

    if problems:
        msg = "; ".join(problems)
        raise DataValidationError(msg)

    zero_dispersion = returns.columns[returns.std(axis=0) == 0].tolist()
    if zero_dispersion:
        logger.warning("Zero dispersion (constant) columns, weighted metrics are undefined: %s", zero_dispersion)


def validate_symbols(returns: pd.DataFrame, symbols: Sequence[str]) -> list[str]:
    """Check that every symbol is present in the returns columns.

    Args:
        returns (pd.DataFrame): Returns dataframe.
        symbols (Sequence[str]): Symbols to validate.

    Returns:
        list[str]: List of validated symbols.

    Raises:
        DataValidationError: If the symbol list is empty.
        UnknownSymbolError: If a symbol is missing from the returns columns.
    """
    symbol_list = list(symbols)
    if not symbol_list:
        msg = "Portfolio must contain at least one symbol"
        raise DataValidationError(msg)
    missing = [symbol for symbol in symbol_list if symbol not in returns.columns]
    if missing:
        raise UnknownSymbolError(missing, returns.columns)
    return symbol_list
