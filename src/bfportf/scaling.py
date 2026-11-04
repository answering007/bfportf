"""Weight scaling helpers."""

from collections.abc import Sequence
from typing import Optional

import pandas as pd

from bfportf.exceptions import DataValidationError, UnknownSymbolError


def scale_return_data(
    df_return: pd.DataFrame,
    target_return: Optional[float] = None,
    target_symbol: Optional[str] = None,
) -> tuple:
    """Scale return (pct_change) data to equal average abs return.

    Args:
        df_return (pd.DataFrame): Source pct_change dataframe.
        target_return (float, optional): Target abs return value (e.g. 0.01 is the 1% return).
            If both target_return and target_symbol are None the average abs of all values is used.
            Defaults to None.
        target_symbol (str, optional): Target symbol to use as the key to get abs return value.
            If it is not None target_return will be ignored. Defaults to None.

    Returns:
        tuple: Scaled returns dataframe and shares series (weights that sum up to one).

    Raises:
        UnknownSymbolError: If ``target_symbol`` is missing from the dataframe.
        DataValidationError: If the goal return or a column mean is zero.
    """
    if target_symbol is not None and target_symbol not in df_return.columns:
        raise UnknownSymbolError([target_symbol], df_return.columns)

    series_mean = df_return.abs().mean()
    goal_return = series_mean.mean()
    if target_symbol is not None:
        goal_return = series_mean[target_symbol]
    elif target_return is not None:
        goal_return = target_return

    if goal_return == 0:
        msg = "goal return is zero, scaling is undefined"
        raise DataValidationError(msg)
    zero_mean = series_mean.index[series_mean == 0].tolist()
    if zero_mean:
        msg = f"zero mean absolute return in columns: {zero_mean}; scaling is undefined"
        raise DataValidationError(msg)

    series_share = goal_return / series_mean
    series_share = series_share / series_share.sum()
    return df_return * series_share, series_share


def add_inverted_return(
    returns: pd.DataFrame,
    inverted_symbols: Sequence[str],
    postfix: str = "-1",
) -> tuple:
    """Add inverted return columns to the dataframe.

    Args:
        returns (pd.DataFrame): Initial dataframe with returns.
        inverted_symbols (Sequence[str]): List of column names to invert.
        postfix (str, optional): Postfix for each inverted column name. Defaults to "-1".

    Returns:
        tuple: Dataframe with inverted returns and list of inverted column names.

    Raises:
        UnknownSymbolError: If an inverted symbol is missing from the dataframe.
        DataValidationError: If an inverted column name already exists.
    """
    symbols = list(inverted_symbols)
    missing = [symbol for symbol in symbols if symbol not in returns.columns]
    if missing:
        raise UnknownSymbolError(missing, returns.columns)

    inverted_names = [symbol + postfix for symbol in symbols]
    duplicates = [name for name in inverted_names if name in returns.columns]
    if duplicates:
        msg = f"inverted column names already exist: {duplicates}"
        raise DataValidationError(msg)

    inverted = pd.DataFrame(data=returns[symbols]) * -1
    inverted.columns = inverted_names
    return pd.concat([returns, inverted], axis=1), inverted_names
