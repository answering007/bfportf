"""Common utilities"""
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd


def return_to_price(
        returns: np.ndarray,
        first_value: np.ndarray = None) -> np.ndarray:
    """Convert return array to price array

    Args:
        return_array (np.ndarray): Source return array
        first_value (np.ndarray, optional): First value. Defaults to np.array(1).

    Returns:
        np.ndarray: Price array
    """
    first_value = first_value if first_value is not None else np.array(1)
    return np.concatenate([first_value, np.add(returns, np.array(1))], axis=None).cumprod()


def price_to_return(prices: np.ndarray) -> np.ndarray:
    """Convert price array to return array

    Args:
        price (np.ndarray): Price array

    Returns:
        np.ndarray: return array
    """
    return np.diff(prices) / prices[:-1]


def scale_return_data(
        df_return: pd.DataFrame,
        target_return: Optional[float] = None,
        target_symbol: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Scale return (pct_change) data to equal average abs return

    Args:
        df_return (pd.DataFrame): Source pct_change dataframe
        target_return (float, optional): Target abs return value (e.g. 0.01 is the 1% return).
        If both target_return and target_symbol is None the average abs of all values will be used.
        Defaults to None.
        target_symbol (str, optional): Target symbol to use as the key to get abs return value.
        If it is not None target_return will be ignored. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Scaled returns dataframe and shares series
    """
    series_mean = df_return.abs().mean()
    goal_return = series_mean.mean()
    if target_symbol is not None:
        goal_return = series_mean[target_symbol]
    elif target_return is not None:
        goal_return = target_return

    series_share = goal_return / series_mean
    series_share = series_share / series_share.sum()
    return df_return * series_share, series_share


def add_inverted_return(
        returns: pd.DataFrame,
        inverted_symbols: List[str],
        postfix: str = "-1") -> Tuple[pd.DataFrame, List[str]]:
    """Add inverted return columns to the dataframe

    Args:
        returns (pd.DataFrame): Initial dataframe with returns
        inverted_symbols (List[str]): List of column names to invert
        postfix (str, optional): Postfix for each inverted column name. Defaults to "-1".

    Returns:
        Tuple[pd.DataFrame, List[str]]: Dataframe with inverted returns
        and list of inverted column names
    """
    inverted_names = [x + postfix for x in inverted_symbols]
    inverted = pd.DataFrame(data=returns[inverted_symbols])*-1
    inverted.columns = inverted_names
    return pd.concat([returns, inverted], axis=1), inverted_names


def drawdown_pct(prices: np.ndarray) -> np.ndarray:
    """Max drawdown array

    Args:
        prices (np.ndarray): Prices

    Returns:
        np.ndarray: Array of max drawdown (percentage)
    """
    np_max = np.maximum.accumulate(prices)
    return prices / np_max - 1


def get_portfolio_return(
        returns: pd.DataFrame,
        portfolio_symbols: List[str],
        return_weights: pd.Series = None) -> pd.Series:
    """Calculate portfolio return series

    Args:
        returns (pd.DataFrame): Dataframe with returns
        portfolio_symbols (List[str]): Portfolio symbols
        return_weights (pd.Series, optional): Symbols weights. Defaults to None. If None, equal weights will be used.

    Returns:
        pd.Series: Portfolio return series
    """
    if return_weights is None:
        return_weights = pd.Series(
            [1] * len(returns.columns), index=returns.columns)
    return returns[portfolio_symbols].sum(axis=1) / return_weights[portfolio_symbols].sum()


def calculate_metric(
        results: pd.DataFrame,
        returns: pd.DataFrame,
        return_weights: pd.Series,
        func: Callable[[np.ndarray], np.float64]) -> pd.Series:
    """Calculate metric for each portfolio symbols

    Args:
        results (pd.DataFrame): Brute force results
        returns (pd.DataFrame): Returns dataframe
        return_weights (pd.Series): Symbols weights
        func (Callable[[np.ndarray], np.float64]): Metric function, use portfolio return as input

    Returns:
        pd.Series: Series of metric values
    """
    def apply_function(
            returns: pd.DataFrame,
            weights: pd.Series,
            symbols: List[str],
            func: Callable[[np.ndarray], np.float64]) -> np.float64:
        portfolio_series = get_portfolio_return(
            returns=returns, portfolio_symbols=symbols, return_weights=weights)
        return func(portfolio_series.values)

    return results["symbols"].apply(lambda symbols_list: apply_function(returns, return_weights, symbols_list, func))
