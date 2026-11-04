"""Common utilities."""

import logging
from collections.abc import Sequence
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from bfportf.exceptions import DataValidationError, UnknownSymbolError

logger = logging.getLogger(__name__)

WeightLike = Optional[Union[pd.Series, np.ndarray]]


def return_to_price(returns: np.ndarray, first_value: np.ndarray = None) -> np.ndarray:
    """Convert return array to price array.

    Args:
        returns (np.ndarray): Source return array.
        first_value (np.ndarray, optional): First (pre-period) value. Defaults to ``np.array(1)``.

    Returns:
        np.ndarray: Price array of ``len(returns) + 1`` elements.
    """
    first = np.asarray(first_value if first_value is not None else np.array(1.0)).ravel()
    return np.concatenate([first, np.asarray(returns).ravel() + 1], axis=None).cumprod()


def price_to_return(prices: np.ndarray) -> np.ndarray:
    """Convert price array to return array.

    Args:
        prices (np.ndarray): Price array.

    Returns:
        np.ndarray: Return array of ``len(prices) - 1`` elements.
    """
    prices = np.asarray(prices).ravel()
    return np.diff(prices) / prices[:-1]


def validate_returns(returns: pd.DataFrame, allow_nan: bool = False) -> None:
    """Validate a returns dataframe.

    Args:
        returns (pd.DataFrame): Returns dataframe, one column per symbol.
        allow_nan (bool, optional): Allow ``NaN`` values. Defaults to False.

    Raises:
        DataValidationError: Collected description of all found problems.

    Notes:
        Columns with zero dispersion are reported via the ``bfportf.utils`` logger instead of
        failing: they are valid data, but their inverse volatility weight is undefined.
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
        raise DataValidationError("; ".join(problems))

    zero_dispersion = returns.columns[returns.std(axis=0) == 0].tolist()
    if zero_dispersion:
        logger.warning("Zero dispersion (constant) columns, weighted metrics are undefined: %s", zero_dispersion)


def _validate_symbols(returns: pd.DataFrame, symbols: Sequence[str]) -> list:
    """Check that every symbol is present in the returns columns.

    Args:
        returns (pd.DataFrame): Returns dataframe.
        symbols (Sequence[str]): Symbols to validate.

    Returns:
        list: List of validated symbols.

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


def _resolve_weights(returns: pd.DataFrame, weights: WeightLike) -> pd.Series:
    """Normalize the weights argument into a series indexed by the returns columns.

    Args:
        returns (pd.DataFrame): Returns dataframe.
        weights (WeightLike): Weights as a series, a positional array or None (equal weights).

    Returns:
        pd.Series: Weights indexed by the returns columns.

    Raises:
        UnknownSymbolError: If the weights series misses a returns column.
        DataValidationError: If the array length does not match the number of columns.
    """
    if weights is None:
        return pd.Series(1.0, index=returns.columns, dtype=float)
    if isinstance(weights, pd.Series):
        missing = [column for column in returns.columns if column not in weights.index]
        if missing:
            raise UnknownSymbolError(missing, weights.index)
        return weights.reindex(returns.columns)
    array = np.asarray(weights, dtype=float)
    if array.shape != (len(returns.columns),):
        msg = f"weights array must have shape ({len(returns.columns)},), got {array.shape}"
        raise DataValidationError(
            msg,
        )
    return pd.Series(array, index=returns.columns)


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
        tuple: Scaled returns dataframe and shares series.

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


def drawdown_pct(prices: np.ndarray) -> np.ndarray:
    """Drawdown array (percentage from the running maximum).

    Args:
        prices (np.ndarray): Prices.

    Returns:
        np.ndarray: Array of drawdown values (percentage, non positive).
    """
    prices = np.asarray(prices)
    np_max = np.maximum.accumulate(prices)
    return prices / np_max - 1


def get_portfolio_return(
    returns: pd.DataFrame,
    portfolio_symbols: Sequence[str],
    return_weights: WeightLike = None,
    normalize: str = "sum",
) -> pd.Series:
    """Calculate the portfolio return series.

    Args:
        returns (pd.DataFrame): Dataframe with returns.
        portfolio_symbols (Sequence[str]): Portfolio symbols.
        return_weights (WeightLike, optional): Symbols weights. Defaults to None (equal weights).
            Weights define the allocation share of every symbol, e.g. inverse volatility shares
            produced by :func:`scale_return_data`.
        normalize (str, optional): ``"sum"`` (default) divides by the total weight of the selected
            symbols, i.e. rescales the sub portfolio to a fully invested position. ``"none"`` keeps
            the original scale (the position is leveraged according to the weights sum).
            Defaults to "sum".

    Returns:
        pd.Series: Portfolio return series.

    Raises:
        UnknownSymbolError: If a symbol is missing from the returns columns.
        DataValidationError: If the symbol list is empty, the normalization is unknown or the
            total weight is zero.

    Notes:
        With ``normalize="sum"`` a single symbol portfolio is rescaled back to the raw symbol
        returns, i.e. the scaling done by :func:`scale_return_data` cancels out.
    """
    symbols = _validate_symbols(returns, portfolio_symbols)
    if normalize not in ("sum", "none"):
        msg = f"normalize must be 'sum' or 'none', got {normalize!r}"
        raise DataValidationError(msg)

    weights = _resolve_weights(returns, return_weights)
    portfolio_series = returns[symbols].sum(axis=1)
    if normalize == "none":
        return portfolio_series

    total_weight = float(weights[symbols].sum())
    if total_weight == 0:
        msg = "total weight of the portfolio symbols is zero, normalization is undefined"
        raise DataValidationError(msg)
    return portfolio_series / total_weight


def calculate_metric(
    results: pd.DataFrame,
    returns: pd.DataFrame,
    return_weights: WeightLike,
    func: Callable[[np.ndarray], np.float64],
) -> pd.Series:
    """Calculate a metric for every portfolio found by the brute force search.

    Args:
        results (pd.DataFrame): Brute force results with the ``symbols`` column.
        returns (pd.DataFrame): Returns dataframe.
        return_weights (WeightLike): Symbols weights.
        func (Callable[[np.ndarray], np.float64]): Metric function, takes the portfolio returns.

    Returns:
        pd.Series: Series of metric values.
    """

    def apply_function(symbols: Sequence[str]) -> np.float64:
        portfolio_series = get_portfolio_return(
            returns=returns,
            portfolio_symbols=symbols,
            return_weights=return_weights,
            normalize="sum",
        )
        return func(portfolio_series.to_numpy())

    return results["symbols"].apply(apply_function)
