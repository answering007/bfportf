"""Portfolio aggregation helpers."""

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from bfportf.data import validate_symbols
from bfportf.exceptions import DataValidationError, UnknownSymbolError

WeightLike = pd.Series | np.ndarray | None


def resolve_weights(returns: pd.DataFrame, weights: WeightLike) -> pd.Series:
    """Normalize the weights argument into a series indexed by the returns columns.

    Args:
        returns (pd.DataFrame): Returns dataframe.
        weights (WeightLike): Weights as a series, a positional array or None (equal weights).

    Returns:
        pd.Series: Weights indexed by the returns columns.

    Raises:
        UnknownSymbolError: If the weights series misses a returns column.
        DataValidationError: If the array shape does not match the number of columns.
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
        raise DataValidationError(msg)
    return pd.Series(array, index=returns.columns)


def normalized_weights(returns: pd.DataFrame, symbols: Sequence[str], weights: WeightLike = None) -> pd.Series:
    """Weights of the portfolio symbols rescaled to sum up to one.

    Args:
        returns (pd.DataFrame): Returns dataframe.
        symbols (Sequence[str]): Portfolio symbols.
        weights (WeightLike, optional): Source weights. Defaults to None (equal weights).

    Returns:
        pd.Series: Weights indexed by the portfolio symbols.

    Raises:
        DataValidationError: If the total weight of the portfolio is zero.
    """
    symbol_list = validate_symbols(returns, symbols)
    resolved = resolve_weights(returns, weights)[symbol_list]
    total = float(resolved.sum())
    if total == 0:
        msg = "total weight of the portfolio symbols is zero"
        raise DataValidationError(msg)
    return resolved / total


def get_portfolio_return(
    returns: pd.DataFrame,
    portfolio_symbols: Sequence[str],
    return_weights: WeightLike = None,
    normalize: str = "sum",
) -> pd.Series:
    """Calculate the portfolio return series.

    Args:
        returns (pd.DataFrame): Dataframe with returns. The returns must already be scaled by the
            weights when the weights should change the result (see the notes below).
        portfolio_symbols (Sequence[str]): Portfolio symbols.
        return_weights (WeightLike, optional): Symbols weights. Defaults to None (equal weights).
            Weights define the allocation share of every symbol, e.g. inverse volatility shares
            produced by :func:`bfportf.scaling.scale_return_data`. They are not multiplied into the
            returns here, they only normalize the aggregate.
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
        The computed series is ``sum(returns[symbols]) / sum(weights[symbols])`` for ``normalize="sum"``
        and ``sum(returns[symbols])`` for ``normalize="none"``. Weights are not applied to the returns
        internally: passing raw returns with unequal weights does not give a weighted average, it only
        changes the divisor. To weight the symbols, scale the returns first (``returns * weights``), as
        :func:`bfportf.scaling.scale_return_data` does.

        With ``normalize="sum"`` a single symbol portfolio is rescaled back to the raw symbol
        returns, i.e. the scaling done by :func:`bfportf.scaling.scale_return_data` cancels out.
    """
    symbols = validate_symbols(returns, portfolio_symbols)
    if normalize not in ("sum", "none"):
        msg = f"normalize must be 'sum' or 'none', got {normalize!r}"
        raise DataValidationError(msg)

    weights = resolve_weights(returns, return_weights)
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
    """Calculate a metric for every portfolio of a results frame.

    Args:
        results (pd.DataFrame): Frame with the ``symbols`` column, e.g. from
            :func:`bfportf.search.results_to_frame`.
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
