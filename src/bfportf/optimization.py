"""Optimization functions."""

from collections.abc import Sequence
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from bfportf.exceptions import DataValidationError, UnknownSymbolError
from bfportf.utils import validate_returns


def _is_improvement(current: float, candidate: float, maximize: bool, tolerance: float) -> bool:
    """Return True when the candidate is better than the current value by more than tolerance."""
    delta = candidate - current if maximize else current - candidate
    return bool(np.isfinite(delta) and delta > tolerance)


def find_next_symbol(
    returns: np.ndarray,
    symbols: np.ndarray,
    weights: np.ndarray,
    initial_returns: np.ndarray,
    initial_weight: np.float64,
    target_function: Callable[[np.ndarray], np.float64],
    maximize: bool = True,
    tolerance: float = 0.0,
) -> Optional[tuple]:
    """Find the best target function value for the specified initial return.

    Args:
        returns (np.ndarray): 2d array of returns (periods x candidates).
        symbols (np.ndarray): 1d array of candidate symbols.
        weights (np.ndarray): 1d array of candidate weights.
        initial_returns (np.ndarray): 1d array of the current portfolio returns.
        initial_weight (np.float64): Total weight of the current portfolio.
        target_function (Callable[[np.ndarray], np.float64]): Target optimization function.
        maximize (bool, optional): True to maximize, otherwise to minimize. Defaults to True.
        tolerance (float, optional): Minimal improvement required to accept a candidate, useful to
            stop the search when the metric does not move. Defaults to 0.0.

    Returns:
        Optional[tuple]: ``(symbol, value)`` of the best candidate that improves the portfolio, or
        ``None`` when there is no valid candidate or the improvement is not significant.

    Notes:
        Candidates whose metric is not finite (``nan``/``inf``) are ignored, so degenerate
        portfolios (for example the profit factor without losing periods) are never selected.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        target_value = float(target_function(initial_returns / initial_weight))
        search_returns = (returns + initial_returns.reshape(initial_returns.shape[0], 1)) / (weights + initial_weight)
        function_results = np.apply_along_axis(target_function, 0, search_returns).astype(float)

    if not np.isfinite(target_value):
        target_value = -np.inf if maximize else np.inf

    valid = np.isfinite(function_results)
    if not valid.any():
        return None

    if maximize:
        candidate_index = int(np.argmax(np.where(valid, function_results, -np.inf)))
    else:
        candidate_index = int(np.argmin(np.where(valid, function_results, np.inf)))

    candidate_value = np.float64(function_results[candidate_index])
    if not _is_improvement(target_value, candidate_value, maximize, tolerance):
        return None
    return str(symbols[candidate_index]), candidate_value


def find_portfolio_for_symbol(
    returns: np.ndarray,
    symbols: np.ndarray,
    weights: np.ndarray,
    target_symbol: str,
    target_function: Callable[[np.ndarray], np.float64],
    maximize: bool = True,
    max_number: Optional[int] = None,
    tolerance: float = 0.0,
) -> tuple:
    """Find the portfolio for the specified symbol with the forward selection.

    Args:
        returns (np.ndarray): 2d array of returns (periods x symbols).
        symbols (np.ndarray): 1d array of symbols.
        weights (np.ndarray): 1d array of symbols weights.
        target_symbol (str): Specified symbol to find portfolio for.
        target_function (Callable[[np.ndarray], np.float64]): Target optimization function.
        maximize (bool, optional): True to maximize, otherwise to minimize. Defaults to True.
        max_number (int, optional): Maximum number of symbols in the portfolio. Defaults to None
            (no limitations).
        tolerance (float, optional): Minimal improvement required to add the next symbol.
            Defaults to 0.0.

    Returns:
        tuple: ``(portfolio_symbols, target_function_value)``.

    Raises:
        UnknownSymbolError: If ``target_symbol`` is missing from ``symbols``.
        DataValidationError: If ``max_number`` is lower than 1.
    """
    available_symbols = symbols.tolist()
    if target_symbol not in available_symbols:
        raise UnknownSymbolError([target_symbol], available_symbols)
    if max_number is not None and max_number < 1:
        msg = f"max_number must be >= 1, got {max_number}"
        raise DataValidationError(msg)

    portfolio_symbol_filter = np.isin(symbols, [target_symbol])
    portfolio_symbols = symbols[portfolio_symbol_filter]
    portfolio_returns = returns[:, portfolio_symbol_filter]
    portfolio_weights = weights[portfolio_symbol_filter]

    with np.errstate(invalid="ignore", divide="ignore"):
        target_function_value = np.float64(
            target_function(portfolio_returns.sum(axis=1) / portfolio_weights.sum()),
        )

    if max_number is not None and len(portfolio_symbols) >= max_number:
        return portfolio_symbols, target_function_value

    candidate_symbol_filter = ~portfolio_symbol_filter
    iterations = int(candidate_symbol_filter.sum())

    for _ in range(iterations):
        find_result = find_next_symbol(
            returns=returns[:, candidate_symbol_filter],
            symbols=symbols[candidate_symbol_filter],
            weights=weights[candidate_symbol_filter],
            initial_returns=portfolio_returns.sum(axis=1),
            initial_weight=portfolio_weights.sum(),
            target_function=target_function,
            maximize=maximize,
            tolerance=tolerance,
        )
        if find_result is None:
            break

        found_symbol, target_function_value = find_result
        portfolio_symbol_filter = portfolio_symbol_filter | np.isin(symbols, [found_symbol])
        candidate_symbol_filter = ~portfolio_symbol_filter
        portfolio_symbols = symbols[portfolio_symbol_filter]
        portfolio_returns = returns[:, portfolio_symbol_filter]
        portfolio_weights = weights[portfolio_symbol_filter]

        if max_number is not None and len(portfolio_symbols) >= max_number:
            break

    return portfolio_symbols, target_function_value


def find_portfolios(
    returns: pd.DataFrame,
    target_function: Callable[[np.ndarray], np.float64],
    maximize: bool = True,
    target_symbols: Optional[Sequence[str]] = None,
    custom_weights: Optional[Union[pd.Series, np.ndarray]] = None,
    max_number: Optional[int] = None,
    verbose: bool = True,
    tolerance: float = 0.0,
    validate: bool = True,
) -> pd.DataFrame:
    """Find the best portfolio for every symbol using the specified target function.

    Args:
        returns (pd.DataFrame): Returns dataframe.
        target_function (Callable[[np.ndarray], np.float64]): Target function to optimize.
        maximize (bool, optional): True to maximize the target function value, otherwise to
            minimize it. Defaults to True.
        target_symbols (Sequence[str], optional): Symbols to use as targets. Defaults to None
            (all columns).
        custom_weights (pd.Series | np.ndarray, optional): Weight of each symbol. Defaults to None
            (equal weights).
        max_number (int, optional): Maximum number of symbols in the portfolio. Defaults to None.
        verbose (bool, optional): True to print progress, otherwise False. Defaults to True.
        tolerance (float, optional): Minimal improvement required to add the next symbol.
            Defaults to 0.0.
        validate (bool, optional): Validate the returns with
            :func:`bfportf.utils.validate_returns`. Defaults to True.

    Returns:
        pd.DataFrame: Found portfolios ordered from the best to the worst.

    Raises:
        UnknownSymbolError: If a target symbol or a weights entry is missing from the columns.
        DataValidationError: If the returns dataframe or the weights do not pass validation.
    """
    if validate:
        validate_returns(returns)

    np_weights = _resolve_search_weights(returns, custom_weights)

    if target_symbols is None:
        symbols_to_search = list(returns.columns)
    else:
        symbols_to_search = list(target_symbols)
        missing = [symbol for symbol in symbols_to_search if symbol not in returns.columns]
        if missing:
            raise UnknownSymbolError(missing, returns.columns)

    np_returns = returns.to_numpy(dtype=float)
    np_symbols = np.array(returns.columns)

    iteration_obj = tqdm(symbols_to_search) if verbose else symbols_to_search

    results = []
    for symbol in iteration_obj:
        portfolio = find_portfolio_for_symbol(
            returns=np_returns,
            symbols=np_symbols,
            weights=np_weights,
            target_symbol=symbol,
            target_function=target_function,
            maximize=maximize,
            max_number=max_number,
            tolerance=tolerance,
        )
        results.append(portfolio)

    results = pd.DataFrame(results, columns=["symbols", "target_function_value"])
    results["Portfolio"] = results["symbols"].apply(";".join)
    results = results.groupby(by="Portfolio", dropna=False).first()
    results = results.sort_values(by="target_function_value", ascending=not maximize, na_position="last")
    return results.reset_index(drop=True)


def _resolve_search_weights(
    returns: pd.DataFrame,
    custom_weights: Optional[Union[pd.Series, np.ndarray]],
) -> np.ndarray:
    """Resolve the custom weights into a positional array aligned with the returns columns.

    Args:
        returns (pd.DataFrame): Returns dataframe.
        custom_weights (pd.Series | np.ndarray, optional): Weights or None for equal weights.

    Returns:
        np.ndarray: 1d array of weights.

    Raises:
        UnknownSymbolError: If the weights series misses a returns column.
        DataValidationError: If the array shape does not match the number of columns.
    """
    if custom_weights is None:
        return np.ones(len(returns.columns), dtype=float)
    if isinstance(custom_weights, pd.Series):
        missing = [column for column in returns.columns if column not in custom_weights.index]
        if missing:
            raise UnknownSymbolError(missing, custom_weights.index)
        return custom_weights.reindex(returns.columns).to_numpy(dtype=float)
    array = np.asarray(custom_weights, dtype=float)
    if array.shape != (len(returns.columns),):
        msg = f"weights array must have shape ({len(returns.columns)},), got {array.shape}"
        raise DataValidationError(
            msg,
        )
    return array
