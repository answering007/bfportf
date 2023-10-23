""" Optimization functions """
import math
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm


def find_next_symbol(
        returns: np.ndarray,
        symbols: np.ndarray,
        weights: np.ndarray,
        initial_returns: np.ndarray,
        initial_weight: np.float64,
        target_function: Callable[[np.ndarray], np.float64],
        maximize: bool = True) -> Union[Tuple[np.ndarray, np.float64], None]:
    """Find best target function value for specified initial return

    Args:
        returns (np.ndarray): 2d array of returns
        symbols (np.ndarray): 1d array of symbols
        weights (np.ndarray): 1d array of weights
        initial_return (np.ndarray): initial 1d array of return
        initial_weight (np.float64): weight of initial return array
        target_function (Callable[[np.ndarray], np.float64]): target optimization function
        maximize (bool, optional): True to maximize, otherwise to False. Defaults to True.

    Returns:
        Union[Tuple[np.ndarray, np.float64], None]: Tuple of target value and symbol if found, else None
    """

    # Calculate target value for the initial returns
    target_value = target_function(initial_returns / initial_weight)
    # Define compare functions
    compare_function = np.max if maximize else np.min
    find_index_function = np.argmax if maximize else np.argmin
    # Add initial returns to returns
    initial_returns = initial_returns.reshape(initial_returns.shape[0], 1)
    search_returns = returns + initial_returns
    # Add initial weight to weights
    search_weights = weights + initial_weight
    # Divide by search weights to get adjusted returns
    search_returns = search_returns / search_weights
    # Apply target function
    function_results = np.apply_along_axis(target_function, 0, search_returns)
    # Select the best result
    candidate_index = find_index_function(function_results)
    candidate_value = function_results[candidate_index]
    candidate_symbol = symbols[candidate_index]

    # Compare target value with candidate value
    compare_result = compare_function((target_value, candidate_value))
    is_equal = math.isclose(target_value, compare_result)

    return (candidate_symbol, candidate_value) if not is_equal else None

def find_portfolio_for_symbol(
        returns: np.ndarray,
        symbols: np.ndarray,
        weights: np.ndarray,
        target_symbol: str,
        target_function: Callable[[np.ndarray], np.float64],
        maximize: bool = True,
        max_number: Optional[int] = None) -> Tuple[np.ndarray, np.float64]:
    """Find portfolio for specified symbol

    Args:
        returns (np.ndarray): 2d array of returns.
        symbols (np.ndarray): 1d array of symbols.
        weights (np.ndarray): 1d array of symbols weights.
        target_symbol (str): Specified symbol to find portfolio for
        target_function (Callable[[np.ndarray], np.float64]): Target optimization function
        maximize (bool, optional): True to maximize, otherwise to False. Defaults to True.
        max_number (int, optional): Maximum number of symbols in portfolio. Defaults to None (no limitations).

    Returns:
        Tuple[np.ndarray, np.float64]: Tuple of portfolio symbols and portfolio target function value
    """

    # Init portfolio data
    portfolio_symbols = np.array([target_symbol])
    portfolio_symbols_filter = np.in1d(symbols, portfolio_symbols)
    portfolio_returns = returns[:, portfolio_symbols_filter]
    portfolio_weights = weights[portfolio_symbols_filter]
    target_function_value = target_function(portfolio_returns.flatten() / portfolio_weights)
    if max_number is not None and len(portfolio_symbols) >= max_number:
        return portfolio_symbols, target_function_value

    # Init candidates data
    candidate_symbols = symbols[~portfolio_symbols_filter]
    candidate_returns = returns[:, ~portfolio_symbols_filter]
    candidate_weights = weights[~portfolio_symbols_filter]
    iterations = len(candidate_symbols)

    for _ in range(iterations):
        initial_returns = np.sum(portfolio_returns, axis=1).flatten()
        initial_weight = portfolio_weights.sum()

        find_result = find_next_symbol(
            returns=candidate_returns,
            symbols=candidate_symbols,
            weights=candidate_weights,
            initial_returns=initial_returns,
            initial_weight=initial_weight,
            target_function=target_function,
            maximize=maximize
        )

        if find_result is not None:
            found_symbol, target_function_value = find_result
            # Add data to portfolio
            found_symbol = np.array([found_symbol])
            portfolio_symbols = np.concatenate([portfolio_symbols, found_symbol], axis=0)
            portfolio_symbols_filter = np.in1d(symbols, portfolio_symbols)
            portfolio_symbols = symbols[portfolio_symbols_filter]
            portfolio_returns = returns[:, portfolio_symbols_filter]
            portfolio_weights = weights[portfolio_symbols_filter]
            # Remove from candidates data
            candidate_symbols_filter = ~np.in1d(symbols, portfolio_symbols)
            candidate_symbols = symbols[candidate_symbols_filter]
            candidate_returns = returns[:, candidate_symbols_filter]
            candidate_weights = weights[candidate_symbols_filter]
            if max_number is not None and len(portfolio_symbols) >= max_number:
                break
        else:
            break
    return portfolio_symbols, target_function_value

def find_portfolios(
        returns: pd.DataFrame,
        target_function: Callable[[np.ndarray], np.float64],
        maximize: bool = True,
        target_symbols: Optional[List[str]] = None,
        custom_weights: pd.Series = None,
        max_number: Optional[int] = None,
        verbose: bool = True) -> pd.DataFrame:
    """Find best portfolio for specified target function

    Args:
        returns (pd.DataFrame): Returns dataframe
        target_function (Callable[[np.ndarray], np.float64]): Target function to optimize
        maximize (bool, optional): True to maximize target function value, otherwise to False. Defaults to True.
        target_symbols (List[str], optional): Symbols to use as a targets. Defaults to None (all columns).
        custom_weights (pd.Series, optional): Weight of each symbol. Defaults to None (equal weights).
        max_number (int, optional): Maximum number of symbols in portfolio. Defaults to None.
        verbose (bool, optional): True to print progress; otherwise False. Defaults to True.

    Returns:
        pd.DataFrame: Founded portfolios
    """

    # Define weights
    if custom_weights is None:
        custom_weights = np.full(len(returns.columns), 1)
        custom_weights = pd.Series(custom_weights, index=returns.columns)
    # Define target symbols
    target_symbols = target_symbols.copy() if target_symbols is not None else returns.columns.tolist()

    # Define numpy variables
    np_returns = returns.values
    np_symbols = np.array(returns.columns)
    np_weights = custom_weights.values

    iteration_obj = tqdm(target_symbols) if verbose else target_symbols

    results = []
    for symbol in iteration_obj:
        portfolio = find_portfolio_for_symbol(
            returns=np_returns,
            symbols=np_symbols,
            weights=np_weights,
            target_symbol=symbol,
            target_function=target_function,
            maximize=maximize,
            max_number=max_number
        )
        results.append(portfolio)

    results = pd.DataFrame(results, columns=["symbols", "target_function_value"])
    results["Portfolio"] = results["symbols"].apply(";".join)
    results = results.groupby(by="Portfolio").first()
    results = results.sort_values(by="target_function_value", ascending= not maximize)
    results.reset_index(inplace=True, drop=True)
    return results
