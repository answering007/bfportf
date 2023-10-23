"""Collection of target functions"""
import numpy as np
from scipy.stats import kurtosis, pearsonr, skew

from bfportf.utils import drawdown_pct, return_to_price


def pnl_target(returns: np.ndarray) -> np.float64:
    """PnL target function

    Args:
        returns (np.ndarray): 1d array of returns

    Returns:
        np.float64: PnL value
    """
    return return_to_price(returns)[-1]-1


def max_drawdown_target(returns: np.ndarray) -> np.float64:
    """Max drawdown target function

    Args:
        returns (np.ndarray): 1d array of returns

    Returns:
        np.float64: Max drawdown value (percentage)
    """
    prices = return_to_price(returns)
    return drawdown_pct(prices).min()


def profit_factor_target(returns: np.ndarray) -> np.float64:
    """Profit factor target function

    Args:
        returns (np.ndarray): 1d array of returns

    Returns:
        np.float64: Profit factor value
    """
    return abs(returns[returns >= 0].sum() / returns[returns < 0].sum())


def recovery_factor_target(returns: np.ndarray) -> np.float64:
    """Recovery factor target function

    Args:
        returns (np.ndarray): 1d array of returns

    Returns:
        np.float64: Recovery factor value
    """
    return returns.sum() / abs(max_drawdown_target(returns))


def sharpe_target(returns: np.ndarray, periods: int = 252) -> np.float64:
    """Sharpe target function

    Args:
        returns (np.ndarray): 1d array of returns
        periods (int, optional): Number of periods. Defaults to 252 (annual).

    Returns:
        np.float64: Sharpe ratio value
    """
    return returns.mean() / returns.std(ddof=1) * np.sqrt(1 if periods is None else periods)


def sortino_target(returns: np.ndarray, periods: int = 252) -> np.float64:
    """Sortino target function

    Args:
        returns (np.ndarray): 1d array of returns
        periods (int, optional): Number of periods. Defaults to 252 (annual).

    Returns:
        np.float64: Sortino ratio value
    """
    underwater = np.sqrt((returns[returns < 0] ** 2).sum() / len(returns))
    result = returns.mean() / underwater * np.sqrt(1 if periods is None else periods)
    return result


def skew_target(returns: np.ndarray) -> np.float64:
    """Skew target function

    Args:
        returns (np.ndarray): 1d array of returns

    Returns:
        np.float64: Skew value
    """
    return skew(returns)


def kurtosis_target(returns: np.ndarray) -> np.float64:
    """Kurtosis target function

    Args:
        returns (np.ndarray): 1d array of returns

    Returns:
        np.float64: Kurtosis value
    """
    return kurtosis(returns)


def autocorrelation_target(
        returns: np.ndarray,
        absolute: bool = True,
        periods: int = 1) -> np.float64:
    """Autocorrelation target function

    Args:
        returns (np.ndarray): 1d array of returns
        absolute (bool): True to calculate absolute value, otherwise to calculate relative.
        Defaults to True.
        periods (int, optional): Number of periods to shift. Defaults to 1.

    Returns:
        np.float64: Autocorrelation value
    """
    result = pearsonr(returns[:-periods], returns[periods:])[0]
    return np.absolute(result) if absolute else result


def cagr_target(returns: np.ndarray, days_per_year: int = 252) -> np.float64:
    """Compound annual growth rate target function

    Args:
        returns (np.ndarray): 1d array of returns
        days_per_year (int, optional): Number of periods. Defaults to 252 (annual).

    Returns:
        np.float64: CAGR value
    """
    prices = return_to_price(returns)
    periods = len(prices) / days_per_year
    return np.power(prices[-1] / prices[0], 1 / periods) - 1
