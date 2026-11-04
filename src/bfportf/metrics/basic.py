"""Built in metrics.

Every metric accepts a 1d array of returns and returns a single ``np.float64``. Degenerate
inputs do not raise: the value is ``np.nan`` when the metric is undefined (for example the
profit factor without losing periods), so search algorithms skip such candidates instead of
selecting them accidentally.

Each metric also has a vectorized twin (``*_matrix``) that accepts a 2d array
(periods x candidates) and returns one value per column; it is used by the search fast path.
"""

from typing import Optional

import numpy as np
from scipy.stats import kurtosis, pearsonr, skew

from bfportf.utils import drawdown_pct, return_to_price

_CONSTANT_TOLERANCE = 1e-12
_MIN_SERIES_SIZE = 2


def _is_constant(values: np.ndarray) -> bool:
    """Return True if the series has no dispersion (dispersion metrics are undefined then).

    The comparison uses a small relative tolerance because ``np.full(n, 0.01)`` has a floating
    point standard deviation around ``1e-18`` instead of exactly zero.
    """
    values = np.asarray(values, dtype=float)
    if values.size < _MIN_SERIES_SIZE:
        return True
    scale = max(float(np.max(np.abs(values))), 1.0)
    return float(np.std(values)) <= _CONSTANT_TOLERANCE * scale


def pnl_target(returns: np.ndarray) -> np.float64:
    """PnL target function.

    Args:
        returns (np.ndarray): 1d array of returns.

    Returns:
        np.float64: Total compounded return (0.1 is 10%), ``np.nan`` for empty input.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return np.float64(np.nan)
    return np.float64(return_to_price(returns)[-1] - 1)


def max_drawdown_target(returns: np.ndarray) -> np.float64:
    """Max drawdown target function.

    Args:
        returns (np.ndarray): 1d array of returns.

    Returns:
        np.float64: Max drawdown value (percentage, non positive), ``np.nan`` for empty input.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return np.float64(np.nan)
    prices = return_to_price(returns)
    return np.float64(drawdown_pct(prices).min())


def profit_factor_target(returns: np.ndarray) -> np.float64:
    """Profit factor target function.

    Args:
        returns (np.ndarray): 1d array of returns.

    Returns:
        np.float64: Gross profit divided by gross loss. ``np.nan`` when there are no losing
        or no winning periods (the ratio is undefined).
    """
    returns = np.asarray(returns, dtype=float)
    gains = float(returns[returns >= 0].sum())
    losses = float(-returns[returns < 0].sum())
    if gains == 0 or losses == 0:
        return np.float64(np.nan)
    return np.float64(gains / losses)


def recovery_factor_target(returns: np.ndarray) -> np.float64:
    """Recovery factor target function.

    Args:
        returns (np.ndarray): 1d array of returns.

    Returns:
        np.float64: Compounded PnL divided by the absolute max drawdown (both in the same units),
        ``np.nan`` when there is no drawdown (the ratio is undefined).
    """
    returns = np.asarray(returns, dtype=float)
    drawdown = max_drawdown_target(returns)
    if not np.isfinite(drawdown) or drawdown == 0:
        return np.float64(np.nan)
    return np.float64(pnl_target(returns) / abs(drawdown))


def sharpe_target(
    returns: np.ndarray,
    periods: Optional[int] = 252,
    risk_free: float = 0.0,
    ddof: int = 1,
) -> np.float64:
    """Sharpe target function.

    Args:
        returns (np.ndarray): 1d array of returns.
        periods (int, optional): Number of periods per year. ``None`` disables annualization.
            Defaults to 252 (annual).
        risk_free (float, optional): Annual risk free rate (0.02 is 2%), converted to a per period
            value by dividing by ``periods``. Defaults to 0.0.
        ddof (int, optional): Delta degrees of freedom of the standard deviation. Defaults to 1.

    Returns:
        np.float64: Sharpe ratio, ``np.nan`` when the dispersion is zero.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size <= ddof:
        return np.float64(np.nan)
    excess = returns - risk_free / periods if periods else returns - risk_free
    if _is_constant(excess):
        return np.float64(np.nan)
    std = excess.std(ddof=ddof)
    if not np.isfinite(std) or std == 0:
        return np.float64(np.nan)
    annualization = np.sqrt(1 if periods is None else periods)
    return np.float64(excess.mean() / std * annualization)


def sortino_target(
    returns: np.ndarray,
    periods: Optional[int] = 252,
    risk_free: float = 0.0,
    ddof: int = 1,
) -> np.float64:
    """Sortino target function.

    Args:
        returns (np.ndarray): 1d array of returns.
        periods (int, optional): Number of periods per year. ``None`` disables annualization.
            Defaults to 252 (annual).
        risk_free (float, optional): Annual risk free rate (0.02 is 2%). Defaults to 0.0.
        ddof (int, optional): Delta degrees of freedom of the downside deviation. Defaults to 1.

    Returns:
        np.float64: Sortino ratio, ``np.nan`` when there are no negative periods or the downside
        deviation is zero.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size <= ddof:
        return np.float64(np.nan)
    excess = returns - risk_free / periods if periods else returns - risk_free
    if _is_constant(excess):
        return np.float64(np.nan)
    downside = excess[excess < 0]
    if downside.size == 0:
        return np.float64(np.nan)
    deviation = np.sqrt(np.square(downside).sum() / (excess.size - ddof))
    if not np.isfinite(deviation) or deviation == 0:
        return np.float64(np.nan)
    annualization = np.sqrt(1 if periods is None else periods)
    return np.float64(excess.mean() / deviation * annualization)


def skew_target(returns: np.ndarray) -> np.float64:
    """Skew target function.

    Args:
        returns (np.ndarray): 1d array of returns.

    Returns:
        np.float64: Skew value, ``np.nan`` for a constant series.
    """
    returns = np.asarray(returns, dtype=float)
    if _is_constant(returns):
        return np.float64(np.nan)
    return np.float64(skew(returns))


def kurtosis_target(returns: np.ndarray) -> np.float64:
    """Kurtosis target function.

    Args:
        returns (np.ndarray): 1d array of returns.

    Returns:
        np.float64: Kurtosis value, ``np.nan`` for a constant series.
    """
    returns = np.asarray(returns, dtype=float)
    if _is_constant(returns):
        return np.float64(np.nan)
    return np.float64(kurtosis(returns))


def autocorrelation_target(
    returns: np.ndarray,
    absolute: bool = True,
    periods: int = 1,
) -> np.float64:
    """Autocorrelation target function.

    Args:
        returns (np.ndarray): 1d array of returns.
        absolute (bool): True to calculate the absolute value, otherwise the relative one.
            Defaults to True.
        periods (int, optional): Number of periods to shift. Defaults to 1.

    Returns:
        np.float64: Autocorrelation value, ``np.nan`` for a constant or too short series.

    Raises:
        ValueError: If ``periods`` is lower than 1.
    """
    if periods < 1:
        msg = f"periods must be >= 1, got {periods}"
        raise ValueError(msg)
    returns = np.asarray(returns, dtype=float)
    if returns.size <= periods:
        return np.float64(np.nan)
    left, right = returns[:-periods], returns[periods:]
    if _is_constant(left) or _is_constant(right):
        return np.float64(np.nan)
    result = pearsonr(left, right)[0]
    if not np.isfinite(result):
        return np.float64(np.nan)
    return np.float64(np.absolute(result) if absolute else result)


def cagr_target(returns: np.ndarray, days_per_year: int = 252) -> np.float64:
    """Compound annual growth rate target function.

    Args:
        returns (np.ndarray): 1d array of returns.
        days_per_year (int, optional): Number of periods per year. Defaults to 252 (annual).

    Returns:
        np.float64: CAGR value, ``np.nan`` for empty input.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0 or days_per_year <= 0:
        return np.float64(np.nan)
    prices = return_to_price(returns)
    periods = returns.size / days_per_year
    if prices[0] == 0:
        return np.float64(np.nan)
    return np.float64(np.power(prices[-1] / prices[0], 1 / periods) - 1)


def _column_scale(matrix: np.ndarray) -> np.ndarray:
    """Per column magnitude used as the reference scale for the constant series check."""
    return np.maximum(np.max(np.abs(matrix), axis=0), 1.0)


def _constant_mask(matrix: np.ndarray) -> np.ndarray:
    """Return a boolean mask of columns without dispersion."""
    return np.std(matrix, axis=0) <= _CONSTANT_TOLERANCE * _column_scale(matrix)


def _growth(matrix: np.ndarray) -> np.ndarray:
    """Cumulative growth of every column, starting from the initial capital (1.0).

    The leading row of ones mirrors :func:`bfportf.utils.return_to_price`, so drawdown metrics
    include the very first period (a loss in the first period is a real drawdown from the start).
    """
    ones = np.ones((1, matrix.shape[1]), dtype=float)
    return np.cumprod(np.vstack([ones, 1.0 + matrix]), axis=0)


def pnl_target_matrix(matrix: np.ndarray) -> np.ndarray:
    """Vectorized :func:`pnl_target`."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[0] == 0:
        return np.full(matrix.shape[1], np.nan)
    prices = _growth(matrix)
    return prices[-1] / prices[0] - 1.0


def cagr_target_matrix(matrix: np.ndarray, days_per_year: int = 252) -> np.ndarray:
    """Vectorized :func:`cagr_target`."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[0] == 0 or days_per_year <= 0:
        return np.full(matrix.shape[1], np.nan)
    growth = _growth(matrix)[-1]
    periods = matrix.shape[0] / days_per_year
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.power(growth, 1.0 / periods) - 1.0
    return np.where(np.isfinite(result), result, np.nan)


def max_drawdown_target_matrix(matrix: np.ndarray) -> np.ndarray:
    """Vectorized :func:`max_drawdown_target`."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[0] == 0:
        return np.full(matrix.shape[1], np.nan)
    prices = _growth(matrix)
    peak = np.maximum.accumulate(prices, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        drawdowns = prices / peak - 1.0
    return np.where(np.isfinite(drawdowns), drawdowns, np.nan).min(axis=0)


def profit_factor_target_matrix(matrix: np.ndarray) -> np.ndarray:
    """Vectorized :func:`profit_factor_target`."""
    matrix = np.asarray(matrix, dtype=float)
    gains = np.where(matrix >= 0, matrix, 0.0).sum(axis=0)
    losses = -np.where(matrix < 0, matrix, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = gains / losses
    return np.where((gains > 0) & (losses > 0), result, np.nan)


def recovery_factor_target_matrix(matrix: np.ndarray) -> np.ndarray:
    """Vectorized :func:`recovery_factor_target`."""
    pnl = pnl_target_matrix(matrix)
    drawdown = max_drawdown_target_matrix(matrix)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = pnl / np.abs(drawdown)
    return np.where(np.isfinite(drawdown) & (drawdown != 0), result, np.nan)


def sharpe_target_matrix(matrix: np.ndarray, periods: int = 252, ddof: int = 1) -> np.ndarray:
    """Vectorized :func:`sharpe_target` (without a risk free rate)."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[0] <= ddof:
        return np.full(matrix.shape[1], np.nan)
    std = matrix.std(axis=0, ddof=ddof)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = matrix.mean(axis=0) / std * np.sqrt(periods)
    valid = np.isfinite(result) & (std > 0) & ~_constant_mask(matrix)
    return np.where(valid, result, np.nan)


def sortino_target_matrix(matrix: np.ndarray, periods: int = 252, ddof: int = 1) -> np.ndarray:
    """Vectorized :func:`sortino_target` (without a risk free rate)."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[0] <= ddof:
        return np.full(matrix.shape[1], np.nan)
    negative = np.where(matrix < 0, matrix, 0.0)
    deviation = np.sqrt(np.square(negative).sum(axis=0) / (matrix.shape[0] - ddof))
    with np.errstate(invalid="ignore", divide="ignore"):
        result = matrix.mean(axis=0) / deviation * np.sqrt(periods)
    valid = (matrix < 0).any(axis=0) & np.isfinite(result) & (deviation > 0) & ~_constant_mask(matrix)
    return np.where(valid, result, np.nan)


def skew_target_matrix(matrix: np.ndarray) -> np.ndarray:
    """Vectorized :func:`skew_target`."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[0] < _MIN_SERIES_SIZE:
        return np.full(matrix.shape[1], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.asarray(skew(matrix, axis=0), dtype=float)
    return np.where(_constant_mask(matrix) | ~np.isfinite(result), np.nan, result)


def kurtosis_target_matrix(matrix: np.ndarray) -> np.ndarray:
    """Vectorized :func:`kurtosis_target`."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[0] < _MIN_SERIES_SIZE:
        return np.full(matrix.shape[1], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.asarray(kurtosis(matrix, axis=0), dtype=float)
    return np.where(_constant_mask(matrix) | ~np.isfinite(result), np.nan, result)
