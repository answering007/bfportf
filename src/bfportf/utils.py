"""Price/return conversions and drawdown helpers."""

import numpy as np


def return_to_price(returns: np.ndarray, first_value: np.ndarray | None = None) -> np.ndarray:
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
