"""Tests for the public API surface."""

from pathlib import Path

import pandas as pd
import pytest

import bfportf
from bfportf import (
    METRICS,
    PortfolioResult,
    __version__,
    autocorrelation_target,
    create_report,
    find_portfolios,
    load_returns,
    resolve_metric,
    sanitize_returns,
    targets,
    validate_returns,
)
from bfportf.metrics import basic
from bfportf.optimization import find_portfolios as legacy_find_portfolios

EXPECTED_EXPORTS = [
    "BeamSearch",
    "DataValidationError",
    "ExhaustiveSearch",
    "GreedySearch",
    "METRICS",
    "Metric",
    "PortfolioResult",
    "SearchConfig",
    "SearchStrategy",
    "UnknownSymbolError",
    "create_report",
    "find_portfolios",
    "get_portfolio_return",
    "load_returns",
    "resolve_metric",
    "results_to_frame",
    "scale_return_data",
    "sharpe_target",
    "validate_returns",
]


def test_version_is_exposed() -> None:
    assert isinstance(__version__, str)
    assert __version__.count(".") == 2


def test_all_names_are_importable() -> None:
    assert bfportf.__all__
    for name in bfportf.__all__:
        assert hasattr(bfportf, name), name


def test_expected_names_are_exported() -> None:
    for name in EXPECTED_EXPORTS:
        assert name in bfportf.__all__, name


def test_targets_module_is_a_backward_compatible_alias() -> None:
    assert targets.sharpe_target is basic.sharpe_target
    assert targets.max_drawdown_target is basic.max_drawdown_target


def test_optimization_module_keeps_find_portfolios() -> None:
    assert legacy_find_portfolios is find_portfolios


def test_py_typed_marker_exists() -> None:
    package_dir = Path(bfportf.__file__).parent
    assert (package_dir / "py.typed").exists()


def test_public_api_is_usable() -> None:
    assert METRICS
    assert callable(find_portfolios)
    assert callable(create_report)
    assert PortfolioResult.__dataclass_fields__


def test_optional_arguments_are_keyword_only(returns: pd.DataFrame) -> None:
    # Options must be passed by name, so a bare `True` cannot land in the wrong slot
    with pytest.raises(TypeError):
        find_portfolios(returns, "sharpe", None)
    with pytest.raises(TypeError):
        find_portfolios(returns, "sharpe", True)
    with pytest.raises(TypeError):
        load_returns(returns, False)
    with pytest.raises(TypeError):
        sanitize_returns(returns, False)
    with pytest.raises(TypeError):
        validate_returns(returns, True)
    with pytest.raises(TypeError):
        resolve_metric("sharpe", False)
    with pytest.raises(TypeError):
        autocorrelation_target(returns.iloc[:, 0].to_numpy(), False)


def test_keyword_only_arguments_still_work(returns: pd.DataFrame) -> None:
    found = find_portfolios(returns, "sharpe", max_number=1, verbose=False, validate=False, maximize=True)
    assert found

    validate_returns(returns, allow_nan=False)
    assert not sanitize_returns(returns, drop_nan_rows=False).isna().any().any()
    assert resolve_metric("sharpe", maximize=True) is METRICS["sharpe"]

    values = returns.iloc[:, 0].to_numpy()
    assert autocorrelation_target(values, absolute=True, periods=1) >= 0
