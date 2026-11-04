"""Tests for the public API surface."""

from pathlib import Path

import bfportf
from bfportf import METRICS, PortfolioResult, __version__, create_report, find_portfolios, targets
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
