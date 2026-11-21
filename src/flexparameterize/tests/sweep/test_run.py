"""Tests for run_sweep and solve_point argument checks (no solver)."""

import sys

import pytest

from flexcore.exceptions import FlexConfigError
from flexparameterize.sweep import run_sweep, solve_point
from flexparameterize.tests.sweep.helpers import (
    build_toy_model,
    initialize_toy,
    optimize_toy,
    toy_config,
)

TOY = (build_toy_model, initialize_toy, optimize_toy)


@pytest.mark.unit
@pytest.mark.parametrize(
    "bounds",
    [{"a": (0.0, 2e-3)}, {"b": (4e5, 6e5)}, {"a": (3e-3, 2e-3)}, {"c": (0.0, 1.0)}],
    ids=["below_range", "above_range", "inverted", "unknown_input"],
)
def test_bounds_outside_the_config_raise(bounds):
    """A bounds box must name config inputs and lie inside their ranges."""
    with pytest.raises(FlexConfigError, match="bounds"):
        run_sweep(toy_config(), *TOY, bounds=bounds)


@pytest.mark.unit
@pytest.mark.parametrize(
    "point",
    [{"a": 2e-3}, {"a": 2e-3, "b": 2e5, "c": 1.0}, {"a": 9e-3, "b": 2e5}],
    ids=["missing_input", "extra_input", "out_of_range"],
)
def test_bad_points_raise(point):
    """A point must give exactly the config inputs, inside their ranges."""
    with pytest.raises(FlexConfigError, match="point"):
        solve_point(toy_config(), *TOY, point)


@pytest.mark.unit
def test_missing_parameter_sweep_raises_with_install_hint(monkeypatch):
    """Without the parameter-sweep package the sweep says how to install it."""
    monkeypatch.setitem(sys.modules, "parameter_sweep", None)

    with pytest.raises(FlexConfigError, match=r"flex-pse\[sweep\]"):
        run_sweep(toy_config(), *TOY)
