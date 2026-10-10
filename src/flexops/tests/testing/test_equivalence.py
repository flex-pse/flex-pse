"""assert_models_equivalent: structural comparison of two Pyomo models."""

import shutil
from pathlib import Path

import pyomo.environ as pyo
import pytest

from flexcore.config.io import load_model_config
from flexops import build_model
from flexops.testing import assert_models_equivalent

_FREEZE = Path(__file__).parent.parent / "fixtures" / "api_freeze"


@pytest.fixture
def config(tmp_path, monkeypatch):
    """Return the API-freeze config as a dict, with its data in the cwd."""
    for path in (_FREEZE / "data").iterdir():
        shutil.copy(path, tmp_path)
    monkeypatch.chdir(tmp_path)
    return load_model_config(_FREEZE / "api_freeze_config.json").model_dump()


def _build(raw):
    """Build a model from a raw config dict."""
    return build_model(load_model_config(raw))


@pytest.mark.component
def test_equivalent_models_pass(config):
    """Building the same config twice gives equivalent models."""
    assert_models_equivalent(_build(config), _build(config))


@pytest.mark.component
def test_equivalence_detects_changed_bound(config):
    """A changed variable bound is reported with the variable's path."""
    a, b = _build(config), _build(config)
    b.waterfacility.tank.volume[0].setub(1.0)

    with pytest.raises(AssertionError, match=r"waterfacility\.tank\.volume\[0\]"):
        assert_models_equivalent(a, b)


@pytest.mark.component
def test_equivalence_detects_fixed_vs_unfixed(config):
    """A var fixed in one model and free in the other is reported."""
    a, b = _build(config), _build(config)
    b.waterfacility.plant.power_electrical[0].fix(1.0)

    with pytest.raises(AssertionError, match=r"power_electrical\[0\].*fixed"):
        assert_models_equivalent(a, b)


@pytest.mark.component
def test_equivalence_detects_changed_coefficient(config):
    """A different energy_intensity changes a constraint body."""
    other = _build(config)
    config["plant"]["units"]["plant"]["construction_options"]["energy_intensity"][
        "value"
    ] = 0.9

    with pytest.raises(AssertionError, match=r"waterfacility\.plant\."):
        assert_models_equivalent(_build(config), other)


@pytest.mark.component
def test_equivalence_detects_extra_constraint(config):
    """An added constraint is reported as present in only one model."""
    a, b = _build(config), _build(config)
    b.waterfacility.extra_limit = pyo.Constraint(
        expr=b.waterfacility.tank.volume[0] <= 5
    )

    with pytest.raises(AssertionError, match=r"Only in b: waterfacility\.extra_limit"):
        assert_models_equivalent(a, b)


@pytest.mark.component
def test_equivalence_detects_objective_sense(config):
    """A flipped objective sense is reported."""
    a, b = _build(config), _build(config)
    b.objective.sense = pyo.maximize

    with pytest.raises(AssertionError, match="objective.*sense"):
        assert_models_equivalent(a, b)


@pytest.mark.component
def test_equivalence_truncates_long_reports(config):
    """Only the first 50 differences are shown, with a count of the rest."""
    a, b = _build(config), _build(config)
    for var in b.waterfacility.tank.component_data_objects(pyo.Var):
        var.setub(123.0)
    for var in b.waterfacility.battery.component_data_objects(pyo.Var):
        var.setub(123.0)

    with pytest.raises(AssertionError, match=r"\.\.\. and \d+ more"):
        assert_models_equivalent(a, b)
