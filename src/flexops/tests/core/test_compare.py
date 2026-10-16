"""Tests for flexops.core.compare: model_differences and assert_models_equivalent."""

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits

from flexops.core.compare import assert_models_equivalent, model_differences
from flexops.testing import dummy_time_block
from flexops.unit_models import ConstantEnergyIntensityModel, Tank


def build(intensity: float = 0.5, n_points: int = 3):
    """Return a small model with a Tank, a constant-intensity unit and an objective."""
    m = dummy_time_block(n_points)
    m.tank = Tank(property_package=m.properties)
    m.plant = ConstantEnergyIntensityModel(
        property_package=m.properties,
        energy_intensity=intensity * pyunits.kWh / pyunits.m**3,
    )
    m.objective = pyo.Objective(expr=m.plant.power_electrical[0])
    return m


@pytest.mark.unit
def test_identical_models_have_no_differences():
    assert model_differences(build(), build()) == []
    assert_models_equivalent(build(), build())


@pytest.mark.unit
def test_extra_constraint_is_listed():
    a, b = build(), build()
    b.extra = pyo.Constraint(expr=b.tank.flow_in[0] >= 1)

    assert model_differences(a, b) == ["Only in b: extra (Constraint)"]


@pytest.mark.unit
def test_changed_bound_is_reported():
    a, b = build(), build()
    b.tank.volume[0].setub(1.0)

    with pytest.raises(AssertionError, match=r"tank\.volume\[0\]"):
        assert_models_equivalent(a, b)


@pytest.mark.unit
def test_fixed_vs_unfixed_is_reported():
    a, b = build(), build()
    b.plant.power_electrical[0].fix(1.0)

    with pytest.raises(AssertionError, match=r"power_electrical\[0\].*fixed"):
        assert_models_equivalent(a, b)


@pytest.mark.unit
def test_changed_coefficient_is_reported():
    with pytest.raises(AssertionError, match=r"plant\."):
        assert_models_equivalent(build(0.9), build())


@pytest.mark.unit
def test_objective_sense_is_reported():
    a, b = build(), build()
    b.objective.sense = pyo.maximize

    with pytest.raises(AssertionError, match="objective.*sense"):
        assert_models_equivalent(a, b)


@pytest.mark.unit
def test_long_reports_are_truncated():
    a, b = build(n_points=10), build(n_points=10)
    for var in b.component_data_objects(pyo.Var):
        var.setub(123.0)

    with pytest.raises(AssertionError, match=r"\.\.\. and \d+ more"):
        assert_models_equivalent(a, b)


@pytest.mark.unit
def test_compare_module_does_not_import_pytest():
    import flexops.core.compare as compare

    assert "pytest" not in vars(compare)
