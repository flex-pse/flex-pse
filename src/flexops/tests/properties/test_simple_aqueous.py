"""Tests for SimpleAqueousFlow: the minimal flow-carrying property package."""

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits
from pyomo.util.check_units import assert_units_consistent, assert_units_equivalent

from flexops.properties.simple_aqueous import SimpleAqueousFlow


@pytest.fixture
def model():
    """A ConcreteModel carrying a fixed-density SimpleAqueousFlow package."""
    m = pyo.ConcreteModel()
    m.props = SimpleAqueousFlow(fixed_density=True)
    return m


@pytest.mark.unit
def test_build_parameter_and_state_block(model):
    """A state block builds and carries flow_vol in m^3/hr."""
    model.state = model.props.build_state_block([0])
    assert model.state[0].flow_vol.is_indexed() is False
    assert_units_equivalent(model.state[0].flow_vol, pyunits.m**3 / pyunits.hr)


@pytest.mark.unit
def test_define_state_vars(model):
    """define_state_vars exposes exactly flow_vol."""
    model.state = model.props.build_state_block([0])
    assert set(model.state[0].define_state_vars().keys()) == {"flow_vol"}


@pytest.mark.unit
def test_fixed_density_param(model):
    """A fixed-density package carries dens_mass ~ 1000 kg/m^3."""
    assert pyo.value(model.props.dens_mass) == pytest.approx(1000.0, rel=1e-6)


@pytest.mark.unit
def test_units_consistent(model):
    """The state block is dimensionally consistent."""
    model.state = model.props.build_state_block([0])
    assert_units_consistent(model.state[0])


@pytest.mark.unit
def test_optional_pressure_temperature():
    """Pressure/temperature are opt-in intensive state vars with correct units."""
    m = pyo.ConcreteModel()
    m.props = SimpleAqueousFlow(has_pressure=True, has_temperature=True)
    m.state = m.props.build_state_block([0])
    sb = m.state[0]
    assert set(sb.define_state_vars().keys()) == {
        "flow_vol",
        "pressure",
        "temperature",
    }
    assert_units_equivalent(sb.pressure, pyunits.Pa)
    assert_units_equivalent(sb.temperature, pyunits.K)
    assert_units_consistent(sb)
