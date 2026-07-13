"""Tests for SimpleGasFlow: the minimal gas-stream property package."""

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits
from pyomo.util.check_units import assert_units_consistent, assert_units_equivalent

from flexops.properties.simple_gas import SimpleGasFlow


@pytest.fixture
def model():
    """A ConcreteModel carrying a SimpleGasFlow package."""
    m = pyo.ConcreteModel()
    m.props = SimpleGasFlow()
    return m


@pytest.mark.unit
def test_build_parameter_and_state_block(model):
    """A state block builds and carries flow_vol in m^3/hr."""
    model.state = model.props.build_state_block([0])
    assert model.state[0].flow_vol.is_indexed() is False
    assert_units_equivalent(model.state[0].flow_vol, pyunits.m**3 / pyunits.hr)


@pytest.mark.unit
def test_define_state_vars(model):
    """define_state_vars exposes flow, density, pressure, and temperature."""
    model.state = model.props.build_state_block([0])
    assert set(model.state[0].define_state_vars().keys()) == {
        "flow_vol",
        "dens_mass",
        "pressure",
        "temperature",
    }


@pytest.mark.unit
def test_state_var_units(model):
    """Density, pressure, and temperature carry the expected units."""
    model.state = model.props.build_state_block([0])
    sb = model.state[0]
    assert_units_equivalent(sb.dens_mass, pyunits.kg / pyunits.m**3)
    assert_units_equivalent(sb.pressure, pyunits.Pa)
    assert_units_equivalent(sb.temperature, pyunits.K)


@pytest.mark.unit
def test_units_consistent(model):
    """The state block is dimensionally consistent."""
    model.state = model.props.build_state_block([0])
    assert_units_consistent(model.state[0])


@pytest.mark.unit
def test_initialize_fixes_and_releases(model):
    """initialize with hold_state fixes all four state vars; release unfixes."""
    model.state = model.props.build_state_block([0])
    sb = model.state[0]
    flags = model.state.initialize(hold_state=True)
    assert all(
        var.fixed for var in (sb.flow_vol, sb.dens_mass, sb.pressure, sb.temperature)
    )
    model.state.release_state(flags)
    assert not any(
        var.fixed for var in (sb.flow_vol, sb.dens_mass, sb.pressure, sb.temperature)
    )
