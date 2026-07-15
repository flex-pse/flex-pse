"""Tests for SimpleAqueousFlow: the minimal flow-carrying property package.

Mirrors the test skeleton of ``test_simple_gas.py``; package-specific extras
(density config options, opt-in pressure/temperature) come last.
"""

import pyomo.environ as pyo
import pytest
from idaes.core import Component, LiquidPhase
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
def test_phase_and_component(model):
    """The package auto-builds the Liq phase and the H2O component."""
    assert isinstance(model.props.Liq, LiquidPhase)
    assert isinstance(model.props.H2O, Component)
    assert list(model.props.phase_list) == ["Liq"]
    assert list(model.props.component_list) == ["H2O"]


@pytest.mark.unit
def test_define_state_vars(model):
    """define_state_vars exposes flow_vol and dens_mass by default."""
    model.state = model.props.build_state_block([0])
    assert set(model.state[0].define_state_vars().keys()) == {
        "flow_vol",
        "dens_mass",
    }


@pytest.mark.unit
def test_state_var_units(model):
    """Flow and density carry the expected units."""
    model.state = model.props.build_state_block([0])
    state_block = model.state[0]
    assert_units_equivalent(state_block.flow_vol, pyunits.m**3 / pyunits.hr)
    assert_units_equivalent(state_block.dens_mass, pyunits.kg / pyunits.m**3)


@pytest.mark.unit
def test_state_var_domains(model):
    """flow_vol is non-negative; dens_mass is strictly positive."""
    model.state = model.props.build_state_block([0])
    state_block = model.state[0]
    assert state_block.flow_vol.domain is pyo.NonNegativeReals
    assert state_block.dens_mass.domain is pyo.PositiveReals


@pytest.mark.unit
def test_metadata_properties_and_units(model):
    """Metadata declares the four properties and the five base default units."""
    meta = model.props.get_metadata()
    supported = {p.name for p in meta.properties.list_supported_properties()}
    assert {"flow_vol", "dens_mass", "pressure", "temperature"} <= supported
    assert meta.default_units["time"] == pyunits.hr
    assert meta.default_units["length"] == pyunits.m
    assert meta.default_units["mass"] == pyunits.kg
    assert meta.default_units["amount"] == pyunits.mol
    assert meta.default_units["temperature"] == pyunits.K


@pytest.mark.unit
def test_define_display_vars(model):
    """Display vars mirror the state vars."""
    model.state = model.props.build_state_block([0])
    state_block = model.state[0]
    assert state_block.define_display_vars() == state_block.define_state_vars()


@pytest.mark.unit
def test_units_consistent(model):
    """The state block is dimensionally consistent."""
    model.state = model.props.build_state_block([0])
    assert_units_consistent(model.state[0])


@pytest.mark.unit
def test_initialize_fixes_and_releases():
    """initialize with hold_state fixes the free state vars; release unfixes."""
    m = pyo.ConcreteModel()
    m.props = SimpleAqueousFlow(fixed_density=False)
    m.state = m.props.build_state_block([0])
    state_block = m.state[0]
    state_vars = state_block.define_state_vars().values()
    flags = m.state.initialize(hold_state=True)
    assert all(var.fixed for var in state_vars)
    m.state.release_state(flags)
    assert not any(var.fixed for var in state_vars)


@pytest.mark.unit
def test_fix_initialization_states(model):
    """fix_initialization_states fixes every state variable."""
    model.state = model.props.build_state_block([0])
    model.state.fix_initialization_states()
    for var in model.state[0].define_state_vars().values():
        assert var.fixed is True


# -- package-specific extras: density config and opt-in intensive states -----


@pytest.mark.unit
def test_fixed_density_fixes_dens_mass(model):
    """fixed_density=True builds dens_mass fixed at the configured density."""
    model.state = model.props.build_state_block([0])
    dens = model.state[0].dens_mass
    assert dens.fixed is True
    assert pyo.value(dens) == pytest.approx(1000.0, rel=1e-6)


@pytest.mark.unit
def test_custom_density():
    """A custom density sets the fixed dens_mass value."""
    m = pyo.ConcreteModel()
    m.props = SimpleAqueousFlow(density=998 * pyunits.kg / pyunits.m**3)
    m.state = m.props.build_state_block([0])
    assert m.state[0].dens_mass.fixed is True
    assert pyo.value(m.state[0].dens_mass) == pytest.approx(998.0, rel=1e-6)


@pytest.mark.unit
def test_free_density():
    """fixed_density=False leaves dens_mass an unfixed state variable."""
    m = pyo.ConcreteModel()
    m.props = SimpleAqueousFlow(fixed_density=False)
    m.state = m.props.build_state_block([0])
    dens = m.state[0].dens_mass
    assert dens.fixed is False
    assert "dens_mass" in m.state[0].define_state_vars()


@pytest.mark.unit
def test_optional_pressure_temperature():
    """Pressure/temperature are opt-in intensive state vars with correct units."""
    m = pyo.ConcreteModel()
    m.props = SimpleAqueousFlow(has_pressure=True, has_temperature=True)
    m.state = m.props.build_state_block([0])
    state_block = m.state[0]
    assert set(state_block.define_state_vars().keys()) == {
        "flow_vol",
        "dens_mass",
        "pressure",
        "temperature",
    }
    assert state_block.pressure.domain is pyo.PositiveReals
    assert state_block.temperature.domain is pyo.PositiveReals
    assert_units_equivalent(state_block.pressure, pyunits.Pa)
    assert_units_equivalent(state_block.temperature, pyunits.K)
    assert_units_consistent(state_block)
