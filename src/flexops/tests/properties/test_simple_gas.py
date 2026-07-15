"""Tests for SimpleGasFlow: the minimal gas-stream property package.

Mirrors the test skeleton of ``test_simple_aqueous.py``; the gas package has no
config options, so there are no package-specific extras.
"""

import pyomo.environ as pyo
import pytest
from idaes.core import Component, VaporPhase
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
def test_phase_and_component(model):
    """The package auto-builds the Vap phase and the gas component."""
    assert isinstance(model.props.Vap, VaporPhase)
    assert isinstance(model.props.gas, Component)
    assert list(model.props.phase_list) == ["Vap"]
    assert list(model.props.component_list) == ["gas"]


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
    """Flow, density, pressure, and temperature carry the expected units."""
    model.state = model.props.build_state_block([0])
    state_block = model.state[0]
    assert_units_equivalent(state_block.flow_vol, pyunits.m**3 / pyunits.hr)
    assert_units_equivalent(state_block.dens_mass, pyunits.kg / pyunits.m**3)
    assert_units_equivalent(state_block.pressure, pyunits.Pa)
    assert_units_equivalent(state_block.temperature, pyunits.K)


@pytest.mark.unit
def test_state_var_domains(model):
    """flow_vol is non-negative; the intensive states are strictly positive."""
    model.state = model.props.build_state_block([0])
    state_block = model.state[0]
    assert state_block.flow_vol.domain is pyo.NonNegativeReals
    for var in (state_block.dens_mass, state_block.pressure, state_block.temperature):
        assert var.domain is pyo.PositiveReals


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
def test_initialize_fixes_and_releases(model):
    """initialize with hold_state fixes all four state vars; release unfixes."""
    model.state = model.props.build_state_block([0])
    state_block = model.state[0]
    state_vars = state_block.define_state_vars().values()
    flags = model.state.initialize(hold_state=True)
    assert all(var.fixed for var in state_vars)
    model.state.release_state(flags)
    assert not any(var.fixed for var in state_vars)


@pytest.mark.unit
def test_fix_initialization_states(model):
    """fix_initialization_states fixes every state variable."""
    model.state = model.props.build_state_block([0])
    model.state.fix_initialization_states()
    for var in model.state[0].define_state_vars().values():
        assert var.fixed is True
