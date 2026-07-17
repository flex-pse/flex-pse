"""Tests for SimpleGasFlow: the minimal gas-stream property package.

Mirrors the test skeleton of ``test_simple_aqueous.py``; the gas package has no
config options, so there are no package-specific extras.

State variables are indexed over time directly (``flow_vol_phase[t, phase]``,
``dens_mass[t]`` and so on); the owning unit passes a ``time_index`` Set and
gets a single scalar state block rather than one block per time point.
"""

import pyomo.environ as pyo
import pytest
from idaes.core import Component, VaporPhase
from pyomo.environ import units as pyunits
from pyomo.util.check_units import assert_units_consistent, assert_units_equivalent

from flexcore.exceptions import FlexConfigError
from flexops.properties.simple_gas import SimpleGasFlow

TIMES = [0, 1, 2]


@pytest.fixture
def model():
    """A ConcreteModel with a SimpleGasFlow package and a 3-point time set."""
    m = pyo.ConcreteModel()
    m.props = SimpleGasFlow()
    m.time = pyo.Set(initialize=TIMES, ordered=True)
    return m


@pytest.mark.unit
def test_build_parameter_and_state_block(model):
    """A state block builds flow_vol_phase indexed by (time, phase) in m^3/hr."""
    model.state = model.props.build_state_block(time_index=model.time)
    flow = model.state.flow_vol_phase
    assert set(flow.index_set()) == {(t, "Vap") for t in TIMES}
    assert_units_equivalent(flow[0, "Vap"], pyunits.m**3 / pyunits.hr)


@pytest.mark.unit
def test_missing_time_index_raises(model):
    """Building a state block without a time_index is a config error."""
    with pytest.raises(FlexConfigError, match="time_index"):
        model.state = model.props.build_state_block()


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
    model.state = model.props.build_state_block(time_index=model.time)
    assert set(model.state.define_state_vars().keys()) == {
        "flow_vol_phase",
        "dens_mass",
        "pressure",
        "temperature",
    }


@pytest.mark.unit
def test_state_var_units(model):
    """Flow, density, pressure, and temperature carry the expected units."""
    model.state = model.props.build_state_block(time_index=model.time)
    state_block = model.state
    assert_units_equivalent(
        state_block.flow_vol_phase[0, "Vap"], pyunits.m**3 / pyunits.hr
    )
    assert_units_equivalent(state_block.dens_mass[0], pyunits.kg / pyunits.m**3)
    assert_units_equivalent(state_block.pressure[0], pyunits.Pa)
    assert_units_equivalent(state_block.temperature[0], pyunits.K)


@pytest.mark.unit
def test_state_var_domains(model):
    """flow_vol_phase is non-negative; the intensive states are strictly positive."""
    model.state = model.props.build_state_block(time_index=model.time)
    state_block = model.state
    assert state_block.flow_vol_phase[0, "Vap"].domain is pyo.NonNegativeReals
    for var in (
        state_block.dens_mass[0],
        state_block.pressure[0],
        state_block.temperature[0],
    ):
        assert var.domain is pyo.PositiveReals


@pytest.mark.unit
def test_metadata_properties_and_units(model):
    """Metadata declares the four properties and the five base default units."""
    meta = model.props.get_metadata()
    supported = {p.name for p in meta.properties.list_supported_properties()}
    assert {"flow_vol_phase", "dens_mass", "pressure", "temperature"} <= supported
    assert meta.default_units["time"] == pyunits.hr
    assert meta.default_units["length"] == pyunits.m
    assert meta.default_units["mass"] == pyunits.kg
    assert meta.default_units["amount"] == pyunits.mol
    assert meta.default_units["temperature"] == pyunits.K


@pytest.mark.unit
def test_define_display_vars(model):
    """Display vars mirror the state vars."""
    model.state = model.props.build_state_block(time_index=model.time)
    state_block = model.state
    assert state_block.define_display_vars() == state_block.define_state_vars()


@pytest.mark.unit
def test_units_consistent(model):
    """The state block is dimensionally consistent."""
    model.state = model.props.build_state_block(time_index=model.time)
    assert_units_consistent(model.state)


@pytest.mark.unit
def test_initialize_fixes_and_releases(model):
    """initialize with hold_state fixes all four state vars; release unfixes."""
    model.state = model.props.build_state_block(time_index=model.time)
    state_block = model.state
    state_data = [
        vd for var in state_block.define_state_vars().values() for vd in var.values()
    ]
    flags = model.state.initialize(hold_state=True)
    assert all(vd.fixed for vd in state_data)
    model.state.release_state(flags)
    assert not any(vd.fixed for vd in state_data)


@pytest.mark.unit
def test_fix_initialization_states(model):
    """fix_initialization_states fixes every state variable."""
    model.state = model.props.build_state_block(time_index=model.time)
    model.state.fix_initialization_states()
    for var in model.state.define_state_vars().values():
        for vd in var.values():
            assert vd.fixed is True
