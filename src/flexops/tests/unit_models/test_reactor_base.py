"""Tests for ReactorBase: shared ports, intensive states, power, thermal lag."""

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits
from pyomo.opt import assert_optimal_termination

from flexcore.exceptions import FlexConfigError
from flexcore.solvers import get_solver
from flexops.properties.simple_aqueous import SimpleAqueousFlow
from flexops.testing import dummy_time_block
from flexops.unit_models.reactor import LumpedReactor, ReactorBase, SpeciesReactor


def _thermal_reactor(n: int = 4, **kwargs):
    """Build a thermal LumpedReactor on a temperature-carrying aqueous package."""
    m = dummy_time_block(n)
    m.warm = SimpleAqueousFlow(has_temperature=True)
    m.unit = LumpedReactor(
        property_package=m.warm,
        has_thermal_dynamics=True,
        thermal_time_constant=1 * pyunits.hr,
        thermal_gain=0.5 * pyunits.K / pyunits.kW,
        initial_temperature=300 * pyunits.K,
        energy_intensity=1.0 * pyunits.kWh / pyunits.m**3,
        **kwargs,
    )
    return m, m.unit


@pytest.mark.unit
def test_bare_reactor_base_cannot_be_built():
    m = dummy_time_block(3)
    with pytest.raises(NotImplementedError):
        m.unit = ReactorBase(property_package=m.properties)


@pytest.mark.unit
@pytest.mark.parametrize(
    "field, names",
    [("inlet_names", ("a", "a")), ("outlet_names", ()), ("inlet_names", ("",))],
)
def test_invalid_port_names_rejected(field, names):
    m = dummy_time_block(3)
    with pytest.raises(FlexConfigError):
        m.unit = SpeciesReactor(property_package=m.properties, **{field: names})


@pytest.mark.unit
def test_thermal_dynamics_require_a_temperature_state():
    m = dummy_time_block(3)
    with pytest.raises(FlexConfigError, match="temperature"):
        m.unit = LumpedReactor(property_package=m.properties, has_thermal_dynamics=True)


@pytest.mark.unit
def test_thermal_relations_and_initial_state_registered():
    m, unit = _thermal_reactor()
    relations = {rec.name for rec in unit._io_registry.relations}
    assert {"thermal_steady_state_relation", "power_electrical_relation"} <= relations
    assert any(p is unit.initial_temperature for p in m.time_block.initial_state_params)
    params = {rec.name: rec.regressable for rec in unit._io_registry.parameters}
    assert params["thermal_time_constant"] is True
    assert params["thermal_gain"] is True
    assert params["initial_temperature"] is False


@pytest.mark.unit
def test_thermal_dynamics_off_builds_no_temperature_state():
    m = dummy_time_block(3)
    m.unit = SpeciesReactor(property_package=m.properties)
    assert m.unit.find_component("reactor_temperature") is None


@pytest.mark.unit
def test_non_reference_inlet_temperature_tied_to_reference():
    m = dummy_time_block(3)
    m.warm = SimpleAqueousFlow(has_temperature=True)
    m.unit = SpeciesReactor(property_package=m.warm, inlet_names=("a", "b"))
    assert m.unit.find_component("inlet_state_equality_temperature") is not None


@pytest.mark.component
@pytest.mark.needs_highs
def test_thermal_lag_step_response_matches_backward_euler():
    m, unit = _thermal_reactor(n=5)
    tb = m.time_block
    for t in tb.time_index:
        unit.flow_in_feed[t].fix(10.0)
        unit.flow_out_product[t].fix(10.0)
        unit.inlet_feed_state.temperature[t].fix(300.0)
    m.obj = pyo.Objective(expr=0)
    assert_optimal_termination(get_solver(model=m, prefer="highs").solve(m))

    # P = 1 kWh/m^3 * 10 m^3/hr = 10 kW; T_ss = 300 + 0.5*10 = 305 K; a = dt/tau.
    a = 0.25
    expected = [300.0]
    for _ in range(1, 5):
        expected.append((expected[-1] + a * 305.0) / (1 + a))
    for t in tb.time_index:
        assert pyo.value(unit.reactor_temperature[t]) == pytest.approx(expected[t])
        assert pyo.value(unit.outlet_product_state.temperature[t]) == pytest.approx(
            expected[t]
        )
