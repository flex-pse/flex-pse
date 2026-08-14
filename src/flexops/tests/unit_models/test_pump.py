"""Harness-driven tests for Pump(SISOBlock)."""

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits

from flexcore.config.schema import SurrogateSpec
from flexcore.exceptions import FlexConfigError
from flexops import SimpleAqueousFlow, TimeBlock
from flexops.testing import UnitModelTestHarness, dummy_time_block
from flexops.unit_models import Pump


class TestPump(UnitModelTestHarness):
    """Fixed inlet flow determines power_electrical via energy_intensity.

    The relation is metered on the outlet flow, which the SISO pass-through
    ties to the fixed inlet flow, so the solved draw is the same either way.
    """

    expected_dof = 0
    expected_solution = {
        "power_electrical[0]": 50.0,
        "power_electrical[1]": 50.0,
        "power_electrical[2]": 50.0,
    }

    def configure(self):
        m = dummy_time_block(3)
        m.unit = Pump(property_package=m.properties)
        for t in m.time_block.time_index:
            m.unit.inlet_state.flow_vol_phase[t, "Liq"].set_value(100.0)
        return m, m.unit


class TestPumpHydraulic(UnitModelTestHarness):
    """Fixed inlet flow/pressures determine power_electrical via the hydraulic
    relation.

    ``power = delta_pressure * flow / efficiency``, with ``delta_pressure =
    outlet.pressure - inlet.pressure``. With ``inlet.pressure=101325 Pa``,
    ``outlet.pressure=301325 Pa`` (a ``delta_pressure`` of 200_000 Pa),
    ``efficiency=0.8``, and inlet flow ``100 m^3/hr = 100/3600 m^3/s``,
    ``power = 200_000 * (100/3600) / 0.8`` W ``= 6944.44...`` W
    ``= 6.94444... kW``.
    """

    expected_dof = 0
    expected_solution = {
        "power_electrical[0]": 6.9444444444,
        "power_electrical[1]": 6.9444444444,
        "power_electrical[2]": 6.9444444444,
    }

    def configure(self):
        m = pyo.ConcreteModel()
        m.time_block = TimeBlock(
            start_date="2025-01-01",
            end_date="2025-01-01T00:45",
            time_step=15 * pyunits.min,
        )
        m.properties = SimpleAqueousFlow(has_pressure=True)
        m.unit = Pump(
            property_package=m.properties,
            power_relation="hydraulic",
            efficiency=0.8,
        )
        for t in m.time_block.time_index:
            m.unit.inlet_state.flow_vol_phase[t, "Liq"].set_value(100.0)
            m.unit.inlet_state.pressure[t].set_value(101325.0)
            m.unit.outlet_state.pressure[t].set_value(301325.0)
        return m, m.unit


@pytest.mark.unit
def test_hydraulic_pump_requires_has_pressure():
    """power_relation='hydraulic' on a no-pressure property package errors loudly."""
    m = dummy_time_block(3)
    with pytest.raises(FlexConfigError):
        m.unit = Pump(property_package=m.properties, power_relation="hydraulic")


@pytest.mark.unit
def test_energy_intensity_is_metered_on_the_outlet_flow():
    """The constant-intensity draw follows the flow the pump delivers."""
    m = dummy_time_block(3)
    m.unit = Pump(property_package=m.properties)
    m.unit.inlet_state.flow_vol_phase[0, "Liq"].set_value(100.0)
    m.unit.outlet_state.flow_vol_phase[0, "Liq"].set_value(40.0)
    m.unit.power_electrical[0].set_value(0.0)

    # power - 0.5 kWh/m^3 * 40 m^3/hr == -20.0 kW
    assert pyo.value(m.unit.power_electrical_relation[0].body) == pytest.approx(-20.0)


@pytest.mark.unit
def test_pump_energy_relation_is_swappable():
    """Folding onto add_constant_intensity_relation makes the relation swappable.

    Unlike the hydraulic power law (a physical relationship, not an energy
    intensity, and never registered), the default relation now goes through
    the same registration/swap path as every other constant-intensity unit.
    """
    m = dummy_time_block(3)
    m.unit = Pump(property_package=m.properties)

    m.unit.swap_relation(
        "power_electrical_relation",
        SurrogateSpec(
            functional_form="linear",
            input_variables=["flow_out"],
            coefficients={"flow_out": 2.0, "intercept": 1.0},
        ),
    )

    assert m.unit.power_electrical_relation[0].active is False
    assert m.unit.find_component("power_electrical_relation_fitted") is not None
