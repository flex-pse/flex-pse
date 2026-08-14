"""ConstantEnergyIntensityModel: harness + the swap-contract constraint name."""

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits

from flexops.testing import UnitModelTestHarness, dummy_time_block
from flexops.unit_models import ConstantEnergyIntensityModel


def _surrogate(n: int = 3, **kwargs):
    """Build a ConstantEnergyIntensityModel on an ``n``-point dummy model."""
    m = dummy_time_block(n)
    m.unit = ConstantEnergyIntensityModel(property_package=m.properties, **kwargs)
    return m, m.unit


class TestConstantEnergyIntensityModel(UnitModelTestHarness):
    """Generic energy-factor-times-flow unit; inlet flow is the only input."""

    expected_dof = 0

    def configure(self):
        return _surrogate(3, energy_intensity=0.5 * pyunits.kWh / pyunits.m**3)


@pytest.mark.unit
def test_power_electrical_relation_constraint_is_named():
    """The energy relation carries the documented swappable name (R11, M10)."""
    _, unit = _surrogate(3)
    assert unit.find_component("power_electrical_relation") is not None


@pytest.mark.unit
def test_energy_intensity_is_metered_on_the_outlet_flow():
    """The draw follows the product (outlet) flow, not what came in."""
    _, unit = _surrogate(3, energy_intensity=0.5 * pyunits.kWh / pyunits.m**3)
    unit.flow_in[0].set_value(10.0)
    unit.flow_out[0].set_value(4.0)
    unit.power_electrical[0].set_value(0.0)

    # power - 0.5 * flow_out == -2.0; flow_in does not enter the relation.
    assert pyo.value(unit.power_electrical_relation[0].body) == pytest.approx(-2.0)
