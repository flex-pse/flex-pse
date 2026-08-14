"""Exchanger: harness subclass on the DIDOBlock topology base (§3.4)."""

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits

from flexops.testing import UnitModelTestHarness, dummy_time_block
from flexops.unit_models import Exchanger


class TestExchanger(UnitModelTestHarness):
    """Two inlet / two outlet streams exchanging mass, with an electrical draw."""

    expected_dof = 0

    def configure(self):
        m = dummy_time_block(3)
        m.unit = Exchanger(
            property_package=m.properties,
            transfer_fraction=0.2,
            energy_intensity=0.1 * pyunits.kWh / pyunits.m**3,
        )
        return m, m.unit


@pytest.mark.unit
def test_energy_intensity_is_metered_on_outlet_a():
    """A two-outlet unit meters its draw against outlet a, not inlet a."""
    m = dummy_time_block(3)
    m.unit = Exchanger(
        property_package=m.properties,
        transfer_fraction=0.2,
        energy_intensity=0.1 * pyunits.kWh / pyunits.m**3,
    )
    m.unit.flow_in_a[0].set_value(10.0)
    m.unit.flow_out_a[0].set_value(4.0)
    m.unit.power_electrical[0].set_value(0.0)

    # power - 0.1 * flow_out_a == -0.4
    assert pyo.value(m.unit.power_electrical_relation[0].body) == pytest.approx(-0.4)
