"""SimpleUtility: the property block a utility takes its name and units from."""

import pyomo.environ as pyo
import pytest

from flexcore.exceptions import FlexConfigError
from flexops import Resource, SimpleUtility
from flexops.properties import PROPERTY_PACKAGES
from flexops.testing import dummy_time_block


@pytest.mark.unit
def test_a_utility_package_holds_its_resource_and_units():
    m = pyo.ConcreteModel()

    m.power = SimpleUtility(resource="electricity", units="kW")

    assert m.power.config.resource == "electricity"
    assert m.power.config.units == "kW"
    assert PROPERTY_PACKAGES["SimpleUtility"] is SimpleUtility


@pytest.mark.unit
@pytest.mark.parametrize(
    "options",
    [{"units": "kW"}, {"resource": "electricity"}, {"resource": "", "units": "kW"}],
)
def test_a_utility_package_needs_a_resource_and_units(options):
    m = pyo.ConcreteModel()

    with pytest.raises(FlexConfigError, match="resource and units"):
        m.power = SimpleUtility(**options)


@pytest.mark.unit
def test_grid_power_and_flow_based_fuel_can_have_different_units():
    m = pyo.ConcreteModel()

    m.power = SimpleUtility(resource="electricity", units="MW")
    m.fuel = SimpleUtility(resource="natural_gas", units="m^3/hr")

    assert m.power.config.units != m.fuel.config.units


@pytest.mark.unit
def test_a_utility_resource_with_nothing_registered_names_its_resource():
    m = dummy_time_block(2)
    m.power = SimpleUtility(resource="electricity", units="kW")

    with pytest.raises(FlexConfigError, match="electricity"):
        m.node = Resource(property_package=m.power)
