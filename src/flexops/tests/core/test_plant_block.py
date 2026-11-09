"""PlantBlock: unit aggregation, construction order, TimeBlock auto-discovery (§3.3)."""

import pyomo.environ as pyo
import pytest
from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexcore import nomenclature as nm
from flexcore.exceptions import FlexConfigError
from flexops import PlantBlock
from flexops.core.ops_block import OpsBlockData
from flexops.core.time_block import TimeBlock
from flexops.testing import dummy_time_block
from flexops.unit_models import BatteryModel, ConstantEnergyIntensityModel

_SURROGATE_KW = 2.0
_BATTERY_KW = 0.5


@declare_process_block_class("DummyFuelUnit")
class DummyFuelUnitData(OpsBlockData):
    """A minimal unit that registers one named fuel-usage flow."""

    CONFIG = OpsBlockData.CONFIG()
    CONFIG.declare(
        "fuel_name",
        ConfigValue(default="natural_gas", domain=str, description="Fuel name."),
    )

    def build(self) -> None:
        super().build()
        tb = self._find_time_block()
        usage = pyo.Var(
            tb.time_index,
            initialize=0.0,
            units=pyunits.m**3 / pyunits.hr,
            doc="Fuel usage flow.",
        )
        self.add_component(f"{nm.FUEL_USAGE}_{self.config.fuel_name}", usage)
        self.register_fuel_usage(usage, fuel_name=self.config.fuel_name)


# ``declare_process_block_class`` injects the constructible ``DummyFuelUnit``
# wrapper into this module's namespace at runtime; bind the name explicitly so
# static tools (ruff) resolve it.
DummyFuelUnit = globals()["DummyFuelUnit"]


def _plant_with_two_units(n: int = 3):
    """Build a plant holding a ConstantEnergyIntensityModel and a BatteryModel."""
    m = dummy_time_block(n)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.surrogate = ConstantEnergyIntensityModel(property_package=m.properties)
    m.plant.battery = BatteryModel(energy_capacity=10 * pyunits.kWh)
    return m


def _fix_power(plant, time_index) -> None:
    """Fix every unit's power draw to a hand-picked value."""
    for t in time_index:
        plant.surrogate.power_electrical[t].fix(_SURROGATE_KW)
        plant.battery.power_electrical[t].fix(_BATTERY_KW)


@pytest.mark.unit
def test_aggregation_two_units():
    """total_electrical/thermal_power equal the hand sum over the plant's units."""
    m = _plant_with_two_units()
    m.plant._build_aggregates()
    _fix_power(m.plant, m.time_block.time_index)

    for t in m.time_block.time_index:
        assert pyo.value(m.plant.total_electrical_power[t]) == pytest.approx(
            _SURROGATE_KW + _BATTERY_KW, rel=1e-6
        )
        assert pyo.value(m.plant.total_thermal_power[t]) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_aggregates_survive_units_added_after_the_first_call():
    """Aggregating before the units exist, then again after, still sums them all.

    The frozen api-freeze script builds the plant (and costing) before its
    units, so aggregation must be re-entrant rather than one-shot (pitfall 3).
    """
    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant._build_aggregates()
    for t in m.time_block.time_index:
        assert pyo.value(m.plant.total_electrical_power[t]) == pytest.approx(
            0.0, abs=1e-9
        )

    m.plant.surrogate = ConstantEnergyIntensityModel(property_package=m.properties)
    m.plant.battery = BatteryModel(energy_capacity=10 * pyunits.kWh)
    m.plant._build_aggregates()
    _fix_power(m.plant, m.time_block.time_index)

    for t in m.time_block.time_index:
        assert pyo.value(m.plant.total_electrical_power[t]) == pytest.approx(
            _SURROGATE_KW + _BATTERY_KW, rel=1e-6
        )


@pytest.mark.unit
def test_fuel_aggregation_two_units_same_fuel():
    """total_fuel_usage[fuel, t] sums two units registered under the same fuel."""
    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.gen_a = DummyFuelUnit(fuel_name="natural_gas")
    m.plant.gen_b = DummyFuelUnit(fuel_name="natural_gas")
    m.plant._build_aggregates()

    for t in m.time_block.time_index:
        m.plant.gen_a.fuel_usage_natural_gas[t].fix(1.5)
        m.plant.gen_b.fuel_usage_natural_gas[t].fix(2.5)

    for t in m.time_block.time_index:
        assert pyo.value(m.plant.total_fuel_usage["natural_gas", t]) == pytest.approx(
            4.0, rel=1e-6
        )


@pytest.mark.unit
def test_fuel_aggregation_keeps_distinct_fuels_separate():
    """Two units registering different fuel names aggregate under separate keys."""
    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.gen_gas = DummyFuelUnit(fuel_name="natural_gas")
    m.plant.gen_oil = DummyFuelUnit(fuel_name="fuel_oil")
    m.plant._build_aggregates()

    for t in m.time_block.time_index:
        m.plant.gen_gas.fuel_usage_natural_gas[t].fix(1.0)
        m.plant.gen_oil.fuel_usage_fuel_oil[t].fix(3.0)

    assert {fuel for fuel, _t in m.plant.total_fuel_usage} == {
        "natural_gas",
        "fuel_oil",
    }
    for t in m.time_block.time_index:
        assert pyo.value(m.plant.total_fuel_usage["natural_gas", t]) == pytest.approx(
            1.0, rel=1e-6
        )
        assert pyo.value(m.plant.total_fuel_usage["fuel_oil", t]) == pytest.approx(
            3.0, rel=1e-6
        )


@pytest.mark.unit
def test_fuel_aggregation_survives_units_added_after_the_first_call():
    """Aggregating before the fuel-registering unit exists, then again after."""
    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant._build_aggregates()
    assert m.plant.component("total_fuel_usage") is None

    m.plant.gen = DummyFuelUnit(fuel_name="natural_gas")
    m.plant._build_aggregates()
    for t in m.time_block.time_index:
        m.plant.gen.fuel_usage_natural_gas[t].fix(2.0)

    for t in m.time_block.time_index:
        assert pyo.value(m.plant.total_fuel_usage["natural_gas", t]) == pytest.approx(
            2.0, rel=1e-6
        )


@pytest.mark.unit
def test_plant_is_steady_state_and_shares_the_time_block_set():
    """dynamic=False and the flowsheet time domain IS the TimeBlock's set (R2)."""
    m = _plant_with_two_units()
    assert m.plant.config.dynamic is False
    assert m.plant.time is m.time_block.time_index


@pytest.mark.unit
def test_time_block_autodiscovery():
    """Omitting time_block= works with one TimeBlock and errors on two."""
    m = dummy_time_block(3)
    m.plant = PlantBlock()
    assert m.plant.time is m.time_block.time_index

    m2 = dummy_time_block(3)
    m2.other_time_block = TimeBlock(
        start_date="2025-02-01", end_date="2025-02-02", time_step=1 * pyunits.hr
    )
    with pytest.raises(FlexConfigError):
        m2.plant = PlantBlock()


# -- boundary (feed/product) aggregation ------------------------------------


def _plant_with_feeds(*feeds):
    """Build a plant carrying one ``Feed`` per ``(block name, resource)`` pair."""
    from flexops.unit_models import Feed

    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    for block_name, resource in feeds:
        m.plant.add_component(
            block_name,
            Feed(property_package=m.properties, resource_name=resource),
        )
    m.plant._build_aggregates()
    return m


@pytest.mark.unit
def test_feed_aggregation_keeps_distinct_resources_separate():
    """Two Feeds with different resource names give two total_feed rows."""
    m = _plant_with_feeds(("raw_water", "raw_water"), ("citric_acid", "citric_acid"))

    assert {resource for resource, _ in m.plant.total_feed} == {
        "raw_water",
        "citric_acid",
    }
    for t in m.time_block.time_index:
        m.plant.raw_water.withdrawal[t].fix(4.0)
        m.plant.citric_acid.withdrawal[t].fix(0.5)
    for t in m.time_block.time_index:
        assert pyo.value(m.plant.total_feed["raw_water", t]) == pytest.approx(4.0)
        assert pyo.value(m.plant.total_feed["citric_acid", t]) == pytest.approx(0.5)


@pytest.mark.unit
def test_feed_aggregation_sums_blocks_sharing_one_resource_name():
    """Two Feeds under one resource_name sum into a single total_feed row."""
    from pyomo.repn import generate_standard_repn

    m = _plant_with_feeds(("north_tap", "city_water"), ("south_tap", "city_water"))

    assert {resource for resource, _ in m.plant.total_feed} == {"city_water"}
    repn = generate_standard_repn(
        m.plant.total_feed["city_water", 0].expr, compute_values=True
    )
    contributors = sorted(id(v) for v in repn.linear_vars)
    assert contributors == sorted(
        id(v)
        for v in (m.plant.north_tap.withdrawal[0], m.plant.south_tap.withdrawal[0])
    )
    assert list(repn.linear_coefs) == [pytest.approx(1.0)] * 2


@pytest.mark.unit
def test_plant_merges_explicit_and_discovered_products_without_double_counting():
    """register_product and a discovered Product both land in total_product."""
    from flexops.unit_models import Product

    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.potable = Product(
        property_package=m.properties, resource_name="potable_water"
    )
    m.plant.surrogate = ConstantEnergyIntensityModel(property_package=m.properties)
    m.plant.register_product(m.plant.surrogate.flow_out, name="permeate")
    m.plant._build_aggregates()

    assert {product for product, _ in m.plant.total_product} == {
        "potable_water",
        "permeate",
    }
    for t in m.time_block.time_index:
        m.plant.potable.delivery[t].fix(3.0)
        m.plant.surrogate.flow_out[t].fix(7.0)
    for t in m.time_block.time_index:
        assert pyo.value(m.plant.total_product["potable_water", t]) == pytest.approx(
            3.0
        )
        assert pyo.value(m.plant.total_product["permeate", t]) == pytest.approx(7.0)


@pytest.mark.unit
def test_feeds_and_products_aggregate_into_disjoint_totals():
    """A Feed never leaks into total_product, nor a Product into total_feed."""
    from flexops.unit_models import Feed, Product

    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.raw_water = Feed(property_package=m.properties)
    m.plant.brine = Product(property_package=m.properties)
    m.plant._build_aggregates()

    assert {resource for resource, _ in m.plant.total_feed} == {"raw_water"}
    assert {product for product, _ in m.plant.total_product} == {"brine"}


@pytest.mark.unit
def test_feed_aggregation_survives_units_added_after_the_first_call():
    """Aggregating before the Feed exists, then again after, still sums it."""
    from flexops.unit_models import Feed

    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant._build_aggregates()
    assert m.plant.component(nm.TOTAL_FEED) is None

    m.plant.raw_water = Feed(property_package=m.properties)
    m.plant._build_aggregates()
    for t in m.time_block.time_index:
        m.plant.raw_water.withdrawal[t].fix(2.0)
    for t in m.time_block.time_index:
        assert pyo.value(m.plant.total_feed["raw_water", t]) == pytest.approx(2.0)
