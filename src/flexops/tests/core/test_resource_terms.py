"""iter_resource_terms: signed per-resource contributions read from registrations."""

import pyomo.environ as pyo
import pytest
from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexcore import nomenclature as nm
from flexcore.exceptions import FlexConfigError
from flexops import NetworkBlock, PlantBlock
from flexops.core.ops_block import OpsBlockData
from flexops.core.registration import iter_resource_terms
from flexops.testing import dummy_time_block
from flexops.unit_models import Feed, GenericRenewables, Product, Pump


@declare_process_block_class("GasBurner")
class GasBurnerData(OpsBlockData):
    """A minimal unit that burns one named fuel."""

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


# ``declare_process_block_class`` injects the constructible ``GasBurner`` wrapper
# at runtime; bind the name so static tools resolve it.
GasBurner = globals()["GasBurner"]


@declare_process_block_class("BareUnit")
class BareUnitData(OpsBlockData):
    """A unit that registers nothing by itself, for registering by hand."""

    CONFIG = OpsBlockData.CONFIG()


BareUnit = globals()["BareUnit"]


def _plant(n: int = 3):
    """Build a plant with a Feed, a Pump, a Product and a gas burner."""
    m = dummy_time_block(n)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.feed = Feed(property_package=m.properties, resource_name="raw_water")
    m.plant.pump = Pump(
        property_package=m.properties,
        energy_intensity=0.5 * pyunits.kWh / pyunits.m**3,
    )
    m.plant.product = Product(property_package=m.properties, resource_name="potable")
    m.plant.burner = GasBurner()
    return m


def _signs(terms) -> dict[str, float]:
    """Return ``{unit local name: coefficient}`` for a list of terms."""
    return {term.unit.local_name: term.coeff for term in terms}


@pytest.mark.unit
def test_terms_are_keyed_by_resource_with_the_registered_signs():
    """Feed supplies, product and fuel and electrical draw consume."""
    m = _plant()

    terms = iter_resource_terms(m.plant)

    assert list(terms) == ["electricity", "natural_gas", "potable", "raw_water"]
    assert _signs(terms["raw_water"]) == {"feed": 1.0}
    assert _signs(terms["potable"]) == {"product": -1.0}
    assert _signs(terms["electricity"]) == {"pump": -1.0}
    assert _signs(terms["natural_gas"]) == {"burner": -1.0}


@pytest.mark.unit
def test_terms_carry_the_registered_variables():
    """A term's var is the object the unit registered, not a copy."""
    m = _plant()
    terms = iter_resource_terms(m.plant)

    assert terms["electricity"][0].var is m.plant.pump.power_electrical
    assert terms["natural_gas"][0].var is m.plant.burner.fuel_usage_natural_gas
    assert terms["raw_water"][0].unit is m.plant.feed
    assert terms["raw_water"][0].offset == 0


@pytest.mark.unit
def test_feed_and_fuel_burner_share_a_resource_with_opposite_signs():
    """A Feed of natural_gas and a burner of natural_gas land on one key."""
    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.gas_feed = Feed(property_package=m.properties, resource_name="natural_gas")
    m.plant.burner = GasBurner()

    gas = iter_resource_terms(m.plant)["natural_gas"]

    assert _signs(gas) == {"gas_feed": 1.0, "burner": -1.0}


@pytest.mark.unit
def test_two_feeds_with_one_resource_name_give_two_terms_on_one_key():
    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.well = Feed(property_package=m.properties, resource_name="raw_water")
    m.plant.river = Feed(property_package=m.properties, resource_name="raw_water")

    terms = iter_resource_terms(m.plant)

    assert list(terms) == ["raw_water"]
    assert _signs(terms["raw_water"]) == {"well": 1.0, "river": 1.0}


@pytest.mark.unit
def test_thermal_power_is_not_mapped_to_a_resource():
    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.heater = BareUnit()
    m.plant.heater.declare_power(nm.PowerKind.THERMAL, temperature=350 * pyunits.K)

    assert m.plant.heater._io_registry.power
    assert iter_resource_terms(m.plant) == {}


@pytest.mark.unit
def test_explicit_registration_is_returned_unchanged():
    """coeff and offset reach the term as given."""
    m = dummy_time_block(4)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.unit = BareUnit()
    recovered = pyo.Var(m.time_block.time_index, initialize=0.0)
    m.plant.unit.recovered = recovered

    m.plant.unit.register_resource(recovered, resource="brine", coeff=-0.45, offset=3)

    (term,) = iter_resource_terms(m.plant)["brine"]
    assert (term.var, term.coeff, term.offset, term.unit) == (
        recovered,
        -0.45,
        3,
        m.plant.unit,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"resource": ""}, "resource"),
        ({"resource": "brine", "offset": -1}, "offset"),
        ({"resource": "brine", "offset": 1.5}, "offset"),
        ({"resource": "brine", "coeff": "1"}, "coeff"),
    ],
)
def test_register_resource_rejects_bad_arguments(kwargs, field):
    m = dummy_time_block(3)
    m.unit = BareUnit()
    m.unit.flow = pyo.Var(m.time_block.time_index)

    with pytest.raises(FlexConfigError) as excinfo:
        m.unit.register_resource(m.unit.flow, **kwargs)

    assert excinfo.value.field == field


@pytest.mark.unit
@pytest.mark.parametrize("indexed_over", ["scalar", "other_set"])
def test_register_resource_requires_a_time_indexed_var(indexed_over):
    """A scalar or wrongly indexed Var is refused with advice on what to pass."""
    m = dummy_time_block(3)
    m.unit = BareUnit()
    m.unit.flow = (
        pyo.Var() if indexed_over == "scalar" else pyo.Var(["a", "b"], initialize=0)
    )

    with pytest.raises(FlexConfigError, match="time points") as excinfo:
        m.unit.register_resource(m.unit.flow, resource="brine")

    assert excinfo.value.field == "var"


@pytest.mark.unit
def test_network_counts_each_unit_once():
    """Terms from two child plants appear once each, with no repeats."""
    m = dummy_time_block(3)
    m.network = NetworkBlock(time_block=m.time_block)
    m.network.north = PlantBlock(time_block=m.time_block)
    m.network.south = PlantBlock(time_block=m.time_block)
    for plant in (m.network.north, m.network.south):
        plant.feed = Feed(property_package=m.properties, resource_name="raw_water")
        plant.pump = Pump(
            property_package=m.properties,
            energy_intensity=0.5 * pyunits.kWh / pyunits.m**3,
        )

    terms = iter_resource_terms(m.network)

    for resource in ("raw_water", "electricity"):
        variables = [id(term.var) for term in terms[resource]]
        assert len(variables) == 2
        assert len(set(variables)) == 2


@pytest.mark.unit
def test_indexed_unit_gives_one_term_per_member():
    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.trains = Pump(
        [0, 1, 2],
        property_package=m.properties,
        energy_intensity=0.5 * pyunits.kWh / pyunits.m**3,
    )

    terms = iter_resource_terms(m.plant)["electricity"]

    assert len(terms) == 3
    assert {term.var for term in terms} == {
        m.plant.trains[i].power_electrical for i in range(3)
    }


@pytest.mark.unit
def test_renewable_export_supplies_electricity():
    """The term is -1 on a power var capped above by 0, so the net is supply."""
    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.solar = GenericRenewables(
        capacity=10 * pyunits.kW, capacity_factor=[0.2, 0.5, 0.8]
    )

    (term,) = iter_resource_terms(m.plant)["electricity"]

    assert term.coeff == -1.0
    assert term.var[0].ub == 0


@pytest.mark.unit
def test_deactivated_unit_contributes_no_terms():
    """A deactivated unit's variables are unconstrained, so they must not balance."""
    m = _plant()
    m.plant.burner.deactivate()

    terms = iter_resource_terms(m.plant)

    assert "natural_gas" not in terms
    assert _signs(terms["raw_water"]) == {"feed": 1.0}


@pytest.mark.unit
def test_registering_the_same_term_twice_raises():
    """A repeated registration would count the variable twice in a balance."""
    m = dummy_time_block(3)
    m.unit = BareUnit()
    m.unit.flow = pyo.Var(m.time_block.time_index)
    m.unit.register_resource(m.unit.flow, resource="brine")

    with pytest.raises(FlexConfigError, match="already registered") as excinfo:
        m.unit.register_resource(m.unit.flow, resource="brine", coeff=2.0)

    assert excinfo.value.field == "var"
