"""Resource balances: flow and stock builders, the resource stage, and solves."""

import copy

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits
from pyomo.network import Arc
from pyomo.util.check_units import assert_units_equivalent

from flexcore.config.spec import SCHEMA_VERSION
from flexcore.exceptions import FlexConfigError
from flexcore.solvers import get_solver
from flexops import PlantBlock, build_model
from flexops.core.resources import add_resource_balance
from flexops.testing import dummy_time_block
from flexops.tests.core.test_resource_terms import BareUnit, GasBurner
from flexops.unit_models import Feed, Product, Pump, Tank

GAS_FLOW = "m^3/hr"


def _gas_plant(n: int = 4):
    """Build a plant with a natural_gas Feed and a unit burning natural_gas."""
    m = dummy_time_block(n)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.gas_feed = Feed(property_package=m.properties, resource_name="natural_gas")
    m.plant.burner = GasBurner()
    return m


@pytest.mark.unit
def test_flow_resource_defaults_to_a_strict_node_balance():
    """With no bounds, net is fixed to zero and balance ties it to the terms."""
    m = _gas_plant()

    block = add_resource_balance(
        m.plant, "gas_balance", resource="natural_gas", units=GAS_FLOW
    )

    assert m.plant.gas_balance is block
    assert all(block.net[t].bounds == (0.0, 0.0) for t in block.net)
    assert len(block.balance) == 4
    assert_units_equivalent(block.net[0], pyunits.m**3 / pyunits.hr)


@pytest.mark.unit
def test_flow_resource_with_one_bound_leaves_the_other_side_open():
    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.pump = Pump(
        property_package=m.properties,
        energy_intensity=0.5 * pyunits.kWh / pyunits.m**3,
    )

    block = add_resource_balance(
        m.plant, "grid", resource="electricity", units="kW", lower=-5000.0
    )

    assert block.net[0].bounds == (-5000.0, None)


@pytest.mark.unit
def test_flow_bound_quantities_are_converted_to_the_resource_units():
    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.pump = Pump(
        property_package=m.properties,
        energy_intensity=0.5 * pyunits.kWh / pyunits.m**3,
    )

    block = add_resource_balance(
        m.plant, "grid", resource="electricity", units="kW", lower=-5 * pyunits.MW
    )

    assert block.net[0].bounds == (pytest.approx(-5000.0), None)


@pytest.mark.unit
def test_stock_resource_follows_the_tank_convention():
    """level[0] is the initial level; flows count from t = 1."""
    m = _gas_plant()

    block = add_resource_balance(
        m.plant,
        "storage",
        resource="natural_gas",
        units="m^3",
        stock=True,
        lower=0.0,
        upper=100.0,
        initial=40.0,
    )

    assert block.level[0].bounds == (0.0, 100.0)
    assert pyo.value(block.initial) == 40.0
    assert block.initial in m.time_block._initial_state_params
    assert list(block.balance.keys()) == [1, 2, 3]
    assert block.initial_level.active


@pytest.mark.unit
def test_empty_resource_error_lists_what_is_registered():
    m = _gas_plant()

    with pytest.raises(FlexConfigError) as excinfo:
        add_resource_balance(m.plant, "water", resource="raw_water", units=GAS_FLOW)

    assert excinfo.value.field == "resource"
    assert "natural_gas" in str(excinfo.value)


@pytest.mark.unit
def test_units_mismatch_names_the_offending_unit():
    """A kW balance cannot absorb a volumetric flow; the error says who and why."""
    m = _gas_plant()

    with pytest.raises(FlexConfigError, match="gas_feed") as excinfo:
        add_resource_balance(m.plant, "gas", resource="natural_gas", units="kW")

    assert excinfo.value.field == "units"


@pytest.mark.unit
def test_bound_with_the_wrong_dimension_is_a_config_error():
    m = _gas_plant()

    with pytest.raises(FlexConfigError) as excinfo:
        add_resource_balance(
            m.plant,
            "gas",
            resource="natural_gas",
            units=GAS_FLOW,
            upper=5 * pyunits.kW,
        )

    assert excinfo.value.field == "upper"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"stock": True}, "initial"),
        ({"initial": 3.0}, "initial"),
        ({"lower": 5.0, "upper": 1.0}, "lower"),
    ],
)
def test_inconsistent_arguments_raise(kwargs, field):
    m = _gas_plant()

    with pytest.raises(FlexConfigError) as excinfo:
        add_resource_balance(
            m.plant, "gas", resource="natural_gas", units=GAS_FLOW, **kwargs
        )

    assert excinfo.value.field == field


@pytest.mark.unit
def test_terms_on_arc_connected_ports_are_refused():
    """An Arc already conserves a port flow; balancing it again counts it twice."""
    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.first = Pump(
        property_package=m.properties, energy_intensity=0.5 * pyunits.kWh / pyunits.m**3
    )
    m.plant.second = Pump(
        property_package=m.properties, energy_intensity=0.5 * pyunits.kWh / pyunits.m**3
    )
    m.plant.link = Arc(source=m.plant.first.outlet, destination=m.plant.second.inlet)
    outflow = pyo.Reference(m.plant.first.outlet_state.flow_vol_phase[:, "Liq"])
    m.plant.first.outflow = outflow
    m.plant.first.register_resource(outflow, resource="water", coeff=1.0)

    with pytest.raises(FlexConfigError, match="Arc") as excinfo:
        add_resource_balance(m.plant, "water_balance", resource="water", units=GAS_FLOW)

    assert excinfo.value.field == "resource"


@pytest.mark.unit
def test_a_term_offset_before_the_horizon_adds_nothing():
    """With offset 2, the first two steps see no contribution."""
    m = dummy_time_block(4)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.unit = BareUnit()
    m.plant.unit.flow = pyo.Var(
        m.time_block.time_index, initialize=1.0, units=pyunits.m**3 / pyunits.hr
    )
    m.plant.unit.register_resource(
        m.plant.unit.flow, resource="delayed", coeff=1.0, offset=2
    )

    block = add_resource_balance(
        m.plant, "delayed", resource="delayed", units=GAS_FLOW, lower=0.0, upper=10.0
    )

    assert "flow" not in str(block.balance[0].expr)
    assert "flow" not in str(block.balance[1].expr)
    assert "flow[0]" in str(block.balance[2].expr)


@pytest.mark.unit
def test_building_a_balance_records_it_on_the_model():
    m = _gas_plant()

    add_resource_balance(m.plant, "gas_balance", resource="natural_gas", units=GAS_FLOW)

    (record,) = m.__dict__["_flex_logic"]
    assert (record.kind, record.fields) == (
        "resource",
        {"name": "plant.gas_balance", "resource": "natural_gas", "units": GAS_FLOW},
    )


# --- the build_model path ---------------------------------------------------

BASE = [
    {
        "kind": "time",
        "start_date": "2025-01-01",
        "end_date": "2025-01-01T05:00",
        "time_step": "1 hr",
    },
    {
        "kind": "property_package",
        "name": "properties",
        "property_class": "SimpleAqueousFlow",
    },
    {"kind": "plant", "name": "site"},
    {
        "kind": "unit",
        "name": "site.well",
        "unit_model_class": "Feed",
        "construction_options": {"resource_name": "raw_water"},
        "costing_package": None,
    },
]


def _spec(*elements):
    return {
        "schema_version": SCHEMA_VERSION,
        "name": "m",
        "elements": copy.deepcopy([*BASE, *elements]),
    }


@pytest.mark.unit
def test_resource_element_builds_after_the_units_it_collects():
    model = build_model(
        _spec({"kind": "resource", "name": "site.raw_water", "units": GAS_FLOW})
    )

    assert hasattr(model.site, "raw_water")
    assert model.site.raw_water.net[0].bounds == (0.0, 0.0)


@pytest.mark.unit
def test_resource_element_names_the_resource_when_it_differs_from_the_block():
    model = build_model(
        _spec(
            {
                "kind": "resource",
                "name": "site.balance",
                "resource": "raw_water",
                "units": GAS_FLOW,
                "upper": {"value": 2.0, "units": "m^3/hr"},
            }
        )
    )

    assert model.site.balance.net[0].bounds == (None, 2.0)


@pytest.mark.unit
def test_resource_element_on_a_unit_is_refused():
    """The parent must be a plant or network, not a unit."""
    with pytest.raises(FlexConfigError, match="parent"):
        build_model(
            _spec({"kind": "resource", "name": "site.well.water", "units": GAS_FLOW})
        )


@pytest.mark.unit
def test_resource_element_with_nothing_registered_lists_the_alternatives():
    with pytest.raises(FlexConfigError, match="raw_water"):
        build_model(_spec({"kind": "resource", "name": "site.fuel", "units": GAS_FLOW}))


# --- solves -----------------------------------------------------------------


def _solve(m):
    results = get_solver(model=m, prefer="highs").solve(m)
    pyo.assert_optimal_termination(results)


@pytest.mark.component
@pytest.mark.needs_highs
def test_gas_balance_forces_withdrawal_to_equal_burn():
    """Without the balance the Feed could draw nothing; with it, it matches."""
    burn = [float(5 + (t % 4)) for t in range(24)]

    def solve(with_balance: bool):
        m = _gas_plant(24)
        steps = list(m.time_block.time_index)
        for t in steps:
            m.plant.burner.fuel_usage_natural_gas[t].fix(burn[t])
            m.plant.gas_feed.withdrawal[t].setlb(0.0)
        if with_balance:
            add_resource_balance(m.plant, "gas", resource="natural_gas", units=GAS_FLOW)
        m.cost = pyo.Objective(
            expr=sum(pyo.value(1) * m.plant.gas_feed.withdrawal[t] for t in steps)
        )
        _solve(m)
        return [pyo.value(m.plant.gas_feed.withdrawal[t]) for t in steps]

    assert solve(False) == pytest.approx([0.0] * 24)
    assert solve(True) == pytest.approx(burn)


@pytest.mark.component
@pytest.mark.needs_highs
def test_electricity_cap_limits_the_net_draw():
    """A 50 kW import cap holds where an uncapped run reaches 100 kW."""

    def peak(cap: float | None):
        m = dummy_time_block(24)
        m.plant = PlantBlock(time_block=m.time_block)
        m.plant.load = BareUnit()
        draw = m.plant.load.declare_power()
        draw.setlb(0.0)
        draw.setub(100.0)
        if cap is not None:
            add_resource_balance(
                m.plant,
                "grid",
                resource="electricity",
                units="kW",
                lower=-cap * pyunits.kW,
            )
        m.reward = pyo.Objective(
            expr=sum(draw[t] for t in m.time_block.time_index), sense=pyo.maximize
        )
        _solve(m)
        return max(pyo.value(draw[t]) for t in m.time_block.time_index)

    assert peak(None) == pytest.approx(100.0)
    assert peak(50.0) == pytest.approx(50.0)


@pytest.mark.component
@pytest.mark.needs_highs
def test_stock_resource_matches_a_tank_with_the_same_bounds():
    """A stock between a Feed and a Product equals a Tank between them."""
    price = [1.0 + 0.5 * (t % 6) for t in range(24)]
    demand = [10.0 if t % 3 else 25.0 for t in range(24)]
    initial, capacity = 30.0, 80.0

    def finish(m, feed, product):
        steps = list(m.time_block.time_index)
        for t in steps:
            m.plant.add_component(
                f"demand_{t}", pyo.Constraint(expr=product.delivery[t] >= demand[t])
            )
        m.cost = pyo.Objective(expr=sum(price[t] * feed.withdrawal[t] for t in steps))
        _solve(m)
        return pyo.value(m.cost)

    def with_stock():
        m = dummy_time_block(24)
        m.plant = PlantBlock(time_block=m.time_block)
        m.plant.feed = Feed(property_package=m.properties, resource_name="water")
        m.plant.product = Product(property_package=m.properties, resource_name="water")
        add_resource_balance(
            m.plant,
            "storage",
            resource="water",
            units="m^3",
            stock=True,
            lower=0.0,
            upper=capacity,
            initial=initial,
        )
        return finish(m, m.plant.feed, m.plant.product)

    def with_tank():
        m = dummy_time_block(24)
        m.plant = PlantBlock(time_block=m.time_block)
        m.plant.feed = Feed(property_package=m.properties, resource_name="water")
        m.plant.tank = Tank(
            property_package=m.properties,
            max_volume=capacity * pyunits.m**3,
            initial_volume=initial * pyunits.m**3,
        )
        m.plant.product = Product(property_package=m.properties, resource_name="water")
        m.plant.into_tank = Arc(
            source=m.plant.feed.outlet_a, destination=m.plant.tank.inlet
        )
        m.plant.out_of_tank = Arc(
            source=m.plant.tank.outlet, destination=m.plant.product.inlet_a
        )
        pyo.TransformationFactory("network.expand_arcs").apply_to(m)
        return finish(m, m.plant.feed, m.plant.product)

    assert with_stock() == pytest.approx(with_tank())
