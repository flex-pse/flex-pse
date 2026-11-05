"""Resource blocks: stream nodes reached by Arcs, utilities read from the registry."""

import copy
import warnings

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits
from pyomo.network import Arc, Port
from pyomo.util.check_units import assert_units_consistent, assert_units_equivalent

from flexcore.config.schema import BalanceType, UnitCommitmentConfig
from flexcore.config.spec import SCHEMA_VERSION
from flexcore.exceptions import FlexConfigError, FlexResourceWarning
from flexcore.solvers import facade, get_solver
from flexops import (
    PlantBlock,
    SimpleAqueousFlow,
    SimpleGasFlow,
    SimpleUtility,
    build_model,
    emit_model,
)
from flexops.core.extensions import extension
from flexops.core.resources import (
    ErrorType,
    Resource,
    ResourceKind,
    resource_graph,
    verify_resources,
)
from flexops.testing import dummy_time_block
from flexops.tests.core.test_resource_terms import GasBurner
from flexops.unit_models import (
    BatteryModel,
    Feed,
    GenericRenewables,
    GridConnection,
    Product,
    Pump,
    Tank,
)

GAS_FLOW = "m^3/hr"
INTENSITY = 0.5 * pyunits.kWh / pyunits.m**3


def _expand(m):
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)


def _gas_plant(n: int = 4):
    """Build a plant with a natural_gas Feed and a unit burning natural_gas."""
    m = dummy_time_block(n)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.gas_feed = Feed(property_package=m.properties, resource_name="natural_gas")
    m.plant.burner = GasBurner()
    return m


def _water_plant(n: int = 4, package=None):
    """Build a Feed and a Product joined by Arcs through a ``line`` stream node."""
    m = dummy_time_block(n)
    m.plant = PlantBlock(time_block=m.time_block)
    package = package or m.properties
    m.plant.feed = Feed(property_package=package, resource_name="well")
    m.plant.product = Product(property_package=package, resource_name="supply")
    m.plant.line = Resource(property_package=package)
    m.plant.into_line = Arc(
        source=m.plant.feed.outlet_a, destination=m.plant.line.inlet
    )
    m.plant.out_of_line = Arc(
        source=m.plant.line.outlet, destination=m.plant.product.inlet_a
    )
    return m


def _utility(m, resource, units=GAS_FLOW):
    """Return a utility package on the model root, building it the first time."""
    name = f"{resource}_package"
    if m.component(name) is None:
        m.add_component(name, SimpleUtility(resource=resource, units=units))
    return m.component(name)


def _pump(m, name):
    """Add a Pump to the plant."""
    m.plant.add_component(
        name, Pump(property_package=m.properties, energy_intensity=INTENSITY)
    )
    return m.plant.find_component(name)


# --- utilities --------------------------------------------------------------


@pytest.mark.unit
def test_utility_sums_explicit_edges_to_zero_at_every_step():
    m = _gas_plant()

    m.plant.gas = Resource(property_package=_utility(m, "natural_gas"))

    assert sorted(m.plant.gas.edges) == [
        "plant.burner.fuel_usage_natural_gas",
        "plant.gas_feed.withdrawal",
    ]
    assert len(m.plant.gas.flow) == 2 * 4
    assert len(m.plant.gas.balance) == 4
    assert m.plant.gas.find_component("inlet") is None


@pytest.mark.unit
def test_edge_flow_is_the_signed_unit_variable():
    """An edge adds no Var: it evaluates to coeff times the unit's own variable."""
    m = _gas_plant()
    m.plant.gas_feed.withdrawal[1].set_value(7.0)
    m.plant.burner.fuel_usage_natural_gas[1].set_value(3.0)

    m.plant.gas = Resource(property_package=_utility(m, "natural_gas"))

    assert pyo.value(m.plant.gas.flow["plant.gas_feed.withdrawal", 1]) == 7.0
    assert pyo.value(m.plant.gas.flow["plant.burner.fuel_usage_natural_gas", 1]) == -3.0
    assert len(list(m.plant.gas.component_objects(pyo.Var))) == 0


@pytest.mark.unit
def test_utility_with_nothing_registered_lists_what_is_registered():
    m = _gas_plant()

    with pytest.raises(FlexConfigError) as excinfo:
        m.plant.other = Resource(property_package=_utility(m, "oil"))

    assert excinfo.value.field == "property_package"
    assert "natural_gas" in str(excinfo.value)


@pytest.mark.unit
def test_units_mismatch_names_the_offending_unit():
    """A kW balance cannot absorb a volumetric flow; the error says who and why."""
    m = _gas_plant()

    with pytest.raises(FlexConfigError, match="gas_feed") as excinfo:
        m.plant.gas = Resource(property_package=_utility(m, "natural_gas", "kW"))

    assert excinfo.value.field == "units"


@pytest.mark.unit
def test_utility_term_on_an_arc_connected_port_is_refused():
    """An Arc already conserves a port flow; balancing it again counts it twice."""
    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    first = _pump(m, "first")
    second = _pump(m, "second")
    m.plant.link = Arc(source=first.outlet, destination=second.inlet)
    first.out_flow = pyo.Reference(first.outlet_state.flow_vol_phase[:, "Liq"])
    first.register_fuel_usage(first.out_flow, fuel_name="water_term")

    with pytest.raises(FlexConfigError, match="Arc") as excinfo:
        m.plant.water = Resource(property_package=_utility(m, "water_term"))

    assert excinfo.value.field == "property_package"


@pytest.mark.unit
def test_building_a_resource_records_it_on_the_model():
    m = _gas_plant()

    m.plant.gas_balance = Resource(property_package=_utility(m, "natural_gas"))

    (record,) = m.__dict__["_flex_logic"]
    assert (record.kind, record.fields) == (
        "resource",
        {
            "name": "plant.gas_balance",
            "property_package": "natural_gas_package",
            "balance": "balanced",
        },
    )


@pytest.mark.unit
def test_a_utility_balances_in_its_packages_units():
    m = _gas_plant()

    m.plant.gas = Resource(property_package=_utility(m, "natural_gas", "m^3/s"))

    flow = m.plant.gas.flow[next(iter(m.plant.gas.flow))]
    assert_units_equivalent(flow, pyunits.m**3 / pyunits.s)


# --- stream nodes -----------------------------------------------------------


@pytest.mark.unit
def test_stream_node_conserves_flow_between_its_ports():
    m = _water_plant()
    _expand(m)

    node = m.plant.line
    assert len(node.balance) == 4
    node.inlet_state.flow_vol_phase[0, "Liq"].set_value(4.0)
    node.outlet_state.flow_vol_phase[0, "Liq"].set_value(4.0)
    assert pyo.value(node.balance[0, "Liq"].body) == pytest.approx(0.0)


@pytest.mark.unit
def test_a_port_flow_is_tagged_with_one_rule_on_units_and_nodes():
    m = _water_plant()

    assert m.plant.feed.outlet_a.rule_for(
        "flow_vol_phase"
    ) is m.plant.line.inlet.rule_for("flow_vol_phase")


@pytest.mark.unit
def test_fan_in_and_fan_out_write_no_split_fraction():
    """A split fraction would be bilinear and hold one ratio over the horizon."""
    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.node = Resource(property_package=m.properties)
    first, second, third, fourth = (_pump(m, n) for n in ("a", "b", "c", "d"))
    m.plant.a1 = Arc(source=first.outlet, destination=m.plant.node.inlet)
    m.plant.a2 = Arc(source=second.outlet, destination=m.plant.node.inlet)
    m.plant.a3 = Arc(source=m.plant.node.outlet, destination=third.inlet)
    m.plant.a4 = Arc(source=m.plant.node.outlet, destination=fourth.inlet)

    _expand(m)

    names = [v.name for v in m.component_data_objects(pyo.Var)]
    assert not [name for name in names if "splitfrac" in name]
    arc_flows = m.plant.a1_expanded.flow_vol_phase.values()
    assert all(flow.domain is pyo.NonNegativeReals for flow in arc_flows)


@pytest.mark.unit
def test_fan_in_arc_flows_carry_units():
    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.node = Resource(property_package=m.properties)
    first, second = _pump(m, "a"), _pump(m, "b")
    m.plant.a1 = Arc(source=first.outlet, destination=m.plant.node.inlet)
    m.plant.a2 = Arc(source=second.outlet, destination=m.plant.node.inlet)

    _expand(m)

    assert_units_consistent(m.plant)


@pytest.mark.unit
def test_a_directed_arc_between_units_writes_the_same_equality():
    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    first, second = _pump(m, "a"), _pump(m, "b")
    m.plant.link = Arc(source=first.outlet, destination=second.inlet)

    _expand(m)

    assert list(m.plant.link_expanded.component_map(pyo.Var)) == []
    assert len(m.plant.link_expanded.flow_vol_phase_equality) == 3


@pytest.mark.unit
def test_an_undirected_arc_between_units_keeps_its_equality():
    """Pyomo's extensive rule alone would write nothing for an undirected arc."""
    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    first, second = _pump(m, "a"), _pump(m, "b")
    m.plant.link = Arc(ports=(first.outlet, second.inlet))

    _expand(m)

    assert len(m.plant.link_expanded.flow_vol_phase_equality) == 3


@pytest.mark.unit
def test_an_undirected_arc_sharing_a_port_with_another_is_refused():
    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    first, second, third = _pump(m, "a"), _pump(m, "b"), _pump(m, "c")
    m.plant.one = Arc(ports=(first.outlet, second.inlet))
    m.plant.two = Arc(source=first.outlet, destination=third.inlet)

    with pytest.raises(FlexConfigError, match="undirected"):
        _expand(m)


@pytest.mark.unit
def test_an_undirected_arc_on_a_node_is_refused():
    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.node = Resource(property_package=m.properties)
    first = _pump(m, "a")
    m.plant.one = Arc(ports=(m.plant.node.outlet, first.inlet))

    with pytest.raises(FlexConfigError, match="undirected"):
        _expand(m)


@pytest.mark.unit
def test_an_arc_running_backwards_through_a_node_port_is_refused():
    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.node = Resource(property_package=m.properties)
    first = _pump(m, "a")
    m.plant.one = Arc(source=first.outlet, destination=m.plant.node.outlet)

    with pytest.raises(FlexConfigError, match="outlet"):
        _expand(m)


@pytest.mark.unit
def test_a_node_side_with_no_arcs_stays_fixed_at_zero():
    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.feed = Feed(property_package=m.properties, resource_name="well")
    m.plant.node = Resource(property_package=m.properties)
    m.plant.one = Arc(source=m.plant.feed.outlet_a, destination=m.plant.node.inlet)

    _expand(m)

    node = m.plant.node
    assert all(
        v.fixed and v.value == 0 for v in node.outlet_state.flow_vol_phase.values()
    )
    assert not any(v.fixed for v in node.inlet_state.flow_vol_phase.values())


@pytest.mark.unit
def test_stream_node_needs_a_property_package():
    m = dummy_time_block(3)

    with pytest.raises(FlexConfigError, match="property_package") as excinfo:
        m.node = Resource()

    assert excinfo.value.field == "property_package"


@pytest.mark.unit
def test_an_arc_whose_port_has_different_units_is_refused():
    m = dummy_time_block(3)
    m.pressurized = SimpleAqueousFlow(has_pressure=True)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.node = Resource(property_package=m.pressurized)
    feed = Feed(property_package=m.pressurized, resource_name="well")
    m.plant.feed = feed
    feed.pressure_kpa = pyo.Var(m.time_block.time_index, units=pyunits.kPa)
    feed.outlet_a.remove("pressure")
    feed.outlet_a.add(feed.pressure_kpa, "pressure")
    m.plant.one = Arc(source=feed.outlet_a, destination=m.plant.node.inlet)

    with pytest.raises(FlexConfigError, match="plant.feed") as excinfo:
        verify_resources(m)

    assert excinfo.value.field == "units"


@pytest.mark.unit
def test_an_arc_whose_port_lacks_a_resource_state_is_refused():
    m = dummy_time_block(3)
    m.pressurized = SimpleAqueousFlow(has_pressure=True)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.node = Resource(property_package=m.pressurized)
    m.plant.feed = Feed(property_package=m.properties, resource_name="well")
    m.plant.one = Arc(source=m.plant.feed.outlet_a, destination=m.plant.node.inlet)

    with pytest.raises(FlexConfigError, match="pressure") as excinfo:
        verify_resources(m)

    assert excinfo.value.field == "units"


@pytest.mark.unit
def test_an_arc_from_a_unit_on_an_incompatible_package_is_refused():
    m = dummy_time_block(3)
    m.gas = SimpleGasFlow()
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.node = Resource(property_package=m.properties)
    m.plant.feed = Feed(property_package=m.gas, resource_name="gas")
    m.plant.one = Arc(source=m.plant.feed.outlet_a, destination=m.plant.node.inlet)

    with pytest.raises(FlexConfigError, match="plant.feed") as excinfo:
        verify_resources(m)

    assert excinfo.value.field == "property_package"


@pytest.mark.unit
def test_a_spec_connecting_a_unit_on_an_incompatible_package_is_refused():
    spec = _process(
        {"kind": "property_package", "name": "gas", "property_class": "SimpleGasFlow"}
    )
    for element in spec["elements"]:
        if element["kind"] == "unit":
            element["property_package"] = "properties"
        if element["kind"] == "resource":
            element["property_package"] = "properties"
    next(e for e in spec["elements"] if e.get("name") == "site.well")[
        "property_package"
    ] = "gas"

    with pytest.raises(FlexConfigError, match="site.well") as excinfo:
        build_model(spec)

    assert excinfo.value.field == "property_package"


@pytest.mark.unit
def test_a_node_gives_its_ports_one_shared_intensive_state():
    m = dummy_time_block(3)
    m.pressurized = SimpleAqueousFlow(has_pressure=True)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.node = Resource(property_package=m.pressurized)

    assert len(m.plant.node.pressure_equal) == 3


# --- resource_graph ---------------------------------------------------------


@pytest.mark.unit
def test_graph_classifies_stream_utility_arc_and_open_resources():
    m = _gas_plant(3)
    m.plant.gas = Resource(property_package=_utility(m, "natural_gas"))
    m.plant.feed = Feed(property_package=m.properties, resource_name="well")
    first, second = _pump(m, "first"), _pump(m, "second")
    m.plant.water = Resource(property_package=m.properties)
    m.plant.into = Arc(source=m.plant.feed.outlet_a, destination=m.plant.water.inlet)
    m.plant.out = Arc(source=m.plant.water.outlet, destination=first.inlet)
    m.plant.link = Arc(source=first.outlet, destination=second.inlet)

    graph = resource_graph(m.plant)

    stream = graph.resources["plant.water"]
    assert stream.kind is ResourceKind.STREAM
    assert [(e.unit, e.coeff) for e in stream.edges] == [
        (m.plant.feed, 1.0),
        (first, -1.0),
    ]
    assert graph.resources["plant.gas"].kind is ResourceKind.UTILITY
    assert graph.resources["plant.link"].kind is ResourceKind.ARC
    assert [e.coeff for e in graph.resources["plant.link"].edges] == [1.0, -1.0]
    assert graph.resources["electricity"].kind is ResourceKind.OPEN
    assert {m.plant.feed, first, second} <= set(graph.tasks)
    assert m.plant.water not in graph.tasks


@pytest.mark.unit
def test_graph_does_not_also_report_a_connected_feeds_total_as_open():
    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.feed = Feed(property_package=m.properties, resource_name="water")
    m.plant.node = Resource(property_package=m.properties)
    m.plant.into = Arc(source=m.plant.feed.outlet_a, destination=m.plant.node.inlet)

    assert "water" not in resource_graph(m.plant).resources


@pytest.mark.unit
def test_graph_still_reports_arcs_after_expansion():
    m = _water_plant(3)
    first, second = _pump(m, "first"), _pump(m, "second")
    m.plant.link = Arc(source=first.outlet, destination=second.inlet)
    _expand(m)

    graph = resource_graph(m.plant)

    assert graph.resources["plant.link"].kind is ResourceKind.ARC
    assert len(graph.resources["plant.line"].edges) == 2


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
RAW_WATER = {
    "kind": "property_package",
    "name": "raw_water",
    "property_class": "SimpleUtility",
    "options": {"resource": "raw_water", "units": GAS_FLOW},
}


def _spec(*elements):
    return {
        "schema_version": SCHEMA_VERSION,
        "name": "m",
        "elements": copy.deepcopy([*BASE, RAW_WATER, *elements]),
    }


@pytest.mark.unit
def test_utility_element_builds_after_the_units_it_collects():
    model = build_model(
        _spec(
            {
                "kind": "resource",
                "name": "site.balance",
                "property_package": "raw_water",
            }
        )
    )

    assert list(model.site.balance.edges) == ["site.well.withdrawal"]
    assert len(model.site.balance.balance) == 5


@pytest.mark.unit
def test_resource_element_on_a_unit_is_refused():
    """The parent can't be a unit."""
    with pytest.raises(FlexConfigError, match="parent"):
        build_model(
            _spec(
                {
                    "kind": "resource",
                    "name": "site.well.water",
                    "property_package": "raw_water",
                }
            )
        )


@pytest.mark.unit
def test_utility_element_with_nothing_registered_lists_the_alternatives():
    with pytest.raises(FlexConfigError, match="raw_water"):
        build_model(
            _spec(
                {
                    "kind": "property_package",
                    "name": "gas",
                    "property_class": "SimpleUtility",
                    "options": {"resource": "natural_gas", "units": GAS_FLOW},
                },
                {"kind": "resource", "name": "site.fuel", "property_package": "gas"},
            )
        )


@pytest.mark.unit
def test_a_connection_to_a_utility_is_refused():
    spec = _spec(
        {"kind": "resource", "name": "site.balance", "property_package": "raw_water"}
    )
    spec["elements"][3]["connections"] = [{"port": "outlet_a", "to": "site.balance"}]

    with pytest.raises(FlexConfigError, match="utility") as excinfo:
        build_model(spec)

    assert excinfo.value.field == "to"


def _pump_element(name, intensity, *, into="site.raw", out_to="site.treated"):
    """Return a Pump element connected to two stream nodes."""
    return {
        "kind": "unit",
        "name": name,
        "unit_model_class": "Pump",
        "construction_options": {
            "energy_intensity": {"value": intensity, "units": "kWh/m^3"}
        },
        "costing_package": None,
        "connections": [
            {"port": "inlet", "to": into},
            {"port": "outlet", "to": out_to},
        ],
    }


def _process(*extra):
    """Return a well, one pump and a product joined through two stream nodes."""
    elements = copy.deepcopy(BASE)
    elements[-1]["connections"] = [{"port": "outlet_a", "to": "site.raw"}]
    return {
        "schema_version": SCHEMA_VERSION,
        "name": "m",
        "elements": [
            *elements,
            _pump_element("site.pump_a", 0.5),
            {
                "kind": "unit",
                "name": "site.supply",
                "unit_model_class": "Product",
                "construction_options": {"resource_name": "potable"},
                "costing_package": None,
                "connections": [{"port": "inlet_a", "to": "site.treated"}],
            },
            {"kind": "resource", "name": "site.raw", "property_package": "properties"},
            {
                "kind": "resource",
                "name": "site.treated",
                "property_package": "properties",
            },
            *extra,
        ],
    }


@pytest.mark.unit
def test_connections_to_a_node_build_arcs_in_the_direction_of_the_port():
    model = build_model(_process())

    graph = resource_graph(model.site)

    assert len(list(model.component_objects(Arc, descend_into=True))) == 4
    assert sorted(
        (e.unit.name, e.coeff) for e in graph.resources["site.raw"].edges
    ) == [
        ("site.pump_a", -1.0),
        ("site.well", 1.0),
    ]
    assert sorted(
        (e.unit.name, e.coeff) for e in graph.resources["site.treated"].edges
    ) == [
        ("site.pump_a", 1.0),
        ("site.supply", -1.0),
    ]


@pytest.mark.unit
def test_a_node_may_sit_at_the_model_root():
    spec = _process()
    for element in spec["elements"]:
        if element["kind"] == "resource":
            element["name"] = element["name"].removeprefix("site.")
        for connection in element.get("connections", []):
            connection["to"] = connection["to"].removeprefix("site.")

    model = build_model(spec)

    assert len(resource_graph(model).resources["raw"].edges) == 2


@pytest.mark.unit
def test_every_member_of_an_indexed_unit_connects_to_the_node():
    spec = _process({"kind": "set", "name": "site.trains", "values": [0, 1]})
    spec["elements"][4] = {
        **_pump_element("site.pump", 0.5),
        "index": "site.trains",
    }

    model = build_model(spec)

    edges = resource_graph(model.site).resources["site.raw"].edges
    assert sorted(e.unit.name for e in edges) == [
        "site.pump[0]",
        "site.pump[1]",
        "site.well",
    ]


@pytest.mark.unit
def test_arcs_to_a_node_emit_as_connections_to_the_node():
    m = _water_plant()

    units = {e.name: e for e in emit_model(m).elements if e.kind == "unit"}

    assert [(c.port, c.to) for c in units["plant.feed"].connections] == [
        ("outlet_a", "plant.line")
    ]
    assert [(c.port, c.to) for c in units["plant.product"].connections] == [
        ("inlet_a", "plant.line")
    ]


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
            m.plant.gas = Resource(property_package=_utility(m, "natural_gas"))
        m.cost = pyo.Objective(
            expr=sum(pyo.value(1) * m.plant.gas_feed.withdrawal[t] for t in steps)
        )
        _solve(m)
        return [pyo.value(m.plant.gas_feed.withdrawal[t]) for t in steps]

    assert solve(False) == pytest.approx([0.0] * 24)
    assert solve(True) == pytest.approx(burn)


@pytest.mark.component
@pytest.mark.needs_highs
def test_a_node_with_supply_and_no_sink_forces_the_flow_to_zero(monkeypatch):
    """A resource cannot vanish at a node: no sink means nothing can be supplied."""
    m = dummy_time_block(4)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.feed = Feed(property_package=m.properties, resource_name="well")
    m.plant.node = Resource(property_package=m.properties)
    m.plant.one = Arc(source=m.plant.feed.outlet_a, destination=m.plant.node.inlet)
    _expand(m)
    steps = list(m.time_block.time_index)
    for t in steps:
        m.plant.feed.withdrawal[t].setub(10.0)
    m.reward = pyo.Objective(
        expr=sum(m.plant.feed.withdrawal[t] for t in steps), sense=pyo.maximize
    )

    monkeypatch.setattr(facade, "PRE_SOLVE_CHECKS", [])

    _solve(m)

    assert [pyo.value(m.plant.feed.withdrawal[t]) for t in steps] == pytest.approx(
        [0.0] * 4
    )


@pytest.mark.component
@pytest.mark.needs_highs
def test_a_node_with_a_sink_and_no_supply_forces_the_flow_to_zero(monkeypatch):
    """A resource cannot appear at a node: no supply means nothing can be drawn."""
    m = dummy_time_block(4)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.product = Product(property_package=m.properties, resource_name="supply")
    m.plant.node = Resource(property_package=m.properties)
    m.plant.one = Arc(source=m.plant.node.outlet, destination=m.plant.product.inlet_a)
    _expand(m)
    steps = list(m.time_block.time_index)
    for t in steps:
        m.plant.product.delivery[t].setub(10.0)
    m.reward = pyo.Objective(
        expr=sum(m.plant.product.delivery[t] for t in steps), sense=pyo.maximize
    )

    monkeypatch.setattr(facade, "PRE_SOLVE_CHECKS", [])

    _solve(m)

    assert [pyo.value(m.plant.product.delivery[t]) for t in steps] == pytest.approx(
        [0.0] * 4
    )


@pytest.mark.component
@pytest.mark.needs_highs
def test_a_node_passes_its_intensive_state_between_ports():
    m = dummy_time_block(3)
    m.pressurized = SimpleAqueousFlow(has_pressure=True)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.feed = Feed(property_package=m.pressurized, resource_name="well")
    m.plant.product = Product(property_package=m.pressurized, resource_name="supply")
    m.plant.node = Resource(property_package=m.pressurized)
    m.plant.one = Arc(source=m.plant.feed.outlet_a, destination=m.plant.node.inlet)
    m.plant.two = Arc(source=m.plant.node.outlet, destination=m.plant.product.inlet_a)
    _expand(m)
    steps = list(m.time_block.time_index)
    for t in steps:
        m.plant.feed.outlet_a_state.pressure[t].fix(250000.0)
        m.plant.feed.withdrawal[t].setlb(0.0)
    m.zero = pyo.Objective(expr=sum(m.plant.feed.withdrawal[t] for t in steps))

    _solve(m)

    assert pyo.value(m.plant.product.inlet_a_state.pressure[1]) == pytest.approx(
        250000.0
    )


@pytest.mark.component
@pytest.mark.needs_highs
def test_a_tank_process_solves_the_same_through_nodes_and_through_arcs():
    """An Arc is a 2-edge resource, so routing through nodes keeps the objective."""
    price = [1.0 + 0.5 * (t % 6) for t in range(24)]
    demand = [10.0 if t % 3 else 25.0 for t in range(24)]

    def solve(connect):
        m = dummy_time_block(24)
        m.plant = PlantBlock(time_block=m.time_block)
        m.plant.feed = Feed(property_package=m.properties, resource_name="well")
        m.plant.tank = Tank(
            property_package=m.properties,
            max_volume=80.0 * pyunits.m**3,
            initial_volume=30.0 * pyunits.m**3,
        )
        m.plant.product = Product(property_package=m.properties, resource_name="supply")
        connect(m)
        _expand(m)
        steps = list(m.time_block.time_index)
        m.demand = pyo.Constraint(
            steps, rule=lambda m, t: m.plant.product.delivery[t] >= demand[t]
        )
        m.cost = pyo.Objective(
            expr=sum(price[t] * m.plant.feed.withdrawal[t] for t in steps)
        )
        _solve(m)
        return pyo.value(m.cost)

    def with_arcs(m):
        m.plant.into_tank = Arc(
            source=m.plant.feed.outlet_a, destination=m.plant.tank.inlet
        )
        m.plant.out_of_tank = Arc(
            source=m.plant.tank.outlet, destination=m.plant.product.inlet_a
        )

    def with_nodes(m):
        p = m.plant
        p.raw = Resource(property_package=m.properties)
        p.stored = Resource(property_package=m.properties)
        p.a1 = Arc(source=p.feed.outlet_a, destination=p.raw.inlet)
        p.a2 = Arc(source=p.raw.outlet, destination=p.tank.inlet)
        p.a3 = Arc(source=p.tank.outlet, destination=p.stored.inlet)
        p.a4 = Arc(source=p.stored.outlet, destination=p.product.inlet_a)

    assert solve(with_nodes) == pytest.approx(solve(with_arcs))


@pytest.mark.component
@pytest.mark.needs_highs
def test_parallel_pumps_between_nodes_can_change_their_split_every_step():
    """One split fraction over the horizon would make these limits infeasible."""
    m = dummy_time_block(4)
    p = m.plant = PlantBlock(time_block=m.time_block)
    p.feed = Feed(property_package=m.properties, resource_name="well")
    p.product = Product(property_package=m.properties, resource_name="supply")
    p.raw = Resource(property_package=m.properties)
    p.treated = Resource(property_package=m.properties)
    first, second = _pump(m, "a"), _pump(m, "b")
    p.a1 = Arc(source=p.feed.outlet_a, destination=p.raw.inlet)
    p.a2 = Arc(source=p.raw.outlet, destination=first.inlet)
    p.a3 = Arc(source=p.raw.outlet, destination=second.inlet)
    p.a4 = Arc(source=first.outlet, destination=p.treated.inlet)
    p.a5 = Arc(source=second.outlet, destination=p.treated.inlet)
    p.a6 = Arc(source=p.treated.outlet, destination=p.product.inlet_a)
    _expand(m)
    steps = list(m.time_block.time_index)
    for t in steps:
        p.product.delivery[t].fix(12.0)
        first.flow_out[t].setub(10.0 if t % 2 else 2.0)
        second.flow_out[t].setub(2.0 if t % 2 else 10.0)
    m.zero = pyo.Objective(expr=sum(p.feed.withdrawal[t] for t in steps))

    _solve(m)

    assert [pyo.value(first.flow_out[t]) for t in steps] == pytest.approx(
        [2.0, 10.0, 2.0, 10.0]
    )


@pytest.mark.component
@pytest.mark.needs_highs
def test_a_candidate_unit_joins_by_connecting_with_no_other_element_edited():
    """Adding a cheaper pump to the spec reroutes the flow through it alone."""

    def solve(*extra):
        model = build_model(_process(*extra))
        steps = list(model.time_block.time_index)
        pumps = [model.site.pump_a]
        if extra:
            pumps.append(model.site.pump_b)
        for t in steps:
            model.site.supply.delivery[t].fix(10.0)
            for pump in pumps:
                pump.flow_in[t].setlb(0.0)
                pump.flow_out[t].setlb(0.0)
        model.cost = pyo.Objective(
            expr=sum(p.power_electrical[t] for p in pumps for t in steps)
        )
        _solve(model)
        return model, pumps

    _, (only_a,) = solve()
    model, (pump_a, pump_b) = solve(_pump_element("site.pump_b", 0.3))

    assert pyo.value(only_a.flow_out[0]) == pytest.approx(10.0)
    assert pyo.value(pump_a.flow_out[0]) == pytest.approx(0.0)
    assert pyo.value(pump_b.flow_out[0]) == pytest.approx(10.0)
    assert pyo.value(model.cost) == pytest.approx(0.3 * 10.0 * 5)


@pytest.mark.component
@pytest.mark.needs_ipopt
def test_ipopt_on_a_closed_recycle_lands_where_its_start_takes_it():
    """Equal counts of rows and variables let IPOPT stop without minimizing."""

    def solve(start):
        m = dummy_time_block(4)
        p = m.plant = PlantBlock(time_block=m.time_block)
        first, second = _pump(m, "a"), _pump(m, "b")
        p.ab = Resource(property_package=m.properties)
        p.ba = Resource(property_package=m.properties)
        p.a1 = Arc(source=first.outlet, destination=p.ab.inlet)
        p.a2 = Arc(source=p.ab.outlet, destination=second.inlet)
        p.a3 = Arc(source=second.outlet, destination=p.ba.inlet)
        p.a4 = Arc(source=p.ba.outlet, destination=first.inlet)
        _expand(m)
        if start is not None:
            for var in m.component_data_objects(pyo.Var):
                if "flow" in var.name and not var.fixed:
                    var.set_value(start)
        target = 5 * pyunits.m**3 / pyunits.hr
        steps = list(m.time_block.time_index)
        m.fit = pyo.Objective(expr=sum((first.flow_in[t] - target) ** 2 for t in steps))
        pyo.assert_optimal_termination(get_solver(model=m, prefer="ipopt").solve(m))
        return pyo.value(second.flow_out[0])

    # IPOPT stops once its scaled error is 0 (here 4.977 at 5.8e-3 dual
    # infeasibility), so the default start lands near, not on, the optimum.
    assert solve(None) == pytest.approx(5.0, abs=0.1)
    assert solve(1000.0) != pytest.approx(5.0, abs=1.0)


# --- verify_resources -------------------------------------------------------


def _both_signs_gas_plant():
    """Return a gas plant whose utility is built and has a supplier and a consumer."""
    m = _gas_plant(2)
    m.plant.gas = Resource(property_package=_utility(m, "natural_gas"))
    return m


@pytest.mark.unit
def test_verify_passes_on_a_utility_with_both_signs(recwarn):
    verify_resources(_both_signs_gas_plant())

    assert not recwarn.list


@pytest.mark.unit
def test_a_unit_added_after_its_utility_is_refused():
    m = _both_signs_gas_plant()
    m.plant.late = GasBurner()

    with pytest.raises(FlexConfigError, match="plant.late") as excinfo:
        verify_resources(m)

    assert excinfo.value.field == "property_package"


@pytest.mark.unit
def test_a_unit_deactivated_after_its_utility_is_refused():
    m = _both_signs_gas_plant()
    m.plant.burner.deactivate()

    with pytest.raises(FlexConfigError, match="plant.burner"):
        verify_resources(m)


@pytest.mark.unit
def test_a_unit_reactivated_after_its_utility_is_refused():
    m = _gas_plant(2)
    m.plant.burner.deactivate()
    m.plant.gas = Resource(property_package=_utility(m, "natural_gas"))
    m.plant.burner.activate()

    with pytest.raises(FlexConfigError, match="plant.burner"):
        verify_resources(m)


@extension
def _add_late_burner(model):
    model.site.late = GasBurner(fuel_name="raw_water")


@pytest.mark.unit
def test_build_model_refuses_a_unit_an_extension_adds_after_a_utility():
    spec = _spec(
        {"kind": "resource", "name": "site.balance", "property_package": "raw_water"},
        {
            "kind": "extension",
            "name": "late_burner",
            "entry_point": "flexops.tests.core.test_resources:_add_late_burner",
        },
    )

    with pytest.raises(FlexConfigError, match="site.late"):
        build_model(spec)


@pytest.mark.component
@pytest.mark.needs_highs
def test_solving_runs_verify_resources():
    m = _both_signs_gas_plant()
    m.plant.late = GasBurner()
    m.cost = pyo.Objective(expr=m.plant.gas_feed.withdrawal[0])

    with pytest.raises(FlexConfigError, match="plant.late"):
        get_solver(model=m, prefer="highs").solve(m)


@pytest.mark.unit
def test_verify_warns_when_an_electricity_utility_has_only_draws():
    m = dummy_time_block(2)
    m.plant = PlantBlock(time_block=m.time_block)
    _pump(m, "pump")
    m.plant.power = Resource(property_package=_utility(m, "electricity", "kW"))

    with pytest.warns(FlexResourceWarning, match="plant.power"):
        verify_resources(m)


@pytest.mark.unit
@pytest.mark.parametrize("side", ["supply", "consume"])
def test_verify_warns_when_a_stream_has_arcs_on_one_side(side):
    m = dummy_time_block(2)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.node = Resource(property_package=m.properties)
    pump = _pump(m, "pump")
    if side == "supply":
        m.plant.one = Arc(source=pump.outlet, destination=m.plant.node.inlet)
    else:
        m.plant.one = Arc(source=m.plant.node.outlet, destination=pump.inlet)

    with pytest.warns(FlexResourceWarning, match="plant.node"):
        verify_resources(m)


@pytest.mark.unit
def test_an_electricity_utility_beside_a_grid_connection_is_refused():
    m = dummy_time_block(2)
    m.plant = PlantBlock(time_block=m.time_block)
    _pump(m, "pump")
    m.plant.grid = GridConnection()
    m.plant.power = Resource(property_package=_utility(m, "electricity", "kW"))

    with pytest.raises(FlexConfigError, match="GridConnection") as excinfo:
        verify_resources(m)

    assert excinfo.value.field == "property_package"


@pytest.mark.unit
def test_a_second_expansion_on_a_node_is_refused():
    """The first equality would survive beside the new arc's and force them equal."""
    m = _water_plant(2)
    _expand(m)
    m.plant.feed_2 = Feed(property_package=m.properties, resource_name="well_2")
    m.plant.second = Arc(source=m.plant.feed_2.outlet_a, destination=m.plant.line.inlet)

    with pytest.raises(FlexConfigError, match="expanded"):
        _expand(m)


@pytest.mark.unit
def test_a_term_registered_twice_under_one_label_is_refused():
    m = dummy_time_block(2)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.burner = GasBurner()
    m.plant.burner.register_fuel_usage(
        m.plant.burner.fuel_usage_natural_gas, fuel_name="natural_gas"
    )

    with pytest.raises(FlexConfigError, match="fuel_usage_natural_gas") as excinfo:
        m.plant.gas = Resource(property_package=_utility(m, "natural_gas"))

    assert excinfo.value.field == "property_package"


@pytest.mark.unit
def test_a_boundary_term_on_a_unit_with_connected_ports_is_refused():
    """A Feed's withdrawal meters its ports, so an Arc already conserves it."""
    m = dummy_time_block(2)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.feed = Feed(property_package=m.properties, resource_name="natural_gas")
    pump = _pump(m, "pump")
    m.plant.one = Arc(source=m.plant.feed.outlet_a, destination=pump.inlet)

    with pytest.raises(FlexConfigError, match="plant.feed") as excinfo:
        m.plant.gas = Resource(property_package=_utility(m, "natural_gas"))

    assert excinfo.value.field == "property_package"


@pytest.mark.unit
def test_an_arc_to_a_deactivated_unit_is_refused():
    m = dummy_time_block(2)
    p = m.plant = PlantBlock(time_block=m.time_block)
    p.node = Resource(property_package=m.properties)
    first, second = _pump(m, "a"), _pump(m, "b")
    p.one = Arc(source=first.outlet, destination=p.node.inlet)
    p.two = Arc(source=second.outlet, destination=p.node.inlet)
    p.three = Arc(source=p.node.outlet, destination=_pump(m, "c").inlet)
    _expand(m)
    second.deactivate()

    with pytest.raises(FlexConfigError, match="plant.b") as excinfo:
        verify_resources(m)

    assert excinfo.value.field == "port"


@pytest.mark.component
@pytest.mark.needs_highs
def test_fixing_a_candidates_flow_to_zero_switches_it_off():
    m = dummy_time_block(2)
    p = m.plant = PlantBlock(time_block=m.time_block)
    p.feed = Feed(property_package=m.properties, resource_name="well")
    p.product = Product(property_package=m.properties, resource_name="supply")
    p.raw = Resource(property_package=m.properties)
    p.treated = Resource(property_package=m.properties)
    first, second = _pump(m, "a"), _pump(m, "b")
    p.a1 = Arc(source=p.feed.outlet_a, destination=p.raw.inlet)
    p.a2 = Arc(source=p.raw.outlet, destination=first.inlet)
    p.a3 = Arc(source=p.raw.outlet, destination=second.inlet)
    p.a4 = Arc(source=first.outlet, destination=p.treated.inlet)
    p.a5 = Arc(source=second.outlet, destination=p.treated.inlet)
    p.a6 = Arc(source=p.treated.outlet, destination=p.product.inlet_a)
    _expand(m)
    steps = list(m.time_block.time_index)
    for t in steps:
        second.flow_in[t].fix(0.0)
        p.product.delivery[t].fix(10.0)
        p.feed.withdrawal[t].setlb(0.0)
    m.cost = pyo.Objective(expr=sum(p.feed.withdrawal[t] for t in steps))

    _solve(m)

    assert pyo.value(p.feed.withdrawal[0]) == pytest.approx(10.0)
    assert pyo.value(first.flow_out[0]) == pytest.approx(10.0)


@pytest.mark.unit
def test_verify_checks_the_package_a_port_actually_uses():
    """A unit with several packages has no single config.property_package."""
    m = dummy_time_block(2)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.node = Resource(property_package=m.properties)
    m.plant.unit = Pump(property_package=m.properties, energy_intensity=INTENSITY)
    m.plant.unit.config.property_package = None
    m.plant.one = Arc(source=m.plant.unit.outlet, destination=m.plant.node.inlet)
    m.plant.two = Arc(source=m.plant.node.outlet, destination=_pump(m, "b").inlet)

    verify_resources(m)


@pytest.mark.unit
def test_an_arc_from_a_plain_block_is_refused_clearly():
    m = dummy_time_block(2)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.node = Resource(property_package=m.properties)
    m.plant.raw_block = pyo.Block()
    m.plant.raw_block.flow = pyo.Var()
    m.plant.raw_block.out = Port()
    m.plant.raw_block.out.add(m.plant.raw_block.flow, "flow_vol_phase")
    m.plant.one = Arc(source=m.plant.raw_block.out, destination=m.plant.node.inlet)

    with pytest.raises(FlexConfigError, match="raw_block") as excinfo:
        verify_resources(m)

    assert excinfo.value.field == "property_package"


@pytest.mark.unit
def test_a_node_flow_fixed_before_expansion_is_refused():
    """Opening a side would otherwise quietly drop the fix."""
    m = _water_plant(2)
    m.plant.line.inlet_state.flow_vol_phase[0, "Liq"].fix(5.0)

    with pytest.raises(FlexConfigError, match="before") as excinfo:
        _expand(m)

    assert excinfo.value.field == "port"


@pytest.mark.unit
def test_an_unknown_balance_type_is_refused():
    m = dummy_time_block(2)

    with pytest.raises(ValueError, match="nonsense"):
        m.node = Resource(property_package=m.properties, balance="nonsense")


@pytest.mark.unit
def test_two_utilities_on_one_resource_in_one_block_are_refused():
    m = _both_signs_gas_plant()

    with pytest.raises(FlexConfigError, match="plant.gas") as excinfo:
        m.plant.gas_2 = Resource(property_package=_utility(m, "natural_gas"))

    assert excinfo.value.field == "property_package"


@pytest.mark.unit
def test_an_indexed_resource_is_refused():
    m = dummy_time_block(2)
    m.s = pyo.Set(initialize=[0, 1])

    with pytest.raises(FlexConfigError, match="indexed") as excinfo:
        m.node = Resource(m.s, property_package=m.properties)

    assert excinfo.value.field == "property_package"


@pytest.mark.unit
def test_an_arc_to_a_node_over_its_own_index_cannot_be_emitted():
    m = _water_plant(2)
    m.s = pyo.Set(initialize=[0])
    m.plant.pump = Pump(property_package=m.properties, energy_intensity=INTENSITY)
    m.plant.indexed = Arc(
        m.s,
        rule=lambda b, i: {"source": b.line.outlet, "destination": b.pump.inlet},
    )

    with pytest.raises(FlexConfigError, match="plant.indexed"):
        emit_model(m, check=False)


# --- balance types ----------------------------------------------------------


def _one_pump_between(inlet_balance, outlet_balance, n: int = 3):
    """Return a pump between two stream resources with the given balance types."""
    m = dummy_time_block(n)
    m.plant = PlantBlock(time_block=m.time_block)
    p = m.plant
    pump = _pump(m, "pump")
    p.source = Resource(property_package=m.properties, balance=inlet_balance)
    p.sink = Resource(property_package=m.properties, balance=outlet_balance)
    p.into_pump = Arc(source=p.source.outlet, destination=pump.inlet)
    p.out_of_pump = Arc(source=pump.outlet, destination=p.sink.inlet)
    _expand(m)
    return m


@pytest.mark.unit
def test_a_resource_is_balanced_by_default():
    m = dummy_time_block(2)
    m.node = Resource(property_package=m.properties)

    assert m.node.config.balance is BalanceType.BALANCED
    assert all(row.equality for row in m.node.balance.values())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("balance", "inflow", "outflow", "feasible"),
    [
        (BalanceType.EXPORT, 3.0, 2.0, True),
        (BalanceType.EXPORT, 2.0, 3.0, False),
        (BalanceType.IMPORT, 2.0, 3.0, True),
        (BalanceType.IMPORT, 3.0, 2.0, False),
    ],
)
def test_a_one_way_stream_balance_allows_only_its_direction(
    balance, inflow, outflow, feasible
):
    m = dummy_time_block(2)
    m.node = Resource(property_package=m.properties, balance=balance)
    for t in m.time_block.time_index:
        m.node.inlet_state.flow_vol_phase[t, "Liq"].set_value(inflow)
        m.node.outlet_state.flow_vol_phase[t, "Liq"].set_value(outflow)

    for row in m.node.balance.values():
        assert (row.lslack() >= 0 and row.uslack() >= 0) is feasible


@pytest.mark.unit
def test_an_unbounded_stream_has_no_flow_balance_but_keeps_one_state():
    m = dummy_time_block(2)
    m.pressurized = SimpleAqueousFlow(has_pressure=True)
    m.node = Resource(property_package=m.pressurized, balance=BalanceType.UNBOUNDED)

    assert m.node.component("balance") is None
    assert m.node.component("pressure_equal") is not None


@pytest.mark.unit
def test_an_unbounded_utility_keeps_its_edges_but_has_no_balance():
    m = _gas_plant()
    m.plant.gas = Resource(
        property_package=_utility(m, "natural_gas"), balance=BalanceType.UNBOUNDED
    )

    assert len(list(m.plant.gas.edges)) == 2
    assert m.plant.gas.component("balance") is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("balance", "sign", "bound"),
    [(BalanceType.EXPORT, ">=", "lower"), (BalanceType.IMPORT, "<=", "upper")],
)
def test_a_one_way_utility_balance_is_an_inequality(balance, sign, bound):
    m = _gas_plant(2)
    m.plant.gas = Resource(property_package=_utility(m, "natural_gas"), balance=balance)

    row = m.plant.gas.balance[0]
    assert not row.equality
    assert pyo.value(row.lower if bound == "lower" else row.upper) == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("balance", "side", "missing"),
    [
        (BalanceType.BALANCED, "outlet", "no import"),
        (BalanceType.BALANCED, "inlet", "no consumer"),
        (BalanceType.EXPORT, "outlet", "no import"),
        (BalanceType.IMPORT, "inlet", "no consumer"),
    ],
)
def test_a_balance_that_forces_a_one_sided_stream_to_zero_is_reported(
    balance, side, missing
):
    m = dummy_time_block(2)
    m.plant = PlantBlock(time_block=m.time_block)
    pump = _pump(m, "pump")
    m.plant.node = Resource(property_package=m.properties, balance=balance)
    if side == "outlet":
        m.plant.one = Arc(source=m.plant.node.outlet, destination=pump.inlet)
    else:
        m.plant.one = Arc(source=pump.outlet, destination=m.plant.node.inlet)

    with pytest.raises(FlexConfigError, match=missing):
        verify_resources(m, error_type="error")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("balance", "side"),
    [
        (BalanceType.IMPORT, "outlet"),
        (BalanceType.EXPORT, "inlet"),
        (BalanceType.UNBOUNDED, "outlet"),
        (BalanceType.UNBOUNDED, "inlet"),
    ],
)
def test_a_boundary_balance_lets_a_one_sided_stream_carry_flow(balance, side, recwarn):
    m = dummy_time_block(2)
    m.plant = PlantBlock(time_block=m.time_block)
    pump = _pump(m, "pump")
    m.plant.node = Resource(property_package=m.properties, balance=balance)
    if side == "outlet":
        m.plant.one = Arc(source=m.plant.node.outlet, destination=pump.inlet)
    else:
        m.plant.one = Arc(source=pump.outlet, destination=m.plant.node.inlet)

    verify_resources(m, error_type="error")

    assert not recwarn.list


@pytest.mark.unit
def test_an_unbounded_electricity_utility_can_be_the_only_supply(recwarn):
    m = dummy_time_block(2)
    m.plant = PlantBlock(time_block=m.time_block)
    _pump(m, "pump")
    m.plant.grid = Resource(
        property_package=_utility(m, "electricity", "kW"),
        balance=BalanceType.UNBOUNDED,
    )

    verify_resources(m, error_type="error")

    assert not recwarn.list


@pytest.mark.unit
def test_a_grid_connection_beside_any_electricity_utility_is_refused():
    m = dummy_time_block(2)
    m.plant = PlantBlock(time_block=m.time_block)
    _pump(m, "pump")
    m.plant.grid_unit = GridConnection()
    m.plant.grid = Resource(
        property_package=_utility(m, "electricity", "kW"),
        balance=BalanceType.UNBOUNDED,
    )

    with pytest.raises(FlexConfigError, match="GridConnection"):
        verify_resources(m)


@pytest.mark.component
@pytest.mark.needs_highs
def test_an_import_source_and_export_sink_replace_boundary_units():
    """A pump with no Feed or Product runs between a source and a sink."""
    m = _one_pump_between(BalanceType.IMPORT, BalanceType.EXPORT)
    pump = m.plant.pump
    for t in m.time_block.time_index:
        pump.inlet_state.flow_vol_phase[t, "Liq"].fix(4.0)
    m.cost = pyo.Objective(expr=0)

    _solve(m)

    assert pyo.value(pump.outlet_state.flow_vol_phase[0, "Liq"]) == pytest.approx(4.0)


@pytest.mark.component
@pytest.mark.needs_highs
@pytest.mark.parametrize(
    ("balance", "cheapest"),
    [
        (BalanceType.BALANCED, 5.0),
        (BalanceType.EXPORT, 5.0),
        (BalanceType.IMPORT, 0.0),
        (BalanceType.UNBOUNDED, 0.0),
    ],
)
def test_a_balance_type_sets_how_much_supply_a_fixed_demand_needs(balance, cheapest):
    """Paying for supply: only import and unbounded let demand exceed it."""
    m = dummy_time_block(2)
    m.plant = PlantBlock(time_block=m.time_block)
    p = m.plant
    p.feed = Feed(property_package=m.properties, resource_name="well")
    p.supply = Product(property_package=m.properties, resource_name="demand")
    p.node = Resource(property_package=m.properties, balance=balance)
    p.into = Arc(source=p.feed.outlet_a, destination=p.node.inlet)
    p.out = Arc(source=p.node.outlet, destination=p.supply.inlet_a)
    _expand(m)
    steps = list(m.time_block.time_index)
    for t in steps:
        p.supply.inlet_a_state.flow_vol_phase[t, "Liq"].fix(5.0)
    m.cost = pyo.Objective(
        expr=sum(p.feed.outlet_a_state.flow_vol_phase[t, "Liq"] for t in steps)
    )

    _solve(m)

    drawn = pyo.value(p.feed.outlet_a_state.flow_vol_phase[steps[0], "Liq"])
    assert drawn == pytest.approx(cheapest)


# --- resource_graph edges and scope -----------------------------------------


@pytest.mark.unit
def test_graph_keeps_a_utility_to_the_block_it_balances():
    m = dummy_time_block(2)
    for name in ("a", "b"):
        plant = PlantBlock(time_block=m.time_block)
        m.add_component(name, plant)
        plant.feed = Feed(property_package=m.properties, resource_name="natural_gas")
        plant.burner = GasBurner()
    m.a.gas = Resource(property_package=_utility(m, "natural_gas"))

    graph = resource_graph(m)

    assert {e.unit for e in graph.resources["a.gas"].edges} == {m.a.feed, m.a.burner}
    assert graph.resources["natural_gas"].kind is ResourceKind.OPEN
    assert {e.unit for e in graph.resources["natural_gas"].edges} == {
        m.b.feed,
        m.b.burner,
    }


@pytest.mark.unit
def test_graph_shows_a_utility_sharing_a_name_with_a_stream():
    m = dummy_time_block(2)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.feed = Feed(property_package=m.properties, resource_name="water")
    m.plant.term = Resource(property_package=_utility(m, "water"))
    m.plant.node = Resource(property_package=m.properties)

    graph = resource_graph(m)

    assert graph.resources["plant.term"].kind is ResourceKind.UTILITY
    assert graph.resources["plant.node"].kind is ResourceKind.STREAM


@pytest.mark.unit
def test_graph_edges_on_a_shared_port_are_the_arc_flows():
    m = dummy_time_block(2)
    p = m.plant = PlantBlock(time_block=m.time_block)
    p.node = Resource(property_package=m.properties)
    first, second = _pump(m, "a"), _pump(m, "b")
    p.one = Arc(source=first.outlet, destination=p.node.inlet)
    p.two = Arc(source=second.outlet, destination=p.node.inlet)

    before = resource_graph(p).resources["plant.node"].edges
    _expand(m)
    after = resource_graph(p).resources["plant.node"].edges

    assert [edge.var for edge in before] == [None, None]
    assert after[0].var is p.one_expanded.flow_vol_phase
    assert after[1].var is p.two_expanded.flow_vol_phase


@pytest.mark.unit
def test_graph_of_a_plant_shows_a_node_outside_it_that_its_units_reach():
    m = dummy_time_block(2)
    m.plant = PlantBlock(time_block=m.time_block)
    m.raw = Resource(property_package=m.properties)
    pump = _pump(m, "pump")
    m.one = Arc(source=m.raw.outlet, destination=pump.inlet)

    graph = resource_graph(m.plant)

    assert graph.resources["raw"].kind is ResourceKind.STREAM
    assert [edge.unit for edge in graph.resources["raw"].edges] == [pump]


# --- one-sided balances -----------------------------------------------------


def _gas_utility(*units):
    """Return a plant with the named gas units and a natural gas utility."""
    m = dummy_time_block(2)
    m.plant = PlantBlock(time_block=m.time_block)
    if "feed" in units:
        m.plant.gas_feed = Feed(
            property_package=m.properties, resource_name="natural_gas"
        )
    if "burner" in units:
        m.plant.burner = GasBurner()
    m.plant.gas = Resource(property_package=_utility(m, "natural_gas"))
    return m


def _one_sided_stream(side):
    """Return a plant whose stream has arcs only on the given side."""
    m = dummy_time_block(2)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.node = Resource(property_package=m.properties)
    pump = _pump(m, "pump")
    if side == "supply":
        m.plant.one = Arc(source=pump.outlet, destination=m.plant.node.inlet)
    else:
        m.plant.one = Arc(source=m.plant.node.outlet, destination=pump.inlet)
    return m


ONE_SIDED = [
    pytest.param(lambda: _gas_utility("burner"), "no import", id="burner_only"),
    pytest.param(lambda: _gas_utility("feed"), "no consumer", id="feed_only"),
    pytest.param(lambda: _one_sided_stream("supply"), "no consumer", id="stream_in"),
    pytest.param(lambda: _one_sided_stream("consume"), "no import", id="stream_out"),
]


@pytest.mark.unit
@pytest.mark.parametrize(("build", "missing"), ONE_SIDED)
def test_a_one_sided_balance_raises_on_error_and_names_what_is_missing(build, missing):
    with pytest.raises(FlexConfigError, match=missing):
        verify_resources(build(), error_type="error")


@pytest.mark.unit
@pytest.mark.parametrize(("build", "missing"), ONE_SIDED)
def test_a_one_sided_balance_warns_by_default(build, missing):
    with pytest.warns(FlexResourceWarning, match=missing):
        verify_resources(build())


@pytest.mark.unit
def test_error_type_accepts_its_enum():
    with pytest.raises(FlexConfigError, match="no import"):
        verify_resources(_gas_utility("burner"), error_type=ErrorType.ERROR)


@pytest.mark.unit
def test_an_unknown_error_type_is_refused():
    with pytest.raises(FlexConfigError, match="error_type") as excinfo:
        verify_resources(_gas_utility("burner", "feed"), error_type="ignore")

    assert excinfo.value.field == "error_type"


@pytest.mark.unit
def test_a_stream_with_no_arcs_neither_warns_nor_raises(recwarn):
    m = dummy_time_block(2)
    m.node = Resource(property_package=m.properties)

    verify_resources(m, error_type="error")

    assert not recwarn.list


@pytest.mark.component
@pytest.mark.needs_highs
def test_solving_a_one_sided_balance_raises_before_the_solver_runs():
    m = _gas_utility("burner")
    m.cost = pyo.Objective(expr=m.plant.burner.fuel_usage_natural_gas[0])

    with pytest.raises(FlexConfigError, match="no import"):
        get_solver(model=m, prefer="highs").solve(m)

    assert m.plant.burner.fuel_usage_natural_gas[0].value == 0.0


# --- electricity ------------------------------------------------------------


def _battery(m):
    """Add a battery with no commitment binary to the plant."""
    m.plant.battery = BatteryModel(
        capacity=100 * pyunits.kWh,
        initial_soc=0.5,
        unit_commitment=UnitCommitmentConfig(status=False),
    )


def _solar(m):
    """Add a renewable generator to the plant."""
    m.plant.solar = GenericRenewables(
        capacity=10 * pyunits.kW, capacity_factor=[0.2, 0.5]
    )


def _plant(*adds):
    """Return a two-step plant with each of the given units added."""
    m = dummy_time_block(2)
    m.plant = PlantBlock(time_block=m.time_block)
    for add in adds:
        add(m)
    return m


def _power_utility_plant(*adds):
    """Return ``_plant(*adds)`` with an electricity utility on it."""
    m = _plant(*adds)
    m.plant.power = Resource(property_package=_utility(m, "electricity", "kW"))
    return m


WRONG_ELECTRICITY = [
    pytest.param([lambda m: _pump(m, "pump")], "no import", id="only_a_draw"),
    pytest.param([_battery], "no consumer", id="battery_alone"),
    pytest.param([_solar], "no consumer", id="generator_alone"),
]


@pytest.mark.unit
@pytest.mark.parametrize(("adds", "missing"), WRONG_ELECTRICITY)
def test_an_electricity_utility_that_forces_zero_raises_on_error(adds, missing):
    m = _power_utility_plant(*adds)

    with pytest.raises(FlexConfigError, match=missing):
        verify_resources(m, error_type="error")


@pytest.mark.unit
@pytest.mark.parametrize(("adds", "missing"), WRONG_ELECTRICITY)
def test_an_electricity_utility_that_forces_zero_warns_by_default(adds, missing):
    m = _power_utility_plant(*adds)

    with pytest.warns(FlexResourceWarning, match=missing):
        verify_resources(m)


def _imported_power_plant(*adds):
    """Return ``_plant(*adds)`` with an IMPORT electricity utility on it."""
    m = _plant(*adds)
    m.plant.power = Resource(
        property_package=_utility(m, "electricity", "kW"),
        balance=BalanceType.IMPORT,
    )
    return m


@pytest.mark.unit
@pytest.mark.parametrize("add", [lambda m: _pump(m, "pump"), _battery])
def test_one_unit_that_can_draw_is_enough_for_an_import_electricity_utility(add):
    verify_resources(_imported_power_plant(add), error_type="error")


@pytest.mark.unit
def test_a_lone_generator_on_an_import_electricity_utility_raises():
    with pytest.raises(FlexConfigError, match="no consumer"):
        verify_resources(_imported_power_plant(_solar), error_type="error")


@pytest.mark.component
@pytest.mark.needs_highs
def test_a_lone_pump_on_an_import_electricity_utility_solves_with_its_draw():
    m = _imported_power_plant(_pumped_water)
    m.cost = pyo.Objective(expr=m.plant.feed.withdrawal[0])

    pyo.assert_optimal_termination(get_solver(model=m, prefer="highs").solve(m))

    assert pyo.value(m.plant.pump.power_electrical[0]) > 0


@pytest.mark.component
@pytest.mark.needs_highs
def test_a_lone_battery_on_an_import_electricity_utility_only_charges():
    m = _imported_power_plant(_battery)
    battery = m.plant.battery
    m.goal = pyo.Objective(expr=battery.power_charge[0], sense=pyo.maximize)
    solver = get_solver(model=m, prefer="highs")

    pyo.assert_optimal_termination(solver.solve(m))
    assert pyo.value(battery.power_charge[0]) > 0

    m.goal.deactivate()
    m.discharge = pyo.Objective(
        expr=sum(battery.power_discharge[t] - battery.power_charge[t] for t in [0, 1]),
        sense=pyo.maximize,
    )

    pyo.assert_optimal_termination(solver.solve(m))
    assert pyo.value(battery.power_electrical[0]) >= -1e-6
    assert pyo.value(battery.power_electrical[1]) >= -1e-6


@pytest.mark.component
@pytest.mark.needs_highs
def test_solving_an_electricity_utility_with_only_draws_raises():
    m = _plant(_pumped_water)
    m.plant.power = Resource(property_package=_utility(m, "electricity", "kW"))
    m.cost = pyo.Objective(expr=m.plant.feed.withdrawal[0])

    with pytest.raises(FlexConfigError, match="no import"):
        get_solver(model=m, prefer="highs").solve(m)


@pytest.mark.unit
def test_a_spec_utility_built_after_a_battery_gives_no_warning():
    """The topology stage runs before a utility exists, so it can't warn yet."""
    spec = {
        "schema_version": SCHEMA_VERSION,
        "name": "m",
        "elements": [
            copy.deepcopy(BASE[0]),
            copy.deepcopy(BASE[1]),
            copy.deepcopy(BASE[2]),
            {
                "kind": "unit",
                "name": "site.battery",
                "unit_model_class": "BatteryModel",
                "construction_options": {
                    "capacity": {"value": 100, "units": "kWh"},
                    "initial_soc": 0.5,
                },
                "costing_package": None,
                "property_package": None,
            },
            {
                "kind": "unit",
                "name": "site.pump",
                "unit_model_class": "Pump",
                "construction_options": {
                    "energy_intensity": {"value": 0.5, "units": "kWh/m^3"}
                },
                "costing_package": None,
            },
            {
                "kind": "property_package",
                "name": "power",
                "property_class": "SimpleUtility",
                "options": {"resource": "electricity", "units": "kW"},
            },
            {"kind": "resource", "name": "site.power", "property_package": "power"},
        ],
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error", FlexResourceWarning)
        build_model(spec)


def _pumped_water(m):
    """Add a Feed, a Pump and a Product delivering 10 at every step."""
    p = m.plant
    p.feed = Feed(property_package=m.properties, resource_name="well")
    p.pump = Pump(property_package=m.properties, energy_intensity=INTENSITY)
    p.product = Product(property_package=m.properties, resource_name="supply")
    p.a1 = Arc(source=p.feed.outlet_a, destination=p.pump.inlet)
    p.a2 = Arc(source=p.pump.outlet, destination=p.product.inlet_a)
    _expand(m)
    for t in m.time_block.time_index:
        p.product.delivery[t].fix(10.0)


@pytest.mark.component
@pytest.mark.needs_highs
def test_a_battery_powers_a_pump_through_an_electricity_utility():
    m = _plant(_pumped_water, _battery)
    m.plant.power = Resource(property_package=_utility(m, "electricity", "kW"))
    m.keep_charge = pyo.Objective(expr=m.plant.battery.charge[1], sense=pyo.maximize)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FlexResourceWarning)
        _solve(m)

    battery, pump = m.plant.battery, m.plant.pump
    assert pyo.value(pump.power_electrical[0]) == pytest.approx(5.0)
    assert pyo.value(battery.power_electrical[0]) == pytest.approx(-5.0)


@pytest.mark.component
@pytest.mark.needs_highs
def test_a_battery_and_a_pump_beside_a_grid_stay_within_its_import_limit():
    m = _plant(_pumped_water, _battery)
    m.plant.grid = GridConnection(max_import=2 * pyunits.kW, max_export=0 * pyunits.kW)
    m.keep_charge = pyo.Objective(expr=m.plant.battery.charge[1], sense=pyo.maximize)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FlexResourceWarning)
        _solve(m)

    grid, battery = m.plant.grid, m.plant.battery
    assert pyo.value(grid.grid_import[0]) == pytest.approx(2.0)
    assert pyo.value(battery.power_electrical[0]) == pytest.approx(-3.0)
