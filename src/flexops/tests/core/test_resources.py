"""Resource nodes: strict balances, explicit edges, attached ports and solves."""

import copy
from types import SimpleNamespace

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits
from pyomo.network import Arc

from flexcore.config.spec import SCHEMA_VERSION
from flexcore.exceptions import FlexConfigError
from flexcore.solvers import get_solver
from flexops import PlantBlock, SimpleAqueousFlow, build_model, emit_model
from flexops.core.resources import (
    ResourceKind,
    add_resource_balance,
    resource_graph,
)
from flexops.testing import dummy_time_block
from flexops.tests.core.test_resource_terms import BareUnit, GasBurner
from flexops.unit_models import Feed, Product, Pump, Tank

GAS_FLOW = "m^3/hr"
INTENSITY = 0.5 * pyunits.kWh / pyunits.m**3


def _gas_plant(n: int = 4):
    """Build a plant with a natural_gas Feed and a unit burning natural_gas."""
    m = dummy_time_block(n)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.gas_feed = Feed(property_package=m.properties, resource_name="natural_gas")
    m.plant.burner = GasBurner()
    return m


def _water_plant(n: int = 4, package=None):
    """Build a Feed and a Product with their ports attached to a ``water`` node."""
    m = dummy_time_block(n)
    m.plant = PlantBlock(time_block=m.time_block)
    package = package or m.properties
    m.plant.feed = Feed(property_package=package, resource_name="well")
    m.plant.product = Product(property_package=package, resource_name="supply")
    m.plant.feed.attach_port("outlet_a", resource="water")
    m.plant.product.attach_port("inlet_a", resource="water")
    return m


@pytest.mark.unit
def test_balance_sums_explicit_edges_to_zero_at_every_step():
    m = _gas_plant()

    block = add_resource_balance(
        m.plant, "gas_balance", resource="natural_gas", units=GAS_FLOW
    )

    assert m.plant.gas_balance is block
    assert sorted(block.edges) == [
        "plant.burner.fuel_usage_natural_gas",
        "plant.gas_feed.withdrawal",
    ]
    assert len(block.flow) == 2 * 4
    assert len(block.balance) == 4
    assert block.find_component("net") is None


@pytest.mark.unit
def test_edge_flow_is_the_signed_unit_variable():
    """An edge adds no Var: it evaluates to coeff times the unit's own variable."""
    m = _gas_plant()
    m.plant.gas_feed.withdrawal[1].set_value(7.0)
    m.plant.burner.fuel_usage_natural_gas[1].set_value(3.0)

    block = add_resource_balance(m.plant, "gas", resource="natural_gas", units=GAS_FLOW)

    assert pyo.value(block.flow["plant.gas_feed.withdrawal", 1]) == 7.0
    assert pyo.value(block.flow["plant.burner.fuel_usage_natural_gas", 1]) == -3.0
    assert len(list(block.component_objects(pyo.Var))) == 0


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
def test_terms_on_arc_connected_ports_are_refused():
    """An Arc already conserves a port flow; balancing it again counts it twice."""
    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.first = Pump(property_package=m.properties, energy_intensity=INTENSITY)
    m.plant.second = Pump(property_package=m.properties, energy_intensity=INTENSITY)
    m.plant.link = Arc(source=m.plant.first.outlet, destination=m.plant.second.inlet)
    m.plant.first.attach_port("outlet", resource="water")

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

    block = add_resource_balance(m.plant, "delayed", resource="delayed", units=GAS_FLOW)

    m.plant.unit.flow[0].set_value(9.0)
    assert list(block.balance.keys()) == [2, 3]
    assert pyo.value(block.flow["plant.unit.flow@2", 1]) == 0.0
    assert pyo.value(block.flow["plant.unit.flow@2", 2]) == 9.0


@pytest.mark.unit
def test_building_a_balance_records_it_on_the_model():
    m = _gas_plant()

    add_resource_balance(m.plant, "gas_balance", resource="natural_gas", units=GAS_FLOW)

    (record,) = m.__dict__["_flex_logic"]
    assert (record.kind, record.fields) == (
        "resource",
        {"name": "plant.gas_balance", "resource": "natural_gas", "units": GAS_FLOW},
    )


# --- attached ports ---------------------------------------------------------


@pytest.mark.unit
def test_an_attached_outlet_supplies_and_an_attached_inlet_consumes():
    m = _water_plant()
    m.plant.feed.outlet_a_state.flow_vol_phase[0, "Liq"].set_value(4.0)
    m.plant.product.inlet_a_state.flow_vol_phase[0, "Liq"].set_value(4.0)

    block = add_resource_balance(m.plant, "water", resource="water", units=GAS_FLOW)

    assert pyo.value(block.flow["plant.feed.outlet_a_state.flow_vol_phase", 0]) == 4.0
    assert (
        pyo.value(block.flow["plant.product.inlet_a_state.flow_vol_phase", 0]) == -4.0
    )


@pytest.mark.unit
def test_attaching_a_port_adds_no_component_to_the_unit():
    m = dummy_time_block(3)
    m.feed = Feed(property_package=m.properties, resource_name="well")
    before = {c.local_name for c in m.feed.component_objects(descend_into=False)}

    m.feed.attach_port("outlet_a", resource="water")

    assert {
        c.local_name for c in m.feed.component_objects(descend_into=False)
    } == before


@pytest.mark.unit
def test_attaching_a_multiphase_port_is_refused():
    m = dummy_time_block(3)
    m.feed = Feed(property_package=m.properties, resource_name="well")
    m.feed._io_registry.port_states["outlet_a"] = SimpleNamespace(
        params=SimpleNamespace(phase_list=["Liq", "Vap"])
    )

    with pytest.raises(FlexConfigError, match="one phase") as excinfo:
        m.feed.attach_port("outlet_a", resource="water")

    assert excinfo.value.field == "port"


@pytest.mark.unit
def test_attaching_an_unknown_port_lists_the_ports():
    m = _water_plant()

    with pytest.raises(FlexConfigError, match="outlet_a") as excinfo:
        m.plant.feed.attach_port("outlet_z", resource="water")

    assert excinfo.value.field == "port"


@pytest.mark.unit
def test_attaching_a_port_twice_is_refused():
    m = _water_plant()

    with pytest.raises(FlexConfigError, match="already attached") as excinfo:
        m.plant.feed.attach_port("outlet_a", resource="other")

    assert excinfo.value.field == "port"


@pytest.mark.unit
def test_a_node_gives_its_ports_one_shared_intensive_state():
    package_block = dummy_time_block(3)
    package_block.pressurized = SimpleAqueousFlow(has_pressure=True)
    m = package_block
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.feed = Feed(property_package=m.pressurized, resource_name="well")
    m.plant.product = Product(property_package=m.pressurized, resource_name="supply")
    m.plant.feed.attach_port("outlet_a", resource="water")
    m.plant.product.attach_port("inlet_a", resource="water")

    block = add_resource_balance(m.plant, "water", resource="water", units=GAS_FLOW)

    assert len(block.pressure) == 3
    assert len(block.pressure_equal) == 2 * 3


@pytest.mark.unit
def test_a_node_without_intensive_state_adds_no_variables():
    m = _water_plant()

    block = add_resource_balance(m.plant, "water", resource="water", units=GAS_FLOW)

    assert len(list(block.component_objects(pyo.Var))) == 0


# --- resource_graph ---------------------------------------------------------


@pytest.mark.unit
def test_graph_classifies_balanced_arc_and_open_resources():
    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.feed = Feed(property_package=m.properties, resource_name="well")
    m.plant.first = Pump(property_package=m.properties, energy_intensity=INTENSITY)
    m.plant.second = Pump(property_package=m.properties, energy_intensity=INTENSITY)
    m.plant.feed.attach_port("outlet_a", resource="water")
    m.plant.first.attach_port("inlet", resource="water")
    m.plant.link = Arc(source=m.plant.first.outlet, destination=m.plant.second.inlet)
    add_resource_balance(m.plant, "water", resource="water", units=GAS_FLOW)

    graph = resource_graph(m.plant)

    assert graph.resources["water"].kind is ResourceKind.BALANCED
    assert len(graph.resources["water"].edges) == 2
    assert graph.resources["plant.link"].kind is ResourceKind.ARC
    assert [e.unit for e in graph.resources["plant.link"].edges] == [
        m.plant.first,
        m.plant.second,
    ]
    assert [e.coeff for e in graph.resources["plant.link"].edges] == [1.0, -1.0]
    assert graph.resources["electricity"].kind is ResourceKind.OPEN
    assert set(graph.tasks) == {m.plant.feed, m.plant.first, m.plant.second}


@pytest.mark.unit
def test_graph_still_reports_an_arc_after_expansion():
    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.first = Pump(property_package=m.properties, energy_intensity=INTENSITY)
    m.plant.second = Pump(property_package=m.properties, energy_intensity=INTENSITY)
    m.plant.link = Arc(source=m.plant.first.outlet, destination=m.plant.second.inlet)
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    graph = resource_graph(m.plant)

    assert graph.resources["plant.link"].kind is ResourceKind.ARC


# --- the build_model path ---------------------------------------------------

SPEC_INTENSITY = {"energy_intensity": {"value": 0.5, "units": "kWh/m^3"}}
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
    assert len(model.site.raw_water.balance) == 5


@pytest.mark.unit
def test_resource_element_names_the_resource_when_it_differs_from_the_block():
    model = build_model(
        _spec(
            {
                "kind": "resource",
                "name": "site.balance",
                "resource": "raw_water",
                "units": GAS_FLOW,
            }
        )
    )

    assert list(model.site.balance.edges) == ["site.well.withdrawal"]


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


def _pump(name, intensity, *, into="site.raw", out_to="site.treated"):
    """Return a Pump element attached to two resource nodes."""
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
    """Return a well, one pump and a product joined through two resource nodes."""
    elements = copy.deepcopy(BASE)
    elements[-1]["connections"] = [{"port": "outlet_a", "to": "site.raw"}]
    return {
        "schema_version": SCHEMA_VERSION,
        "name": "m",
        "elements": [
            *elements,
            _pump("site.pump_a", 0.5),
            {
                "kind": "unit",
                "name": "site.supply",
                "unit_model_class": "Product",
                "construction_options": {"resource_name": "potable"},
                "costing_package": None,
                "connections": [{"port": "inlet_a", "to": "site.treated"}],
            },
            {"kind": "resource", "name": "site.raw", "units": GAS_FLOW},
            {"kind": "resource", "name": "site.treated", "units": GAS_FLOW},
            *extra,
        ],
    }


@pytest.mark.unit
def test_connections_to_a_node_attach_ports_and_build_no_arcs():
    model = build_model(_process())

    assert len(list(model.component_objects(Arc, descend_into=True))) == 0
    assert sorted(model.site.raw.edges) == [
        "site.pump_a.inlet_state.flow_vol_phase",
        "site.well.outlet_a_state.flow_vol_phase",
    ]


@pytest.mark.unit
def test_every_member_of_an_indexed_unit_attaches_to_the_node():
    pump = _pump("site.pump", 0.5)
    pump["index"] = "trains"
    spec = _process(
        {"kind": "set", "name": "site.trains", "values": [0, 1]},
    )
    spec["elements"][4] = {**pump, "index": "site.trains"}

    model = build_model(spec)

    assert sorted(model.site.raw.edges) == [
        "site.pump[0].inlet_state.flow_vol_phase",
        "site.pump[1].inlet_state.flow_vol_phase",
        "site.well.outlet_a_state.flow_vol_phase",
    ]


@pytest.mark.unit
def test_attached_ports_emit_as_connections_to_the_node():
    m = _water_plant()
    add_resource_balance(m.plant, "line", resource="water", units=GAS_FLOW)

    units = {e.name: e for e in emit_model(m).elements if e.kind == "unit"}

    assert [(c.port, c.to) for c in units["plant.feed"].connections] == [
        ("outlet_a", "plant.line")
    ]
    assert [(c.port, c.to) for c in units["plant.product"].connections] == [
        ("inlet_a", "plant.line")
    ]


@pytest.mark.unit
def test_an_attached_port_with_no_balance_cannot_be_emitted():
    m = _water_plant()

    with pytest.raises(FlexConfigError, match="no balance"):
        emit_model(m)


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
def test_a_node_with_supply_and_no_sink_forces_the_flow_to_zero():
    """A resource cannot vanish at a node: no sink means nothing can be supplied."""
    m = dummy_time_block(4)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.feed = Feed(property_package=m.properties, resource_name="well")
    m.plant.feed.attach_port("outlet_a", resource="water")
    add_resource_balance(m.plant, "water", resource="water", units=GAS_FLOW)
    steps = list(m.time_block.time_index)
    for t in steps:
        m.plant.feed.withdrawal[t].setub(10.0)
    m.reward = pyo.Objective(
        expr=sum(m.plant.feed.withdrawal[t] for t in steps), sense=pyo.maximize
    )

    _solve(m)

    assert [pyo.value(m.plant.feed.withdrawal[t]) for t in steps] == pytest.approx(
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
    m.plant.feed.attach_port("outlet_a", resource="water")
    m.plant.product.attach_port("inlet_a", resource="water")
    add_resource_balance(m.plant, "water", resource="water", units=GAS_FLOW)
    steps = list(m.time_block.time_index)
    for t in steps:
        m.plant.feed.outlet_a_state.pressure[t].fix(250000.0)
    m.zero = pyo.Objective(expr=sum(m.plant.feed.withdrawal[t] for t in steps))
    for t in steps:
        m.plant.feed.withdrawal[t].setlb(0.0)

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
        pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    def with_nodes(m):
        m.plant.feed.attach_port("outlet_a", resource="raw")
        m.plant.tank.attach_port("inlet", resource="raw")
        m.plant.tank.attach_port("outlet", resource="stored")
        m.plant.product.attach_port("inlet_a", resource="stored")
        add_resource_balance(m.plant, "raw", resource="raw", units=GAS_FLOW)
        add_resource_balance(m.plant, "stored", resource="stored", units=GAS_FLOW)

    assert solve(with_nodes) == pytest.approx(solve(with_arcs))


@pytest.mark.component
@pytest.mark.needs_highs
def test_a_candidate_unit_joins_by_attaching_with_no_other_element_edited():
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
    model, (pump_a, pump_b) = solve(_pump("site.pump_b", 0.3))

    assert pyo.value(only_a.flow_out[0]) == pytest.approx(10.0)
    assert pyo.value(pump_a.flow_out[0]) == pytest.approx(0.0)
    assert pyo.value(pump_b.flow_out[0]) == pytest.approx(10.0)
    assert pyo.value(model.cost) == pytest.approx(0.3 * 10.0 * 5)


@pytest.mark.component
@pytest.mark.needs_ipopt
def test_ipopt_converges_on_a_closed_recycle_with_a_redundant_balance():
    """One balance is implied by the rest, so IPOPT regularizes and loses accuracy."""
    m = dummy_time_block(4)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.a = Pump(property_package=m.properties, energy_intensity=INTENSITY)
    m.plant.b = Pump(property_package=m.properties, energy_intensity=INTENSITY)
    m.plant.a.attach_port("outlet", resource="ab")
    m.plant.b.attach_port("inlet", resource="ab")
    m.plant.b.attach_port("outlet", resource="ba")
    m.plant.a.attach_port("inlet", resource="ba")
    add_resource_balance(m.plant, "ab", resource="ab", units=GAS_FLOW)
    add_resource_balance(m.plant, "ba", resource="ba", units=GAS_FLOW)
    target = 5 * pyunits.m**3 / pyunits.hr
    steps = list(m.time_block.time_index)
    m.fit = pyo.Objective(expr=sum((m.plant.a.flow_in[t] - target) ** 2 for t in steps))

    results = get_solver(model=m, prefer="ipopt").solve(m)

    pyo.assert_optimal_termination(results)
    assert pyo.value(m.plant.b.flow_out[0]) == pytest.approx(5.0, abs=0.01)
