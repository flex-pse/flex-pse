"""Tests for the ramping, logic, extensions and costing stages of build_model."""

import copy

import pyomo.environ as pyo
import pytest
from pyomo.network import Arc

from flexcore.config.spec import SCHEMA_VERSION
from flexcore.exceptions import FlexConfigError
from flexops import apply_spec, build_model
from flexops.core.compare import model_differences
from flexops.core.ops_block import OpsBlockData
from flexops.core.stages import LOGIC_ORDER
from flexops.unit_models.general.battery import BatteryModelData
from flexops.unit_models.general.tank import TankData

KW = {"value": 0.0, "units": "kW"}
KW_MAX = {"value": 10.0, "units": "kW"}
FLOW = {"value": 1.0, "units": "m^3/h"}
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
    {
        "kind": "costing",
        "name": "costing",
        "energy_prices": {"electrical": {"value": 0.1, "units": "USD/kWh"}},
    },
    {"kind": "plant", "name": "p"},
    {"kind": "unit", "name": "p.a", "unit_model_class": "Pump"},
    {"kind": "unit", "name": "p.b", "unit_model_class": "Pump"},
    {"kind": "objective"},
]


def status(unit):
    """Return a status element on ``unit``'s electrical power."""
    return {
        "kind": "status",
        "unit": unit,
        "variable": "power_electrical",
        "min_output": KW,
        "max_output": KW_MAX,
    }


ELEMENTS = {
    "status": status("p.a"),
    "startup_shutdown": {"kind": "startup_shutdown", "unit": "p.a", "min_up": 2},
    "dwell": {"kind": "dwell", "unit": "p.a", "variable": "flow_in", "k": 2},
    "bypass": {"kind": "bypass", "unit": "p.a", "variable": "flow_in", "max": FLOW},
    "ramp": {
        "kind": "ramp",
        "unit": "p.a",
        "variable": "power_electrical",
        "ramp_up": 1.0,
        "units": "kW/h",
    },
    "startup_delay": {
        "kind": "startup_delay",
        "unit": "p.b",
        "upstream": "p.a",
        "k": 1,
    },
    "conditional": {"kind": "conditional", "unit": "p.a", "implies": "p.b"},
    "parallel_group": {"kind": "parallel_group", "units": ["p.a", "p.b"]},
    "extension": {
        "kind": "extension",
        "entry_point": "flexops.tests.sample_extensions:cap_power",
        "options": {"unit": "p.a", "max_kw": 5.0},
        "name": "cap",
    },
}


def spec(*elements, statuses=("p.a", "p.b"), base=BASE) -> dict:
    """Return a spec dict of ``base``, statuses on ``statuses``, and ``elements``."""
    return {
        "schema_version": SCHEMA_VERSION,
        "name": "m",
        "elements": copy.deepcopy([*base, *map(status, statuses), *elements]),
    }


@pytest.mark.unit
def test_logic_order_constant():
    assert LOGIC_ORDER == (
        "status",
        "bypass",
        "dwell",
        "startup_shutdown",
        "startup_delay",
        "conditional",
        "parallel_group",
    )


@pytest.mark.component
@pytest.mark.parametrize("kind", sorted(ELEMENTS))
def test_every_logic_kind_changes_the_model(kind):
    statuses = ("p.b",) if kind == "status" else ("p.a", "p.b")

    without = build_model(spec(statuses=statuses))
    with_element = build_model(spec(ELEMENTS[kind], statuses=statuses))

    assert model_differences(without, with_element)


@pytest.mark.unit
@pytest.mark.parametrize("kind", sorted(ELEMENTS))
def test_apply_spec_rejects_every_logic_kind(kind):
    model = build_model(spec())

    with pytest.raises(FlexConfigError, match="immutable"):
        apply_spec(model, [ELEMENTS[kind]])


@pytest.mark.unit
def test_logic_builds_whatever_order_the_file_lists_it_in():
    elements = [ELEMENTS[k] for k in reversed(sorted(ELEMENTS)) if k != "status"]

    model = build_model(spec(*elements))

    assert model.p.a.find_component("startup") is not None
    assert model.p.b.find_component("startup_delay") is not None


INDEXED = [
    *BASE[:4],
    {"kind": "set", "name": "p.trains", "values": [0, 1, 2]},
    {"kind": "unit", "name": "p.ro", "unit_model_class": "Pump", "index": "p.trains"},
    {"kind": "unit", "name": "p.a", "unit_model_class": "Pump"},
]


@pytest.mark.unit
def test_logic_on_an_indexed_unit_applies_to_every_member():
    model = build_model(
        spec(
            {"kind": "dwell", "unit": "p.ro", "variable": "flow_in", "k": 2},
            statuses=(),
            base=INDEXED,
        )
    )

    assert all(
        m.find_component("flow_in_dwell") is not None for m in model.p.ro.values()
    )


@pytest.mark.unit
def test_logic_on_a_member_applies_to_that_member_only():
    model = build_model(
        spec(
            {"kind": "dwell", "unit": "p.ro[1]", "variable": "flow_in", "k": 2},
            statuses=(),
            base=INDEXED,
        )
    )

    assert [
        m.find_component("flow_in_dwell") is not None for m in model.p.ro.values()
    ] == [False, True, False]


@pytest.mark.unit
def test_cross_unit_logic_on_an_indexed_unit_uses_its_members_in_order():
    model = build_model(
        spec(
            {"kind": "parallel_group", "units": ["p.ro", "p.a"]},
            statuses=("p.ro", "p.a"),
            base=INDEXED,
        )
    )

    assert model.p.ro[0].find_component("conditional") is None
    assert model.p.ro[1].find_component("conditional") is not None
    assert model.p.ro[2].find_component("conditional") is not None
    assert model.p.a.find_component("conditional") is not None


@pytest.mark.unit
def test_conditional_from_an_indexed_unit_to_a_scalar_unit_pairs_each_member():
    model = build_model(
        spec(
            {"kind": "conditional", "unit": "p.ro", "implies": "p.a"},
            statuses=("p.ro", "p.a"),
            base=INDEXED,
        )
    )

    assert all(m.find_component("conditional") is not None for m in model.p.ro.values())


@pytest.mark.unit
def test_logic_variable_must_exist_on_the_unit():
    bad = {"kind": "dwell", "unit": "p.a", "variable": "nope", "k": 2}

    with pytest.raises(FlexConfigError, match="nope"):
        build_model(spec(bad))


@pytest.mark.unit
def test_build_expands_arcs_by_default():
    connected = copy.deepcopy(BASE)
    connected[4]["connections"] = [{"port": "outlet", "to": "p.b.inlet"}]

    model = build_model(spec(statuses=(), base=connected))

    (arc,) = model.component_data_objects(Arc, active=None, descend_into=True)
    assert not arc.active
    assert arc.expanded_block is not None


@pytest.mark.unit
def test_extension_objective_without_an_extension_objective_is_an_error():
    no_cost_objective = [*BASE[:-1], {"kind": "objective", "expression": "extension"}]

    with pytest.raises(
        FlexConfigError, match="needs an extension that adds an Objective"
    ):
        build_model(spec(statuses=(), base=no_cost_objective))


@pytest.mark.unit
def test_extension_objective_built_by_an_extension():
    elements = [
        *BASE[:-1],
        {"kind": "objective", "expression": "extension"},
        {
            "kind": "extension",
            "entry_point": "flexops.tests.sample_extensions:minimize_power",
            "options": {"units": ["p.a", "p.b"]},
            "name": "objective",
        },
    ]

    model = build_model(spec(statuses=(), base=elements))

    (objective,) = model.component_data_objects(pyo.Objective, active=True)
    assert objective.local_name == "power_objective"
    assert model.costing.find_component("aggregate_operating_cost") is not None


def battery(name="p.b", **fields):
    """Return a battery unit element on ``name``."""
    return {
        "kind": "unit",
        "name": name,
        "unit_model_class": "BatteryModel",
        "property_package": None,
        "construction_options": {"capacity": {"value": 1.0, "units": "kWh"}},
        **fields,
    }


@pytest.mark.unit
def test_only_storage_units_refuse_a_status_element():
    assert OpsBlockData.allows_status_element
    assert not TankData.allows_status_element
    assert not BatteryModelData.allows_status_element


@pytest.mark.component
@pytest.mark.parametrize(
    "element",
    [
        {"kind": "startup_shutdown", "unit": "p.b"},
        {"kind": "conditional", "unit": "p.a", "implies": "p.b"},
        {"kind": "parallel_group", "units": ["p.a", "p.b"]},
        {"kind": "startup_delay", "unit": "p.b", "upstream": "p.a", "k": 1},
    ],
    ids=["startup_shutdown", "conditional", "parallel_group", "startup_delay"],
)
def test_status_logic_needs_a_status_on_each_unit(element):
    with pytest.raises(FlexConfigError, match=r"'p\.b' needs a status"):
        build_model(spec(element, statuses=("p.a",)))


@pytest.mark.component
def test_parallel_group_without_status_ordering_needs_no_status():
    group = {
        "kind": "parallel_group",
        "units": ["p.a", "p.b"],
        "variables": ["flow_in"],
        "order_status": False,
    }

    build_model(spec(group, statuses=()))


@pytest.mark.component
def test_a_status_on_the_indexed_unit_covers_its_members():
    model = build_model(
        spec(
            {"kind": "startup_shutdown", "unit": "p.ro[0]"},
            statuses=("p.ro",),
            base=INDEXED,
        )
    )

    assert model.p.ro[0].find_component("startup") is not None


@pytest.mark.component
def test_battery_with_its_own_status_needs_no_status_element():
    base = [*BASE[:-2], battery(unit_commitment={"status": True}), BASE[-1]]

    model = build_model(
        spec({"kind": "startup_shutdown", "unit": "p.b"}, statuses=(), base=base)
    )

    assert model.p.b.find_component("startup") is not None


@pytest.mark.component
def test_battery_without_its_own_status_still_needs_one():
    base = [*BASE[:-2], battery(), BASE[-1]]

    with pytest.raises(FlexConfigError, match="needs a status"):
        build_model(
            spec({"kind": "startup_shutdown", "unit": "p.b"}, statuses=(), base=base)
        )


@pytest.mark.component
@pytest.mark.parametrize(
    "unit",
    [{"kind": "unit", "name": "p.b", "unit_model_class": "Tank"}, battery()],
    ids=["Tank", "BatteryModel"],
)
def test_status_is_not_allowed_on_units_that_manage_their_own(unit):
    base = [*BASE[:-2], unit, BASE[-1]]

    with pytest.raises(FlexConfigError, match="manages its own status"):
        build_model(spec(statuses=("p.b",), base=base))
