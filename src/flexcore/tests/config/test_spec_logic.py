"""Tests for the logic, ramp and extension element kinds of the flat spec."""

import pytest

from flexcore.config.io import load_spec
from flexcore.config.spec import KINDS, SCHEMA_VERSION, FlowsheetSpec
from flexcore.tests.config.test_spec import (
    PACKAGE,
    TIME,
    plant,
    rejects,
    spec_dict,
    unit,
)

KW = {"value": 0.0, "units": "kW"}
KW_MAX = {"value": 10.0, "units": "kW"}


def pump(name="p.a", **fields):
    """Return a pump unit element dict."""
    return unit(name, unit_model_class="Pump", **fields)


def status(unit_name="p.a", **fields):
    """Return a status element dict on ``unit_name``."""
    return {
        "kind": "status",
        "unit": unit_name,
        "variable": "power_electrical",
        "min_output": KW,
        "max_output": KW_MAX,
        **fields,
    }


def trains():
    """Return a set and an indexed pump over it, inside plant p."""
    return (
        {"kind": "set", "name": "p.trains", "values": [0, 1]},
        pump("p.ro", index="p.trains"),
    )


def accepts(*elements) -> FlowsheetSpec:
    """Validate a spec of plant p plus ``elements``."""
    return FlowsheetSpec.model_validate(spec_dict(plant(), *elements))


@pytest.mark.unit
def test_new_kinds_are_registered_immutable():
    new = {
        "status": "logic",
        "startup_shutdown": "logic",
        "dwell": "logic",
        "bypass": "logic",
        "ramp": "ramping",
        "startup_delay": "logic",
        "conditional": "logic",
        "parallel_group": "logic",
        "extension": "extensions",
    }

    assert {kind: KINDS[kind] for kind in new} == {
        kind: (stage, False) for kind, stage in new.items()
    }


@pytest.mark.unit
def test_every_logic_kind_validates():
    spec = accepts(
        pump(),
        pump("p.b"),
        status(),
        status("p.b"),
        {"kind": "startup_shutdown", "unit": "p.a", "min_up": 3, "min_down": 2},
        {"kind": "dwell", "unit": "p.a", "variable": "flow_in", "k": 2},
        {"kind": "bypass", "unit": "p.a", "variable": "flow_in", "max": KW},
        {
            "kind": "ramp",
            "unit": "p.a",
            "variable": "power_electrical",
            "ramp_up": 1.0,
            "units": "kW/h",
        },
        {"kind": "startup_delay", "unit": "p.b", "upstream": "p.a", "k": 1},
        {"kind": "conditional", "unit": "p.a", "implies": "p.b", "state": "off"},
        {"kind": "parallel_group", "units": ["p.a", "p.b"]},
        {"kind": "extension", "entry_point": "pkg.mod:fn", "name": "extra"},
    )

    assert len(spec.of_kind(*KINDS)) == len(spec.elements)


@pytest.mark.unit
def test_quantity_needs_exactly_value_and_units():
    rejects(
        spec_dict(plant(), pump(), status(min_output={"value": 0.0})),
        "value",
        "units",
    )


@pytest.mark.unit
def test_quantity_may_be_a_plain_number():
    accepts(pump(), status(min_output=0.0, max_output=10.0))


@pytest.mark.unit
@pytest.mark.parametrize(
    "element",
    [
        {"kind": "startup_shutdown", "unit": "p.a", "min_up": 0},
        {"kind": "dwell", "unit": "p.a", "variable": "flow_in", "k": 0},
        {
            "kind": "ramp",
            "unit": "p.a",
            "variable": "x",
            "ramp_up": 1,
            "units": "kW/h",
            "window": 0,
        },
    ],
    ids=["min_up", "dwell_k", "window"],
)
def test_counts_must_be_at_least_one(element):
    rejects(spec_dict(plant(), pump(), status(), element), "greater than or equal to 1")


@pytest.mark.unit
def test_unit_reference_must_name_a_unit_element():
    rejects(spec_dict(plant(), pump(), status("p.nope")), "p.nope", "no such unit")


@pytest.mark.unit
def test_unit_reference_may_name_a_member_of_an_indexed_unit():
    accepts(*trains(), status("p.ro[1]"))


@pytest.mark.unit
def test_member_reference_must_be_in_the_set():
    rejects(spec_dict(plant(), *trains(), status("p.ro[5]")), "p.ro[5]")


@pytest.mark.unit
def test_member_reference_needs_an_indexed_unit():
    rejects(spec_dict(plant(), pump(), status("p.a[0]")), "p.a[0]")


@pytest.mark.unit
@pytest.mark.parametrize("field", ["implies", "upstream"])
def test_cross_unit_references_must_resolve(field):
    kind = "conditional" if field == "implies" else "startup_delay"
    element = {"kind": kind, "unit": "p.a", field: "p.zz"}
    if kind == "startup_delay":
        element["k"] = 1
    rejects(spec_dict(plant(), pump(), status(), element), "p.zz")


@pytest.mark.unit
def test_parallel_group_units_must_resolve():
    group = {"kind": "parallel_group", "units": ["p.a", "p.zz"], "order_status": False}
    rejects(spec_dict(plant(), pump(), group), "p.zz")


@pytest.mark.unit
def test_ramp_needs_a_direction():
    ramp = {"kind": "ramp", "unit": "p.a", "variable": "x", "units": "kW/h"}
    rejects(spec_dict(plant(), pump(), ramp), "ramp_up", "ramp_down")


@pytest.mark.unit
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
    rejects(spec_dict(plant(), pump(), pump("p.b"), status(), element), "p.b", "status")


@pytest.mark.unit
def test_parallel_group_without_status_ordering_needs_no_status():
    group = {
        "kind": "parallel_group",
        "units": ["p.a", "p.b"],
        "variables": ["flow_in"],
        "order_status": False,
    }
    accepts(pump(), pump("p.b"), group)


@pytest.mark.unit
def test_a_status_on_the_indexed_unit_covers_its_members():
    accepts(
        *trains(),
        status("p.ro"),
        {"kind": "startup_shutdown", "unit": "p.ro[0]"},
    )


@pytest.mark.unit
def test_battery_with_its_own_status_needs_no_status_element():
    battery = unit(
        "p.b",
        unit_model_class="BatteryModel",
        unit_commitment={"status": True},
    )
    accepts(battery, {"kind": "startup_shutdown", "unit": "p.b"})


@pytest.mark.unit
def test_battery_without_its_own_status_still_needs_one():
    battery = unit("p.b", unit_model_class="BatteryModel")
    rejects(
        spec_dict(plant(), battery, {"kind": "startup_shutdown", "unit": "p.b"}),
        "status",
    )


@pytest.mark.unit
@pytest.mark.parametrize("unit_class", ["Tank", "BatteryModel"])
def test_status_is_not_allowed_on_units_that_manage_their_own(unit_class):
    rejects(
        spec_dict(plant(), unit("p.a", unit_model_class=unit_class), status()),
        f"{unit_class} manages its own status",
        "unit_commitment.status",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "element",
    [
        status(),
        {"kind": "startup_shutdown", "unit": "p.a"},
        {"kind": "bypass", "unit": "p.a", "variable": "flow_in", "max": KW},
        {"kind": "dwell", "unit": "p.a", "variable": "flow_in", "k": 2},
        {
            "kind": "ramp",
            "unit": "p.a",
            "variable": "flow_in",
            "ramp_up": 1.0,
            "units": "m^3/h^2",
        },
        {"kind": "conditional", "unit": "p.a", "implies": "p.b"},
    ],
    ids=["status", "startup_shutdown", "bypass", "dwell", "ramp", "conditional"],
)
def test_duplicate_logic_is_an_error(element):
    base = [pump(), pump("p.b"), status("p.b")]
    if element["kind"] != "status":
        base.append(status())
    rejects(spec_dict(plant(), *base, element, dict(element)), "twice")


@pytest.mark.unit
def test_logic_on_an_indexed_unit_and_its_member_is_a_duplicate():
    rejects(spec_dict(plant(), *trains(), status("p.ro"), status("p.ro[0]")), "twice")


@pytest.mark.unit
def test_ramps_with_different_component_names_are_not_duplicates():
    ramp = {
        "kind": "ramp",
        "unit": "p.a",
        "variable": "flow_in",
        "ramp_up": 1.0,
        "units": "m^3/h^2",
    }
    accepts(pump(), ramp, {**ramp, "component_name": "slow", "window": 2})


@pytest.mark.unit
def test_extension_names_are_unique():
    ext = {"kind": "extension", "entry_point": "pkg.mod:fn", "name": "extra"}
    rejects(spec_dict(plant(), ext, dict(ext)), "extra")


@pytest.mark.unit
def test_extension_objective_needs_no_costing_element():
    without_costing = spec_dict(
        {"kind": "objective", "expression": "extension"}, base=(TIME, PACKAGE)
    )
    FlowsheetSpec.model_validate(without_costing)


@pytest.mark.unit
def test_a_0_1_0_spec_migrates_to_the_current_version():
    data = spec_dict(plant())
    data["schema_version"] = "0.1.0"

    assert load_spec(data).schema_version == SCHEMA_VERSION
