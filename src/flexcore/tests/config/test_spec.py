"""Tests for the flat FlowsheetSpec schema and its validators."""

import copy
import json
import typing

import pytest
from pydantic import ValidationError

from flexcore.config.spec import (
    KINDS,
    SCHEMA_VERSION,
    Element,
    FlowsheetSpec,
)

TIME = {
    "kind": "time",
    "start_date": "2025-01-01",
    "end_date": "2025-01-02",
    "time_step": "1 hr",
}
COSTING = {
    "kind": "costing",
    "name": "costing",
    "energy_prices": {"electrical": {"value": 0.1, "units": "USD/kWh"}},
}
PACKAGE = {
    "kind": "property_package",
    "name": "properties",
    "property_class": "SimpleAqueousFlow",
}


def unit(name, **fields):
    """Return a unit element dict."""
    return {"kind": "unit", "name": name, "unit_model_class": "Tank", **fields}


def spec_dict(*elements, base=(TIME, COSTING, PACKAGE)) -> dict:
    """Return a spec dict holding the base elements plus ``elements``."""
    return {
        "schema_version": SCHEMA_VERSION,
        "name": "m",
        "elements": copy.deepcopy([*base, *elements]),
    }


def plant(name="p"):
    """Return a plant element dict."""
    return {"kind": "plant", "name": name}


def rejects(data, *fragments):
    """Assert the spec dict fails validation with every fragment in the message."""
    with pytest.raises(ValidationError) as excinfo:
        FlowsheetSpec.model_validate(data)
    for fragment in fragments:
        assert fragment in str(excinfo.value)


@pytest.mark.unit
def test_kinds_registry_complete():
    """KINDS covers exactly the kind literals of the Element union."""
    members = typing.get_args(typing.get_args(Element)[0])
    literals = {
        typing.get_args(cls.model_fields["kind"].annotation)[0] for cls in members
    }

    assert set(KINDS) == literals


@pytest.mark.unit
def test_minimal_spec_validates():
    """A spec of just the required elements is valid."""
    spec = FlowsheetSpec.model_validate(spec_dict(plant(), unit("p.tank")))

    assert [el.kind for el in spec.elements] == [
        "time",
        "costing",
        "property_package",
        "plant",
        "unit",
    ]


@pytest.mark.unit
def test_unknown_field_rejected():
    """An undocumented key on an element is an error."""
    rejects(spec_dict(plant(), unit("p.tank", bogus=1)), "bogus")


@pytest.mark.unit
def test_missing_time_rejected():
    """A spec needs exactly one time element."""
    rejects(spec_dict(base=(COSTING, PACKAGE)), "time")


@pytest.mark.unit
def test_two_time_elements_rejected():
    """Two time elements are an error."""
    rejects(spec_dict(TIME), "time")


@pytest.mark.unit
def test_duplicate_names_rejected():
    """Two unit elements with one name name the element and its kind."""
    rejects(spec_dict(plant(), unit("p.tank"), unit("p.tank")), "p.tank", "unit")


@pytest.mark.unit
def test_name_shared_across_kinds_rejected():
    """A plant and a property package may not share a name."""
    rejects(spec_dict(plant("properties")), "properties")


@pytest.mark.unit
def test_unit_parent_must_exist():
    """A unit under a missing plant names the unit and the missing parent."""
    rejects(spec_dict(unit("ghost.tank")), "ghost.tank", "ghost")


@pytest.mark.unit
def test_unit_cannot_nest_under_unit():
    """A unit's parent must be a plant or network, not another unit."""
    rejects(spec_dict(plant(), unit("p.tank"), unit("p.tank.inner")), "p.tank.inner")


@pytest.mark.unit
def test_network_must_be_at_root():
    """A network under a plant is an error."""
    rejects(spec_dict(plant(), {"kind": "network", "name": "p.net"}), "p.net")


@pytest.mark.unit
def test_plant_may_sit_under_network_but_not_plant():
    """A plant can be under a network; under a plant it is an error."""
    net = {"kind": "network", "name": "net"}
    FlowsheetSpec.model_validate(spec_dict(net, plant("net.p")))
    rejects(spec_dict(plant(), plant("p.q")), "p.q")


@pytest.mark.unit
def test_root_level_element_name_has_no_dot():
    """Property packages, costing and objectives live at the model root."""
    rejects(spec_dict({**PACKAGE, "name": "a.b"}), "a.b")


@pytest.mark.unit
def test_unit_at_root_is_allowed():
    """A unit may sit directly on the model."""
    FlowsheetSpec.model_validate(spec_dict(unit("tank")))


@pytest.mark.unit
def test_auto_property_package_needs_exactly_one():
    """'auto' with two packages is an error naming the unit."""
    other = {**PACKAGE, "name": "other"}
    rejects(spec_dict(plant(), unit("p.tank"), other), "p.tank", "property_package")


@pytest.mark.unit
def test_named_property_package_must_exist():
    """A unit naming a missing package names both."""
    rejects(spec_dict(plant(), unit("p.tank", property_package="nope")), "nope")


@pytest.mark.unit
def test_null_packages_allowed_with_no_package_elements():
    """A unit may opt out of packages entirely."""
    data = spec_dict(
        plant(),
        unit("p.tank", property_package=None, costing_package=None),
        base=(TIME,),
    )
    FlowsheetSpec.model_validate(data)


@pytest.mark.unit
def test_auto_costing_package_needs_exactly_one():
    """'auto' costing with no costing element is an error naming the unit."""
    rejects(spec_dict(plant(), unit("p.tank"), base=(TIME, PACKAGE)), "p.tank")


@pytest.mark.unit
def test_unit_index_must_name_a_set():
    """An index naming something other than a set element is an error."""
    rejects(spec_dict(plant(), unit("p.tank", index="p")), "p.tank", "index")


@pytest.mark.unit
def test_set_values_nonempty_and_unique():
    """A set needs at least one value and no repeats."""
    rejects(spec_dict({"kind": "set", "name": "s", "values": []}), "values")
    rejects(spec_dict({"kind": "set", "name": "s", "values": [1, 1]}), "values")


def connected(*connections, extra=()):
    """Return a spec of two units in a plant, with connections on the first."""
    return spec_dict(
        plant(),
        unit("p.a", connections=list(connections)),
        unit("p.b"),
        unit("p.b_bypass"),
        *extra,
    )


@pytest.mark.unit
def test_connection_to_must_name_a_unit():
    """A connection to a path under no unit names the source unit."""
    rejects(
        connected({"port": "outlet", "to": "p.ghost.inlet"}), "p.a", "p.ghost.inlet"
    )


@pytest.mark.unit
def test_connection_longest_unit_prefix_wins():
    """With units 'p.b' and 'p.b_bypass', 'p.b_bypass.inlet' is the second."""
    FlowsheetSpec.model_validate(
        connected({"port": "outlet", "to": "p.b_bypass.inlet"})
    )


@pytest.mark.unit
def test_connection_index_must_name_a_set():
    """A connection index that is not a set element is an error."""
    rejects(connected({"port": "outlet", "to": "p.b.inlet", "index": "p"}), "p.a")


@pytest.mark.unit
def test_placeholder_needs_an_index():
    """'{i}' with no index on the connection or its unit is an error."""
    rejects(connected({"port": "outlet", "to": "p.b[{i}].inlet"}), "p.a", "{i}")


@pytest.mark.unit
def test_placeholder_allowed_with_unit_index():
    """A connection inherits its unit's index, so '{i}' is allowed."""
    data = spec_dict(
        plant(),
        {"kind": "set", "name": "p.s", "values": [0, 1]},
        unit(
            "p.a",
            index="p.s",
            connections=[{"port": "outlet", "to": "p.b[{i}].inlet"}],
        ),
        unit("p.b", index="p.s"),
    )
    FlowsheetSpec.model_validate(data)


@pytest.mark.unit
def test_duplicate_connection_source_rejected():
    """Two connections out of one port is a mistake."""
    rejects(
        connected(
            {"port": "outlet", "to": "p.b.inlet"},
            {"port": "outlet", "to": "p.b_bypass.inlet"},
        ),
        "p.a",
        "outlet",
    )


@pytest.mark.unit
def test_duplicate_connection_destination_rejected():
    """Two connections into one port is a mistake."""
    rejects(
        connected(
            {"port": "outlet", "to": "p.b.inlet"},
            {"port": "waste", "to": "p.b.inlet"},
        ),
        "p.b.inlet",
    )


@pytest.mark.unit
def test_surrogate_and_dispatch_unit_must_exist():
    """Surrogates and dispatches name a unit element."""
    surrogate = {
        "kind": "surrogate",
        "unit": "p.ghost",
        "surrogate_type": "multilinear",
    }
    dispatch = {"kind": "dispatch", "unit": "p.ghost", "variable": "v", "values": [1.0]}
    rejects(spec_dict(plant(), surrogate), "p.ghost", "surrogate")
    rejects(spec_dict(plant(), dispatch), "p.ghost", "dispatch")


@pytest.mark.unit
def test_two_surrogates_same_relation_rejected():
    """No file order means no 'last one wins': a repeated (unit, relation) fails."""
    surrogate = {"kind": "surrogate", "unit": "p.tank", "surrogate_type": "multilinear"}
    rejects(
        spec_dict(plant(), unit("p.tank"), surrogate, surrogate), "p.tank", "surrogate"
    )


@pytest.mark.unit
def test_two_dispatches_same_variable_rejected():
    """A repeated (unit, variable) dispatch fails for the same reason."""
    dispatch = {"kind": "dispatch", "unit": "p.tank", "variable": "v", "values": [1.0]}
    rejects(
        spec_dict(plant(), unit("p.tank"), dispatch, dispatch), "p.tank", "dispatch"
    )


@pytest.mark.unit
def test_two_objectives_rejected():
    """At most one objective."""
    objective = {"kind": "objective", "name": "o1"}
    rejects(spec_dict(objective, {**objective, "name": "o2"}), "objective")


@pytest.mark.unit
def test_objective_costing_package_must_resolve():
    """An objective naming a missing costing element is an error."""
    rejects(spec_dict({"kind": "objective", "costing_package": "nope"}), "nope")


@pytest.mark.unit
def test_costing_needs_a_pricing_source():
    """A costing element with neither tariff nor prices is an error."""
    rejects(
        spec_dict(base=(TIME, {"kind": "costing", "name": "costing"}, PACKAGE)),
        "costing",
    )


@pytest.mark.unit
def test_spec_roundtrips_through_json():
    """Dumping with aliases and reloading gives an equal spec."""
    data = spec_dict(plant(), unit("p.tank"))
    spec = FlowsheetSpec.model_validate(data)

    reloaded = FlowsheetSpec.model_validate(
        json.loads(spec.model_dump_json(by_alias=True))
    )

    assert reloaded == spec
