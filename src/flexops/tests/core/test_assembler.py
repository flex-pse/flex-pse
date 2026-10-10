"""Tests for the flat-spec assembler: build order, connections, apply_spec."""

import copy
import importlib.util
import json
import random
import shutil
from pathlib import Path

import pyomo.environ as pyo
import pytest
from pyomo.network import Arc
from pyomo.opt import assert_optimal_termination

from flexcore.config.io import load_model_config, load_spec
from flexcore.config.spec import KINDS, STAGE_ORDER, DispatchElement, SurrogateElement
from flexcore.exceptions import FlexConfigError
from flexcore.solvers import get_solver
from flexops import apply_spec, build_model
from flexops.core import stages as stages_module
from flexops.core.stages import STAGES, elements_of
from flexops.testing import assert_models_equivalent, model_fingerprint

FIXTURES = Path(__file__).parent.parent / "fixtures"
FREEZE = FIXTURES / "api_freeze"
SNAPSHOTS = FIXTURES / "legacy_snapshots"

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
OBJECTIVE = {"kind": "objective", "name": "objective", "expression": "cost"}
INTENSITY = {"energy_intensity": {"value": 0.5, "units": "kWh/m^3"}}
SURROGATE_DATA = {
    "input_variables": {"flow_out": "m^3/hr"},
    "output_variables": {"power_electrical": "kW"},
    "coefficients": {"flow_out": 0.8, "intercept": 0.0},
}


def unit(name, unit_model_class="Tank", **fields):
    """Return a unit element dict."""
    options = INTENSITY if unit_model_class == "ConstantEnergyIntensityModel" else {}
    return {
        "kind": "unit",
        "name": name,
        "unit_model_class": unit_model_class,
        "construction_options": options,
        **fields,
    }


def spec_dict(*elements, name="site", objective=True) -> dict:
    """Return a spec of the base elements plus ``elements``."""
    base = [TIME, COSTING, PACKAGE, *([OBJECTIVE] if objective else [])]
    return {
        "schema_version": "0.1.0",
        "name": name,
        "elements": copy.deepcopy([*base, *elements]),
    }


def pump(name, **fields):
    """Return a constant-intensity unit element dict (ports inlet and outlet)."""
    return unit(name, "ConstantEnergyIntensityModel", **fields)


def freeze_spec(directory: Path) -> Path:
    """Write the API-freeze model as a scrambled flat spec; return its path."""
    for path in (FREEZE / "data").iterdir():
        shutil.copy(path, directory)
    elements = [
        {
            "kind": "unit",
            "name": "waterfacility.plant",
            "unit_model_class": "ConstantEnergyIntensityModel",
            "construction_options": {
                "energy_intensity": {"value": 0.5, "units": "kWh/m^3"}
            },
        },
        {"kind": "objective", "name": "objective", "expression": "cost"},
        {
            "kind": "unit",
            "name": "waterfacility.tank",
            "unit_model_class": "Tank",
            "connections": [{"port": "outlet", "to": "waterfacility.plant.inlet"}],
        },
        {
            "kind": "costing",
            "name": "costing",
            "tariff_source": "tariff.json",
            "dr": {"events_source": "dr_events.json"},
        },
        {"kind": "plant", "name": "waterfacility"},
        {
            "kind": "time",
            "start_date": "2025-01-01",
            "end_date": "2025-01-30",
            "time_step": "15 min",
        },
        {
            "kind": "unit",
            "name": "waterfacility.battery",
            "unit_model_class": "BatteryModel",
            "property_package": None,
            "construction_options": {"capacity": {"value": 1.0, "units": "kWh"}},
        },
        {
            "kind": "property_package",
            "name": "properties",
            "property_class": "SimpleAqueousFlow",
        },
    ]
    path = directory / "freeze.json"
    path.write_text(
        json.dumps(
            {"schema_version": "0.1.0", "name": "waterfacility", "elements": elements}
        )
    )
    return path


def network_spec(directory: Path) -> Path:
    """Write a two-plant network spec with in-plant and cross-plant arcs."""
    elements = [
        {"kind": "network", "name": "net"},
        {"kind": "plant", "name": "net.a"},
        {"kind": "plant", "name": "net.b"},
        pump("net.a.pump", connections=[{"port": "outlet", "to": "net.b.tank.inlet"}]),
        unit("net.a.tank", connections=[{"port": "outlet", "to": "net.a.pump.inlet"}]),
        pump("net.b.pump"),
        unit("net.b.tank", connections=[{"port": "outlet", "to": "net.b.pump.inlet"}]),
    ]
    path = directory / "network.json"
    path.write_text(json.dumps(spec_dict(*elements, name="net")))
    return path


def shuffled(path: Path, seed: int, directory: Path) -> Path:
    """Write ``path``'s spec with its elements shuffled by ``seed``."""
    data = json.loads(path.read_text())
    random.Random(seed).shuffle(data["elements"])
    out = directory / f"shuffled_{path.stem}_{seed}.json"
    out.write_text(json.dumps(data))
    return out


def creation_order(model) -> list[str]:
    """Return component names in the order the model created them."""
    return [c.name for c in model.component_objects(descend_into=True)]


@pytest.mark.component
@pytest.mark.parametrize("make_spec", [freeze_spec, network_spec])
def test_order_agnostic(make_spec, tmp_path):
    """A shuffled element list builds the same model, in the same creation order."""
    path = make_spec(tmp_path)
    reference = build_model(path)

    for seed in range(10):
        other = build_model(shuffled(path, seed, tmp_path))

        assert_models_equivalent(reference, other)
        assert creation_order(reference) == creation_order(other)


@pytest.mark.unit
def test_flat_freeze_spec_matches_nested_freeze_config(tmp_path):
    """The flat API-freeze spec builds the model the frozen nested JSON builds."""
    flat_path = freeze_spec(tmp_path)
    data = json.loads(flat_path.read_text())
    tank = next(e for e in data["elements"] if e["name"] == "waterfacility.tank")
    tank["connections"][0]["name"] = "arc_0"
    flat_path.write_text(json.dumps(data))
    shutil.copy(FREEZE / "api_freeze_config.json", tmp_path)
    nested = build_model(load_model_config(tmp_path / "api_freeze_config.json"))
    flat = build_model(flat_path)

    assert_models_equivalent(nested, flat)


def snapshot_cases():
    """Return the legacy snapshot case names."""
    return sorted(p.stem for p in SNAPSHOTS.glob("*.json"))


@pytest.mark.unit
@pytest.mark.parametrize("case", snapshot_cases())
def test_legacy_nested_builds_unchanged(case):
    """Each nested config still builds the model recorded before the flat spec."""
    script = SNAPSHOTS / "make_snapshots.py"
    module_spec = importlib.util.spec_from_file_location("make_snapshots", script)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    built = model_fingerprint(module.build_case(case))

    recorded = json.loads((SNAPSHOTS / f"{case}.json").read_text())
    assert built["components"] == recorded["components"]
    for kind in ("vars", "constraints"):
        changed = sorted(
            name
            for name in recorded[kind]
            if built[kind].get(name) != recorded[kind][name]
        )
        assert not changed, f"{kind} changed: {changed[:10]}"
        assert built[kind].keys() == recorded[kind].keys()


@pytest.mark.unit
def test_stage_order_matches_registry_order():
    """The spec registry lists stages in the order the build runs them."""
    assert [s for s in STAGES if s in STAGE_ORDER] == list(STAGE_ORDER)


@pytest.mark.unit
def test_every_mutable_kind_has_an_applier():
    """A kind marked mutable can be applied to a live model."""
    mutable = {kind for kind, (_, flag) in KINDS.items() if flag}

    assert mutable == set(stages_module.APPLIERS)


@pytest.mark.unit
def test_elements_of_sorts_by_depth_then_name():
    """Elements come back parents first, then by name, whatever the file order."""
    spec = load_spec(
        spec_dict(
            {"kind": "plant", "name": "b"},
            {"kind": "plant", "name": "a"},
            {"kind": "network", "name": "n"},
            {"kind": "plant", "name": "n.p"},
        )
    )

    names = [el.name for el in elements_of(spec, "plant", "network")]

    assert names == ["a", "b", "n", "n.p"]


@pytest.mark.unit
def test_build_context_records_members_and_units():
    """The context maps each unit element to its block and member blocks."""
    spec = spec_dict(
        {"kind": "plant", "name": "p"},
        {"kind": "set", "name": "p.s", "values": [0, 1, 2]},
        pump("p.ro", index="p.s"),
        unit("p.tank"),
    )

    model = build_model(spec)

    context = model._flex_build_context
    assert context.units["p.tank"] is model.p.tank
    assert context.members["p.tank"] == [model.p.tank]
    assert context.members["p.ro"] == [model.p.ro[0], model.p.ro[1], model.p.ro[2]]
    assert model._flex_spec.name == "site"


@pytest.mark.unit
def test_indexed_unit_and_arc():
    """A set, an indexed unit and an indexed connection wire member to member."""
    spec = spec_dict(
        {"kind": "plant", "name": "p"},
        {"kind": "set", "name": "p.s", "values": [0, 1, 2]},
        pump("p.ro", index="p.s"),
        unit(
            "p.tank",
            index="p.s",
            connections=[{"port": "outlet", "to": "p.ro[{i}].inlet", "name": "feed"}],
        ),
    )

    model = build_model(spec)

    assert len(model.p.ro) == 3
    assert len(model.p.feed) == 3
    for i in (0, 1, 2):
        assert model.p.feed[i].source is model.p.tank[i].outlet
        assert model.p.feed[i].destination is model.p.ro[i].inlet
        assert model.p.feed[i].directed


@pytest.mark.unit
def test_undirected_connection():
    """directed=false builds undirected arcs holding both ports, scalar and indexed."""
    spec = spec_dict(
        {"kind": "plant", "name": "p"},
        {"kind": "set", "name": "p.s", "values": [0, 1]},
        unit(
            "p.tank",
            connections=[
                {
                    "port": "outlet",
                    "to": "p.pump.inlet",
                    "name": "scalar",
                    "directed": False,
                }
            ],
        ),
        pump("p.pump"),
        unit(
            "p.many",
            index="p.s",
            connections=[
                {
                    "port": "outlet",
                    "to": "p.many_pump[{i}].inlet",
                    "name": "indexed",
                    "directed": False,
                }
            ],
        ),
        pump("p.many_pump", index="p.s"),
    )

    model = build_model(spec)

    assert model.p.scalar.directed is False
    assert set(model.p.scalar.ports) == {model.p.tank.outlet, model.p.pump.inlet}
    for i in (0, 1):
        assert model.p.indexed[i].directed is False
        assert set(model.p.indexed[i].ports) == {
            model.p.many[i].outlet,
            model.p.many_pump[i].inlet,
        }


@pytest.mark.unit
def test_string_indexed_set():
    """A set of strings builds members addressable by name."""
    spec = spec_dict(
        {"kind": "plant", "name": "p"},
        {"kind": "set", "name": "p.s", "values": ["x", "y"]},
        pump("p.ro", index="p.s"),
        unit(
            "p.tank",
            index="p.s",
            connections=[{"port": "outlet", "to": "p.ro[{i}].inlet", "name": "feed"}],
        ),
    )

    model = build_model(spec)

    assert model.find_component("p.ro[x].inlet") is model.p.ro["x"].inlet
    assert model.p.feed["y"].destination is model.p.ro["y"].inlet


@pytest.mark.unit
def test_connection_auto_name_independent_of_order():
    """An unnamed connection gets the same derived name however elements are ordered."""
    elements = [
        {"kind": "plant", "name": "p"},
        unit("p.tank", connections=[{"port": "outlet", "to": "p.pump.inlet"}]),
        pump("p.pump"),
    ]
    names = set()
    for seed in range(5):
        shuffled_elements = copy.deepcopy(elements)
        random.Random(seed).shuffle(shuffled_elements)
        model = build_model(spec_dict(*shuffled_elements))
        names.add(tuple(a.local_name for a in model.p.component_objects(Arc)))

    assert names == {("tank_outlet_to_pump_inlet",)}


@pytest.mark.unit
def test_connection_name_collision_asks_for_a_name():
    """A derived arc name that is already taken is an error asking for a name."""
    spec = spec_dict(
        {"kind": "plant", "name": "p"},
        unit(
            "p.tank",
            connections=[
                {"port": "outlet", "to": "p.pump.inlet"},
                {"port": "inlet", "to": "p.pump.outlet"},
            ],
        ),
        pump("p.pump"),
    )
    spec["elements"].append(unit("p.tank_outlet_to_pump_inlet"))

    with pytest.raises(FlexConfigError, match="name"):
        build_model(spec)


@pytest.mark.unit
def test_arc_between_plants_lives_on_the_network(tmp_path):
    """The arc sits on the deepest block above both units."""
    model = build_model(network_spec(tmp_path))

    assert model.net.a_pump_outlet_to_b_tank_inlet.source is model.net.a.pump.outlet
    assert model.net.a.tank_outlet_to_pump_inlet is not None


@pytest.mark.unit
def test_unit_swap_missing_port_lists_ports():
    """Swapping to a class without the used port names the ports it does have."""
    spec = spec_dict(
        {"kind": "plant", "name": "p"},
        unit("p.tank", connections=[{"port": "outlet", "to": "p.pump.inlet"}]),
        pump("p.pump"),
    )
    spec["elements"][-2]["unit_model_class"] = "BatteryModel"
    spec["elements"][-2]["property_package"] = None
    spec["elements"][-2]["construction_options"] = {
        "capacity": {"value": 1.0, "units": "kWh"}
    }

    with pytest.raises(FlexConfigError) as excinfo:
        build_model(spec)

    message = str(excinfo.value)
    assert "p.tank" in message and "outlet" in message
    assert "Available ports: []" in message


@pytest.mark.unit
def test_missing_destination_port_lists_destination_ports():
    """A destination port that does not exist lists that unit's ports."""
    spec = spec_dict(
        {"kind": "plant", "name": "p"},
        unit("p.tank", connections=[{"port": "outlet", "to": "p.pump.nope"}]),
        pump("p.pump"),
    )

    with pytest.raises(FlexConfigError) as excinfo:
        build_model(spec)

    assert "p.pump" in str(excinfo.value)
    assert "['inlet', 'outlet']" in str(excinfo.value)


def surrogate_element(unit_name="p.pump"):
    """Return a multilinear surrogate element dict for a pump."""
    return {
        "kind": "surrogate",
        "unit": unit_name,
        "surrogate_type": "multilinear",
        "data": SURROGATE_DATA,
    }


@pytest.mark.unit
def test_surrogate_applied_post_construction():
    """A surrogate element swaps the relation; the constructor never saw it."""
    spec = spec_dict(
        {"kind": "plant", "name": "p"}, pump("p.pump"), surrogate_element()
    )

    model = build_model(spec)

    assert model.p.pump.config.flexops_config.surrogate is None
    assert not model.p.pump.power_electrical_relation[0].active
    assert model.p.pump.surrogate_power_electrical.fitted is not None


def dispatch_spec(values) -> dict:
    """Return a spec dispatching the pump's power."""
    return spec_dict(
        {"kind": "plant", "name": "p"},
        pump("p.pump"),
        {
            "kind": "dispatch",
            "unit": "p.pump",
            "variable": "power_electrical",
            "values": values,
        },
    )


@pytest.mark.unit
def test_dispatch_inline_list_and_mapping():
    """A list is one value per time point; a mapping is keyed by time index."""
    from_list = build_model(dispatch_spec([float(t) for t in range(24)]))
    from_mapping = build_model(dispatch_spec({str(t): float(t) for t in range(24)}))

    assert from_list.p.pump.power_electrical[5].value == 5.0
    assert from_mapping.p.pump.power_electrical[5].value == 5.0


@pytest.mark.unit
def test_dispatch_unknown_variable_raises():
    """A dispatched variable the unit lacks is an error naming both."""
    data = dispatch_spec([1.0])
    data["elements"][-1]["variable"] = "nope"

    with pytest.raises(FlexConfigError, match="nope"):
        build_model(data)


@pytest.mark.unit
def test_dispatch_indexed_unit_fixes_every_member():
    """A dispatch on an indexed unit applies to each member."""
    data = spec_dict(
        {"kind": "plant", "name": "p"},
        {"kind": "set", "name": "p.s", "values": [0, 1]},
        pump("p.pump", index="p.s"),
        {
            "kind": "dispatch",
            "unit": "p.pump",
            "variable": "power_electrical",
            "values": [3.0] * 24,
        },
    )

    model = build_model(data)

    assert all(model.p.pump[i].power_electrical[0].fixed for i in (0, 1))


@pytest.mark.unit
def test_no_objective_element():
    """A spec without an objective element builds a model without an Objective."""
    model = build_model(
        spec_dict({"kind": "plant", "name": "p"}, pump("p.pump"), objective=False)
    )

    assert not list(model.component_objects(pyo.Objective))
    assert model.costing.aggregate_operating_cost is not None


@pytest.mark.unit
def test_objective_sense_and_name():
    """The objective element sets the name and the direction."""
    data = spec_dict({"kind": "plant", "name": "p"}, pump("p.pump"), objective=False)
    data["elements"].append({"kind": "objective", "name": "goal", "sense": "maximize"})

    model = build_model(data)

    assert model.goal.sense == pyo.maximize


@pytest.mark.unit
def test_two_costing_elements_each_get_processed():
    """Every costing element is processed, and units pick theirs by name."""
    data = spec_dict(
        {"kind": "plant", "name": "p"},
        pump("p.pump", costing_package="other_costing"),
        {**COSTING, "name": "other_costing"},
        objective=False,
    )
    data["elements"].append({**OBJECTIVE, "costing_package": "costing"})

    model = build_model(data)

    assert model.p.pump.config.costing_package is model.other_costing
    assert model.other_costing.aggregate_operating_cost is not None
    assert model.costing.aggregate_operating_cost is not None


@pytest.mark.unit
def test_apply_spec_rejects_immutable():
    """An immutable element is refused before anything changes."""
    spec = spec_dict({"kind": "plant", "name": "p"}, pump("p.pump"))
    model = build_model(spec)
    before = build_model(spec)

    with pytest.raises(FlexConfigError, match="'unit' elements are immutable"):
        apply_spec(model, [surrogate_element(), pump("p.other")])

    assert_models_equivalent(before, model)


@pytest.mark.unit
def test_apply_spec_unknown_unit_raises():
    """A unit the live model lacks is an error naming it."""
    model = build_model(spec_dict({"kind": "plant", "name": "p"}, pump("p.pump")))

    with pytest.raises(FlexConfigError, match="p.ghost"):
        apply_spec(model, [surrogate_element("p.ghost")])


@pytest.mark.unit
def test_apply_spec_accepts_elements_and_dicts():
    """Elements may be models or dicts and are recorded as applied."""
    model = build_model(spec_dict({"kind": "plant", "name": "p"}, pump("p.pump")))
    element = SurrogateElement.model_validate(surrogate_element())
    dispatch = {
        "kind": "dispatch",
        "unit": "p.pump",
        "variable": "power_electrical",
        "values": [1.0] * 24,
    }

    apply_spec(model, [element, dispatch])

    assert model.p.pump.power_electrical[0].fixed
    assert isinstance(model._flex_applied[0], SurrogateElement)
    assert isinstance(model._flex_applied[1], DispatchElement)


@pytest.mark.component
@pytest.mark.needs_highs
def test_apply_spec_surrogate_on_live_model():
    """A surrogate applied after costing is processed leaves the model solvable."""
    model = build_model(
        spec_dict(
            {"kind": "plant", "name": "p"},
            unit("p.tank", connections=[{"port": "outlet", "to": "p.pump.inlet"}]),
            pump("p.pump"),
        )
    )

    apply_spec(model, [surrogate_element()])

    pyo.TransformationFactory("network.expand_arcs").apply_to(model)
    results = get_solver(model=model, prefer="highs").solve(model)
    assert_optimal_termination(results)
    assert len(model._flex_applied) == 1


@pytest.mark.unit
def test_second_surrogate_deactivates_the_first():
    """Applying a second surrogate to the same relation replaces the first."""
    model = build_model(
        spec_dict({"kind": "plant", "name": "p"}, pump("p.pump"), surrogate_element())
    )
    second = surrogate_element()
    second["data"] = {
        **SURROGATE_DATA,
        "coefficients": {"flow_out": 0.9, "intercept": 0.0},
    }

    apply_spec(model, [second])

    names = model.p.pump.list_surrogate_blocks("power_electrical_relation")
    assert len(names) == 2
    assert [model.p.pump.find_component(n).active for n in names] == [False, True]
