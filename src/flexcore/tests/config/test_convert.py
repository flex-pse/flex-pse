"""Tests for converting the legacy nested ModelConfig into a flat FlowsheetSpec."""

import json

import pytest

from flexcore.config.convert import nested_to_spec
from flexcore.config.schema import (
    ExternalDispatchSpec,
    ModelConfig,
    PropertyPackageSpec,
    SurrogateSpec,
)
from flexcore.config.spec import SCHEMA_VERSION
from flexcore.exceptions import FlexConfigError

TIME = {"start_date": "2025-01-01", "end_date": "2025-01-02", "time_step": "1 hr"}
COSTING = {
    "energy_prices": {"electrical": {"value": 0.1, "units": "USD/kWh"}},
    "solver": "highs",
}


def nested(**overrides) -> ModelConfig:
    """Return a one-plant nested config with top-level keys overridden."""
    data = {
        "schema_version": "0.0.4",
        "time": TIME,
        "costing": COSTING,
        "plant": {
            "name": "site",
            "units": {
                "tank": {"unit_model_class": "Tank"},
                "pump": {"unit_model_class": "Pump", "costing": False},
            },
            "arcs": [{"source": "tank.outlet", "destination": "pump.inlet"}],
        },
        **overrides,
    }
    return ModelConfig.model_validate(data)


def by_name(spec, kind):
    """Return the spec's elements of one kind keyed by name."""
    return {el.name: el for el in spec.of_kind(kind)}


@pytest.mark.unit
def test_nested_to_spec_version_and_name():
    """The result carries the spec version and the plant's name."""
    spec = nested_to_spec(nested())

    assert spec.schema_version == SCHEMA_VERSION
    assert spec.name == "site"


@pytest.mark.unit
def test_nested_to_spec_time_element():
    """The time config becomes one time element."""
    (time,) = nested_to_spec(nested()).of_kind("time")

    assert (time.start_date, time.end_date, time.time_step) == (
        "2025-01-01",
        "2025-01-02",
        "1 hr",
    )


@pytest.mark.unit
def test_nested_to_spec_property_packages():
    """Each properties key becomes a property_package element of that name."""
    cfg = nested()
    cfg.properties = {
        "water": PropertyPackageSpec(property_class="SimpleAqueousFlow"),
        "gas": PropertyPackageSpec(property_class="SimpleGasFlow", options={"a": 1}),
    }
    for unit in cfg.plant.units.values():
        unit.property_package = "water"

    packages = by_name(nested_to_spec(cfg), "property_package")

    assert set(packages) == {"water", "gas"}
    assert packages["gas"].options == {"a": 1}


@pytest.mark.unit
def test_nested_to_spec_costing_objective_and_solver():
    """Costing becomes a costing element plus an objective; solver moves up."""
    spec = nested_to_spec(nested())

    (costing,) = spec.of_kind("costing")
    (objective,) = spec.of_kind("objective")
    assert costing.name == "costing"
    assert costing.energy_prices["electrical"].value == 0.1
    assert objective.expression == "cost"
    assert spec.solver == "highs"


@pytest.mark.unit
def test_nested_to_spec_plant_and_unit_names():
    """The plant keeps its name and each unit is named plant.key."""
    spec = nested_to_spec(nested())

    assert set(by_name(spec, "plant")) == {"site"}
    assert set(by_name(spec, "unit")) == {"site.tank", "site.pump"}


@pytest.mark.unit
def test_nested_to_spec_network_and_plant_names():
    """A network gives network, network.plant and network.plant.unit names."""
    cfg = nested()
    plant = cfg.plant
    cfg = nested(
        plant=None, network={"name": "net", "plants": {"a": plant.model_dump()}}
    )

    spec = nested_to_spec(cfg)

    assert spec.name == "net"
    assert set(by_name(spec, "network")) == {"net"}
    assert set(by_name(spec, "plant")) == {"net.a"}
    assert set(by_name(spec, "unit")) == {"net.a.tank", "net.a.pump"}


@pytest.mark.unit
def test_nested_to_spec_unit_fields_and_costing_flag():
    """Unit fields are copied and costing true/false becomes auto/null."""
    cfg = nested()
    cfg.plant.units["tank"].construction_options = {"max_volume": 5}
    units = by_name(nested_to_spec(cfg), "unit")

    assert units["site.tank"].unit_model_class == "Tank"
    assert units["site.tank"].construction_options == {"max_volume": 5}
    assert units["site.tank"].costing_package == "auto"
    assert units["site.pump"].costing_package is None
    assert units["site.tank"].property_package == "auto"


@pytest.mark.unit
def test_nested_to_spec_surrogate_inline():
    """A unit surrogate becomes a surrogate element on the power relation."""
    cfg = nested()
    cfg.plant.units["tank"].surrogate = SurrogateSpec(
        surrogate_type="multilinear", data={"a": 1}, provenance={"r2": 0.9}
    )

    (surrogate,) = nested_to_spec(cfg).of_kind("surrogate")

    assert surrogate.unit == "site.tank"
    assert surrogate.relation == "power_electrical_relation"
    assert surrogate.data == {"a": 1}
    assert surrogate.provenance == {"r2": 0.9}


@pytest.mark.unit
def test_nested_to_spec_digestor_surrogate_uses_biogas_relation():
    """A Digestor's relation is the biogas relation, not the power relation."""
    cfg = nested()
    digestor = cfg.plant.units["tank"]
    digestor.unit_model_class = "Digestor"
    digestor.surrogate = SurrogateSpec(surrogate_type="multilinear")

    (surrogate,) = nested_to_spec(cfg).of_kind("surrogate")

    assert surrogate.relation == "biogas_relation"


@pytest.mark.unit
def test_nested_to_spec_surrogate_sidecar_is_inlined(tmp_path):
    """A surrogate sidecar's data is read into the surrogate element."""
    (tmp_path / "fit.json").write_text(json.dumps({"data": {"a": 2}}))
    cfg = nested()
    cfg._base_dir = tmp_path
    cfg.plant.units["tank"].surrogate = SurrogateSpec(
        surrogate_type="multilinear", source="fit.json"
    )

    (surrogate,) = nested_to_spec(cfg).of_kind("surrogate")

    assert surrogate.data == {"a": 2}


@pytest.mark.unit
def test_nested_to_spec_external_dispatch_is_inlined(tmp_path):
    """External dispatch becomes a dispatch element holding the file's values."""
    (tmp_path / "series.json").write_text(json.dumps({"0": 1.5, "1": 2.5}))
    cfg = nested()
    cfg._base_dir = tmp_path
    cfg.plant.units["tank"].external_dispatch = ExternalDispatchSpec(
        variable="power_electrical", source="series.json", fix=False
    )

    (dispatch,) = nested_to_spec(cfg).of_kind("dispatch")

    assert dispatch.unit == "site.tank"
    assert dispatch.variable == "power_electrical"
    assert dispatch.values == {"0": 1.5, "1": 2.5}
    assert dispatch.fix is False


@pytest.mark.unit
def test_nested_to_spec_missing_dispatch_file_raises(tmp_path):
    """An unreadable external-dispatch file is a config error."""
    cfg = nested()
    cfg.plant.units["tank"].external_dispatch = ExternalDispatchSpec(
        variable="power_electrical", source=str(tmp_path / "missing.json")
    )

    with pytest.raises(FlexConfigError, match="Could not read"):
        nested_to_spec(cfg)


@pytest.mark.unit
def test_nested_to_spec_plant_arc_becomes_named_connection():
    """A plant arc is a connection on its source unit, named arc_<n>."""
    units = by_name(nested_to_spec(nested()), "unit")

    (connection,) = units["site.tank"].connections
    assert (connection.port, connection.to, connection.name) == (
        "outlet",
        "site.pump.inlet",
        "arc_0",
    )
    assert units["site.pump"].connections == []


@pytest.mark.unit
def test_nested_to_spec_sub_port_path_stays_with_port():
    """In 'u.sub.port' the unit is the first segment, the rest is the port."""
    cfg = nested()
    cfg.plant.arcs[0].source = "tank.outlet.flow"

    (connection,) = by_name(nested_to_spec(cfg), "unit")["site.tank"].connections

    assert connection.port == "outlet.flow"


@pytest.mark.unit
def test_nested_to_spec_network_arc_uses_first_two_segments():
    """A network arc's unit is plant.unit; arcs keep their list position as names."""
    plant = nested().plant.model_dump()
    plant["arcs"] = []
    cfg = nested(
        plant=None,
        network={
            "name": "net",
            "plants": {"a": plant, "b": plant},
            "arcs": [{"source": "a.tank.outlet", "destination": "b.pump.inlet"}],
        },
    )

    units = by_name(nested_to_spec(cfg), "unit")

    (connection,) = units["net.a.tank"].connections
    assert (connection.port, connection.to, connection.name) == (
        "outlet",
        "net.b.pump.inlet",
        "arc_0",
    )


@pytest.mark.unit
def test_nested_to_spec_arc_from_unknown_unit_raises():
    """An arc whose source is not a unit of the plant is a config error."""
    cfg = nested()
    cfg.plant.arcs[0].source = "ghost.outlet"

    with pytest.raises(FlexConfigError, match="ghost"):
        nested_to_spec(cfg)


@pytest.mark.unit
def test_nested_to_spec_carries_base_dir(tmp_path):
    """The config's base directory carries over to the spec."""
    cfg = nested()
    cfg._base_dir = tmp_path

    assert nested_to_spec(cfg)._base_dir == tmp_path
