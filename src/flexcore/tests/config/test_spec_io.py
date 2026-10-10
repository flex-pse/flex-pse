"""Tests for loading and dumping flat specs."""

import json
from pathlib import Path

import pytest

from flexcore.config import io as io_module
from flexcore.config.io import (
    dump_spec,
    load_model_config,
    load_spec,
)
from flexcore.config.schema import ModelConfig
from flexcore.config.spec import SCHEMA_VERSION, FlowsheetSpec
from flexcore.exceptions import FlexConfigError

NESTED = {
    "schema_version": "0.0.4",
    "time": {
        "start_date": "2025-01-01",
        "end_date": "2025-01-02",
        "time_step": "1 hr",
    },
    "costing": {"energy_prices": {"electrical": {"value": 0.1, "units": "USD/kWh"}}},
    "plant": {
        "name": "site",
        "units": {"tank": {"unit_model_class": "Tank"}},
    },
}
FLAT = {
    "schema_version": SCHEMA_VERSION,
    "name": "site",
    "elements": [
        {"kind": "objective"},
        {"kind": "unit", "name": "site.tank", "unit_model_class": "Tank"},
        {"kind": "plant", "name": "site"},
        {
            "kind": "costing",
            "name": "costing",
            "energy_prices": {"electrical": {"value": 0.1, "units": "USD/kWh"}},
        },
        {
            "kind": "property_package",
            "name": "p",
            "property_class": "SimpleAqueousFlow",
        },
        {
            "kind": "time",
            "start_date": "2025-01-01",
            "end_date": "2025-01-02",
            "time_step": "1 hr",
        },
        {
            "kind": "surrogate",
            "unit": "site.tank",
            "surrogate_type": "multilinear",
            "data": {"coefficients": {"intercept": 1.0}},
        },
    ],
}
CONFIGS = Path(__file__).parent.parent / "fixtures" / "configs"


@pytest.mark.unit
def test_load_spec_accepts_every_form(tmp_path):
    """Path, dict, FlowsheetSpec and ModelConfig all load."""
    (tmp_path / "flat.json").write_text(json.dumps(FLAT))
    (tmp_path / "nested.json").write_text(json.dumps(NESTED))
    spec = load_spec(FLAT)

    sources = [
        tmp_path / "flat.json",
        FLAT,
        spec,
        tmp_path / "nested.json",
        NESTED,
        ModelConfig.model_validate(NESTED),
    ]

    for source in sources:
        assert isinstance(load_spec(source), FlowsheetSpec)
    assert load_spec(spec) is spec
    assert load_spec(tmp_path / "flat.json")._base_dir == tmp_path


@pytest.mark.unit
@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.json")), ids=lambda p: p.stem)
def test_every_nested_fixture_converts(path):
    """Each stored nested config converts to a flat spec."""
    spec = load_spec(path)

    assert spec.schema_version == SCHEMA_VERSION


@pytest.mark.unit
def test_load_spec_dict_is_not_mutated():
    """Loading leaves the input mapping alone."""
    before = json.loads(json.dumps(FLAT))

    load_spec(FLAT)

    assert FLAT == before


@pytest.mark.unit
def test_load_spec_old_nested_version_migrates_then_converts():
    """A 0.0.1 nested config runs the nested migrations before conversion."""
    old = {**NESTED, "schema_version": "0.0.1", "properties": {}}

    spec = load_spec(old)

    assert [el.name for el in spec.of_kind("property_package")] == ["properties"]


@pytest.mark.unit
def test_load_spec_newer_version_raises():
    """A spec newer than this build says to upgrade flex-pse."""
    with pytest.raises(FlexConfigError, match="Upgrade flex-pse"):
        load_spec({**FLAT, "schema_version": "9.0.0"})


@pytest.mark.unit
def test_load_spec_runs_spec_migrations(monkeypatch):
    """An older flat spec steps through MIGRATIONS to the current version."""
    monkeypatch.setattr(io_module, "SCHEMA_VERSION", "0.1.1")
    monkeypatch.setitem(
        io_module.MIGRATIONS,
        "0.1.0",
        lambda data, base_dir: {**data, "schema_version": "0.1.1"},
    )

    assert load_spec(FLAT).schema_version == "0.1.1"


@pytest.mark.unit
def test_load_spec_missing_migration_raises():
    """A flat version with no migration path is an error naming it."""
    with pytest.raises(FlexConfigError, match="0.0.9"):
        load_spec({**FLAT, "schema_version": "0.0.9"})


@pytest.mark.unit
def test_load_spec_missing_version_raises():
    """A document with no schema_version is an error."""
    data = {k: v for k, v in FLAT.items() if k != "schema_version"}

    with pytest.raises(FlexConfigError, match="schema_version"):
        load_spec(data)


@pytest.mark.unit
def test_load_spec_validation_error_names_field():
    """A bad element is a FlexConfigError naming the field path."""
    bad = {**FLAT, "elements": [*FLAT["elements"], {"kind": "plant"}]}

    with pytest.raises(FlexConfigError, match="elements"):
        load_spec(bad)


@pytest.mark.unit
def test_load_spec_rejects_other_suffix(tmp_path):
    """Only .json loads."""
    path = tmp_path / "spec.yaml"
    path.write_text(json.dumps(FLAT))

    with pytest.raises(FlexConfigError):
        load_spec(path)


@pytest.mark.unit
def test_load_model_config_rejects_flat():
    """The nested loader points a flat spec at load_spec."""
    with pytest.raises(FlexConfigError, match="load_spec"):
        load_model_config(FLAT)


@pytest.mark.unit
def test_dump_spec_sorted_and_minimal(tmp_path):
    """Elements are sorted, defaults are omitted, and a reload is equal."""
    spec = load_spec(FLAT)
    path = tmp_path / "spec.json"

    dump_spec(spec, path)

    written = json.loads(path.read_text())
    assert [el["kind"] for el in written["elements"]] == [
        "costing",
        "property_package",
        "time",
        "plant",
        "unit",
        "surrogate",
        "objective",
    ]
    unit = written["elements"][4]
    assert "costing_package" not in unit and "connections" not in unit
    assert written["elements"][5]["data"] == {"coefficients": {"intercept": 1.0}}
    reloaded = load_spec(path)
    assert sorted(el.model_dump_json() for el in reloaded.elements) == sorted(
        el.model_dump_json() for el in spec.elements
    )
    assert reloaded.model_dump(exclude={"elements"}) == spec.model_dump(
        exclude={"elements"}
    )


@pytest.mark.unit
def test_dump_spec_orders_elements_with_equal_kind_by_name(tmp_path):
    """Within a kind, elements sort by name, whatever the in-memory order."""
    data = {
        **FLAT,
        "elements": [
            *FLAT["elements"],
            {"kind": "unit", "name": "site.a", "unit_model_class": "Tank"},
        ],
    }
    path = tmp_path / "spec.json"

    dump_spec(load_spec(data), path)

    names = [
        e["name"]
        for e in json.loads(path.read_text())["elements"]
        if e["kind"] == "unit"
    ]
    assert names == ["site.a", "site.tank"]


@pytest.mark.unit
def test_dump_spec_does_not_reorder_the_spec(tmp_path):
    """Only the written file is sorted; the in-memory spec keeps its order."""
    spec = load_spec(FLAT)

    dump_spec(spec, tmp_path / "spec.json")

    assert [el.kind for el in spec.elements] == [e["kind"] for e in FLAT["elements"]]
