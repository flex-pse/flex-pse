"""Tests for spec references: $source sidecar files and $package options."""

import copy
import gzip
import json

import pytest
from pydantic import ValidationError

from flexcore.config.io import read_source, resolve_sources
from flexcore.config.spec import SCHEMA_VERSION, FlowsheetSpec, SourceRef
from flexcore.exceptions import FlexConfigError

TIME = {
    "kind": "time",
    "start_date": "2025-01-01",
    "end_date": "2025-01-02",
    "time_step": "1 hr",
}
PACKAGE = {
    "kind": "property_package",
    "name": "properties",
    "property_class": "SimpleAqueousFlow",
}


def spec_dict(*elements) -> dict:
    """Return a spec dict holding time, a package and ``elements``."""
    return {
        "schema_version": SCHEMA_VERSION,
        "name": "m",
        "elements": copy.deepcopy([TIME, PACKAGE, *elements]),
    }


def costing(value) -> dict:
    """Return a costing element pricing electricity at ``value``."""
    return {
        "kind": "costing",
        "name": "costing",
        "energy_prices": {"electrical": {"value": value, "units": "USD/kWh"}},
    }


def unit(**options) -> dict:
    """Return a Tank unit element with ``options`` as construction options."""
    return {
        "kind": "unit",
        "name": "tank",
        "unit_model_class": "Tank",
        "construction_options": options,
    }


@pytest.mark.unit
def test_source_ref_reads_and_writes_the_dollar_source_key():
    ref = SourceRef.model_validate({"$source": "data/p.json.gz"})

    assert ref.source == "data/p.json.gz"
    assert ref.model_dump(by_alias=True) == {"$source": "data/p.json.gz"}


@pytest.mark.unit
def test_price_value_may_be_a_source_ref():
    spec = FlowsheetSpec.model_validate(spec_dict(costing({"$source": "p.json.gz"})))

    price = spec.of_kind("costing")[0].energy_prices["electrical"]
    assert price.value == SourceRef(source="p.json.gz")


@pytest.mark.unit
def test_dispatch_values_may_be_a_source_ref():
    dispatch = {
        "kind": "dispatch",
        "unit": "tank",
        "variable": "flow_in",
        "values": {"$source": "d.json.gz"},
    }
    spec = FlowsheetSpec.model_validate(spec_dict(costing(0.1), unit(), dispatch))

    assert spec.of_kind("dispatch")[0].values == SourceRef(source="d.json.gz")


@pytest.mark.unit
def test_package_reference_must_name_a_property_package():
    data = spec_dict(costing(0.1), unit(extra={"feed": {"$package": "missing"}}))

    with pytest.raises(ValidationError, match="missing"):
        FlowsheetSpec.model_validate(data)


@pytest.mark.unit
def test_package_reference_to_a_known_package_validates():
    data = spec_dict(costing(0.1), unit(extra={"feed": {"$package": "properties"}}))

    FlowsheetSpec.model_validate(data)


@pytest.mark.unit
@pytest.mark.parametrize("name", ["v.json", "v.json.gz"])
def test_read_source_reads_json_and_gzipped_json(tmp_path, name):
    text = json.dumps([1.0, 2.0])
    path = tmp_path / name
    if name.endswith(".gz"):
        path.write_bytes(gzip.compress(text.encode()))
    else:
        path.write_text(text)

    assert read_source(name, tmp_path) == [1.0, 2.0]


@pytest.mark.unit
def test_read_source_rejects_other_formats(tmp_path):
    (tmp_path / "v.csv").write_text("1,2")

    with pytest.raises(FlexConfigError, match="v.csv"):
        read_source("v.csv", tmp_path)


@pytest.mark.unit
def test_read_source_names_a_missing_file(tmp_path):
    with pytest.raises(FlexConfigError, match="nope.json.gz"):
        read_source("nope.json.gz", tmp_path)


@pytest.mark.unit
def test_resolve_sources_replaces_refs_at_any_depth(tmp_path):
    (tmp_path / "a.json").write_text("[1, 2]")
    value = {
        "x": SourceRef(source="a.json"),
        "y": [{"$source": "a.json"}, 3],
        "z": "kept",
    }

    assert resolve_sources(value, tmp_path) == {
        "x": [1, 2],
        "y": [[1, 2], 3],
        "z": "kept",
    }
