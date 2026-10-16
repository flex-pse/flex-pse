"""Round trip: build in code, emit_model, dump JSON, build_model, compare."""

import json
import random
import warnings
from pathlib import Path

import jsonschema
import pyomo.environ as pyo
import pytest

from flexcore.config.io import dump_spec
from flexcore.exceptions import FlexEmitWarning
from flexops import build_model, emit_model
from flexops.core.compare import assert_models_equivalent
from flexops.tests.fixtures.roundtrip_corpus import CASES

SCHEMA = (
    Path(__file__).parents[2]
    / "flexcore"
    / "config"
    / "schemas"
    / "flowsheet_spec.schema.json"
)


def corpus_cases() -> list:
    """Return each corpus case name, marked with its test tier."""
    return [
        pytest.param(name, marks=getattr(pytest.mark, tier))
        for name, (_, _, tier) in CASES.items()
    ]


def emit_case(name: str, directory: Path):
    """Build a corpus case, emit it with warnings as errors, and dump it."""
    build, expand_arcs, _ = CASES[name]
    model = build()
    if expand_arcs:
        pyo.TransformationFactory("network.expand_arcs").apply_to(model)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FlexEmitWarning)
        spec = emit_model(model, data_dir=directory / "data", relative_to=directory)
    path = directory / "spec.json"
    dump_spec(spec, path)
    return model, path, expand_arcs


@pytest.mark.parametrize("name", corpus_cases())
def test_corpus_case_round_trips(name, tmp_path):
    model, path, expand_arcs = emit_case(name, tmp_path)

    rebuilt = build_model(path, expand_arcs=expand_arcs)

    assert_models_equivalent(model, rebuilt)


@pytest.mark.parametrize("name", corpus_cases())
def test_emitted_spec_matches_the_json_schema(name, tmp_path):
    _, path, _ = emit_case(name, tmp_path)

    jsonschema.validate(json.loads(path.read_text()), json.loads(SCHEMA.read_text()))


@pytest.mark.parametrize("name", corpus_cases())
def test_shuffled_emitted_spec_builds_the_same_model(name, tmp_path):
    model, path, expand_arcs = emit_case(name, tmp_path)
    data = json.loads(path.read_text())
    random.Random(0).shuffle(data["elements"])
    shuffled = tmp_path / "shuffled.json"
    shuffled.write_text(json.dumps(data))

    assert_models_equivalent(model, build_model(shuffled, expand_arcs=expand_arcs))


@pytest.mark.unit
def test_long_prices_are_written_to_a_sidecar(tmp_path):
    _, path, _ = emit_case("long_prices", tmp_path)

    costing = next(
        e for e in json.loads(path.read_text())["elements"] if e["kind"] == "costing"
    )
    source = costing["energy_prices"]["electrical"]["value"]["$source"]
    assert source.startswith("data/")
    assert (tmp_path / source).is_file()
