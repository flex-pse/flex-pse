"""dumps: a built model pickles and plain pickle.loads reads it back."""

import pickle

import pyomo.environ as pyo
import pytest
from pyomo.opt import assert_optimal_termination

from flexcore.exceptions import FlexError
from flexcore.pickling import dumps
from flexcore.solvers import get_solver
from flexops import build_model
from flexops.core.compare import assert_models_equivalent
from flexops.tests.test_roundtrip import corpus_cases, emit_case


def built_case(name: str, directory) -> pyo.ConcreteModel:
    """Build a corpus case through emit_model and build_model."""
    _, path, expand_arcs = emit_case(name, directory)
    return build_model(path, expand_arcs=expand_arcs)


@pytest.mark.parametrize("name", corpus_cases())
def test_dumps_round_trips_every_corpus_spec(name, tmp_path):
    model = built_case(name, tmp_path)

    loaded = pickle.loads(dumps(model))

    assert_models_equivalent(model, loaded)


@pytest.mark.component
def test_loaded_model_solves_to_same_objective(tmp_path):
    model = built_case("battery_status", tmp_path)
    loaded = pickle.loads(dumps(model))

    for m in (model, loaded):
        assert_optimal_termination(get_solver(model=m, prefer="highs").solve(m))

    assert pyo.value(loaded.objective) == pytest.approx(pyo.value(model.objective))


@pytest.mark.unit
def test_dropped_local_rule_raises_flex_error(tmp_path):
    loaded = pickle.loads(dumps(built_case("battery_status", tmp_path)))
    constraint = loaded.costing.eq_aggregate_power

    with pytest.raises(FlexError, match="was a local function"):
        constraint._rule._fcn(loaded.costing, 0, "electrical")
