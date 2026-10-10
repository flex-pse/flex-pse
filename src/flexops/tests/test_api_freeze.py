"""The frozen public API (PLAN.md §2).

**Breaking-change tripwire.** ``fixtures/api_freeze/api_freeze.py`` is the
frozen user-facing API script. Any pull request that has to edit it — or this
test — to stay green is making a **breaking change** and must say so in its
description.
"""

import math
import runpy
import shutil
from pathlib import Path

import pyomo.environ as pyo
import pytest
from pyomo.opt import assert_optimal_termination

from flexcore.config.io import load_model_config
from flexcore.solvers import get_solver
from flexops import build_model

_FIXTURES = Path(__file__).parent / "fixtures" / "api_freeze"
_SCRIPT = _FIXTURES / "api_freeze.py"
_CONFIG = _FIXTURES / "api_freeze_config.json"
_DATA = _FIXTURES / "data"


def _in_fixture_dir(tmp_path, monkeypatch) -> None:
    """Copy the tariff/DR data fixtures into ``tmp_path`` and chdir there.

    The frozen script loads bare filenames (``"tariff.json"``), so it runs with
    the working directory set to where those fixtures live.
    """
    for path in _DATA.iterdir():
        shutil.copy(path, tmp_path)
    monkeypatch.chdir(tmp_path)


def _solve(model) -> float:
    """Expand the model's arcs, solve it, and return the objective value.

    Arc expansion is the caller's job in v0: no library code applies the
    ``network.expand_arcs`` transformation implicitly.
    """
    pyo.TransformationFactory("network.expand_arcs").apply_to(model)
    results = get_solver(model=model, prefer="highs").solve(model)
    assert_optimal_termination(results)
    return pyo.value(model.objective)


@pytest.mark.component
@pytest.mark.needs_highs
def test_api_freeze_runs_and_solves(tmp_path, monkeypatch):
    """The frozen script runs top to bottom and its model solves optimally."""
    _in_fixture_dir(tmp_path, monkeypatch)
    model = runpy.run_path(str(_SCRIPT))["m"]
    _solve(model)
    assert math.isfinite(pyo.value(model.costing.aggregate_operating_cost))


@pytest.mark.component
@pytest.mark.needs_highs
def test_api_freeze_config_matches_imperative(tmp_path, monkeypatch):
    """The config-driven twin solves to the same objective as the script."""
    _in_fixture_dir(tmp_path, monkeypatch)
    imperative = _solve(runpy.run_path(str(_SCRIPT))["m"])
    from_config = _solve(build_model(load_model_config(_CONFIG)))
    assert from_config == pytest.approx(imperative, rel=1e-6)


@pytest.mark.unit
def test_api_freeze_config_is_schema_valid():
    """The checked-in config validates against the exported JSON Schema."""
    import json

    import jsonschema

    from flexcore import config as flexcore_config

    schemas = Path(flexcore_config.__file__).parent / "schemas"
    schema = json.loads((schemas / "model_config.schema.json").read_text())
    jsonschema.validate(json.loads(_CONFIG.read_text()), schema)


@pytest.mark.component
@pytest.mark.xfail(
    strict=True,
    reason="Known script-vs-config difference: arc names, script "
    "'tank_to_plant' vs config 'arc_0', and so their expanded constraints.",
)
def test_api_freeze_script_and_config_are_equivalent(tmp_path, monkeypatch):
    """The frozen script and its config twin build structurally equal models."""
    from flexops.testing import assert_models_equivalent

    _in_fixture_dir(tmp_path, monkeypatch)
    script = runpy.run_path(str(_SCRIPT))["m"]
    twin = build_model(load_model_config(_CONFIG))
    pyo.TransformationFactory("network.expand_arcs").apply_to(script)
    pyo.TransformationFactory("network.expand_arcs").apply_to(twin)

    assert_models_equivalent(script, twin)
