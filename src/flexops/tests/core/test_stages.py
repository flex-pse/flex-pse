"""The ordered build stages and apply_stages."""

import json
import shutil
from pathlib import Path

import pytest
from pyomo.network import Arc

from flexcore.config.io import load_model_config
from flexcore.exceptions import FlexConfigError
from flexops import apply_stages, build_model
from flexops.core import stages as stages_module
from flexops.core.stages import STAGE_FUNCTIONS, STAGES

_FIXTURES = Path(__file__).parent.parent / "fixtures"
_FREEZE = _FIXTURES / "api_freeze"
_DEMO = _FIXTURES / "plant_config_demo.json"


@pytest.fixture
def freeze_config(tmp_path, monkeypatch):
    """Return the API-freeze config, with its bare-filename data in the cwd."""
    for path in (_FREEZE / "data").iterdir():
        shutil.copy(path, tmp_path)
    monkeypatch.chdir(tmp_path)
    return load_model_config(_FREEZE / "api_freeze_config.json")


@pytest.mark.unit
def test_stages_order_is_fixed():
    """Changing the build order is a schema-contract change; this is the tripwire."""
    assert STAGES == (
        "declare",
        "topology",
        "surrogates",
        "degradation",
        "ramping",
        "logic",
        "state",
        "extensions",
        "costing",
    )
    assert stages_module.POST_TOPOLOGY_STAGES == STAGES[2:]
    assert set(STAGE_FUNCTIONS) == set(STAGES)


@pytest.mark.component
def test_build_model_runs_every_stage_once(freeze_config, monkeypatch):
    """build_model calls each stage function exactly once, in STAGES order."""
    calls = []

    def recorder(name, original):
        def wrapped(model, cfg, ctx):
            calls.append(name)
            original(model, cfg, ctx)

        return wrapped

    for name in STAGES:
        monkeypatch.setitem(
            STAGE_FUNCTIONS, name, recorder(name, STAGE_FUNCTIONS[name])
        )

    build_model(freeze_config)

    assert calls == list(STAGES)


@pytest.mark.component
def test_expand_arcs_flag(freeze_config):
    """expand_arcs=True deactivates every Arc; the default leaves them active."""
    default = build_model(freeze_config)
    expanded = build_model(freeze_config, expand_arcs=True)

    default_arcs = list(default.component_data_objects(Arc, descend_into=True))
    expanded_arcs = list(expanded.component_data_objects(Arc, descend_into=True))
    assert default_arcs and all(arc.active for arc in default_arcs)
    assert expanded_arcs and not any(arc.active for arc in expanded_arcs)


@pytest.mark.unit
def test_build_context_maps_unit_paths():
    """The build context records each unit under its config path."""
    model = build_model(load_model_config(_DEMO))

    units = model._flex_build_context.units

    assert set(units) == {"demo.tank", "demo.surrogate", "demo.battery"}
    assert units["demo.tank"] is model.demo.tank


@pytest.mark.unit
def test_external_dispatch_still_applied(tmp_path):
    """The state stage fixes a dispatched variable to its series."""
    raw = json.loads(_DEMO.read_text())
    raw["plant"]["units"]["surrogate"]["external_dispatch"] = {
        "variable": "power_electrical",
        "source": "series.json",
    }
    shutil.copy(_FIXTURES / "tariff_tou_demo.json", tmp_path)
    (tmp_path / "series.json").write_text(json.dumps({str(t): 2.5 for t in range(24)}))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))

    model = build_model(load_model_config(path))

    power = model.demo.surrogate.power_electrical
    assert power[0].fixed and power[0].value == 2.5


@pytest.mark.unit
@pytest.mark.parametrize("stage", ["declare", "topology"])
def test_apply_stages_rejects_topology(stage):
    """declare and topology build the model; they cannot be re-run on it."""
    model = build_model(load_model_config(_DEMO))

    with pytest.raises(FlexConfigError, match=stage):
        apply_stages(model, load_model_config(_DEMO), [stage])


@pytest.mark.unit
def test_apply_stages_rejects_unknown():
    """A stage name outside STAGES is a config error naming it."""
    model = build_model(load_model_config(_DEMO))

    with pytest.raises(FlexConfigError, match="nonsense"):
        apply_stages(model, load_model_config(_DEMO), ["nonsense"])


@pytest.mark.unit
def test_apply_stages_rejects_double_costing():
    """Costing cannot run twice; it would double-register costs."""
    model = build_model(load_model_config(_DEMO))

    with pytest.raises(FlexConfigError, match="costing"):
        apply_stages(model, load_model_config(_DEMO), ["costing"])


@pytest.mark.unit
def test_apply_stages_runs_noop_stages_on_live_model():
    """The default post-topology stages leave a built model unchanged."""
    cfg = load_model_config(_DEMO)
    model = build_model(cfg)
    reference = build_model(cfg)

    apply_stages(model, cfg, [s for s in STAGES[2:] if s != "costing"])

    from flexops.testing import assert_models_equivalent

    assert_models_equivalent(model, reference)


@pytest.mark.unit
def test_apply_stages_resolves_units_without_build_context():
    """A model with no stored context gets one by resolving config unit paths."""
    cfg = load_model_config(_DEMO)
    model = build_model(cfg)
    del model._flex_build_context

    apply_stages(model, cfg, ["state"])

    bad = cfg.model_copy(deep=True)
    bad.plant.units["ghost"] = bad.plant.units["tank"]
    with pytest.raises(FlexConfigError, match="demo.ghost"):
        apply_stages(model, bad, ["state"])
