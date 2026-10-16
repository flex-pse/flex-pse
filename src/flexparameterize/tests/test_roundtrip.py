"""The headline invariant: a fit applied in place round-trips through a spec."""

import pyomo.environ as pyo
import pytest

from flexcore.config.io import dump_spec, load_spec
from flexops import build_model, emit_model
from flexops.unit_models import Pump
from flexparameterize.apply import apply_to_model
from flexparameterize.tags import TagMap, model_alias
from flexparameterize.tests.helpers import INTENSITY, build_plant, evaluate_data
from flexparameterize.validate import check_sufficiency


@pytest.mark.component
def test_constant_intensity_round_trip(tmp_path):
    """Known intensity -> data -> tags -> fit in place -> spec -> rebuilt model."""
    _, truth = build_plant(Pump)
    data = evaluate_data(truth)
    m, pump = build_plant(Pump, intensity=1.0)
    tags = {
        "FT_0231.PV": model_alias(pump.inlet_state.flow_vol_phase),
        "FT_0232.PV": model_alias(pump.outlet_state.flow_vol_phase),
        "MTR_KW_04": model_alias(pump.power_electrical),
    }
    historian = data.rename(columns={alias: tag for tag, alias in tags.items()})
    assert check_sufficiency(m, TagMap(tags).apply(historian), m.time_block).sufficient

    pump.energy_intensity.unfix()
    apply_to_model(m, historian, TagMap(tags))
    path = tmp_path / "emitted.json"
    dump_spec(emit_model(m), path)
    rebuilt = build_model(load_spec(path)).facility.plant

    assert pyo.value(rebuilt.energy_intensity) == pytest.approx(INTENSITY, rel=1e-6)
    for probe in (3.0, 11.0):
        for unit in (pump, rebuilt):
            unit.flow_out[0].set_value(probe)
            unit.power_electrical[0].set_value(0.0)
        assert pyo.value(pump.power_electrical_relation[0].body) == pytest.approx(
            pyo.value(rebuilt.power_electrical_relation[0].body), rel=1e-6
        )
