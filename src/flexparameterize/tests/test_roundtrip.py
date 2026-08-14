"""The headline invariants: data round-trips, and both directions agree."""

import pyomo.environ as pyo
import pytest

from flexcore.config.io import dump_model_config, load_model_config
from flexops.core.build import build_model
from flexops.unit_models import Pump
from flexparameterize.apply import apply_to_model
from flexparameterize.emit import emit_model_config
from flexparameterize.tags import TagMap, model_alias
from flexparameterize.tests.helpers import (
    INTENSITY,
    build_plant,
    evaluate_data,
    fit_intensity,
)
from flexparameterize.validate import check_sufficiency


@pytest.mark.component
def test_constant_intensity_round_trip(tmp_path):
    """Known intensity -> data -> tags -> fit -> config -> rebuilt model."""
    m, pump = build_plant(Pump)
    data = evaluate_data(pump)
    tags = {
        "FT_0231.PV": model_alias(pump.inlet_state.flow_vol_phase),
        "FT_0232.PV": model_alias(pump.outlet_state.flow_vol_phase),
        "MTR_KW_04": model_alias(pump.power_electrical),
    }
    historian = data.rename(columns={alias: tag for tag, alias in tags.items()})

    aliased = TagMap(tags).apply(historian)
    assert check_sufficiency(m, aliased, m.time_block).sufficient is True

    cfg = emit_model_config(
        pump, fit_intensity(pump, aliased), {"data_source": "round-trip test"}
    )
    path = tmp_path / "emitted.json"
    dump_model_config(cfg, path)
    rebuilt = build_model(load_model_config(path))

    assert pyo.value(rebuilt.facility.plant.energy_intensity) == pytest.approx(
        INTENSITY, rel=1e-6
    )


@pytest.mark.component
def test_apply_and_emit_agree():
    """The in-place mutation and the emitted-config rebuild describe one model."""
    m_applied, applied = build_plant()
    applied_data = evaluate_data(applied)
    applied.energy_intensity.unfix()
    apply_to_model(m_applied, applied_data, TagMap({}))

    _, emitted = build_plant()
    cfg = emit_model_config(emitted, fit_intensity(emitted, evaluate_data(emitted)))
    rebuilt = build_model(cfg).facility.plant

    assert pyo.value(rebuilt.energy_intensity) == pytest.approx(
        pyo.value(applied.energy_intensity), rel=1e-6
    )
    for probe in (3.0, 11.0):
        for unit in (applied, rebuilt):
            unit.flow_out[0].set_value(probe)
            unit.power_electrical[0].set_value(0.0)
        assert pyo.value(applied.power_electrical_relation[0].body) == pytest.approx(
            pyo.value(rebuilt.power_electrical_relation[0].body), rel=1e-6
        )
