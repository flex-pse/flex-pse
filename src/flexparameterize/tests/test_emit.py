"""Tests for emit_model_config: the serializable FlexParameterize direction."""

import json

import pytest

from flexcore.config.io import dump_model_config, load_model_config
from flexcore.config.schema import SurrogateSpec
from flexparameterize.emit import emit_model_config
from flexparameterize.tests.helpers import build_plant, evaluate_data, fit_intensity


@pytest.mark.unit
def test_emitted_config_validates(tmp_path):
    """An emitted config round-trips through dump/load (pydantic validates it)."""
    _, unit = build_plant()
    regressor = fit_intensity(unit, evaluate_data(unit))

    cfg = emit_model_config(unit, regressor)
    path = tmp_path / "emitted.json"
    dump_model_config(cfg, path)
    reloaded = load_model_config(path)

    unit_cfg = reloaded.plant.units["plant"]
    assert unit_cfg.unit_model_class == "ConstantEnergyIntensityModel"
    assert unit_cfg.surrogate.functional_form == "constant_intensity"
    assert {spec.name for spec in unit_cfg.io_variables} == {
        "flow_vol_phase",
        "power_electrical",
    }


@pytest.mark.unit
def test_provenance_fields_present():
    """Fit metrics, data window and package versions are present and JSON-safe."""
    _, unit = build_plant()
    data = evaluate_data(unit)
    regressor = fit_intensity(unit, data)

    cfg = emit_model_config(unit, regressor, {"data_source": "unit test"})

    provenance = cfg.plant.units["plant"].surrogate.provenance
    assert provenance["n_samples"] == len(data)
    assert provenance["r2"] == pytest.approx(1.0)
    assert provenance["rmse"] == pytest.approx(0.0, abs=1e-12)
    assert provenance["data_window"] == [
        data.index.min().isoformat(),
        data.index.max().isoformat(),
    ]
    assert set(provenance["versions"]) == {"flex-pse", "pyomo", "pandas"}
    assert provenance["data_source"] == "unit test"
    json.dumps(provenance)


@pytest.mark.unit
def test_emit_from_hand_built_surrogate():
    """A hand-built spec needs no data, no fit, and documents its own source."""
    _, unit = build_plant()
    spec = SurrogateSpec(
        functional_form="linear",
        input_variables=["flow_in"],
        coefficients={"flow_in": 0.42, "intercept": 1.0},
    )

    cfg = emit_model_config(unit, spec, {"source": "vendor_datasheet"})

    surrogate = cfg.plant.units["plant"].surrogate
    assert surrogate.functional_form == "linear"
    assert surrogate.provenance["source"] == "vendor_datasheet"
    assert not {"n_samples", "r2", "rmse"} & set(surrogate.provenance)
    assert load_model_config(cfg.model_dump()) == cfg
