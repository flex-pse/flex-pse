"""Tests for the versioned config schema and its YAML/JSON load-dump round-trip."""

import json

import pytest
from pydantic import ValidationError

from flexcore.config.io import (
    dump_model_config,
    export_json_schemas,
    load_model_config,
)
from flexcore.config.schema import (
    CURRENT_SCHEMA_VERSION,
    CostingConfig,
    DRConfig,
    ExternalDispatchSpec,
    IOVariableSpec,
    ModelConfig,
    PlantConfig,
    SurrogateSpec,
    TimeConfig,
    UnitCommitmentConfig,
    UnitConfig,
)
from flexcore.exceptions import FlexConfigError


def _model_config() -> ModelConfig:
    """A representative full ModelConfig: a plant of two contrasting units."""
    tank = UnitConfig(
        unit_model_class="Tank",
        io_variables=[
            IOVariableSpec(name="flow_in", role="input", units="m^3/hr"),
            IOVariableSpec(name="flow_out", role="output", units="m^3/hr"),
        ],
        surrogate=SurrogateSpec(
            functional_form="linear",
            coefficients={"slope": 0.5, "intercept": 0.1},
            input_variables=["flow_in"],
            output_variables=["electrical_power"],
        ),
    )
    plant_unit = UnitConfig(
        unit_model_class="ConstantEnergyIntensityModel",
        external_dispatch=ExternalDispatchSpec(
            variable="electrical_power", source="dispatch.csv"
        ),
        unit_commitment=UnitCommitmentConfig(
            startup_shutdown=True, dwell=True, min_up=4, min_down=2
        ),
    )
    return ModelConfig(
        schema_version=CURRENT_SCHEMA_VERSION,
        time=TimeConfig(
            start_date="2025-01-01", end_date="2025-01-30", time_step="15 min"
        ),
        costing=CostingConfig(tariff_source="tariff.json", dr=DRConfig()),
        plant=PlantConfig(name="svcw", units={"tank": tank, "plant": plant_unit}),
    )


@pytest.mark.unit
def test_model_config_yaml_roundtrip(tmp_path):
    """A full config round-trips through the canonical YAML format."""
    cfg = _model_config()
    path = tmp_path / "model.yaml"
    dump_model_config(cfg, path)
    reloaded = load_model_config(path)
    assert reloaded.model_dump() == cfg.model_dump()


@pytest.mark.unit
def test_model_config_json_roundtrip(tmp_path):
    """The same config also round-trips through the accepted JSON format."""
    cfg = _model_config()
    path = tmp_path / "model.json"
    dump_model_config(cfg, path)
    reloaded = load_model_config(path)
    assert reloaded.model_dump() == cfg.model_dump()


@pytest.mark.unit
def test_invalid_role_names_field_path(tmp_path):
    """An invalid IO role surfaces the full field path in the error text."""
    cfg = _model_config()
    data = cfg.model_dump(mode="json")
    data["plant"]["units"]["tank"]["io_variables"][0]["role"] = "both"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(FlexConfigError) as exc:
        load_model_config(path)
    assert "plant.units.tank.io_variables.0.role" in str(exc.value)


@pytest.mark.unit
def test_missing_schema_version_raises(tmp_path):
    """A config without schema_version is rejected at load."""
    cfg = _model_config()
    data = cfg.model_dump(mode="json")
    del data["schema_version"]
    path = tmp_path / "noversion.json"
    path.write_text(json.dumps(data))
    with pytest.raises(FlexConfigError):
        load_model_config(path)


@pytest.mark.unit
def test_schema_version_too_new_raises(tmp_path):
    """A schema_version newer than this build is rejected."""
    cfg = _model_config()
    data = cfg.model_dump(mode="json")
    data["schema_version"] = CURRENT_SCHEMA_VERSION + 1
    path = tmp_path / "future.json"
    path.write_text(json.dumps(data))
    with pytest.raises(FlexConfigError):
        load_model_config(path)


@pytest.mark.unit
def test_network_or_plant_exactly_one():
    """Exactly one of network/plant must be set."""
    base = _model_config()
    plant = base.plant

    # Neither.
    with pytest.raises(ValidationError):
        ModelConfig(
            schema_version=CURRENT_SCHEMA_VERSION,
            time=base.time,
            costing=base.costing,
        )
    # Both.
    with pytest.raises(ValidationError):
        ModelConfig(
            schema_version=CURRENT_SCHEMA_VERSION,
            time=base.time,
            costing=base.costing,
            plant=plant,
            network={"name": "n", "plants": {"p": plant.model_dump()}},
        )
    # Exactly one passes.
    assert base.plant is not None


@pytest.mark.unit
def test_unknown_key_rejected():
    """extra='forbid' rejects an undocumented key on a nested model."""
    with pytest.raises(ValidationError):
        IOVariableSpec(name="x", role="input", units="m^3/hr", mystery_key=True)


@pytest.mark.unit
def test_exported_schema_up_to_date():
    """The checked-in JSON Schema matches the in-memory model."""
    from importlib.resources import files

    current = json.dumps(ModelConfig.model_json_schema(), indent=2, sort_keys=True)
    checked_in = (
        files("flexcore.config.schemas")
        .joinpath("model_config.schema.json")
        .read_text()
    )
    assert current == checked_in, (
        "Checked-in JSON Schema is stale; re-run export_json_schemas and commit "
        "src/flexcore/config/schemas/model_config.schema.json."
    )


@pytest.mark.unit
def test_export_json_schemas_writes_stable_output(tmp_path):
    """export_json_schemas writes deterministically sorted, indented JSON."""
    export_json_schemas(tmp_path)
    written = (tmp_path / "model_config.schema.json").read_text()
    expected = json.dumps(ModelConfig.model_json_schema(), indent=2, sort_keys=True)
    assert written == expected
