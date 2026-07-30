"""Tests for the versioned config schema and its JSON load-dump round-trip."""

import json

import pytest
from pydantic import BaseModel, ValidationError

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
    PriceSpec,
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
            output_variables=["power_electrical"],
        ),
    )
    plant_unit = UnitConfig(
        unit_model_class="ConstantEnergyIntensityModel",
        external_dispatch=ExternalDispatchSpec(
            variable="power_electrical", source="dispatch.csv"
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
def test_model_config_roundtrip(tmp_path):
    """A full config round-trips through the canonical JSON format."""
    cfg = _model_config()
    path = tmp_path / "model.json"
    dump_model_config(cfg, path)
    reloaded = load_model_config(path)
    assert reloaded.model_dump() == cfg.model_dump()


@pytest.mark.unit
@pytest.mark.parametrize("suffix", [".yaml", ".yml", ".toml"])
def test_non_json_suffix_rejected(tmp_path, suffix):
    """JSON is the only on-disk format: other suffixes are rejected on load
    and dump."""
    cfg = _model_config()
    path = tmp_path / f"model{suffix}"
    with pytest.raises(FlexConfigError):
        dump_model_config(cfg, path)
    path.write_text(json.dumps(cfg.model_dump(mode="json")))
    with pytest.raises(FlexConfigError):
        load_model_config(path)


@pytest.mark.unit
def test_load_model_config_from_dict():
    """load_model_config accepts a plain dict and leaves the input unmutated."""
    cfg = _model_config()
    data = cfg.model_dump(mode="json")
    snapshot = json.loads(json.dumps(data))
    loaded = load_model_config(data)
    assert loaded.model_dump() == cfg.model_dump()
    assert data == snapshot


@pytest.mark.unit
def test_load_model_config_dict_missing_version_raises():
    """A dict config without schema_version is rejected like a file config."""
    data = _model_config().model_dump(mode="json")
    del data["schema_version"]
    with pytest.raises(FlexConfigError):
        load_model_config(data)


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
    data["schema_version"] = "999.0.0"
    path = tmp_path / "future.json"
    path.write_text(json.dumps(data))
    with pytest.raises(FlexConfigError):
        load_model_config(path)


@pytest.mark.unit
def test_current_schema_version_is_semver():
    """The build's schema version is a semantic-version string."""
    major, minor, patch = CURRENT_SCHEMA_VERSION.split(".")
    assert all(part.isdigit() for part in (major, minor, patch))


@pytest.mark.unit
@pytest.mark.parametrize("bad_version", [1, "1", "0.1", "v0.0.1", True])
def test_malformed_schema_version_raises(bad_version):
    """A schema_version that is not an X.Y.Z string is rejected at load."""
    data = _model_config().model_dump(mode="json")
    data["schema_version"] = bad_version
    with pytest.raises(FlexConfigError):
        load_model_config(data)


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
def test_price_spec_accepts_scalar_or_per_period_list():
    """A price is one number, or one number per period of the horizon."""
    scalar = PriceSpec(value=0.12, units="USD/kWh")
    series = PriceSpec(value=[0.10, 0.20, 0.15], units="USD/kWh")
    assert scalar.value == 0.12
    assert series.value == [0.10, 0.20, 0.15]

    cfg = CostingConfig(energy_prices={"electrical": series})
    assert cfg.energy_prices["electrical"].value == [0.10, 0.20, 0.15]


@pytest.mark.unit
def test_price_spec_rejects_empty_series():
    """An empty price list is rejected: it can never align with the horizon."""
    with pytest.raises(ValidationError):
        PriceSpec(value=[], units="USD/kWh")


@pytest.mark.unit
def test_unknown_key_rejected():
    """extra='forbid' rejects an undocumented key on a nested model.

    The counterfactual shows the strict base is what rejects: a plain pydantic
    model without extra='forbid' accepts the same unknown key and silently
    drops it.
    """
    with pytest.raises(ValidationError):
        IOVariableSpec(name="x", role="input", units="m^3/hr", mystery_key=True)

    class LooseSpec(BaseModel):
        name: str

    loose = LooseSpec(name="x", mystery_key=True)
    assert not hasattr(loose, "mystery_key")


@pytest.mark.unit
def test_exported_schema_up_to_date(tmp_path):
    """The checked-in JSON Schema matches the in-memory model."""
    from importlib.resources import files

    export_json_schemas(tmp_path)
    current = (tmp_path / "model_config.schema.json").read_text()
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
    assert written == json.dumps(json.loads(written), indent=2, sort_keys=True)


@pytest.mark.unit
def test_export_json_schemas_custom_filename(tmp_path):
    """A filename override writes side by side without clobbering the default."""
    export_json_schemas(tmp_path)
    export_json_schemas(tmp_path, filename="model_config.schema.v1.json")
    default = (tmp_path / "model_config.schema.json").read_text()
    versioned = (tmp_path / "model_config.schema.v1.json").read_text()
    assert versioned == default


@pytest.mark.unit
def test_exported_descriptions_are_plain_text(tmp_path):
    """No exported description carries newlines or formatting codes (§, RST)."""
    export_json_schemas(tmp_path)
    schema = json.loads((tmp_path / "model_config.schema.json").read_text())

    def walk(node):
        if isinstance(node, dict):
            desc = node.get("description")
            if isinstance(desc, str):
                yield desc
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)

    for desc in walk(schema):
        assert "\n" not in desc, f"newline in description: {desc!r}"
        assert "§" not in desc, f"section sign in description: {desc!r}"
        assert "``" not in desc, f"RST literal in description: {desc!r}"
