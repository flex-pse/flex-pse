"""Tests for the shared config models and the exported JSON Schema."""

import json

import pytest
from pydantic import BaseModel, ValidationError

from flexcore.config.io import export_json_schemas
from flexcore.config.schema import (
    CostingConfig,
    DRBaseline,
    DRBaselineMethod,
    DRConfig,
    DREvent,
    DRPayment,
    DRPaymentKind,
    DRPaymentRegion,
    HistoricalPower,
    IOVariableSpec,
    PriceSpec,
    SurrogateSpec,
)

SCHEMA_FILENAME = "flowsheet_spec.schema.json"


@pytest.mark.unit
def test_surrogate_type_rejects_an_unknown_name():
    """Unlike the old open ``functional_form`` string, an unknown enum member
    is rejected at config-validation time, not build time."""
    with pytest.raises(ValidationError):
        SurrogateSpec(surrogate_type="vendor_curve_v3")


@pytest.mark.unit
def test_consumption_estimate_rejects_an_unknown_utility():
    """A misspelled utility key fails config validation."""
    with pytest.raises(ValidationError):
        CostingConfig(
            tariff_source="tariff.json", consumption_estimate={"electricity": 1.0}
        )


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
    current = (tmp_path / SCHEMA_FILENAME).read_text()
    checked_in = files("flexcore.config.schemas").joinpath(SCHEMA_FILENAME)
    assert current == checked_in.read_text(), (
        "Checked-in JSON Schema is stale; re-run export_json_schemas and commit "
        f"src/flexcore/config/schemas/{SCHEMA_FILENAME}."
    )


@pytest.mark.unit
def test_export_json_schemas_writes_stable_output(tmp_path):
    """export_json_schemas writes deterministically sorted, indented JSON."""
    export_json_schemas(tmp_path)
    written = (tmp_path / SCHEMA_FILENAME).read_text()
    assert written == json.dumps(json.loads(written), indent=2, sort_keys=True)


@pytest.mark.unit
def test_export_json_schemas_custom_filename(tmp_path):
    """A filename override writes side by side without clobbering the default."""
    export_json_schemas(tmp_path)
    export_json_schemas(tmp_path, filename="flowsheet_spec.schema.v1.json")
    default = (tmp_path / SCHEMA_FILENAME).read_text()
    versioned = (tmp_path / "flowsheet_spec.schema.v1.json").read_text()
    assert versioned == default


@pytest.mark.unit
def test_exported_descriptions_are_plain_text(tmp_path):
    """No exported description carries newlines or formatting codes (§, RST)."""
    export_json_schemas(tmp_path)
    schema = json.loads((tmp_path / SCHEMA_FILENAME).read_text())

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


def _dr_event(**changes) -> dict:
    """Return one valid demand-response event as plain data."""
    event = {
        "event_date": "2025-07-08",
        "start_hour": 16,
        "duration_hours": 3,
        "baseline_days": ["2025-07-07"],
        "bid_capacity_kW": 100.0,
        "capacity_price": 10.0,
    }
    return event | changes


def _dr_config(**changes) -> dict:
    """Return one valid demand-response config as plain data."""
    config = {
        "events": [_dr_event()],
        "historical_power": {
            "start": "2025-07-07T00:00",
            "time_step_minutes": 60,
            "values_kW": [100.0] * 24,
        },
        "baseline": {"method": "fixed_level", "firm_level_kW": 150.0},
        "payment": {"kind": "capacity", "regions": [{"x1": 0.0, "y1": 0.0, "y2": 0.0}]},
    }
    return config | changes


@pytest.mark.unit
def test_dr_config_parses_events_baseline_and_payment():
    """A demand-response config parses into typed events, baseline and payment."""
    dr = DRConfig.model_validate(_dr_config())
    assert isinstance(dr.events[0], DREvent)
    assert dr.events[0].notification_hours == 0.0
    assert dr.events[0].region_x1 is None
    assert isinstance(dr.historical_power, HistoricalPower)
    assert dr.baseline.method is DRBaselineMethod.FIXED_LEVEL
    assert dr.payment.kind is DRPaymentKind.CAPACITY
    assert dr.payment.settlement == "interval"


@pytest.mark.unit
def test_dr_payment_region_none_bound_means_unbounded():
    """A missing region bound means unbounded, since JSON cannot hold infinity."""
    region = DRPaymentRegion(x1=1.0, y1=1.0, y2=1.0)
    assert region.x1 == 1.0
    assert region.x2 is None


@pytest.mark.unit
def test_dr_fixed_level_baseline_needs_a_firm_level():
    """A fixed-level baseline without its contracted level is rejected."""
    with pytest.raises(ValidationError, match="firm_level_kW"):
        DRBaseline(method="fixed_level")


@pytest.mark.unit
def test_dr_capacity_energy_payment_needs_an_energy_price():
    """A capacity-plus-energy payment without an energy price is rejected."""
    with pytest.raises(ValidationError, match="energy_price"):
        DRPayment(kind="capacity_energy", regions=None)


@pytest.mark.unit
def test_dr_rejects_empty_baseline_days_and_history():
    """An event with no baseline days, or an empty meter history, is rejected."""
    with pytest.raises(ValidationError):
        DREvent.model_validate(_dr_event(baseline_days=[]))
    with pytest.raises(ValidationError):
        HistoricalPower(start="2025-07-07", time_step_minutes=60, values_kW=[])


@pytest.mark.unit
def test_costing_config_takes_a_dr_config():
    """CostingConfig.dr holds the full demand-response config."""
    cfg = CostingConfig(tariff_source="tariff.json", dr=_dr_config())
    assert isinstance(cfg.dr, DRConfig)
