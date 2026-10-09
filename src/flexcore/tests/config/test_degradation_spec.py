"""Tests for the persisted degradation-penalty config models."""

import pytest
from pydantic import ValidationError

from flexcore.config.io import load_model_config
from flexcore.config.schema import (
    CURRENT_SCHEMA_VERSION,
    DegradationSpec,
    DegradationTerm,
    DegradationTermSpec,
    UnitConfig,
)


def _variation(**overrides) -> DegradationTermSpec:
    """A valid variation term, with any field overridden."""
    fields = {"kind": DegradationTerm.VARIATION, "variable": "speed", "price": 2.0}
    return DegradationTermSpec(**{**fields, **overrides})


@pytest.mark.unit
def test_term_kind_rejects_an_unknown_name():
    """An unknown term kind is rejected at validation time."""
    with pytest.raises(ValidationError):
        DegradationTermSpec(kind="rainflow", variable="speed", price=1.0)


@pytest.mark.unit
def test_term_defaults_are_no_deadband_and_one_step_window():
    """A minimal term charges every unit of change between adjacent steps."""
    term = _variation()
    assert term.deadband == 0.0
    assert term.window == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [{"price": -1.0}, {"deadband": -0.5}, {"window": 0}],
)
def test_term_rejects_negative_price_deadband_and_empty_window(overrides):
    """Prices and deadbands are non-negative and a window spans at least one step."""
    with pytest.raises(ValidationError):
        _variation(**overrides)


@pytest.mark.unit
def test_deviation_requires_a_reference():
    """A deviation term measures distance from a reference, so one is required."""
    with pytest.raises(ValidationError):
        DegradationTermSpec(kind=DegradationTerm.DEVIATION, variable="flow", price=1.0)
    term = DegradationTermSpec(
        kind=DegradationTerm.DEVIATION, variable="flow", price=1.0, reference=80.0
    )
    assert term.reference == 80.0


@pytest.mark.unit
def test_exceedance_requires_a_bound_and_rejects_an_deadband():
    """An exceedance needs at least one bound; the bound is already its deadband."""
    with pytest.raises(ValidationError):
        DegradationTermSpec(kind=DegradationTerm.EXCEEDANCE, variable="p", price=1.0)
    with pytest.raises(ValidationError):
        DegradationTermSpec(
            kind=DegradationTerm.EXCEEDANCE,
            variable="p",
            price=1.0,
            upper=10.0,
            deadband=1.0,
        )
    term = DegradationTermSpec(
        kind=DegradationTerm.EXCEEDANCE, variable="p", price=1.0, lower=2.0
    )
    assert term.upper is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [{"reference": 1.0}, {"upper": 1.0}, {"lower": 1.0}],
)
def test_term_rejects_fields_belonging_to_another_kind(overrides):
    """A field another kind uses is rejected rather than silently ignored."""
    with pytest.raises(ValidationError):
        _variation(**overrides)


@pytest.mark.unit
def test_window_is_only_for_variation():
    """Only a variation term compares against an earlier step."""
    with pytest.raises(ValidationError):
        DegradationTermSpec(
            kind=DegradationTerm.THROUGHPUT, variable="flow", price=1.0, window=2
        )


@pytest.mark.unit
def test_spec_requires_at_least_one_term():
    """A degradation penalty with no terms is rejected."""
    with pytest.raises(ValidationError):
        DegradationSpec(name="wear", terms=[])


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [{"covered_cost": -1.0}, {"horizon_budget": -1.0}, {"period_hours": 0.0}],
)
def test_spec_rejects_negative_horizon_limits_and_empty_period(overrides):
    """covered cost and budget are non-negative; a period has positive length."""
    with pytest.raises(ValidationError):
        DegradationSpec(name="wear", terms=[_variation()], **overrides)


@pytest.mark.unit
def test_unit_config_defaults_to_no_degradation():
    """A unit carries no degradation penalty unless one is configured."""
    assert UnitConfig(unit_model_class="Pump").degradation == []


@pytest.mark.unit
def test_unit_config_rejects_duplicate_degradation_names():
    """Two penalties on one unit cannot share a name (their components would clash)."""
    spec = DegradationSpec(name="wear", terms=[_variation()])
    with pytest.raises(ValidationError):
        UnitConfig(unit_model_class="Pump", degradation=[spec, spec])


@pytest.mark.unit
def test_previous_schema_version_with_degradation_free_units_migrates():
    """A 0.0.3 document (which predates degradation) loads at the current version."""
    data = {
        "schema_version": "0.0.3",
        "time": {
            "start_date": "2025-07-08",
            "end_date": "2025-07-09",
            "time_step": "1 hr",
        },
        "costing": {"tariff_source": "tariff.json"},
        "plant": {"name": "demo", "units": {"pump": {"unit_model_class": "Pump"}}},
    }
    loaded = load_model_config(data)
    assert loaded.schema_version == CURRENT_SCHEMA_VERSION == "0.0.4"
    assert loaded.plant.units["pump"].degradation == []
