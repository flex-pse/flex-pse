"""Tests for the persisted degradation-penalty config models."""

import pytest
from pydantic import ValidationError

from flexcore.config.schema import (
    DegradationSpec,
    DegradationTerm,
    DegradationTermSpec,
)
from flexcore.config.spec import FlowsheetSpec
from flexcore.tests.config.test_spec import rejects, spec_dict, unit


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
def test_exceedance_rejects_a_lower_bound_above_the_upper():
    """An inverted band would charge wear at every value, so it is rejected."""
    with pytest.raises(ValidationError):
        DegradationTermSpec(
            kind=DegradationTerm.EXCEEDANCE, variable="p", price=1.0, lower=5, upper=2
        )


@pytest.mark.unit
@pytest.mark.parametrize("value", [float("inf"), float("nan")])
@pytest.mark.parametrize(
    "fields, field",
    [
        ({"kind": DegradationTerm.VARIATION}, "price"),
        ({"kind": DegradationTerm.VARIATION}, "deadband"),
        ({"kind": DegradationTerm.DEVIATION}, "reference"),
        ({"kind": DegradationTerm.EXCEEDANCE, "upper": 1.0}, "lower"),
        ({"kind": DegradationTerm.EXCEEDANCE, "lower": 0.0}, "upper"),
    ],
)
def test_term_rejects_non_finite_values(fields, field, value):
    """inf and nan never reach the solver as coefficients."""
    with pytest.raises(ValidationError):
        DegradationTermSpec(**{"variable": "p", "price": 1.0, **fields, field: value})


@pytest.mark.unit
@pytest.mark.parametrize("value", [float("inf"), float("nan")])
@pytest.mark.parametrize("field", ["covered_cost", "horizon_budget", "period_hours"])
def test_spec_rejects_non_finite_values(field, value):
    """Horizon limits and the period must be finite numbers."""
    with pytest.raises(ValidationError):
        DegradationSpec(name="wear", terms=[_variation()], **{field: value})


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


def _element(**overrides) -> dict:
    """A valid degradation element dict on unit 'tank', with any field overridden."""
    fields = {
        "kind": "degradation",
        "unit": "tank",
        "name": "wear",
        "terms": [{"kind": "variation", "variable": "speed", "price": 2.0}],
    }
    return {**fields, **overrides}


@pytest.mark.unit
def test_degradation_element_bills_the_only_costing_by_default():
    """A degradation element validates and defaults to the model's only costing."""
    spec = FlowsheetSpec.model_validate(spec_dict(unit("tank"), _element()))
    (element,) = spec.of_kind("degradation")
    assert element.costing_package == "auto"


@pytest.mark.unit
def test_degradation_element_rejects_duplicate_names_on_a_unit():
    """Two penalties on one unit cannot share a name (their components would clash)."""
    rejects(
        spec_dict(unit("tank"), _element(), _element()),
        "degradation on 'tank' is set twice",
    )


@pytest.mark.unit
def test_degradation_element_rejects_unknown_unit_and_costing():
    """A degradation element must name an existing unit and costing element."""
    rejects(spec_dict(_element()), "no such unit element")
    rejects(
        spec_dict(unit("tank"), _element(costing_package="nope")),
        "costing 'nope' is not an element",
    )
