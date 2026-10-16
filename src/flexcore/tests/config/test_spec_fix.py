"""Tests for the fix and unfix element kinds and the spec's fix policy fields."""

import pytest

from flexcore.config.spec import KINDS, FlowsheetSpec, SourceRef
from flexcore.tests.config.test_spec import plant, rejects, spec_dict, unit


def fix(variable="p.tank.volume", **fields):
    """Return a fix element dict."""
    return {"kind": "fix", "variable": variable, "value": 1.0, **fields}


def accepts(*elements) -> FlowsheetSpec:
    """Validate a spec of plant p, tank p.tank and ``elements``."""
    return FlowsheetSpec.model_validate(spec_dict(plant(), unit("p.tank"), *elements))


@pytest.mark.unit
def test_fix_and_unfix_are_mutable_state_kinds():
    assert KINDS["fix"] == ("state", True)
    assert KINDS["unfix"] == ("state", True)


@pytest.mark.unit
def test_fix_policy_defaults_to_error():
    spec = accepts()

    assert spec.fix_on_violation == "error"
    assert spec.fix_tolerance == 1e-9


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [0.5, [1.0, 2.0], {"0": 1.0, "3": 2.0}, {"$source": "soc.json"}],
    ids=["number", "list", "mapping", "source"],
)
def test_fix_value_forms_validate(value):
    (element,) = accepts(fix(value=value)).of_kind("fix")

    if isinstance(value, dict) and "$source" in value:
        assert element.value == SourceRef(source="soc.json")
    else:
        assert element.value == value


@pytest.mark.unit
def test_fix_fields_validate():
    (element,) = accepts(
        fix(
            "p.tank.volume[*]",
            value=[1.0, 2.0],
            at=[0, 3],
            units="m^3",
            on_violation="skip",
            tolerance=0.01,
            name="levels",
        )
    ).of_kind("fix")

    assert (element.at, element.on_violation, element.tolerance) == (
        [0, 3],
        "skip",
        0.01,
    )


@pytest.mark.unit
def test_unfix_validates():
    (element,) = accepts({"kind": "unfix", "variable": "p.ro[*].recovery"}).of_kind(
        "unfix"
    )

    assert element.variable == "p.ro[*].recovery"


@pytest.mark.unit
@pytest.mark.parametrize("variable", ["", "p.tank. volume", "p.tank.volume\t"])
def test_fix_variable_must_be_a_path_without_whitespace(variable):
    rejects(spec_dict(plant(), fix(variable)), "variable")


@pytest.mark.unit
def test_unfix_variable_must_not_be_empty():
    rejects(spec_dict(plant(), {"kind": "unfix", "variable": ""}), "variable")


@pytest.mark.unit
def test_fix_list_length_must_match_at():
    rejects(spec_dict(plant(), fix(value=[1.0, 2.0], at=[0])), "at", "2", "1")


@pytest.mark.unit
def test_bad_policy_and_negative_tolerance_rejected():
    rejects(spec_dict(plant(), fix(on_violation="clamp")), "on_violation")
    rejects(spec_dict(plant(), fix(tolerance=-1.0)), "tolerance")


@pytest.mark.unit
def test_spec_policy_fields_validate():
    data = spec_dict(plant())
    data.update(fix_on_violation="warn", fix_tolerance=0.5)

    spec = FlowsheetSpec.model_validate(data)

    assert (spec.fix_on_violation, spec.fix_tolerance) == ("warn", 0.5)
