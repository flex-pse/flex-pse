"""DesignSpec: validation of a design study and loading it from files."""

import json

import pytest
from pydantic import ValidationError

from flexcore.config.design import DesignSpec, load_design_spec
from flexcore.config.spec import SCHEMA_VERSION
from flexcore.exceptions import FlexConfigError


def _base() -> dict:
    """Return a minimal flat spec with one costing element and a named fix."""
    return {
        "schema_version": SCHEMA_VERSION,
        "name": "base",
        "elements": [
            {
                "kind": "time",
                "start_date": "2025-01-01",
                "end_date": "2025-02-01",
                "time_step": "1 hr",
            },
            {
                "kind": "costing",
                "name": "costing",
                "energy_prices": {"electrical": {"value": 0.1, "units": "USD/kWh"}},
            },
            {"kind": "fix", "name": "load", "variable": "pump.flow", "value": 1.0},
        ],
    }


def _study(**overrides) -> dict:
    """Return a valid study document with the given fields replaced."""
    study = {
        "schema_version": "0.1.0",
        "name": "study",
        "base": _base(),
        "years": [{"year": 2026, "represents": 5}, {"year": 2031, "represents": 5}],
        "months": [{"month": 1, "weight": 6}, {"month": 7, "weight": 6}],
        "changes": [],
        "capacities": [
            {
                "name": "solar",
                "variable": "pv.capacity",
                "capital_cost": 1000.0,
                "lifetime_years": 25,
                "upper": 100.0,
            }
        ],
    }
    return {**study, **overrides}


def _capacity(**overrides) -> dict:
    """Return a valid capacity decision with the given fields replaced."""
    return {**_study()["capacities"][0], **overrides}


@pytest.mark.unit
def test_valid_study_is_accepted():
    """A consistent study validates, with every field reachable."""
    spec = DesignSpec.model_validate(_study())
    assert [year.year for year in spec.years] == [2026, 2031]
    assert spec.capacities[0].lower == 0.0


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"years": [{"year": 2026, "represents": 5}, {"year": 2030}]}, "must follow"),
        ({"months": [{"month": 1, "weight": 6}, {"month": 7, "weight": 5}]}, "sum to"),
        ({"months": [{"month": 13, "weight": 12}]}, "less than or equal to 12"),
        ({"months": [{"month": 1, "weight": 6}, {"month": 1, "weight": 6}]}, "unique"),
        ({"capacities": [_capacity(), _capacity()]}, "unique"),
        ({"capacities": [_capacity(lower=5.0, upper=1.0)]}, "lower > upper"),
        ({"months": [{"month": 1, "weight": 12, "days": 0}]}, "greater than or equal"),
        ({"months": [{"month": 1, "weight": 12, "days": 29}]}, "less than or equal"),
        ({"unexpected": 1}, "Extra inputs"),
        ({"changes": [{"element": "nope", "values": {}}]}, "unknown element"),
        (
            {"changes": [{"element": "load", "values": {"name": "other"}}]},
            "name or kind",
        ),
        (
            {"changes": [{"element": "load", "values": {}, "years": [2027]}]},
            "unmodeled year",
        ),
        (
            {"changes": [{"element": "load", "values": {}, "months": [2]}]},
            "unmodeled month",
        ),
    ],
)
def test_invalid_study_is_rejected(overrides, message):
    """Each inconsistent field raises a message naming the problem."""
    with pytest.raises(ValidationError, match=message):
        DesignSpec.model_validate(_study(**overrides))


@pytest.mark.unit
@pytest.mark.parametrize("field", ["upper", "lifetime_years"])
def test_capacity_requires_upper_and_lifetime(field):
    """A decision without an upper bound or lifetime is rejected."""
    capacity = _capacity()
    del capacity[field]
    with pytest.raises(ValidationError, match=field):
        DesignSpec.model_validate(_study(capacities=[capacity]))


@pytest.mark.unit
@pytest.mark.parametrize("count", [0, 2])
def test_base_needs_exactly_one_costing_element(count):
    """Zero or two costing elements are rejected."""
    base = _base()
    base["elements"] = [el for el in base["elements"] if el["kind"] != "costing"]
    base["elements"] += [
        {
            "kind": "costing",
            "name": f"costing_{i}",
            "energy_prices": {"electrical": {"value": 0.1, "units": "USD/kWh"}},
        }
        for i in range(count)
    ]
    with pytest.raises(ValidationError, match="one costing element"):
        DesignSpec.model_validate(_study(base=base))


@pytest.mark.unit
def test_loader_resolves_base_path_relative_to_study(tmp_path):
    """A string base loads relative to the study, keeping its own directory."""
    data = tmp_path / "data"
    data.mkdir()
    (data / "base.json").write_text(json.dumps(_base()))
    path = tmp_path / "study.json"
    path.write_text(json.dumps(_study(base="data/base.json")))
    assert load_design_spec(path).base._base_dir == data


@pytest.mark.unit
def test_loader_sets_inline_base_directory_to_the_study_directory(tmp_path):
    """An inline base resolves its data files beside the study."""
    path = tmp_path / "study.json"
    path.write_text(json.dumps(_study()))
    assert load_design_spec(path).base._base_dir == tmp_path


@pytest.mark.unit
def test_loader_reports_invalid_study_as_config_error(tmp_path):
    """A study failing validation raises FlexConfigError naming the problem."""
    path = tmp_path / "study.json"
    path.write_text(json.dumps(_study(name=None)))
    with pytest.raises(FlexConfigError, match="name"):
        load_design_spec(path)


@pytest.mark.unit
def test_loader_reports_missing_base_as_config_error(tmp_path):
    """A study without a base raises FlexConfigError."""
    document = _study()
    del document["base"]
    path = tmp_path / "study.json"
    path.write_text(json.dumps(document))
    with pytest.raises(FlexConfigError, match="base"):
        load_design_spec(path)
