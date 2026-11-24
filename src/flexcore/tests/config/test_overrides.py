"""apply_overrides: matching, merging, removing and adding spec elements."""

import copy
from pathlib import Path

import pytest

from flexcore.config.io import load_spec
from flexcore.config.overrides import apply_overrides, deep_merge
from flexcore.config.spec import SCHEMA_VERSION
from flexcore.exceptions import FlexConfigError


def _base() -> dict:
    """Return a spec with named and unnamed elements."""
    return {
        "schema_version": SCHEMA_VERSION,
        "name": "site",
        "elements": [
            {
                "kind": "time",
                "start_date": "2025-01-01",
                "end_date": "2025-01-02",
                "time_step": "1 hr",
            },
            {
                "kind": "costing",
                "name": "costing",
                "energy_prices": {"electrical": {"value": 0.1, "units": "USD/kWh"}},
            },
            {
                "kind": "unit",
                "name": "pv",
                "unit_model_class": "GenericRenewables",
                "property_package": None,
                "construction_options": {
                    "capacity": {"value": 10.0, "units": "kW"},
                    "capacity_factor": {"$source": "pv.json"},
                },
            },
            {"kind": "fix", "variable": "pv.capacity", "value": 10.0},
            {"kind": "objective"},
        ],
    }


def _element(spec, kind: str):
    """Return the spec's only element of ``kind``."""
    (element,) = spec.of_kind(kind)
    return element


@pytest.mark.unit
def test_named_element_is_deep_merged_and_the_base_is_untouched():
    """Nested dicts merge, so units and other options survive."""
    data = _base()
    before = copy.deepcopy(data)
    base = load_spec(data)
    override = {"name": "pv", "construction_options": {"capacity": {"value": 25.0}}}
    spec = apply_overrides(base, [override])
    options = _element(spec, "unit").construction_options
    assert options["capacity"] == {"value": 25.0, "units": "kW"}
    assert options["capacity_factor"] == {"$source": "pv.json"}
    assert data == before
    assert base == load_spec(before)


@pytest.mark.unit
def test_overrides_apply_in_order():
    """A later override of the same field wins."""
    spec = apply_overrides(
        _base(),
        [
            {"kind": "time", "time_step": "2 hr"},
            {"kind": "time", "time_step": "3 hr"},
        ],
    )
    assert _element(spec, "time").time_step == "3 hr"
    assert _element(spec, "time").end_date == "2025-01-02"


@pytest.mark.unit
def test_source_reference_is_replaced_not_merged():
    """Values and a $source can't be merged into one another."""
    for factor in ([0.5] * 24, {"$source": "other.json"}):
        spec = apply_overrides(
            _base(),
            [{"name": "pv", "construction_options": {"capacity_factor": factor}}],
        )
        assert _element(spec, "unit").construction_options["capacity_factor"] == factor
    target = {"value": {"$source": "a.json"}}
    deep_merge(target, {"value": {"value": 1.0}})
    assert target == {"value": {"value": 1.0}}


@pytest.mark.unit
def test_unnamed_element_is_matched_by_what_it_sets():
    """A fix is found by its variable, so emitted fixes need no name."""
    match = {"kind": "fix", "variable": "pv.capacity"}
    assert _element(
        apply_overrides(_base(), [{**match, "value": 3.0}]), "fix"
    ).value == (3.0)
    assert not apply_overrides(_base(), [{**match, "$remove": True}]).of_kind("fix")


@pytest.mark.unit
def test_unmatched_override_adds_an_element():
    """An override with a kind and a new name is appended."""
    spec = apply_overrides(
        _base(), [{"kind": "fix", "variable": "pv.other", "value": 1.0}]
    )
    assert [el.variable for el in spec.of_kind("fix")] == ["pv.capacity", "pv.other"]


@pytest.mark.unit
def test_same_name_on_two_units_is_told_apart_by_kind_and_unit():
    """Degradation names repeat across units, so the name alone is ambiguous."""
    data = _base()
    pv = next(el for el in data["elements"] if el.get("name") == "pv")
    data["elements"].append({**copy.deepcopy(pv), "name": "pv2"})
    term = {"kind": "throughput", "variable": "power_generated", "price": 1.0}
    for unit in ("pv", "pv2"):
        data["elements"].append(
            {"kind": "degradation", "unit": unit, "name": "wear", "terms": [term]}
        )
    change = {"name": "wear", "terms": [{**term, "price": 2.0}]}
    with pytest.raises(FlexConfigError, match="matches 2 elements"):
        apply_overrides(data, [change])
    spec = apply_overrides(data, [{**change, "kind": "degradation", "unit": "pv2"}])
    prices = {el.unit: el.terms[0].price for el in spec.of_kind("degradation")}
    assert prices == {"pv": 1.0, "pv2": 2.0}


@pytest.mark.unit
@pytest.mark.parametrize(
    "override, message",
    [
        ({"value": 1.0}, "needs a 'name'"),
        ({"name": "missing", "$remove": True}, "no such element"),
        ({"name": "missing", "value": 1.0}, "needs a 'kind'"),
        ({"name": "pv", "unknown_field": 1}, "unknown_field"),
    ],
)
def test_bad_overrides_raise(override, message):
    """Each way an override can be wrong names what to change."""
    with pytest.raises(FlexConfigError, match=message):
        apply_overrides(_base(), [override])


@pytest.mark.unit
def test_base_directory_is_kept():
    """Relative $source paths still resolve against the base spec's directory."""
    base = load_spec(_base())
    base._base_dir = Path("somewhere")
    assert apply_overrides(base, [])._base_dir == Path("somewhere")
