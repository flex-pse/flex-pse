"""expand_horizons: one flowsheet spec per year and month."""

import pytest

from flexcore.config.spec import SourceRef
from flexdesign import expand_horizons
from flexdesign.tests.fixtures import solar_only_study


def _study(changes=(), years=None, months=(1, 7)):
    """Return a solar study with the given changes, years and two months."""
    spec = solar_only_study(
        years=years or [{"year": 2025}],
        weights=(6.0, 6.0),
        changes=list(changes),
        capital_cost=100.0,
        lifetime_years=10,
        rate=0.0,
    )
    spec.months[0].month, spec.months[1].month = months
    return spec


def _load(spec) -> float:
    """Return the value of a spec's named load element."""
    return next(el for el in spec.elements if getattr(el, "name", None) == "load").value


def _time(spec):
    """Return a spec's time element."""
    return spec.of_kind("time")[0]


@pytest.mark.unit
def test_horizons_cover_years_times_months_with_correct_dates():
    """Each year and month gets a spec spanning exactly that month."""
    years = [{"year": 2025}, {"year": 2026}]
    expanded = expand_horizons(_study(years=years))
    assert list(expanded) == [(2025, 1), (2025, 7), (2026, 1), (2026, 7)]
    assert _time(expanded[2026, 7]).start_date == "2026-07-01"
    assert _time(expanded[2026, 7]).end_date == "2026-08-01"


@pytest.mark.unit
def test_december_and_leap_february_end_on_the_next_month():
    """Month ends roll over the year and respect leap years."""
    expanded = expand_horizons(_study(years=[{"year": 2028}], months=(2, 12)))
    assert _time(expanded[2028, 2]).end_date == "2028-03-01"
    assert _time(expanded[2028, 12]).end_date == "2029-01-01"


@pytest.mark.unit
def test_month_selector_changes_only_that_month():
    """A month-selected change applies to that month in every year."""
    change = {"element": "load", "months": [7], "values": {"value": 60.0}}
    years = [{"year": 2025}, {"year": 2026}]
    expanded = expand_horizons(_study([change], years=years))
    assert _load(expanded[2025, 7]) == _load(expanded[2026, 7]) == 60.0
    assert _load(expanded[2025, 1]) == _load(expanded[2026, 1]) == 30.0


@pytest.mark.unit
def test_year_selector_changes_only_that_year():
    """A year-selected change applies to every month of that year."""
    change = {"element": "load", "years": [2026], "values": {"value": 60.0}}
    years = [{"year": 2025}, {"year": 2026}]
    expanded = expand_horizons(_study([change], years=years))
    assert _load(expanded[2026, 1]) == _load(expanded[2026, 7]) == 60.0
    assert _load(expanded[2025, 1]) == _load(expanded[2025, 7]) == 30.0


@pytest.mark.unit
def test_year_and_month_selectors_change_only_that_pair():
    """A change selecting both applies to exactly one horizon."""
    change = {"element": "load", "years": [2026], "months": [7], "values": {"value": 9}}
    years = [{"year": 2025}, {"year": 2026}]
    expanded = expand_horizons(_study([change], years=years))
    values = {key: _load(spec) for key, spec in expanded.items()}
    assert values == {(2025, 1): 30, (2025, 7): 30, (2026, 1): 30, (2026, 7): 9}


@pytest.mark.unit
def test_later_change_wins():
    """When two changes match one field, the later entry is used."""
    changes = [
        {"element": "load", "values": {"value": 1.0}},
        {"element": "load", "months": [7], "values": {"value": 2.0}},
    ]
    expanded = expand_horizons(_study(changes))
    assert _load(expanded[2025, 1]) == 1.0
    assert _load(expanded[2025, 7]) == 2.0


@pytest.mark.unit
def test_change_merges_nested_fields_and_keeps_siblings():
    """A nested change leaves the element's other fields in place."""
    change = {
        "element": "pv",
        "values": {"construction_options": {"capacity_factor": [0.1] * 744}},
    }
    pv = next(
        el
        for el in expand_horizons(_study([change]))[2025, 1].elements
        if getattr(el, "name", None) == "pv"
    )
    assert pv.construction_options["capacity_factor"] == [0.1] * 744
    assert pv.construction_options["capacity"] == {"value": 500.0, "units": "kW"}


@pytest.mark.unit
def test_unchanged_horizon_matches_base_apart_from_dates():
    """A horizon no change selects keeps every other base element."""
    spec = _study()
    horizon = expand_horizons(spec)[2025, 7]

    def untimed(document):
        return [el for el in document["elements"] if el["kind"] != "time"]

    assert untimed(horizon.model_dump()) == untimed(spec.base.model_dump())


@pytest.mark.unit
def test_base_spec_is_not_mutated():
    """Expanding leaves the base spec untouched."""
    spec = _study([{"element": "load", "values": {"value": 60.0}}])
    before = spec.base.model_dump()
    expand_horizons(spec)
    assert spec.base.model_dump() == before


@pytest.mark.unit
def test_source_references_survive_expansion():
    """A $source set by a change validates into the horizon's spec."""
    change = {"element": "load", "values": {"value": {"$source": "load.json"}}}
    load = next(
        el
        for el in expand_horizons(_study([change]))[2025, 1].elements
        if getattr(el, "name", None) == "load"
    )
    assert isinstance(load.value, SourceRef)
    assert load.value.source == "load.json"


@pytest.mark.unit
def test_base_directory_is_carried_to_every_horizon(tmp_path):
    """Each horizon resolves relative paths against the base's directory."""
    spec = _study()
    spec.base._base_dir = tmp_path
    assert all(s._base_dir == tmp_path for s in expand_horizons(spec).values())


@pytest.mark.unit
def test_days_shorten_a_horizon_from_the_start_of_the_month():
    """A prorated month ends the given number of days after it starts."""
    spec = _study()
    spec.months[1].days = 2
    expanded = expand_horizons(spec)
    assert _time(expanded[2025, 7]).end_date == "2025-07-03"
    assert _time(expanded[2025, 1]).end_date == "2025-02-01"
