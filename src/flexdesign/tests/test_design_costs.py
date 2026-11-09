"""Annualized cost expressions and reports, evaluated with values set by hand."""

import pyomo.environ as pyo
import pytest

from flexdesign import build_design_model, design_results, present_worth_factors
from flexdesign.tests.fixtures import pump_battery_solar_study, solar_only_study
from flexops.costing import capital_recovery_factor

BILLS = {(2026, 1): 10.0, (2026, 7): 20.0, (2031, 1): 30.0, (2031, 7): 40.0}
SIZES = {"battery_energy": 100.0, "battery_power": 25.0, "solar": 50.0}


def _set_values(m, bills, sizes) -> None:
    """Set each horizon's bill and each decision's size."""
    for key, bill in bills.items():
        m.horizon[key].costing.aggregate_operating_cost.set_value(bill)
    for name, size in sizes.items():
        m.capacity[name].size.set_value(size)


@pytest.fixture
def valued():
    """Return the shared study's model with bills and sizes set by hand."""
    m = build_design_model(pump_battery_solar_study())
    _set_values(m, BILLS, SIZES)
    return m


def _solar_model(lifetime_years=20, rate=0.0, years=None):
    """Return a one-day-per-month solar model with a 100 per kW capital cost."""
    return build_design_model(
        solar_only_study(
            years=years or [{"year": 2026}],
            weights=(6.0, 6.0),
            changes=[],
            capital_cost=100.0,
            lifetime_years=lifetime_years,
            rate=rate,
            days=1,
        )
    )


@pytest.mark.unit
def test_annual_cost_scales_each_day_to_its_month_and_applies_weights(valued):
    """January and July both have 31 days, so a bill scales by 31 and its weight."""
    assert pyo.value(valued.annual_operating_cost[2026]) == pytest.approx(
        6.0 * 31 * (10.0 + 20.0)
    )
    assert pyo.value(valued.annual_operating_cost[2031]) == pytest.approx(
        6.0 * 31 * (30.0 + 40.0)
    )


@pytest.mark.unit
def test_february_scales_by_its_leap_year_length():
    """A prorated February scales by the days that year actually has."""
    spec = pump_battery_solar_study(
        years=[{"year": 2028}],
        months=[{"month": 2, "weight": 12.0, "days": 1}],
        changes=[],
    )
    m = build_design_model(spec)
    m.horizon[2028, 2].costing.aggregate_operating_cost.set_value(1.0)
    assert pyo.value(m.annual_operating_cost[2028]) == pytest.approx(12.0 * 29)


@pytest.mark.unit
def test_annualized_operating_cost_is_the_discounted_average(valued):
    """Annual costs average under each modeled year's present worth factor."""
    spec = valued._design_spec
    factors = present_worth_factors(spec.years, 0.07)
    expected = (
        factors[2026] * 6.0 * 31 * 30.0 + factors[2031] * 6.0 * 31 * 70.0
    ) / sum(factors.values())
    assert pyo.value(valued.annualized_operating_cost) == pytest.approx(expected)


@pytest.mark.unit
def test_each_decision_annualizes_over_its_own_lifetime(valued):
    """Battery and solar capital use their own capital recovery factors."""
    energy = capital_recovery_factor(0.07, 15) * 300.0 * 100.0
    power = capital_recovery_factor(0.07, 15) * 200.0 * 25.0
    solar = capital_recovery_factor(0.07, 25) * 1200.0 * 50.0
    block = valued.capacity
    assert pyo.value(block["battery_energy"].annualized_capital_cost) == pytest.approx(
        energy
    )
    assert pyo.value(block["solar"].annualized_capital_cost) == pytest.approx(solar)
    assert pyo.value(valued.annualized_capital_cost) == pytest.approx(
        energy + power + solar
    )


@pytest.mark.unit
def test_objective_is_annualized_operating_plus_capital(valued):
    """The objective adds the two annualized totals."""
    assert pyo.value(valued.annualized_cost) == pytest.approx(
        pyo.value(valued.annualized_operating_cost)
        + pyo.value(valued.annualized_capital_cost)
    )


@pytest.mark.unit
def test_one_year_study_reduces_to_annual_opex_plus_crf_capital():
    """With one modeled year, discounting cancels out of the operating cost."""
    m = _solar_model(rate=0.07)
    _set_values(m, {(2026, 1): 1.0, (2026, 7): 2.0}, {"solar": 10.0})
    annual = 6.0 * 31 * 3.0
    capital = capital_recovery_factor(0.07, 20) * 100.0 * 10.0
    assert pyo.value(m.annualized_cost) == pytest.approx(annual + capital)


@pytest.mark.unit
def test_doubling_lifetime_halves_annualized_capital_at_zero_rate():
    """At a zero rate, annualized capital is overnight cost over lifetime."""
    for lifetime_years, expected in ((10, 100.0), (20, 50.0)):
        m = _solar_model(lifetime_years=lifetime_years)
        m.capacity["solar"].size.set_value(10.0)
        assert pyo.value(m.annualized_capital_cost) == pytest.approx(expected)


@pytest.mark.unit
def test_results_report_sizes_costs_and_binding_bounds(valued):
    """Results read the model's values, and flag a size at its upper bound."""
    valued.capacity["solar"].size.set_value(1500.0)
    results = design_results(valued)
    assert results.study_years == 10
    assert results.effective_rate == pytest.approx(0.07)
    assert results.operating_cost_by_horizon == pytest.approx(BILLS)
    assert results.capacities["solar"].at_upper_bound
    assert not results.capacities["battery_energy"].at_upper_bound
    assert results.capacities["battery_energy"].units == "kWh"
    assert results.annualized_cost == pytest.approx(pyo.value(valued.annualized_cost))


@pytest.mark.unit
def test_summary_lists_each_capacity_and_annualized_total(valued):
    """The text report names every decision, year and annualized total."""
    results = design_results(valued)
    text = results.summary()
    for name in ("battery_energy", "battery_power", "solar", "2026", "2031"):
        assert name in text
    assert f"{results.annualized_cost:,.2f}" in text
