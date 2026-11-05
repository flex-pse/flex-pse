"""Solved design models, checked against hand calculations."""

from pathlib import Path

import pyomo.environ as pyo
import pytest
from pyomo.opt import assert_optimal_termination

import flexops
from flexcore.solvers import get_solver
from flexdesign import (
    build_design_model,
    build_horizons,
    design_results,
    present_worth_factors,
)
from flexdesign.tests.fixtures import pump_battery_solar_study, solar_only_study
from flexops.costing import capital_recovery_factor

pytestmark = [pytest.mark.component, pytest.mark.needs_highs]

JAN_KW, JUL_KW = 30 / 3600 * 3e5 / 1000 / 0.75, 60 / 3600 * 3e5 / 1000 / 0.75
HOURS = 744


def _solve(spec):
    """Build and solve a study, returning the model and its results."""
    m = build_design_model(spec)
    assert_optimal_termination(get_solver(m, prefer="highs").solve(m))
    return m, design_results(m)


@pytest.fixture(scope="module")
def solved():
    """Return the shared study, solved once."""
    return _solve(pump_battery_solar_study())


@pytest.mark.parametrize(
    "weights, capital_cost, expected",
    [
        ((11.0, 1.0), lambda all_, jul: 10 * (all_ + jul) / 2, JAN_KW / 0.5),
        ((1.0, 11.0), lambda all_, jul: 10 * jul * 0.5, JUL_KW / 0.5),
    ],
)
def test_month_weights_flip_solar_size_to_hand_computed_value(
    weights, capital_cost, expected
):
    """Solar sizes to the load whose weighted energy value covers its capital cost."""
    value_all = 0.1 * 0.5 * sum(weights) * HOURS
    value_jul = 0.1 * 0.5 * weights[1] * HOURS
    spec = solar_only_study(
        years=[{"year": 2025}],
        weights=weights,
        changes=[{"element": "load", "months": [7], "values": {"value": 60.0}}],
        capital_cost=capital_cost(value_all, value_jul),
        lifetime_years=10,
        rate=0.0,
    )
    _, results = _solve(spec)
    assert results.capacities["solar"].size == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize(
    "target, expected",
    [
        (lambda all_, late: 0.5 * late, JUL_KW / 0.5),
        (lambda all_, late: (late + all_) / 2, JAN_KW / 0.5),
    ],
)
def test_later_year_load_growth_sizes_to_hand_computed_value(target, expected):
    """Later-year load is worth its discounted share of the annualized cost."""
    years = [{"year": 2025, "represents": 5}, {"year": 2030, "represents": 5}]
    late = sum(1.07**-k for k in range(6, 11)) / sum(1.07**-k for k in range(1, 11))
    value_all = 0.1 * 0.5 * 12 * HOURS
    spec = solar_only_study(
        years=years,
        weights=(6.0, 6.0),
        changes=[{"element": "load", "years": [2030], "values": {"value": 60.0}}],
        capital_cost=target(value_all, value_all * late)
        / capital_recovery_factor(0.07, 20),
        lifetime_years=20,
        rate=0.07,
    )
    _, results = _solve(spec)
    assert results.capacities["solar"].size == pytest.approx(expected, rel=1e-5)


def test_annualized_costs_match_hand_calculation(solved):
    """Reported totals follow the discounted-average and CRF formulas."""
    m, results = solved
    spec = m._design_spec
    factors = present_worth_factors(spec.years, 0.07)
    by_year = {
        year: 6.0 * sum(results.operating_cost_by_horizon[year, mo] for mo in (1, 7))
        for year in (2026, 2031)
    }
    assert results.annual_operating_cost_by_year == pytest.approx(by_year)
    expected_opex = sum(factors[y] * by_year[y] for y in by_year) / sum(
        factors.values()
    )
    assert results.annualized_operating_cost == pytest.approx(expected_opex)
    expected_capital = sum(
        capital_recovery_factor(0.07, decision.lifetime_years)
        * decision.capital_cost
        * results.capacities[decision.name].size
        for decision in spec.capacities
    )
    assert results.annualized_capital_cost == pytest.approx(expected_capital)
    assert results.annualized_cost == pytest.approx(expected_opex + expected_capital)
    assert results.annualized_cost == pytest.approx(pyo.value(m.annualized_cost))


def test_one_year_study_reduces_to_annual_opex_plus_crf_capital():
    """With one year, annualized operating cost is that year's operating cost."""
    spec = solar_only_study(
        years=[{"year": 2025}],
        weights=(6.0, 6.0),
        changes=[],
        capital_cost=500.0,
        lifetime_years=20,
        rate=0.07,
    )
    _, results = _solve(spec)
    crf = capital_recovery_factor(0.07, 20)
    capital = results.total_capital_cost
    annual = results.annual_operating_cost_by_year[2025]
    assert results.study_years == 1
    assert results.annualized_cost == pytest.approx(annual + crf * capital)


def test_doubling_lifetime_halves_annualized_capital_at_zero_rate():
    """At a zero rate, annualized capital is overnight cost over lifetime."""

    def annualized(lifetime_years):
        spec = solar_only_study(
            years=[{"year": 2025}],
            weights=(6.0, 6.0),
            changes=[],
            capital_cost=100.0,
            lifetime_years=lifetime_years,
            rate=0.0,
            lower=500.0,
        )
        return _solve(spec)[1].annualized_capital_cost

    assert annualized(10) == pytest.approx(5000.0)
    assert annualized(20) == pytest.approx(2500.0)


def test_decisions_use_their_own_lifetimes(solved):
    """Battery and solar annualize over their own lifetimes."""
    _, results = solved
    for name, lifetime in (("battery_energy", 15), ("solar", 25)):
        capacity = results.capacities[name]
        assert capacity.lifetime_years == lifetime
        assert capacity.annualized_capital_cost == pytest.approx(
            capital_recovery_factor(0.07, lifetime) * capacity.capital_cost
        )


def test_every_horizon_shares_capacity_but_dispatches_differently(solved):
    """All horizons hold one size, and months operate differently."""
    m, results = solved
    for key in m.horizon:
        battery = m.horizon[key].battery
        energy = results.capacities["battery_energy"].size
        power = results.capacities["battery_power"].size
        assert pyo.value(battery.capacity) == pytest.approx(energy, abs=1e-6)
        assert pyo.value(battery.power_capacity) == pytest.approx(power, abs=1e-6)
    january = [pyo.value(m.horizon[2026, 1].pv.power_generated[t]) for t in range(24)]
    july = [pyo.value(m.horizon[2026, 7].pv.power_generated[t]) for t in range(24)]
    assert january != pytest.approx(july)
    assert results.annual_operating_cost_by_year[2026] != pytest.approx(
        results.annual_operating_cost_by_year[2031]
    )


def test_fixed_capacity_horizon_solves_match_merged_operating_cost(solved):
    """Solving each horizon alone at the merged sizes reproduces its cost."""
    m, results = solved
    spec = m._design_spec
    for key, horizon in build_horizons(spec).items():
        for decision in spec.capacities:
            size = results.capacities[decision.name].size
            horizon.find_component(decision.variable).fix(size)
        assert_optimal_termination(get_solver(horizon, prefer="highs").solve(horizon))
        assert pyo.value(horizon.costing.aggregate_operating_cost) == pytest.approx(
            results.operating_cost_by_horizon[key], rel=1e-5
        )


def test_energy_and_power_size_independently(solved):
    """Raising the power price lowers power and shifts the energy-to-power ratio."""
    _, base = solved
    _, costly = _solve(pump_battery_solar_study(power_cost=2000.0))
    base_energy = base.capacities["battery_energy"].size
    base_power = base.capacities["battery_power"].size
    energy = costly.capacities["battery_energy"].size
    power = costly.capacities["battery_power"].size
    assert power < base_power
    assert energy / power != pytest.approx(base_energy / base_power, rel=1e-3)


def test_expensive_capacity_sizes_to_zero():
    """Capacity that can't pay for itself isn't built."""
    _, results = _solve(
        pump_battery_solar_study(energy_cost=1e6, power_cost=1e6, solar_cost=1e6)
    )
    for capacity in results.capacities.values():
        assert capacity.size == pytest.approx(0.0, abs=1e-6)


def test_upper_bound_is_reported_when_binding():
    """A capacity pinned at its limit is flagged, and others aren't."""
    _, results = _solve(pump_battery_solar_study(solar_upper=1.0))
    assert results.capacities["solar"].at_upper_bound
    assert not results.capacities["battery_energy"].at_upper_bound


def test_tariff_months_solve_with_linked_capacity():
    """Whole months on a tariff bill and link like native prices."""
    tariff = Path(flexops.__file__).parent / "tests/fixtures/tariff_tou_demo.json"
    costing = {
        "kind": "costing",
        "name": "costing",
        "tariff_source": str(tariff),
        "discount_rate": 0.07,
    }
    m, results = _solve(pump_battery_solar_study(costing=costing))
    assert all(cost > 0 for cost in results.operating_cost_by_horizon.values())
    solar = results.capacities["solar"].size
    for key in m.horizon:
        assert pyo.value(m.horizon[key].pv.capacity) == pytest.approx(solar, abs=1e-6)


def test_summary_lists_each_capacity_and_annualized_total(solved):
    """The text report names every decision, year and annualized total."""
    _, results = solved
    text = results.summary()
    for name in ("battery_energy", "battery_power", "solar", "2026", "2031"):
        assert name in text
    assert f"{results.annualized_cost:,.2f}" in text
