"""investment_metrics: NPV and IRR of a design's savings against a baseline."""

import pytest

from flexdesign import CapacityResult, DesignResults, investment_metrics
from flexops.costing import capital_recovery_factor


def _results(
    capital_cost: float,
    lifetime_years: float,
    operating_cost_by_year: dict[int, float],
    study_years: int,
    rate: float = 0.08,
) -> DesignResults:
    """Return hand-built results with one capacity."""
    capacity = CapacityResult(
        size=1.0,
        units="kW",
        at_upper_bound=False,
        capital_cost=capital_cost,
        lifetime_years=lifetime_years,
        annualized_capital_cost=0.0,
    )
    return DesignResults(
        study_years=study_years,
        effective_rate=rate,
        currency="USD",
        capacities={"battery": capacity},
        annual_operating_cost_by_year=operating_cost_by_year,
        operating_cost_by_horizon={},
        annualized_operating_cost=0.0,
        annualized_capital_cost=0.0,
        annualized_cost=0.0,
        total_capital_cost=capital_cost,
    )


@pytest.mark.unit
def test_savings_equal_to_the_capital_annuity_give_zero_npv_and_irr_at_the_rate():
    """Savings of capex times CRF over the lifetime exactly repay the capex."""
    savings = 1000.0 * capital_recovery_factor(0.08, 10)
    results = _results(1000.0, 10, {2026: 0.0}, study_years=10)
    metrics = investment_metrics(results, {2026: savings})
    assert metrics.cash_flows[0] == pytest.approx(-1000.0)
    assert metrics.npv == pytest.approx(0.0, abs=1e-6)
    assert metrics.irr == pytest.approx(0.08, abs=1e-6)


@pytest.mark.unit
def test_asset_wearing_out_inside_the_horizon_is_bought_again():
    """A 5-year asset in a 10-year horizon is replaced in year 5, with no salvage."""
    results = _results(1000.0, 5, {2026: 0.0}, study_years=10)
    metrics = investment_metrics(results, {2026: 300.0})
    assert metrics.cash_flows[5] == pytest.approx(300.0 - 1000.0)
    assert metrics.cash_flows[10] == pytest.approx(300.0)


@pytest.mark.unit
def test_asset_outliving_the_horizon_earns_straight_line_salvage():
    """A 25-year asset in a 10-year horizon gets back 15/25 of its cost in year 10."""
    results = _results(1000.0, 25, {2026: 0.0}, study_years=10)
    metrics = investment_metrics(results, {2026: 300.0})
    assert metrics.cash_flows[10] == pytest.approx(300.0 + 1000.0 * 15 / 25)


@pytest.mark.unit
def test_each_modeled_year_covers_the_calendar_years_it_represents():
    """2026 covers years 1 to 5 and 2031 covers years 6 to 10."""
    results = _results(1000.0, 10, {2026: 50.0, 2031: 50.0}, study_years=10)
    metrics = investment_metrics(results, {2026: 150.0, 2031: 250.0})
    assert metrics.cash_flows[1:] == pytest.approx([100.0] * 5 + [200.0] * 5)


@pytest.mark.unit
def test_horizon_past_the_study_repeats_the_last_modeled_year():
    """Years beyond the study reuse the last modeled year's savings."""
    results = _results(1000.0, 12, {2026: 50.0, 2031: 50.0}, study_years=10)
    metrics = investment_metrics(results, {2026: 150.0, 2031: 250.0}, horizon_years=12)
    assert metrics.horizon_years == 12
    assert metrics.cash_flows[11:] == pytest.approx([200.0, 200.0])


@pytest.mark.unit
def test_design_that_never_saves_has_no_irr():
    """Without a sign change in the cash flows there is no IRR."""
    results = _results(1000.0, 10, {2026: 500.0}, study_years=10)
    metrics = investment_metrics(results, {2026: 400.0})
    assert metrics.irr is None
    assert metrics.npv < 0


@pytest.mark.unit
def test_baseline_years_must_match_the_modeled_years():
    """A baseline missing a modeled year, or an empty horizon, is rejected."""
    results = _results(1000.0, 10, {2026: 0.0, 2031: 0.0}, study_years=10)
    with pytest.raises(ValueError, match="2031"):
        investment_metrics(results, {2026: 100.0})
    with pytest.raises(ValueError, match="horizon_years"):
        investment_metrics(results, {2026: 100.0, 2031: 100.0}, horizon_years=0)


@pytest.mark.unit
def test_summary_reports_npv_irr_and_currency():
    """The summary names the currency and shows NPV and IRR."""
    savings = 1000.0 * capital_recovery_factor(0.08, 10)
    results = _results(1000.0, 10, {2026: 0.0}, study_years=10)
    summary = investment_metrics(results, {2026: savings}).summary()
    assert "USD" in summary
    assert "NPV" in summary
    assert "IRR" in summary and "8.00%" in summary
