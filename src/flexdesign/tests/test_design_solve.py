"""One short solve, checking only that an assembled study is feasible."""

import pytest
from pyomo.opt import assert_optimal_termination

from flexcore.solvers import get_solver
from flexdesign import build_design_model, design_results
from flexdesign.tests.fixtures import pump_battery_solar_study


@pytest.mark.component
@pytest.mark.needs_highs
def test_prorated_study_solves_to_optimality():
    """A one-day-per-month study terminates optimally; its values aren't checked."""
    m = build_design_model(pump_battery_solar_study())
    results = get_solver(m, prefer="highs").solve(m, time_limit=60)
    assert_optimal_termination(results)


@pytest.mark.component
@pytest.mark.needs_highs
def test_capex_budget_binds_total_overnight_capital():
    """A budget below the free optimum's capital is met with equality."""
    free = build_design_model(pump_battery_solar_study())
    get_solver(free, prefer="highs").solve(free, time_limit=60)
    free_capital = design_results(free).total_capital_cost
    assert free_capital > 0

    budget = 0.5 * free_capital
    m = build_design_model(pump_battery_solar_study(capex_budget=budget))
    assert_optimal_termination(get_solver(m, prefer="highs").solve(m, time_limit=60))
    assert design_results(m).total_capital_cost == pytest.approx(budget, rel=1e-4)
