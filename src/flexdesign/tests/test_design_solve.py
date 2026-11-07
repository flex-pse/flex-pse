"""One short solve, checking only that an assembled study is feasible."""

import pytest
from pyomo.opt import assert_optimal_termination

from flexcore.solvers import get_solver
from flexdesign import build_design_model
from flexdesign.tests.fixtures import pump_battery_solar_study


@pytest.mark.component
@pytest.mark.needs_highs
def test_prorated_study_solves_to_optimality():
    """A one-day-per-month study terminates optimally; its values aren't checked."""
    m = build_design_model(pump_battery_solar_study())
    results = get_solver(m, prefer="highs").solve(m, time_limit=60)
    assert_optimal_termination(results)
