"""Component-tier test: a separate export price still solves as an LP."""

import pyomo.environ as pyo
import pytest
from pyomo.opt import assert_optimal_termination

from flexcore.solvers import ProblemClass, classify, get_solver
from flexops.tests.costing.test_flex_costing import _NET_KW, _export_priced_costing


@pytest.mark.component
@pytest.mark.needs_highs
def test_export_price_solves_as_lp_with_split_cost():
    """Net export earns the lower export price, with no simultaneous import/export."""
    m = _export_priced_costing()
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
    assert classify(m) == ProblemClass.LP
    results = get_solver(model=m, prefer="highs").solve(m)
    assert_optimal_termination(results)

    # 12 h x 100 kW x $0.10 - 12 h x 40 kW x $0.05 = 120 - 24
    assert pyo.value(m.costing.opex.electricity_cost) == pytest.approx(96.0)
    for t, kw in _NET_KW.items():
        assert pyo.value(m.costing.grid_import[t]) == pytest.approx(max(kw, 0.0))
        assert pyo.value(m.costing.grid_export[t]) == pytest.approx(max(-kw, 0.0))
