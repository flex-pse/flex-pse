"""Component-tier test: separate export prices still solve as an LP."""

import pyomo.environ as pyo
import pytest
from pyomo.opt import assert_optimal_termination

from flexcore.solvers import ProblemClass, classify, get_solver
from flexops.tests.costing.test_flex_costing import (
    _NET_BIOGAS,
    _NET_KW,
    _export_priced_costing,
    _fuel_export_costing,
)


def _solve_lp(m) -> None:
    """Minimize operating cost, asserting the model is an LP that solves."""
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
    assert classify(m) == ProblemClass.LP
    assert_optimal_termination(get_solver(model=m, prefer="highs").solve(m))


@pytest.mark.component
@pytest.mark.needs_highs
def test_export_price_solves_as_lp_with_split_cost():
    """Net export earns the lower export price, with no simultaneous import/export."""
    m = _export_priced_costing()
    _solve_lp(m)

    # 12 h x 100 kW x $0.10 - 12 h x 40 kW x $0.05 = 120 - 24
    opex = m.costing.opex
    assert pyo.value(opex.electricity_cost) == pytest.approx(96.0)
    for t, kw in _NET_KW.items():
        assert pyo.value(opex.import_electrical[t]) == pytest.approx(max(kw, 0.0))
        assert pyo.value(opex.export_electrical[t]) == pytest.approx(max(-kw, 0.0))


@pytest.mark.component
@pytest.mark.needs_highs
def test_fuel_export_price_solves_as_lp_with_split_cost():
    """A sold fuel earns its export price, with no simultaneous buy and sell."""
    m = _fuel_export_costing()
    _solve_lp(m)

    # 12 h x 10 m3/hr x $0.50 - 12 h x 4 m3/hr x $0.20 = 60 - 9.6
    opex = m.costing.opex
    assert pyo.value(opex.fuel_cost_biogas) == pytest.approx(50.4)
    for t, usage in _NET_BIOGAS.items():
        assert pyo.value(opex.import_biogas[t]) == pytest.approx(max(usage, 0.0))
        assert pyo.value(opex.export_biogas[t]) == pytest.approx(max(-usage, 0.0))
