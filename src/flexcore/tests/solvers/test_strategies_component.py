"""Component-tier facade solves: warm starts, strategies, options, reports."""

import logging

import pyomo.environ as pyo
import pytest

from flexcore.logger import DedupHandler
from flexcore.solvers import ProblemClass, RelaxAndFix, get_solver

TC = pyo.TerminationCondition


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _knapsack() -> pyo.ConcreteModel:
    """A small 0-1 knapsack MILP whose capacity is a mutable Param."""
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, 11)
    m.capacity = pyo.Param(initialize=30, mutable=True)
    m.x = pyo.Var(m.I, domain=pyo.Binary)
    m.c = pyo.Constraint(expr=sum((i + 3) * m.x[i] for i in m.I) <= m.capacity)
    m.obj = pyo.Objective(
        expr=sum(((i * i) % 7 + 1) * m.x[i] for i in m.I), sense=pyo.maximize
    )
    return m


def _unit_commitment() -> pyo.ConcreteModel:
    """Three semicontinuous units covering a demand profile at least cost."""
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, 7)
    m.U = pyo.RangeSet(0, 2)
    demand = [40, 60, 90, 120, 110, 70, 50, 30]
    cap = {0: 60, 1: 50, 2: 40}
    floor = {0: 20, 1: 15, 2: 10}
    cost = {0: 1.0, 1: 1.5, 2: 2.5}
    fixed_cost = {0: 30, 1: 20, 2: 10}
    m.on = pyo.Var(m.U, m.T, domain=pyo.Binary)
    m.p = pyo.Var(m.U, m.T, bounds=(0, None))
    m.lo = pyo.Constraint(
        m.U, m.T, rule=lambda m, u, t: m.p[u, t] >= floor[u] * m.on[u, t]
    )
    m.hi = pyo.Constraint(
        m.U, m.T, rule=lambda m, u, t: m.p[u, t] <= cap[u] * m.on[u, t]
    )
    m.dem = pyo.Constraint(
        m.T, rule=lambda m, t: sum(m.p[u, t] for u in m.U) >= demand[t]
    )
    m.obj = pyo.Objective(
        expr=sum(
            cost[u] * m.p[u, t] + fixed_cost[u] * m.on[u, t] for u in m.U for t in m.T
        )
    )
    return m


@pytest.mark.component
@pytest.mark.parametrize(
    "solver_name",
    [
        pytest.param("highs", marks=pytest.mark.needs_highs),
        pytest.param("gurobi", marks=pytest.mark.needs_gurobi),
    ],
)
def test_warm_resolve_after_param_retarget_matches_cold_optimum(solver_name):
    cold = _knapsack()
    warm = _knapsack()
    solver = get_solver(model=warm, prefer=solver_name)
    assert solver.name == solver_name
    assert pyo.check_optimal_termination(solver.solve(warm))

    warm.capacity.set_value(25)
    cold.capacity.set_value(25)
    assert pyo.check_optimal_termination(solver.solve(warm, warmstart=True))
    assert pyo.check_optimal_termination(
        get_solver(model=cold, prefer=solver_name).solve(cold)
    )
    assert pyo.value(warm.obj) == pytest.approx(pyo.value(cold.obj))


@pytest.mark.component
@pytest.mark.needs_scip
def test_warmstart_on_unsupported_solver_warns_and_solves_cold():
    logger = logging.getLogger("flexcore.solvers.facade")
    for handler in logger.handlers:
        if isinstance(handler, DedupHandler):
            handler.dedup_enabled[logging.WARNING] = False
    capture = _LogCapture()
    logger.addHandler(capture)
    try:
        m = _knapsack()
        results = get_solver(model=m, prefer="scip").solve(m, warmstart=True)
    finally:
        logger.removeHandler(capture)
    assert pyo.check_optimal_termination(results)
    assert any("warm" in record.getMessage() for record in capture.records)


@pytest.mark.component
@pytest.mark.needs_highs
def test_infeasible_solve_returns_termination_instead_of_raising():
    m = pyo.ConcreteModel()
    m.b = pyo.Var(domain=pyo.Binary)
    m.c = pyo.Constraint(expr=m.b >= 2)
    m.obj = pyo.Objective(expr=m.b)
    solver = get_solver(model=m, prefer="highs")
    results = solver.solve(m)
    assert results.solver.termination_condition is TC.infeasible
    assert m.b.value is None
    assert solver.last_report.steps[0].termination is TC.infeasible


@pytest.mark.component
@pytest.mark.needs_highs
def test_direct_solve_reports_wall_time():
    m = _knapsack()
    solver = get_solver(model=m, prefer="highs")
    solver.solve(m)
    report = solver.last_report
    assert report.strategy == "direct"
    assert report.wall_time > 0
    assert report.objective == pytest.approx(pyo.value(m.obj))


@pytest.mark.component
@pytest.mark.parametrize(
    "solver_name",
    [
        pytest.param("highs", marks=pytest.mark.needs_highs),
        pytest.param("gurobi", marks=pytest.mark.needs_gurobi),
    ],
)
def test_relax_and_fix_matches_exact_unit_commitment_optimum(solver_name):
    exact = _unit_commitment()
    assert pyo.check_optimal_termination(
        get_solver(model=exact, prefer=solver_name).solve(exact)
    )

    m = _unit_commitment()
    solver = get_solver(model=m, prefer=solver_name, strategy=RelaxAndFix())
    assert pyo.check_optimal_termination(solver.solve(m))
    report = solver.last_report
    assert report.steps[0].label == "relaxation"
    assert report.bound <= pyo.value(m.obj) + 1e-6
    assert report.gap >= 0
    assert pyo.value(m.obj) == pytest.approx(pyo.value(exact.obj), rel=0.05)
    assert not any(v.fixed for v in m.on.values())
    assert all(v.domain is pyo.Binary for v in m.on.values())


@pytest.mark.component
@pytest.mark.needs_scip
def test_relax_and_fix_on_scip_solves_relaxation_with_an_lp_solver():
    m = _unit_commitment()
    solver = get_solver(model=m, prefer="scip", strategy=RelaxAndFix())
    assert solver.problem_class is ProblemClass.MILP
    assert pyo.check_optimal_termination(solver.solve(m))
    assert solver.last_report.steps[0].solver != "scip"
    assert solver.last_report.steps[-1].solver == "scip"


@pytest.mark.component
@pytest.mark.parametrize(
    "solver_name",
    [
        pytest.param("highs", marks=pytest.mark.needs_highs),
        pytest.param("gurobi", marks=pytest.mark.needs_gurobi),
        pytest.param("scip", marks=pytest.mark.needs_scip),
        pytest.param("cbc", marks=pytest.mark.needs_cbc),
    ],
)
def test_generic_options_are_accepted_by_each_solver(solver_name):
    m = _knapsack()
    solver = get_solver(model=m, prefer=solver_name)
    assert solver.name == solver_name
    results = solver.solve(m, mip_gap=0.01, time_limit=30)
    assert pyo.check_optimal_termination(results)
