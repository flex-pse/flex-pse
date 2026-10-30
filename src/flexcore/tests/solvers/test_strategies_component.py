"""Component-tier facade solves: warm starts, strategies, options, reports."""

import logging
from types import SimpleNamespace

import pyomo.environ as pyo
import pytest

from flexcore.logger import DedupHandler
from flexcore.solvers import ProblemClass, RelaxAndFix, SolverFacade, get_solver

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


@pytest.mark.component
@pytest.mark.needs_highs
def test_feasibility_model_without_objective_loads_solution_on_highs():
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.c1 = pyo.Constraint(expr=m.x + m.y == 5)
    m.c2 = pyo.Constraint(expr=m.x - m.y == 1)
    solver = get_solver(model=m, prefer="highs")
    assert pyo.check_optimal_termination(solver.solve(m))
    assert pyo.value(m.x) == pytest.approx(3.0)
    assert pyo.value(m.y) == pytest.approx(2.0)
    assert solver.last_report.objective is None


@pytest.fixture
def facade_log():
    logger = logging.getLogger("flexcore.solvers.facade")
    for handler in logger.handlers:
        if isinstance(handler, DedupHandler):
            handler.dedup_enabled[logging.WARNING] = False
    capture = _LogCapture()
    logger.addHandler(capture)
    yield capture
    logger.removeHandler(capture)


class _FakeSolver:
    """Records each call's options and returns scripted terminations."""

    def __init__(self, terminations, clock=None, step_seconds=0.0):
        self.terminations = list(terminations)
        self.clock = clock
        self.step_seconds = step_seconds
        self.calls = []

    def solve(self, model, tee, options, load_solutions, **extra):
        self.calls.append({"options": dict(options), **extra})
        if self.clock is not None:
            self.clock.now += self.step_seconds
        return SimpleNamespace(
            solver=SimpleNamespace(termination_condition=self.terminations.pop(0)),
            solution=[],
        )


def _use_fake(monkeypatch, available, fake):
    from flexcore.solvers import facade

    monkeypatch.setattr(facade, "available_solvers", lambda: available)
    monkeypatch.setattr(facade, "_pyomo_solver", lambda name, warmstart: fake)


@pytest.mark.component
def test_native_options_are_dropped_on_a_fallback_solver(monkeypatch, facade_log):
    fake = _FakeSolver([TC.optimal])
    _use_fake(
        monkeypatch, {"scip": {ProblemClass.MILP}, "highs": {ProblemClass.LP}}, fake
    )
    solver = SolverFacade(name="scip", problem_class=ProblemClass.LP)
    solver.solve(_knapsack(), options={"limits/nodes": 10})
    assert fake.calls[0]["options"] == {}
    assert solver.last_report.steps[0].solver == "highs"
    assert any("not passed" in r.getMessage() for r in facade_log.records)


@pytest.mark.component
def test_time_limit_is_a_deadline_across_relax_and_fix_steps(monkeypatch):
    from flexcore.solvers import facade

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(facade, "time", SimpleNamespace(perf_counter=lambda: clock.now))
    fake = _FakeSolver(
        [TC.optimal, TC.infeasible, TC.infeasible, TC.infeasible, TC.optimal],
        clock=clock,
        step_seconds=30.0,
    )
    _use_fake(monkeypatch, {"highs": {ProblemClass.LP, ProblemClass.MILP}}, fake)
    solver = SolverFacade(
        name="highs", problem_class=ProblemClass.MILP, strategy=RelaxAndFix()
    )
    solver.solve(_knapsack(), time_limit=100)
    limits = [call["options"]["time_limit"] for call in fake.calls]
    assert limits == [100, 70, 40, 10, 0]


@pytest.mark.component
def test_failed_termination_logs_warning_and_resets_report(monkeypatch, facade_log):
    fake = _FakeSolver([TC.optimal, TC.infeasible])
    _use_fake(monkeypatch, {"highs": {ProblemClass.MILP}}, fake)
    solver = SolverFacade(name="highs", problem_class=ProblemClass.MILP)
    solver.solve(_knapsack())
    assert solver.last_report is not None
    solver.solve(_knapsack())
    warnings = [r for r in facade_log.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "infeasible" in warnings[0].getMessage()

    _use_fake(monkeypatch, {"highs": {ProblemClass.MILP}}, None)
    with pytest.raises(AttributeError):
        solver.solve(_knapsack())
    assert solver.last_report is None


@pytest.mark.component
def test_highs_warmstart_solves_cold_without_appsi_highs(monkeypatch, facade_log):
    from flexcore.solvers import facade

    fake = _FakeSolver([TC.optimal])
    _use_fake(monkeypatch, {"highs": {ProblemClass.MILP}}, fake)
    monkeypatch.setattr(
        facade.pyo,
        "SolverFactory",
        lambda name: SimpleNamespace(available=lambda exception_flag: False),
    )
    SolverFacade(name="highs", problem_class=ProblemClass.MILP).solve(
        _knapsack(), warmstart=True
    )
    assert "warmstart" not in fake.calls[0]
    assert any("warm" in r.getMessage() for r in facade_log.records)
