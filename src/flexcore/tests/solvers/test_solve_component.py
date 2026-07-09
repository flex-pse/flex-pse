"""Component-tier smoke solves through :func:`flexcore.solvers.get_solver`.

Each solves a tiny 2-variable problem and asserts optimal termination plus the
known optimum. Guarded by ``needs_*`` markers so they skip cleanly when the
solver is absent.
"""

import pyomo.environ as pyo
import pytest

from flexcore.solvers import ProblemClass, get_solver


@pytest.mark.component
@pytest.mark.needs_highs
def test_highs_lp_smoke():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, None))
    m.y = pyo.Var(bounds=(0, None))
    m.c = pyo.Constraint(expr=m.x + m.y >= 2)
    m.obj = pyo.Objective(expr=m.x + 2 * m.y)  # min at x=2, y=0 -> 2
    solver = get_solver(problem_class=ProblemClass.LP, prefer="highs")
    results = solver.solve(m)
    assert pyo.check_optimal_termination(results)
    assert pyo.value(m.obj) == pytest.approx(2.0, rel=1e-6)


@pytest.mark.component
@pytest.mark.needs_highs
def test_highs_milp_smoke():
    m = pyo.ConcreteModel()
    m.b = pyo.Var(domain=pyo.Binary)
    m.x = pyo.Var(bounds=(0, 5))
    m.c = pyo.Constraint(expr=m.x <= 3 * m.b)
    m.obj = pyo.Objective(expr=m.x - 0.5 * m.b, sense=pyo.maximize)  # b=1, x=3 -> 2.5
    solver = get_solver(problem_class=ProblemClass.MILP, prefer="highs")
    results = solver.solve(m)
    assert pyo.check_optimal_termination(results)
    assert pyo.value(m.obj) == pytest.approx(2.5, rel=1e-6)


@pytest.mark.component
@pytest.mark.needs_scip
def test_scip_minlp_smoke():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 10))
    m.b = pyo.Var(domain=pyo.Binary)
    m.c = pyo.Constraint(expr=m.x**2 <= 4 * m.b)  # nonlinear + binary -> MINLP
    m.obj = pyo.Objective(expr=m.x - 0.5 * m.b, sense=pyo.maximize)  # b=1, x=2 -> 1.5
    solver = get_solver(model=m)  # classify -> MINLP -> SCIP
    assert solver.name == "scip"
    results = solver.solve(m)
    assert pyo.check_optimal_termination(results)
    assert pyo.value(m.obj) == pytest.approx(1.5, rel=1e-6)


@pytest.mark.component
@pytest.mark.needs_ipopt
def test_ipopt_nlp_smoke():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(-10, 10), initialize=0.0)
    m.c = pyo.Constraint(expr=m.x >= 0)
    m.obj = pyo.Objective(expr=(m.x - 3) ** 2 + 1)  # min at x=3 -> 1
    solver = get_solver(problem_class=ProblemClass.NLP, prefer="ipopt")
    results = solver.solve(m)
    assert pyo.check_optimal_termination(results)
    assert pyo.value(m.obj) == pytest.approx(1.0, rel=1e-6)
