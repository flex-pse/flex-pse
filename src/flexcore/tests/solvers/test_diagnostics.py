"""find_redundant_equalities and the facade's warning about what it finds."""

import logging

import pyomo.environ as pyo
import pytest

from flexcore.solvers import find_redundant_equalities, get_solver


def _recycle(n: int = 3, *, closed: bool = True) -> pyo.ConcreteModel:
    """Build a chain of equal flows, closed into a loop if ``closed``."""
    m = pyo.ConcreteModel()
    m.t = pyo.RangeSet(0, n - 1)
    m.a = pyo.Var(m.t, initialize=1.0, bounds=(0.0, 10.0))
    m.b = pyo.Var(m.t, initialize=1.0, bounds=(0.0, 10.0))
    m.c = pyo.Var(m.t, initialize=1.0, bounds=(0.0, 10.0))
    m.ab = pyo.Constraint(m.t, rule=lambda m, t: m.a[t] == m.b[t])
    m.bc = pyo.Constraint(m.t, rule=lambda m, t: m.b[t] == m.c[t])
    if closed:
        m.ca = pyo.Constraint(m.t, rule=lambda m, t: m.c[t] == m.a[t])
    return m


@pytest.mark.unit
def test_finds_the_row_a_closed_recycle_implies():
    m = _recycle()

    assert find_redundant_equalities(m) == [m.ca[0], m.ca[1], m.ca[2]]


@pytest.mark.unit
def test_independent_rows_are_not_flagged():
    assert find_redundant_equalities(_recycle(closed=False)) == []


@pytest.mark.unit
def test_a_dependency_across_scaled_coefficients_is_found():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=1.0)
    m.y = pyo.Var(initialize=1.0)
    m.first = pyo.Constraint(expr=0.001 * m.x + 3.6 * m.y == 1.0)
    m.second = pyo.Constraint(expr=2000.0 * m.x + 7.2e6 * m.y == 2.0e6)

    assert find_redundant_equalities(m) == [m.second]


@pytest.mark.unit
def test_inequalities_and_inactive_rows_are_ignored():
    m = _recycle(1)
    m.ca[0].deactivate()
    m.limit = pyo.Constraint(expr=m.a[0] + m.b[0] <= 4.0)
    m.twin = pyo.Constraint(expr=m.a[0] + m.b[0] <= 5.0)

    assert find_redundant_equalities(m) == []


@pytest.mark.unit
def test_nonlinear_rows_are_skipped():
    m = _recycle(1)
    m.ca[0].deactivate()
    m.curve = pyo.Constraint(expr=m.c[0] * m.c[0] == m.a[0] * m.a[0])

    assert find_redundant_equalities(m) == []


@pytest.mark.unit
def test_fixing_a_variable_can_make_a_row_redundant():
    """With ``a`` and ``c`` fixed, ``a + b == 2`` and ``b + c == 2`` both set ``b``."""
    m = pyo.ConcreteModel()
    m.a = pyo.Var(initialize=1.0)
    m.b = pyo.Var(initialize=1.0)
    m.c = pyo.Var(initialize=1.0)
    m.first = pyo.Constraint(expr=m.a + m.b == 2.0)
    m.second = pyo.Constraint(expr=m.b + m.c == 2.0)

    assert find_redundant_equalities(m) == []

    m.a.fix(1.0)
    m.c.fix(1.0)

    assert find_redundant_equalities(m) == [m.second]


@pytest.mark.unit
def test_a_row_with_no_free_variable_is_not_flagged():
    """The row leaves the Jacobian entirely, so nothing is rank deficient."""
    m = pyo.ConcreteModel()
    m.a = pyo.Var(initialize=1.0)
    m.b = pyo.Var(initialize=1.0)
    m.row = pyo.Constraint(expr=m.a + m.b == 2.0)
    m.a.fix(1.0)
    m.b.fix(1.0)

    assert find_redundant_equalities(m) == []


def _objective(m):
    m.fit = pyo.Objective(expr=sum((m.a[t] - 5.0) ** 2 for t in m.t))
    return m


@pytest.mark.component
@pytest.mark.needs_ipopt
def test_ipopt_warns_names_the_rows_and_still_solves(caplog):
    m = _objective(_recycle())
    facade = get_solver(model=m, prefer="ipopt")

    with caplog.at_level(logging.WARNING):
        results = facade.solve(m)

    pyo.assert_optimal_termination(results)
    assert "ca[0]" in caplog.text
    assert facade.last_report.redundant_equalities == ["ca[0]", "ca[1]", "ca[2]"]


@pytest.mark.component
@pytest.mark.needs_ipopt
def test_ipopt_reports_nothing_when_the_rows_are_independent(caplog):
    m = _objective(_recycle(closed=False))
    facade = get_solver(model=m, prefer="ipopt")

    with caplog.at_level(logging.WARNING):
        facade.solve(m)

    assert "redundant" not in caplog.text
    assert facade.last_report.redundant_equalities == []


@pytest.mark.component
@pytest.mark.needs_highs
def test_highs_solves_a_redundant_lp_without_the_check(caplog):
    """HiGHS presolve handles dependent rows, so the facade doesn't spend the time."""
    m = _recycle()
    m.reward = pyo.Objective(expr=sum(m.a[t] for t in m.t), sense=pyo.maximize)
    facade = get_solver(model=m, prefer="highs")

    with caplog.at_level(logging.WARNING):
        results = facade.solve(m)

    pyo.assert_optimal_termination(results)
    assert pyo.value(m.reward) == pytest.approx(30.0)
    assert "redundant" not in caplog.text
    assert facade.last_report.redundant_equalities == []
