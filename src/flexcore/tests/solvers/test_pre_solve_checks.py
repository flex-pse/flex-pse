"""Checks other packages register to run before every solve."""

import pyomo.environ as pyo
import pytest

from flexcore.solvers import facade, get_solver


@pytest.mark.component
@pytest.mark.needs_highs
def test_a_registered_check_runs_before_the_solve_and_can_stop_it(monkeypatch):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 1))
    m.o = pyo.Objective(expr=m.x)
    seen = []

    def check(model):
        seen.append(model)
        raise ValueError("stop")

    monkeypatch.setattr(facade, "PRE_SOLVE_CHECKS", [check])

    with pytest.raises(ValueError, match="stop"):
        get_solver(model=m, prefer="highs").solve(m)

    assert seen == [m]
    assert m.x.value is None
