"""Unit tests for :func:`flexcore.solvers.classify.classify`.

Six synthetic 2-3 variable models, one test each, asserting the returned
:class:`ProblemClass`.
"""

import pyomo.environ as pyo
import pytest

from flexcore.solvers import ProblemClass, classify


def _lp_model() -> pyo.ConcreteModel:
    """Linear objective + linear constraint."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.c = pyo.Constraint(expr=m.x + m.y <= 1)
    m.obj = pyo.Objective(expr=m.x + 2 * m.y)
    return m


def _qp_model() -> pyo.ConcreteModel:
    """Quadratic objective + linear constraint."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.c = pyo.Constraint(expr=m.x + m.y <= 1)
    m.obj = pyo.Objective(expr=m.x**2 + 2 * m.y**2)
    return m


def _nlp_model() -> pyo.ConcreteModel:
    """Nonlinear constraint + linear objective."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.c = pyo.Constraint(expr=pyo.exp(m.x) <= 2)
    m.obj = pyo.Objective(expr=m.x + m.y)
    return m


def _milp_model() -> pyo.ConcreteModel:
    """LP with one unfixed Binary appearing in a constraint."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.b = pyo.Var(domain=pyo.Binary)
    m.c = pyo.Constraint(expr=m.x + m.y <= 1)
    m.c_b = pyo.Constraint(expr=m.x <= m.b)
    m.obj = pyo.Objective(expr=m.x + 2 * m.y)
    return m


def _minlp_model() -> pyo.ConcreteModel:
    """NLP with one unfixed Binary appearing in a constraint."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.b = pyo.Var(domain=pyo.Binary)
    m.c = pyo.Constraint(expr=m.x + m.y <= 1)
    m.c_b = pyo.Constraint(expr=m.x <= m.b)
    m.obj = pyo.Objective(expr=m.x**2 + 2 * m.y**2)
    return m


@pytest.mark.unit
def test_classify_lp():
    assert classify(_lp_model()) is ProblemClass.LP


@pytest.mark.unit
def test_classify_qp():
    assert classify(_qp_model()) is ProblemClass.QP


@pytest.mark.unit
def test_classify_nlp():
    assert classify(_nlp_model()) is ProblemClass.NLP


@pytest.mark.unit
def test_classify_milp():
    assert classify(_milp_model()) is ProblemClass.MILP


@pytest.mark.unit
def test_classify_minlp():
    assert classify(_minlp_model()) is ProblemClass.MINLP


@pytest.mark.unit
def test_classify_minlp_fix_binary():
    """A minlp model with all binaries fixed is classified as qp."""
    m = _minlp_model()
    m.b.fix(1)
    assert classify(m) is ProblemClass.QP
