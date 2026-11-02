"""Unit tests for :func:`flexcore.solvers.classify.classify`.

Six synthetic 2-3 variable models, one test each, asserting the returned
:class:`ProblemClass`.
"""

import importlib

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


def _count_shape_calls(monkeypatch) -> list[int]:
    """Patch the classifier's degree ranking to count how often the walk runs."""
    classify_module = importlib.import_module("flexcore.solvers.classify")
    calls: list[int] = []
    original = classify_module._shape

    def _counting(degree):
        calls.append(degree)
        return original(degree)

    monkeypatch.setattr(classify_module, "_shape", _counting)
    return calls


@pytest.mark.unit
def test_classify_reuses_cached_result_for_unchanged_model(monkeypatch):
    m = _milp_model()
    calls = _count_shape_calls(monkeypatch)
    assert classify(m) is ProblemClass.MILP
    walked = len(calls)
    assert walked > 0
    assert classify(m) is ProblemClass.MILP
    assert len(calls) == walked


@pytest.mark.unit
def test_classify_cache_invalidated_by_fixing_a_binary():
    m = _milp_model()
    assert classify(m) is ProblemClass.MILP
    m.b.fix(1)
    assert classify(m) is ProblemClass.LP
    m.b.unfix()
    assert classify(m) is ProblemClass.MILP


@pytest.mark.unit
def test_classify_cache_invalidated_by_new_constraint():
    m = _lp_model()
    assert classify(m) is ProblemClass.LP
    m.c_nl = pyo.Constraint(expr=pyo.exp(m.x) <= 2)
    assert classify(m) is ProblemClass.NLP


@pytest.mark.unit
def test_classify_cache_invalidated_by_deactivating_a_constraint():
    m = _nlp_model()
    assert classify(m) is ProblemClass.NLP
    m.c.deactivate()
    assert classify(m) is ProblemClass.LP


@pytest.mark.unit
def test_classify_cache_invalidated_by_domain_relaxation():
    m = _milp_model()
    assert classify(m) is ProblemClass.MILP
    m.b.domain = pyo.UnitInterval
    assert classify(m) is ProblemClass.LP


@pytest.mark.unit
def test_classify_cache_invalidated_by_unfixing_a_continuous_var():
    """Unfixing a var that multiplies another makes the model bilinear."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.r = pyo.Var(initialize=0.5)
    m.r.fix()
    m.c = pyo.Constraint(expr=m.r * m.x <= 1)
    m.obj = pyo.Objective(expr=m.x)
    assert classify(m) is ProblemClass.LP
    m.r.unfix()
    assert classify(m) is ProblemClass.NLP


@pytest.mark.unit
def test_classify_cache_invalidated_by_swapping_a_constraint():
    """Deactivating one constraint and adding another keeps the count equal."""
    m = _lp_model()
    assert classify(m) is ProblemClass.LP
    m.c.deactivate()
    m.c_nl = pyo.Constraint(expr=pyo.exp(m.x) <= 2)
    assert classify(m) is ProblemClass.NLP


@pytest.mark.unit
def test_classify_cache_invalidated_by_swapping_which_var_is_fixed():
    """Fixing one bilinear factor while unfixing the other keeps the count equal."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=1.0)
    m.y = pyo.Var(initialize=1.0)
    m.z = pyo.Var(initialize=1.0)
    m.x.fix()
    m.c = pyo.Constraint(expr=m.x * m.y + pyo.exp(m.z) <= 4)
    m.obj = pyo.Objective(expr=m.y)
    m.z.fix()
    assert classify(m) is ProblemClass.LP
    m.z.unfix()
    m.y.fix()
    assert classify(m) is ProblemClass.NLP
