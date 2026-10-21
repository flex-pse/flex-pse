"""Unit tests for :mod:`flexcore.solvers.strategies` with a scripted fake runner."""

import logging

import pyomo.environ as pyo
import pytest

from flexcore.logger import DedupHandler
from flexcore.solvers import (
    Direct,
    ProblemClass,
    RelaxAndFix,
    SolveReport,
    StepResult,
)

TC = pyo.TerminationCondition


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def strategies_log():
    logger = logging.getLogger("flexcore.solvers.strategies")
    for handler in logger.handlers:
        if isinstance(handler, DedupHandler):
            handler.dedup_enabled[logging.WARNING] = False
    capture = _LogCapture()
    logger.addHandler(capture)
    yield capture
    logger.removeHandler(capture)


class _ScriptedRunner:
    """Fake step runner: each call pops one scripted step and records what it saw."""

    def __init__(self, model, script):
        self.model = model
        self.script = list(script)
        self.calls = []

    def __call__(self, problem_class, label, warmstart):
        termination, values, objective = self.script.pop(0)
        b = self.model.b
        self.calls.append(
            {
                "problem_class": problem_class,
                "label": label,
                "warmstart": warmstart,
                "domains": [b[i].domain for i in b],
                "fixed": {i: b[i].value for i in b if b[i].fixed},
            }
        )
        if values is not None:
            for i, value in values.items():
                if not b[i].fixed:
                    b[i].set_value(value, skip_validation=True)
        step = StepResult(
            label=label,
            solver="fake",
            termination=termination,
            objective=objective,
            wall_time=1.0,
            solver_time=0.5,
        )
        return f"results-{len(self.calls)}", step


def _model() -> pyo.ConcreteModel:
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, 3)
    m.b = pyo.Var(m.I, domain=pyo.Binary)
    m.x = pyo.Var(bounds=(0, 10))
    m.c = pyo.Constraint(expr=m.x <= sum(m.b[i] for i in m.I))
    m.obj = pyo.Objective(expr=m.x)
    return m


_RELAXED = {0: 0.02, 1: 0.97, 2: 0.5, 3: 0.3}
_INTEGRAL = {0: 0, 1: 1, 2: 1, 3: 0}


@pytest.mark.unit
def test_direct_runs_one_step_and_forwards_warmstart():
    m = _model()
    runner = _ScriptedRunner(m, [(TC.optimal, _INTEGRAL, 5.0)])
    results, report = Direct().solve(m, ProblemClass.MILP, runner, warmstart=True)
    assert results == "results-1"
    assert runner.calls[0]["problem_class"] is ProblemClass.MILP
    assert runner.calls[0]["warmstart"] is True
    assert report.strategy == "direct"
    assert report.bound is None
    assert report.gap is None
    assert report.objective == 5.0
    assert len(report.steps) == 1


@pytest.mark.unit
def test_relax_and_fix_relaxes_then_fixes_near_integral_binaries():
    m = _model()
    runner = _ScriptedRunner(
        m, [(TC.optimal, _RELAXED, 10.0), (TC.optimal, _INTEGRAL, 11.0)]
    )
    results, report = RelaxAndFix(fix_tol=0.1).solve(
        m, ProblemClass.MILP, runner, warmstart=False
    )

    relaxation, fixed_mip = runner.calls
    assert relaxation["problem_class"] is ProblemClass.LP
    assert all(domain is pyo.UnitInterval for domain in relaxation["domains"])
    assert relaxation["fixed"] == {}

    assert fixed_mip["problem_class"] is ProblemClass.MILP
    assert all(domain is pyo.Binary for domain in fixed_mip["domains"])
    assert fixed_mip["fixed"] == {0: 0, 1: 1}

    assert results == "results-2"
    assert report.strategy == "relax_and_fix"
    assert report.bound == 10.0
    assert report.objective == 11.0
    assert report.gap == pytest.approx(1.0 / 11.0)
    assert report.steps[1].n_fixed == 2
    assert report.wall_time == pytest.approx(2.0)


@pytest.mark.unit
def test_relax_and_fix_leaves_model_unfixed_and_binary_with_solution_loaded():
    m = _model()
    runner = _ScriptedRunner(
        m, [(TC.optimal, _RELAXED, 10.0), (TC.optimal, _INTEGRAL, 11.0)]
    )
    RelaxAndFix().solve(m, ProblemClass.MILP, runner, warmstart=False)
    assert not any(m.b[i].fixed for i in m.I)
    assert all(m.b[i].domain is pyo.Binary for i in m.I)
    assert {i: m.b[i].value for i in m.I} == _INTEGRAL


@pytest.mark.unit
def test_relax_and_fix_steps_down_the_ladder_after_infeasible_rungs():
    m = _model()
    runner = _ScriptedRunner(
        m,
        [
            (TC.optimal, _RELAXED, 10.0),
            (TC.infeasible, None, None),
            (TC.infeasibleOrUnbounded, None, None),
            (TC.optimal, _INTEGRAL, 12.0),
        ],
    )
    _, report = RelaxAndFix(fix_tol=0.1).solve(
        m, ProblemClass.MILP, runner, warmstart=False
    )
    # tol=0.1 fixes b0, b1; tol=0.05 fixes b0, b1; tol=0.025 fixes b0 only.
    assert [call["fixed"] for call in runner.calls[1:]] == [
        {0: 0, 1: 1},
        {0: 0, 1: 1},
        {0: 0},
    ]
    assert [step.n_fixed for step in report.steps[1:]] == [2, 2, 1]
    assert report.objective == 12.0


@pytest.mark.unit
def test_relax_and_fix_last_rung_fixes_nothing():
    m = _model()
    script = [(TC.optimal, _RELAXED, 10.0)] + [(TC.infeasible, None, None)] * 4
    runner = _ScriptedRunner(m, script)
    results, report = RelaxAndFix(fix_tol=0.1).solve(
        m, ProblemClass.MILP, runner, warmstart=False
    )
    assert len(runner.calls) == 5
    assert runner.calls[-1]["fixed"] == {}
    assert report.steps[-1].termination is TC.infeasible
    assert results == "results-5"


@pytest.mark.unit
def test_relax_and_fix_returns_relaxation_when_it_is_infeasible():
    m = _model()
    runner = _ScriptedRunner(m, [(TC.infeasible, None, None)])
    results, report = RelaxAndFix().solve(m, ProblemClass.MILP, runner, warmstart=False)
    assert results == "results-1"
    assert len(report.steps) == 1
    assert report.bound is None
    assert all(m.b[i].domain is pyo.Binary for i in m.I)


@pytest.mark.unit
def test_relax_and_fix_restores_model_when_a_step_raises():
    m = _model()

    def _runner(problem_class, label, warmstart):
        if problem_class is ProblemClass.LP:
            for i, value in _RELAXED.items():
                m.b[i].set_value(value, skip_validation=True)
            return "relaxed", StepResult(label, "fake", TC.optimal, 10.0, 1.0, None)
        raise RuntimeError("solver crashed")

    with pytest.raises(RuntimeError, match="solver crashed"):
        RelaxAndFix().solve(m, ProblemClass.MILP, _runner, warmstart=False)
    assert not any(m.b[i].fixed for i in m.I)
    assert all(m.b[i].domain is pyo.Binary for i in m.I)


@pytest.mark.unit
def test_relax_and_fix_leaves_user_fixed_binaries_alone():
    m = _model()
    m.b[3].fix(1)
    runner = _ScriptedRunner(
        m, [(TC.optimal, _RELAXED, 10.0), (TC.optimal, _INTEGRAL, 11.0)]
    )
    RelaxAndFix().solve(m, ProblemClass.MILP, runner, warmstart=False)
    assert runner.calls[0]["domains"][3] is pyo.Binary
    assert m.b[3].fixed and m.b[3].value == 1


@pytest.mark.unit
def test_relax_and_fix_keeps_general_integers_in_the_relaxation():
    m = _model()
    m.n = pyo.Var(domain=pyo.NonNegativeIntegers, bounds=(0, 5))
    runner = _ScriptedRunner(
        m, [(TC.optimal, _RELAXED, 10.0), (TC.optimal, _INTEGRAL, 11.0)]
    )
    RelaxAndFix().solve(m, ProblemClass.MILP, runner, warmstart=False)
    assert runner.calls[0]["problem_class"] is ProblemClass.MILP
    assert m.n.domain is pyo.NonNegativeIntegers


@pytest.mark.unit
def test_relax_and_fix_relaxes_minlp_to_nlp():
    m = _model()
    runner = _ScriptedRunner(
        m, [(TC.optimal, _RELAXED, 10.0), (TC.optimal, _INTEGRAL, 11.0)]
    )
    RelaxAndFix().solve(m, ProblemClass.MINLP, runner, warmstart=False)
    assert runner.calls[0]["problem_class"] is ProblemClass.NLP
    assert runner.calls[1]["problem_class"] is ProblemClass.MINLP


@pytest.mark.unit
def test_relax_and_fix_on_continuous_class_is_a_single_direct_step():
    m = _model()
    runner = _ScriptedRunner(m, [(TC.optimal, None, 3.0)])
    _, report = RelaxAndFix().solve(m, ProblemClass.LP, runner, warmstart=False)
    assert len(runner.calls) == 1
    assert runner.calls[0]["problem_class"] is ProblemClass.LP
    assert report.bound is None


@pytest.mark.unit
def test_relax_and_fix_warns_and_runs_cold_when_warmstart_requested(strategies_log):
    m = _model()
    runner = _ScriptedRunner(
        m, [(TC.optimal, _RELAXED, 10.0), (TC.optimal, _INTEGRAL, 11.0)]
    )
    RelaxAndFix().solve(m, ProblemClass.MILP, runner, warmstart=True)
    assert all(call["warmstart"] is False for call in runner.calls)
    assert any("warm" in record.getMessage() for record in strategies_log.records)


@pytest.mark.unit
@pytest.mark.parametrize("fix_tol", [-0.1, 0.5, 0.7])
def test_relax_and_fix_rejects_out_of_range_tolerance(fix_tol):
    with pytest.raises(ValueError, match="fix_tol"):
        RelaxAndFix(fix_tol=fix_tol)


@pytest.mark.unit
def test_report_gap_is_none_without_objective():
    step = StepResult("direct", "fake", TC.infeasible, None, 1.0, None)
    report = SolveReport(strategy="relax_and_fix", steps=[step], bound=10.0)
    assert report.gap is None
