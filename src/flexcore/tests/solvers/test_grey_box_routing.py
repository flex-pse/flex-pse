"""Grey-box models through the solver facade: classify, routing, dive, range check."""

import numpy as np
import pyomo.environ as pyo
import pytest
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
    ExternalGreyBoxBlock,
    ExternalGreyBoxModel,
)
from scipy.sparse import coo_matrix

from flexcore.exceptions import FlexSolverError
from flexcore.solvers import (
    ProblemClass,
    RelaxAndFix,
    SolverFacade,
    check_training_range,
    classify,
    facade,
    get_solver,
)

TC = pyo.TerminationCondition


class _Doubler(ExternalGreyBoxModel):
    """Stub external model: y = 2 * x."""

    def __init__(self, training_range=None):
        if training_range is not None:
            self.training_range = training_range
        self.evaluations = 0
        self._x = np.zeros(1)

    def input_names(self):
        return ["x"]

    def output_names(self):
        return ["y"]

    def set_input_values(self, input_values):
        self._x = np.array(input_values, dtype=float)

    def evaluate_outputs(self):
        self.evaluations += 1
        return np.array([2.0 * self._x[0]])

    def evaluate_jacobian_outputs(self):
        return coo_matrix(([2.0], ([0], [0])), shape=(1, 1))


def _model(training_range=None) -> pyo.ConcreteModel:
    """Minimize (y - 3)^2 with y = 2 * u, so u = 1.5 at the optimum."""
    m = pyo.ConcreteModel()
    m.sub = pyo.Block()
    m.sub.egb = ExternalGreyBoxBlock(external_model=_Doubler(training_range))
    m.u = pyo.Var(bounds=(0, 10), initialize=1.0)
    m.link = pyo.Constraint(expr=m.sub.egb.inputs["x"] == m.u)
    m.obj = pyo.Objective(expr=(m.sub.egb.outputs["y"] - 3) ** 2)
    return m


def _model_with_binaries(n: int = 3) -> pyo.ConcreteModel:
    """As :func:`_model`, with u the sum of ``n`` binaries."""
    m = _model()
    m.on = pyo.Var(range(n), domain=pyo.Binary, initialize=0)
    m.total = pyo.Constraint(expr=m.u == sum(m.on[i] for i in range(n)))
    return m


def _forbid_real_solve(monkeypatch):
    """Report cyipopt as installed and fail if it is ever called."""

    def _no_solve(name, warmstart):
        raise AssertionError("cyipopt must not run on a discrete model")

    monkeypatch.setattr(
        facade, "available_solvers", lambda: {"cyipopt": {ProblemClass.NLP}}
    )
    monkeypatch.setattr(facade, "_pyomo_solver", _no_solve)


# -- classify ------------------------------------------------------------------


@pytest.mark.unit
def test_classify_reports_nlp_for_grey_box_with_linear_constraints():
    assert classify(_model()) is ProblemClass.NLP


@pytest.mark.unit
def test_classify_reports_nlp_for_grey_box_with_no_other_constraints():
    m = pyo.ConcreteModel()
    m.egb = ExternalGreyBoxBlock(external_model=_Doubler())
    assert classify(m) is ProblemClass.NLP


@pytest.mark.unit
def test_classify_reports_minlp_for_grey_box_with_binaries():
    assert classify(_model_with_binaries()) is ProblemClass.MINLP


@pytest.mark.unit
def test_classify_cache_updates_when_the_grey_box_is_deactivated():
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.c = pyo.Constraint(expr=m.x <= 1)
    m.egb = ExternalGreyBoxBlock(external_model=_Doubler())
    assert classify(m) is ProblemClass.NLP
    m.egb.deactivate()
    assert classify(m) is ProblemClass.LP


@pytest.mark.unit
def test_classify_ignores_grey_box_under_a_deactivated_parent():
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.c = pyo.Constraint(expr=m.x <= 1)
    m.alt = pyo.Block()
    m.alt.egb = ExternalGreyBoxBlock(external_model=_Doubler())
    m.alt.deactivate()
    assert classify(m) is ProblemClass.LP


# -- get_solver routing --------------------------------------------------------


@pytest.mark.unit
def test_get_solver_returns_cyipopt_for_grey_box_model(monkeypatch):
    monkeypatch.setattr(
        facade,
        "available_solvers",
        lambda: {"cyipopt": {ProblemClass.NLP}, "ipopt": {ProblemClass.NLP}},
    )
    chosen = get_solver(model=_model())
    assert chosen.name == "cyipopt"
    assert chosen.problem_class is ProblemClass.NLP


@pytest.mark.unit
def test_get_solver_uses_normal_selection_once_grey_box_is_deactivated(monkeypatch):
    monkeypatch.setattr(
        facade,
        "available_solvers",
        lambda: {
            "cyipopt": {ProblemClass.NLP},
            "ipopt": {ProblemClass.NLP, ProblemClass.QP},
        },
    )
    m = _model()
    m.sub.egb.deactivate()
    assert get_solver(model=m).name == "ipopt"


@pytest.mark.unit
def test_get_solver_names_the_extra_when_cyipopt_is_missing(monkeypatch):
    monkeypatch.setattr(facade, "available_solvers", dict)
    with pytest.raises(FlexSolverError, match=r"flex-pse\[greybox\]"):
        get_solver(model=_model())


@pytest.mark.unit
def test_get_solver_overrides_prefer_for_grey_box_model(monkeypatch):
    monkeypatch.setattr(
        facade,
        "available_solvers",
        lambda: {"cyipopt": {ProblemClass.NLP}, "gurobi": {ProblemClass.NLP}},
    )
    chosen = get_solver(model=_model(), prefer="gurobi")
    assert chosen.name == "cyipopt"


@pytest.mark.unit
def test_ordinary_nlp_does_not_select_cyipopt(monkeypatch):
    monkeypatch.setattr(
        facade,
        "available_solvers",
        lambda: {"cyipopt": {ProblemClass.NLP}, "ipopt": {ProblemClass.NLP}},
    )
    assert get_solver(problem_class=ProblemClass.NLP).name == "ipopt"


# -- discrete models never reach cyipopt ---------------------------------------


@pytest.mark.component
def test_direct_solve_with_unfixed_binaries_raises_before_cyipopt(monkeypatch):
    _forbid_real_solve(monkeypatch)
    m = _model_with_binaries()
    with pytest.raises(FlexSolverError, match="RelaxAndFix"):
        get_solver(model=m).solve(m)


@pytest.mark.component
def test_problem_class_path_with_unfixed_binaries_raises_before_cyipopt(monkeypatch):
    _forbid_real_solve(monkeypatch)
    solver = SolverFacade(name="ipopt", problem_class=ProblemClass.NLP)
    with pytest.raises(FlexSolverError, match="RelaxAndFix"):
        solver.solve(_model_with_binaries())


@pytest.mark.component
def test_relax_and_fix_with_general_integers_raises_before_cyipopt(monkeypatch):
    _forbid_real_solve(monkeypatch)
    m = _model()
    m.count = pyo.Var(domain=pyo.NonNegativeIntegers, initialize=1)
    m.total = pyo.Constraint(expr=m.u == m.count)
    solver = get_solver(model=m, strategy=RelaxAndFix())
    with pytest.raises(FlexSolverError, match="integer"):
        solver.solve(m)
    assert m.count.is_integer()


# -- real cyipopt solves -------------------------------------------------------


@pytest.mark.component
@pytest.mark.needs_cyipopt
def test_direct_solve_loads_solution_and_records_objective():
    m = _model()
    solver = get_solver(model=m)
    results = solver.solve(m, options={"print_level": 0, "sb": "yes"})
    assert results.solver.termination_condition is TC.optimal
    assert pyo.value(m.u) == pytest.approx(1.5, abs=1e-5)
    step = solver.last_report.steps[-1]
    assert step.solver == "cyipopt"
    assert step.objective == pytest.approx(0.0, abs=1e-6)


@pytest.mark.component
@pytest.mark.needs_cyipopt
def test_ipopt_facade_reroutes_grey_box_model_to_cyipopt():
    m = _model()
    solver = SolverFacade(name="ipopt", problem_class=ProblemClass.NLP)
    solver.solve(m)
    assert solver.last_report.steps[-1].solver == "cyipopt"
    assert pyo.value(m.u) == pytest.approx(1.5, abs=1e-5)


@pytest.mark.component
@pytest.mark.needs_cyipopt
def test_time_limit_is_passed_to_cyipopt():
    m = _model()
    solver = get_solver(model=m)
    solver.solve(m, time_limit=30.0)
    assert pyo.value(m.u) == pytest.approx(1.5, abs=1e-5)


@pytest.mark.component
@pytest.mark.needs_cyipopt
def test_spent_time_limit_does_not_raise_an_invalid_option_error():
    m = _model_with_binaries()
    solver = get_solver(model=m, strategy=RelaxAndFix())
    solver.solve(m, time_limit=1e-9)
    assert solver.last_report is not None


@pytest.mark.component
@pytest.mark.needs_cyipopt
def test_infeasible_solve_restores_prior_values():
    m = _model()
    m.u.setlb(5)
    m.cap = pyo.Constraint(expr=m.u <= 1)
    m.u.set_value(7.0)
    m.sub.egb.inputs["x"].set_value(7.0)
    solver = get_solver(model=m)
    results = solver.solve(m)
    assert results.solver.termination_condition is TC.infeasible
    assert m.u.value == 7.0
    assert m.sub.egb.inputs["x"].value == 7.0
    assert solver.last_report.objective is None


@pytest.mark.component
@pytest.mark.needs_cyipopt
def test_grey_box_under_deactivated_parent_is_left_out_of_the_solve():
    m = _model()
    m.alt = pyo.Block()
    orphan = _Doubler()
    m.alt.egb = ExternalGreyBoxBlock(external_model=orphan)
    m.alt.egb.inputs["x"].set_value(7.0)
    orphan.evaluations = 0
    m.alt.deactivate()

    get_solver(model=m).solve(m)

    assert m.alt.egb.active
    assert m.alt.egb.inputs["x"].value == 7.0
    assert orphan.evaluations == 0
    assert pyo.value(m.u) == pytest.approx(1.5, abs=1e-5)


# -- RelaxAndFix dive ----------------------------------------------------------


@pytest.mark.component
@pytest.mark.needs_cyipopt
def test_dive_fixes_every_binary_then_restores_the_model():
    m = _model_with_binaries()
    solver = get_solver(model=m, strategy=RelaxAndFix())
    solver.solve(m)

    report = solver.last_report
    assert report.steps[0].label == "relaxation"
    assert report.steps[-1].n_fixed == 3
    assert all(step.label.startswith("dive") for step in report.steps[1:])
    for i in m.on:
        assert m.on[i].value in (0, 1)
        assert m.on[i].is_binary()
        assert not m.on[i].fixed
    assert pyo.value(m.u) == pytest.approx(sum(m.on[i].value for i in m.on))


@pytest.mark.component
@pytest.mark.needs_cyipopt
def test_dive_reports_no_bound_because_a_local_optimum_is_not_one():
    m = _model_with_binaries()
    solver = get_solver(model=m, strategy=RelaxAndFix())
    solver.solve(m)
    assert solver.last_report.bound is None
    assert solver.last_report.gap is None


@pytest.mark.component
@pytest.mark.needs_cyipopt
def test_dive_makes_progress_when_nothing_is_within_fix_tol():
    m = _model_with_binaries()
    solver = get_solver(model=m, strategy=RelaxAndFix(fix_tol=0.0))
    solver.solve(m)
    report = solver.last_report
    assert report.steps[-1].n_fixed == 3
    assert len(report.steps) > 2


@pytest.mark.component
@pytest.mark.needs_cyipopt
def test_dive_returns_report_and_cleans_up_when_a_step_is_infeasible():
    m = _model_with_binaries()
    m.band = pyo.Constraint(expr=pyo.inequality(1.4, m.u, 1.6))
    solver = get_solver(model=m, strategy=RelaxAndFix())
    solver.solve(m)
    assert solver.last_report.steps[-1].termination is TC.infeasible
    for i in m.on:
        assert m.on[i].is_binary()
        assert not m.on[i].fixed


# -- training range ------------------------------------------------------------


@pytest.mark.unit
def test_check_training_range_returns_inputs_outside_the_range():
    m = _model(training_range={"x": (0.0, 1.0)})
    m.sub.egb.inputs["x"].set_value(1.5)
    violations = check_training_range(m)
    assert len(violations) == 1
    assert violations[0].block == "sub.egb"
    assert violations[0].input_name == "x"
    assert violations[0].value == 1.5
    assert (violations[0].lower, violations[0].upper) == (0.0, 1.0)


@pytest.mark.unit
def test_check_training_range_passes_inputs_inside_the_range():
    m = _model(training_range={"x": (0.0, 2.0)})
    m.sub.egb.inputs["x"].set_value(1.5)
    assert check_training_range(m) == []


@pytest.mark.unit
def test_check_training_range_skips_inactive_grey_boxes():
    m = _model(training_range={"x": (0.0, 1.0)})
    m.sub.egb.inputs["x"].set_value(1.5)
    m.sub.egb.deactivate()
    assert check_training_range(m) == []


@pytest.mark.unit
def test_check_training_range_skips_models_with_no_declared_range():
    m = _model()
    m.sub.egb.inputs["x"].set_value(1.5)
    assert check_training_range(m) == []


@pytest.mark.component
@pytest.mark.needs_cyipopt
def test_report_lists_out_of_range_inputs_of_the_final_solution():
    m = _model(training_range={"x": (0.0, 1.0)})
    solver = get_solver(model=m)
    solver.solve(m)
    assert [v.input_name for v in solver.last_report.out_of_range] == ["x"]


@pytest.mark.component
@pytest.mark.needs_cyipopt
def test_training_range_is_checked_once_across_a_multi_step_dive(monkeypatch):
    calls = []
    monkeypatch.setattr(
        facade, "check_training_range", lambda model: calls.append(model) or []
    )
    m = _model_with_binaries()
    solver = get_solver(model=m, strategy=RelaxAndFix())
    solver.solve(m)
    assert len(solver.last_report.steps) > 1
    assert len(calls) == 1


@pytest.mark.component
@pytest.mark.needs_cyipopt
def test_training_range_is_not_checked_after_an_infeasible_solve(monkeypatch):
    calls = []
    monkeypatch.setattr(
        facade, "check_training_range", lambda model: calls.append(model) or []
    )
    m = _model()
    m.u.setlb(5)
    m.cap = pyo.Constraint(expr=m.u <= 1)
    get_solver(model=m).solve(m)
    assert calls == []


# -- constraints left with no free variables -----------------------------------


@pytest.mark.component
@pytest.mark.needs_cyipopt
def test_dive_solves_a_final_step_with_a_binary_only_constraint():
    m = _model_with_binaries()
    m.count = pyo.Constraint(expr=sum(m.on[i] for i in m.on) == 2)
    solver = get_solver(model=m, strategy=RelaxAndFix())
    solver.solve(m)
    report = solver.last_report
    assert report.steps[-1].termination is TC.optimal
    assert report.steps[-1].n_fixed == 3
    assert sum(m.on[i].value for i in m.on) == 2
    assert pyo.value(m.u) == pytest.approx(2.0, abs=1e-5)
    assert m.count.active


@pytest.mark.component
@pytest.mark.needs_cyipopt
def test_dive_reports_infeasible_when_fixing_violates_a_binary_only_constraint():
    m = _model_with_binaries()
    m.count = pyo.Constraint(expr=sum(m.on[i] for i in m.on) == 1.5)
    solver = get_solver(model=m, strategy=RelaxAndFix())
    solver.solve(m)
    assert solver.last_report.steps[-1].termination is TC.infeasible
    assert m.count.active
    assert not hasattr(m, "_tmp_trivial_deactivated_constrs")
    for i in m.on:
        assert m.on[i].is_binary()
        assert not m.on[i].fixed


@pytest.mark.component
@pytest.mark.needs_cyipopt
def test_too_few_degrees_of_freedom_loads_nothing_and_skips_the_range_check(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        facade, "check_training_range", lambda model: calls.append(model) or []
    )
    m = _model()
    m.v = pyo.Var(initialize=4.0)
    m.a = pyo.Constraint(expr=m.u == m.v)
    m.b = pyo.Constraint(expr=m.u == 2 * m.v)
    m.c = pyo.Constraint(expr=m.u == 3 * m.v)
    m.d = pyo.Constraint(expr=m.u == 4 * m.v)
    m.u.set_value(7.0)
    solver = get_solver(model=m)
    solver.solve(m)
    assert solver.last_report.steps[-1].termination is TC.invalidProblem
    assert solver.last_report.objective is None
    assert m.u.value == 7.0
    assert calls == []
