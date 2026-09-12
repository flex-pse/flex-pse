"""Unit tests for :mod:`flexcore.solvers.facade`.

Availability is monkeypatched so no real solver is touched. These tests import
``get_solver``/``SolverFacade`` from the package (``flexcore.solvers``); the
unit-tier guard patches ``flexcore.solvers.facade.get_solver`` and
``SolverFacade.solve``, so ``get_solver`` still classifies/selects here while an
actual ``solve`` remains forbidden (see ``test_unit_tier_forbids_solve``).
"""

import logging

import pyomo.environ as pyo
import pytest

from flexcore.exceptions import FlexSolverError
from flexcore.logger import DedupHandler
from flexcore.solvers import ProblemClass, SolverFacade, facade, get_solver


@pytest.fixture(autouse=True)
def _reset_facade_logger_dedup():
    logger = logging.getLogger("flexcore.solvers.facade")
    for handler in logger.handlers:
        if isinstance(handler, DedupHandler):
            handler._deque.clear()
            handler._map.clear()
            handler.dedup_enabled[logging.WARNING] = False
    yield
    for handler in logger.handlers:
        if isinstance(handler, DedupHandler):
            handler._deque.clear()
            handler._map.clear()
            handler.dedup_enabled[logging.WARNING] = False


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.mark.unit
def test_no_capable_solver_message(monkeypatch):
    monkeypatch.setattr(facade, "available_solvers", dict)
    with pytest.raises(FlexSolverError) as excinfo:
        get_solver(problem_class=ProblemClass.NLP)
    message = str(excinfo.value)
    assert "NLP" in message
    assert "idaes get-extensions" in message


@pytest.mark.unit
def test_minlp_error_mentions_solve_sequence(monkeypatch):
    monkeypatch.setattr(facade, "available_solvers", dict)
    with pytest.raises(FlexSolverError) as excinfo:
        get_solver(problem_class=ProblemClass.MINLP)
    message = str(excinfo.value)
    assert "flexschedule.SolveSequence" in message
    assert "relax" in message


@pytest.mark.unit
def test_prefer_respected_and_fallback(monkeypatch):
    available = {
        "cbc": {ProblemClass.LP, ProblemClass.MILP},
        "ipopt": {ProblemClass.NLP},
    }
    monkeypatch.setattr(facade, "available_solvers", lambda: available)

    # prefer is capable -> it is picked.
    chosen = get_solver(problem_class=ProblemClass.LP, prefer="cbc")
    assert chosen.name == "cbc"
    assert chosen.problem_class is ProblemClass.LP

    # prefer is incapable -> fall back to the priority list, log a warning.
    capture = _LogCapture()
    logger = logging.getLogger("flexcore.solvers.facade")
    logger.addHandler(capture)
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        fallback = get_solver(problem_class=ProblemClass.LP, prefer="ipopt")
    finally:
        logger.removeHandler(capture)
        logger.setLevel(old_level)
    assert fallback.name == "cbc"
    assert any("ipopt" in record.getMessage() for record in capture.records)


@pytest.mark.unit
def test_minlp_routes_to_scip_when_available(monkeypatch):
    """SCIP is the default MINLP solver when installed (a capable solver)."""
    available = {"scip": {ProblemClass.MILP, ProblemClass.MINLP}}
    monkeypatch.setattr(facade, "available_solvers", lambda: available)
    chosen = get_solver(problem_class=ProblemClass.MINLP)
    assert chosen.name == "scip"
    assert chosen.problem_class is ProblemClass.MINLP


@pytest.mark.unit
def test_default_routing_by_class(monkeypatch):
    """HiGHS for LP, SCIP for MILP/MINLP, IPOPT for NLP when all are installed.

    SCIP is preferred over HiGHS for MILP (benchmark-driven); HiGHS still wins
    LP because SCIP is not LP-capable in the registry.
    """
    available = {
        "highs": {ProblemClass.LP, ProblemClass.MILP},
        "ipopt": {ProblemClass.NLP},
        "scip": {ProblemClass.MILP, ProblemClass.MINLP},
    }
    monkeypatch.setattr(facade, "available_solvers", lambda: available)
    assert get_solver(problem_class=ProblemClass.LP).name == "highs"
    assert get_solver(problem_class=ProblemClass.MILP).name == "scip"
    assert get_solver(problem_class=ProblemClass.NLP).name == "ipopt"
    assert get_solver(problem_class=ProblemClass.MINLP).name == "scip"


@pytest.mark.unit
def test_model_vs_problem_class_exclusive():
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.obj = pyo.Objective(expr=m.x)
    with pytest.raises(ValueError):
        get_solver(model=m, problem_class=ProblemClass.LP)


@pytest.mark.unit
def test_ipopt_routes_through_idaes(monkeypatch):
    """IPOPT is built from idaes (HSL ma27) when idaes is importable."""
    sentinel = object()
    monkeypatch.setattr(facade, "_idaes_ipopt", lambda: sentinel)
    assert facade._pyomo_solver("ipopt") is sentinel


@pytest.mark.unit
def test_ipopt_falls_back_to_solver_factory(monkeypatch):
    """When idaes is unavailable, IPOPT falls back to stock SolverFactory."""
    monkeypatch.setattr(facade, "_idaes_ipopt", lambda: None)
    made = []
    monkeypatch.setattr(facade.pyo, "SolverFactory", lambda name: made.append(name))
    facade._pyomo_solver("ipopt")
    assert made == ["ipopt"]


@pytest.mark.unit
def test_non_ipopt_uses_solver_factory(monkeypatch):
    """Every non-IPOPT solver is constructed with pyo.SolverFactory, not idaes."""
    monkeypatch.setattr(
        facade, "_idaes_ipopt", lambda: pytest.fail("idaes path used for non-ipopt")
    )
    made = []
    monkeypatch.setattr(facade.pyo, "SolverFactory", lambda name: made.append(name))
    facade._pyomo_solver("highs")
    assert made == ["highs"]


@pytest.mark.unit
def test_unit_tier_forbids_solve():
    """The unit-tier guard blocks any actual solve (DoD)."""
    fac = SolverFacade(name="highs", problem_class=ProblemClass.LP)
    with pytest.raises(RuntimeError, match="forbidden"):
        fac.solve(pyo.ConcreteModel())


@pytest.mark.component
def test_get_solver_raises_clear_error_for_grey_box_model():
    """get_solver refuses a model with an ExternalGreyBoxBlock, pointing at
    cyipopt -- classify() alone would misclassify it as LP (documented here,
    not gated on cyipopt/torch: this guard protects every normal install)."""
    from pyomo.contrib.pynumero.interfaces.external_grey_box import (
        ExternalGreyBoxBlock,
        ExternalGreyBoxModel,
    )

    from flexcore.solvers.classify import ProblemClass, classify

    class _StubExternalModel(ExternalGreyBoxModel):
        def input_names(self):
            return ["x"]

        def output_names(self):
            return ["y"]

        def set_input_values(self, input_values):
            pass

        def evaluate_outputs(self):
            import numpy as np

            return np.array([0.0])

    m = pyo.ConcreteModel()
    m.egb = ExternalGreyBoxBlock(external_model=_StubExternalModel())

    assert classify(m) is ProblemClass.LP

    with pytest.raises(FlexSolverError, match="cyipopt"):
        get_solver(model=m)
