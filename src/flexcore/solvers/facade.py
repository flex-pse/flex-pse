"""The ``get_solver`` facade and the thin :class:`SolverFacade` wrapper.

``get_solver`` picks the best available solver for a problem class and errors
loudly when none qualifies — it **never** transforms the model (decision R5,
``plan/01_architecture.md`` §2.2): no integrality relaxation, no decomposition,
no trust regions. MINLP routes to SCIP (the default open-source MINLP solver)
when installed; with no MINLP-capable solver available it is a hard error
pointing at SCIP or ``flexschedule.SolveSequence``, not silent model surgery.
"""

import dataclasses
import logging

import pyomo.environ as pyo

from flexcore.exceptions import FlexSolverError
from flexcore.solvers.classify import ProblemClass, classify
from flexcore.solvers.registry import _solver_runtime_env, available_solvers

_log = logging.getLogger(__name__)

# Fixed fallback priority (implementer's choice): commercial first, then the
# open-source LP/MILP solvers (HiGHS preferred), then SCIP (the default
# open-source MINLP solver, also MILP-capable), then the NLP solver.
_PRIORITY = ["gurobi", "highs", "cbc", "scip", "ipopt"]


@dataclasses.dataclass
class SolverFacade:
    """A thin wrapper around a selected Pyomo solver.

    Attributes:
        name: The Pyomo solver name (e.g. ``"highs"``).
        problem_class: The class this solver was selected for.
    """

    name: str
    problem_class: ProblemClass

    def solve(self, model: pyo.ConcreteModel, **kwargs):
        """Solve ``model`` with the selected solver.

        No option translation is performed in v0; keyword arguments are passed
        straight through to Pyomo's ``solve``, defaulting ``tee`` to ``False``.

        Args:
            model: The Pyomo model to solve.
            **kwargs: Forwarded to the underlying Pyomo solver's ``solve``.

        Returns:
            The Pyomo results object returned by the solver.
        """
        kwargs.setdefault("tee", False)
        with _solver_runtime_env(self.name):
            return pyo.SolverFactory(self.name).solve(model, **kwargs)


def get_solver(
    model: pyo.ConcreteModel | None = None,
    problem_class: ProblemClass | None = None,
    prefer: str | None = None,
) -> SolverFacade:
    """Select the best available solver for a problem class.

    Exactly one of ``model`` or ``problem_class`` may be given. If ``model`` is
    given it is classified via :func:`classify`; if neither is given the class
    defaults to ``ProblemClass.LP`` (smallest choice, implementer's).
    Candidate order is ``prefer`` (if named and capable) then the fixed
    priority list ``["gurobi", "highs", "cbc", "scip", "ipopt"]``. If ``prefer``
    is named but unavailable or incapable, a warning is logged and selection
    falls through to the priority list. In particular MINLP routes to SCIP (the
    default open-source MINLP solver) when it is installed; only when no
    MINLP-capable solver is available does MINLP raise (decision R5).

    Args:
        model: A Pyomo model to classify; mutually exclusive with
            ``problem_class``.
        problem_class: The :class:`ProblemClass` to solve; mutually exclusive
            with ``model``.
        prefer: A solver name to try first.

    Returns:
        A :class:`SolverFacade` bound to the chosen solver.

    Raises:
        ValueError: If both ``model`` and ``problem_class`` are given.
        FlexSolverError: If no available solver supports the problem class. For
            ``MINLP`` the message points at ``flexschedule.SolveSequence``
            (decision R5); the facade never relaxes integrality itself.

    Example:
        >>> from flexcore.solvers import get_solver, ProblemClass
        >>> solver = get_solver(problem_class=ProblemClass.LP)  # doctest: +SKIP
        >>> results = solver.solve(model)  # doctest: +SKIP
    """
    if model is not None and problem_class is not None:
        raise ValueError(
            "Pass exactly one of `model` or `problem_class`, not both. "
            "Pass `model` to classify it automatically, or `problem_class` to "
            "select for a known class."
        )
    if model is not None:
        problem_class = classify(model)
    elif problem_class is None:
        problem_class = ProblemClass.LP

    available = available_solvers()

    def _capable(name: str) -> bool:
        caps = available.get(name)
        return caps is not None and problem_class in caps

    if prefer is not None:
        if _capable(prefer):
            return SolverFacade(name=prefer, problem_class=problem_class)
        _log.warning(
            "Preferred solver %r cannot solve %s (unavailable or incapable); "
            "falling back to the default priority list.",
            prefer,
            problem_class.value,
        )

    for name in _PRIORITY:
        if _capable(name):
            return SolverFacade(name=name, problem_class=problem_class)

    if problem_class is ProblemClass.MINLP:
        raise FlexSolverError(
            "this model is MINLP; install a MINLP-capable solver (SCIP: "
            "'conda install -c conda-forge scip') or compose a "
            "`flexschedule.SolveSequence` (relax -> MIP -> fix -> NLP).",
            problem_class=ProblemClass.MINLP.value,
        )
    raise FlexSolverError(
        f"No available solver supports {problem_class.value}. Install a capable "
        "solver (HiGHS via the 'highspy' wheel, or IPOPT with "
        "'idaes get-extensions') or pass prefer=<installed solver>.",
        problem_class=problem_class.value,
    )
