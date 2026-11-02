"""The ``get_solver`` facade and the thin :class:`SolverFacade` wrapper.

``get_solver`` picks the best available solver for a problem class and errors
loudly when none qualifies — it **never** transforms the model: no integrality
relaxation, no decomposition, no trust regions. MINLP routes to SCIP (the
default open-source MINLP solver) when installed; with no MINLP-capable
solver available it is a hard error pointing at SCIP or
``flexschedule.SolveSequence``, not silent model surgery.

IPOPT is bound to idaes's HSL-built binary (``idaes.bin_directory/ipopt``, whose
default linear solver is ``ma27``) when idaes is importable, rather than the
MUMPS-linked stock IPOPT — an order of magnitude faster and more robust on stiff
NLPs. It falls back to ``pyo.SolverFactory`` when idaes is unavailable, keeping
idaes an optional dependency (see :func:`_idaes_ipopt`).

How a facade solves is set by its strategy (:mod:`flexcore.solvers.strategies`),
and its options by settings plus solver-neutral keywords
(:mod:`flexcore.solvers.options`).
"""

import dataclasses
import os
import time
from typing import Any

import pyomo.environ as pyo

from flexcore.exceptions import FlexSolverError
from flexcore.logger import get_logger
from flexcore.solvers.classify import ProblemClass, classify
from flexcore.solvers.options import SolverSettings, solver_options
from flexcore.solvers.registry import (
    WARMSTART_SOLVERS,
    _solver_runtime_env,
    available_solvers,
)
from flexcore.solvers.strategies import (
    Direct,
    SolveReport,
    SolveStrategy,
    StepResult,
    _is_infeasible,
)

_log = get_logger(__name__)

# Fixed fallback priority (implementer's choice): commercial first, then SCIP
# (benchmark-preferred over HiGHS for MILP, and the default open-source MINLP
# solver), then the open-source LP/MILP solvers, then the NLP solver. SCIP is
# not LP-capable in the registry, so pure LP still resolves to HiGHS.
_PRIORITY = ["gurobi", "scip", "highs", "cbc", "ipopt"]


def _idaes_ipopt():
    """Return idaes's HSL-linked IPOPT solver, or ``None`` if idaes is absent.

    ``idaes get-extensions`` installs an IPOPT built against HSL under
    ``idaes.bin_directory``; its default linear solver is ``ma27`` — an order of
    magnitude faster and more robust on stiff NLPs than the MUMPS-linked stock
    IPOPT. The binary is targeted **explicitly** rather than via
    :func:`idaes.core.solvers.get_solver`: in a conda env that also ships an
    IPOPT (``scip`` pulls one in), idaes's ``get_solver`` resolves to the conda
    binary on ``PATH`` while ``import idaes`` prepends idaes's libraries to the
    loader path — the mismatched ``libipopt`` then segfaults (the same clash
    :func:`~flexcore.solvers.registry._clean_idaes_from_libpath` handles for
    SCIP). idaes is optional, so a missing import or binary returns ``None`` and
    the caller falls back to the stock IPOPT.

    Returns:
        A Pyomo IPOPT solver bound to idaes's binary, or ``None`` when idaes or
        its IPOPT binary is unavailable.
    """
    try:
        import idaes
    except ImportError:
        _log.debug("idaes not importable; using stock IPOPT via SolverFactory.")
        return None
    executable = os.path.join(idaes.bin_directory, "ipopt")
    if not os.path.isfile(executable):
        _log.debug("no idaes IPOPT binary at %s; using stock IPOPT.", executable)
        return None
    solver = pyo.SolverFactory("ipopt", executable=executable)
    solver.options["linear_solver"] = "ma27"  # ensure ma27 is default
    return solver


def _pyomo_solver(name: str, warmstart: bool = False):
    """Construct the Pyomo solver object for a solver name.

    IPOPT is routed through idaes (see :func:`_idaes_ipopt`) when available so
    the faster HSL ``ma27`` build is used. A warm-started ``highs`` uses the
    ``appsi_highs`` interface, the one HiGHS interface that accepts a warm
    start. Every other solver is constructed with ``pyo.SolverFactory``.

    Args:
        name: The Pyomo solver name.
        warmstart: Whether the solve will pass ``warmstart=True``.

    Returns:
        The constructed Pyomo solver object.
    """
    if warmstart and name == "highs":
        return pyo.SolverFactory("appsi_highs")
    if name == "ipopt":
        solver = _idaes_ipopt()
        if solver is not None:
            return solver
    return pyo.SolverFactory(name)


@dataclasses.dataclass
class SolverFacade:
    """A selected solver plus how to tune it and how to solve with it.

    Attributes:
        name: The Pyomo solver name (e.g. ``"highs"``).
        problem_class: The class this solver was selected for.
        settings: The tuning settings applied to every solver call.
        strategy: How one ``solve`` call is turned into solver calls.
        last_report: The :class:`SolveReport` of the most recent ``solve``.
    """

    name: str
    problem_class: ProblemClass
    settings: SolverSettings = SolverSettings.DEFAULT
    strategy: SolveStrategy = dataclasses.field(default_factory=Direct)
    last_report: SolveReport | None = dataclasses.field(default=None, init=False)

    def solve(
        self,
        model: pyo.ConcreteModel,
        *,
        tee: bool = False,
        warmstart: bool = False,
        mip_gap: float | None = None,
        time_limit: float | None = None,
        options: dict[str, Any] | None = None,
    ):
        """Solve ``model`` with the facade's strategy, solver and settings.

        The solution is loaded into the model only when the solver found one;
        an infeasible model returns its results without raising. The step-by-step
        record is kept on :attr:`last_report`.

        Args:
            model: The Pyomo model to solve.
            tee: Stream the solver's log to stdout.
            warmstart: Start from the model's current values. A solver that
                cannot warm start logs a warning and solves cold.
            mip_gap: Relative MIP gap at which to stop, applied to every step.
            time_limit: Time limit in seconds for the whole solve, across all steps.
            options: Native solver options; each must not also be set elsewhere.
                A step run on a different solver than the facade's drops them.

        Returns:
            The Pyomo results object of the step that produced the solution.
        """
        self.last_report = None
        deadline = None if time_limit is None else time.perf_counter() + time_limit

        def run_step(
            problem_class: ProblemClass, label: str, step_warmstart: bool
        ) -> tuple[Any, StepResult]:
            name = self.name
            if problem_class not in available_solvers().get(name, set()):
                name = get_solver(problem_class=problem_class).name
            step_options = options
            if options and name != self.name:
                _log.warning(
                    "Native options for %r are not passed to %r.", self.name, name
                )
                step_options = None
            warm_capable = name in WARMSTART_SOLVERS and (
                name != "highs"
                or pyo.SolverFactory("appsi_highs").available(exception_flag=False)
            )
            if step_warmstart and not warm_capable:
                _log.warning("Solver %r cannot warm start; solving cold.", name)
                step_warmstart = False
            remaining = (
                None if deadline is None else max(deadline - time.perf_counter(), 0.0)
            )
            native = solver_options(
                name, self.settings, mip_gap, remaining, step_options
            )
            extra = {"warmstart": True} if step_warmstart else {}
            active = next(
                model.component_data_objects(
                    pyo.Objective, active=True, descend_into=True
                ),
                None,
            )
            start = time.perf_counter()
            with _solver_runtime_env(name):
                # HiGHS keeps no solution to load later when there is no objective.
                results = _pyomo_solver(name, step_warmstart).solve(
                    model,
                    tee=tee,
                    options=native,
                    load_solutions=active is None,
                    **extra,
                )
            wall_time = time.perf_counter() - start
            termination = results.solver.termination_condition
            objective = None
            if (
                active is not None
                and len(results.solution) > 0
                and not _is_infeasible(termination)
            ):
                model.solutions.load_from(results)
                objective = pyo.value(active)
            solver_time = getattr(results.solver, "wall_time", None)
            step = StepResult(
                label=label,
                solver=name,
                termination=termination,
                objective=objective,
                wall_time=wall_time,
                solver_time=None if solver_time is None else float(solver_time),
            )
            ok = termination in (
                pyo.TerminationCondition.optimal,
                pyo.TerminationCondition.locallyOptimal,
                pyo.TerminationCondition.globallyOptimal,
                pyo.TerminationCondition.feasible,
            )
            (_log.info if ok else _log.warning)(
                "%s solve with %s: %s in %.2f s.", label, name, termination, wall_time
            )
            return results, step

        results, self.last_report = self.strategy.solve(
            model, self.problem_class, run_step, warmstart
        )
        return results


def get_solver(
    model: pyo.ConcreteModel | None = None,
    problem_class: ProblemClass | None = None,
    prefer: str | None = None,
    settings: SolverSettings = SolverSettings.DEFAULT,
    strategy: SolveStrategy | None = None,
) -> SolverFacade:
    """Select the best available solver for a problem class.

    Exactly one of ``model`` or ``problem_class`` may be given. If ``model`` is
    given it is classified via :func:`classify`; if neither is given the class
    defaults to ``ProblemClass.LP`` (smallest choice, implementer's).
    Candidate order is ``prefer`` (if named and capable) then the fixed
    priority list ``["gurobi", "scip", "highs", "cbc", "ipopt"]``. If ``prefer``
    is named but unavailable or incapable, a warning is logged and selection
    falls through to the priority list. SCIP is tried before HiGHS, so MILP
    routes to SCIP when installed (benchmark-preferred); pure LP still routes to
    HiGHS since SCIP is not LP-capable. In particular MINLP routes to SCIP (the
    default open-source MINLP solver) when it is installed; only when no
    MINLP-capable solver is available does MINLP raise.

    Args:
        model: A Pyomo model to classify; mutually exclusive with
            ``problem_class``.
        problem_class: The :class:`ProblemClass` to solve; mutually exclusive
            with ``model``.
        prefer: A solver name to try first.
        settings: The tuning settings applied to every solver call.
        strategy: How to solve, e.g. :class:`RelaxAndFix`; defaults to :class:`Direct`.

    Returns:
        A :class:`SolverFacade` bound to the chosen solver.

    Raises:
        ValueError: If both ``model`` and ``problem_class`` are given.
        FlexSolverError: If no available solver supports the problem class. For
            ``MINLP`` the message points at ``flexschedule.SolveSequence``;
            the facade never relaxes integrality itself.

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
    strategy = Direct() if strategy is None else strategy

    def _facade(name: str) -> SolverFacade:
        return SolverFacade(
            name=name, problem_class=problem_class, settings=settings, strategy=strategy
        )

    def _capable(name: str) -> bool:
        caps = available.get(name)
        return caps is not None and problem_class in caps

    if prefer is not None:
        if _capable(prefer):
            return _facade(prefer)
        _log.warning(
            "Preferred solver %r cannot solve %s (unavailable or incapable); "
            "falling back to the default priority list.",
            prefer,
            problem_class.value,
        )

    for name in _PRIORITY:
        if _capable(name):
            return _facade(name)

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
