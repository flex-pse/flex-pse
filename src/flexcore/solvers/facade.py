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
"""

import dataclasses
import os

import pyomo.environ as pyo

from flexcore.exceptions import FlexSolverError
from flexcore.logger import get_logger
from flexcore.solvers.classify import ProblemClass, classify
from flexcore.solvers.registry import _solver_runtime_env, available_solvers

_log = get_logger(__name__)

# Fixed fallback priority (implementer's choice): commercial first, then SCIP
# (benchmark-preferred over HiGHS for MILP, and the default open-source MINLP
# solver), then the open-source LP/MILP solvers, then the NLP solver. SCIP is
# not LP-capable in the registry, so pure LP still resolves to HiGHS.
_PRIORITY = ["gurobi", "scip", "highs", "cbc", "ipopt"]


def _has_grey_box_block(model: pyo.ConcreteModel) -> bool:
    """Report whether ``model`` contains an ``ExternalGreyBoxBlock``.

    ``classify`` sees no Pyomo constraints inside a grey-box block and reports
    the smallest class (often ``LP``), so the priority list below would
    silently hand back an ASL/NL-file solver -- none of which can call back
    into Python to evaluate the grey box. This check is core Pyomo (present
    regardless of whether ``cyipopt``/``torch`` are installed), so it runs
    unconditionally, before ``classify``.

    Args:
        model: The Pyomo model to inspect.

    Returns:
        ``True`` if any active or inactive block on the model is an
        ``ExternalGreyBoxBlockData``.

    Note:
        ``descend_into=True`` alone defaults to ``ctype=(Block,)``: an
        ``ExternalGreyBoxBlock`` is registered as its own distinct ctype (via
        ``declare_custom_block``), so plain ``block_data_objects(descend_into=True)``
        would neither descend into nor return one -- a deviation from the
        milestone's literal snippet, verified against the installed Pyomo
        source. ``ExternalGreyBoxBlock`` must be named explicitly so the walk
        still descends through the ordinary ``Block``s (unit, surrogate
        block) above it.
    """
    from pyomo.contrib.pynumero.interfaces.external_grey_box import (
        ExternalGreyBoxBlock,
        ExternalGreyBoxBlockData,
    )

    return any(
        isinstance(b, ExternalGreyBoxBlockData)
        for b in model.block_data_objects(
            descend_into=(pyo.Block, ExternalGreyBoxBlock)
        )
    )


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


def _pyomo_solver(name: str):
    """Construct the Pyomo solver object for a solver name.

    IPOPT is routed through idaes (see :func:`_idaes_ipopt`) when available so
    the faster HSL ``ma27`` build is used; every other solver — and IPOPT when
    idaes is unavailable — is constructed with ``pyo.SolverFactory``.

    Args:
        name: The Pyomo solver name.

    Returns:
        The constructed Pyomo solver object.
    """
    if name == "ipopt":
        solver = _idaes_ipopt()
        if solver is not None:
            return solver
    return pyo.SolverFactory(name)


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
            return _pyomo_solver(self.name).solve(model, **kwargs)


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
        if _has_grey_box_block(model):
            raise FlexSolverError(
                "this model contains an ExternalGreyBoxBlock (an "
                "ExternalModelSurrogate); get_solver cannot select for it -- "
                "its whole priority list is ASL/NL-file solvers, none of "
                "which can call back into Python. Solve directly with "
                "SolverFactory('cyipopt').",
                solver="cyipopt",
            )
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
