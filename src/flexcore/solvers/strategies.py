"""Solve strategies: how one ``solve`` call becomes one or more solver calls.

A strategy receives a ``run_step`` callable from the facade that solves the
model once as a given problem class, so strategies never pick solvers or
options themselves. Add a strategy by writing a frozen dataclass with a
``solve`` method matching :class:`SolveStrategy`.
"""

import dataclasses
from collections.abc import Callable
from typing import Any, Protocol

import pyomo.environ as pyo

from flexcore.logger import get_logger
from flexcore.solvers.classify import ProblemClass

_log = get_logger(__name__)


@dataclasses.dataclass
class StepResult:
    """The outcome of one solver call made by a strategy.

    Attributes:
        label: What the step was, e.g. ``"direct"`` or ``"relaxation"``.
        solver: The Pyomo solver name that ran it.
        termination: The solver's termination condition.
        objective: The active objective's value, or ``None`` if no solution loaded.
        wall_time: Seconds for the whole call, including model write and read.
        solver_time: Seconds the solver reported for itself, if it reports one.
        n_fixed: Number of variables the strategy fixed for this step.
    """

    label: str
    solver: str
    termination: pyo.TerminationCondition
    objective: float | None
    wall_time: float
    solver_time: float | None
    n_fixed: int = 0


@dataclasses.dataclass
class SolveReport:
    """Every step a strategy ran for one ``solve`` call, plus its bound.

    Attributes:
        strategy: The strategy's name.
        steps: The steps in the order they ran; the last produced the solution.
        bound: A bound on the optimal objective, if the strategy computed one.
    """

    strategy: str
    steps: list[StepResult]
    bound: float | None = None

    @property
    def objective(self) -> float | None:
        """The objective value of the final step."""
        return self.steps[-1].objective

    @property
    def gap(self) -> float | None:
        """Relative distance of the final objective from ``bound``, or ``None``."""
        if self.bound is None or self.objective is None:
            return None
        if not self.objective:
            return 0.0
        return abs(self.objective - self.bound) / abs(self.objective)

    @property
    def wall_time(self) -> float:
        """Total seconds across all steps."""
        return sum(step.wall_time for step in self.steps)


StepRunner = Callable[[ProblemClass, str, bool], tuple[Any, StepResult]]


class SolveStrategy(Protocol):
    """The interface every solve strategy implements."""

    def solve(
        self,
        model: pyo.ConcreteModel,
        problem_class: ProblemClass,
        run_step: StepRunner,
        warmstart: bool,
    ) -> tuple[Any, SolveReport]:
        """Solve ``model`` using ``run_step`` for every solver call.

        Args:
            model: The Pyomo model to solve.
            problem_class: The model's class as classified by the facade.
            run_step: Solves the model once as ``(problem_class, label, warmstart)``.
            warmstart: Whether the caller asked to warm start from current values.

        Returns:
            The Pyomo results of the step that produced the solution, and the report.
        """
        ...


def _is_infeasible(termination: pyo.TerminationCondition) -> bool:
    """Report whether a termination condition proves no feasible point exists."""
    return termination in (
        pyo.TerminationCondition.infeasible,
        pyo.TerminationCondition.infeasibleOrUnbounded,
    )


@dataclasses.dataclass(frozen=True)
class Direct:
    """Solve the model once, exactly as built."""

    def solve(
        self,
        model: pyo.ConcreteModel,
        problem_class: ProblemClass,
        run_step: StepRunner,
        warmstart: bool,
    ) -> tuple[Any, SolveReport]:
        """Solve ``model`` in one step; see :meth:`SolveStrategy.solve`."""
        results, step = run_step(problem_class, "direct", warmstart)
        return results, SolveReport(strategy="direct", steps=[step])


@dataclasses.dataclass(frozen=True)
class RelaxAndFix:
    """Solve the binary relaxation, fix what it decided, then solve the rest exactly.

    A heuristic: the returned schedule is integer-feasible but not proven
    optimal. ``SolveReport.gap`` measures it against the relaxation's bound.

    Attributes:
        fix_tol: A relaxed binary below ``fix_tol`` is fixed to 0 and one above
            ``1 - fix_tol`` to 1. Must be in ``[0, 0.5)``; ``0`` fixes nothing.

    Raises:
        ValueError: If ``fix_tol`` is outside ``[0, 0.5)``.
    """

    fix_tol: float = 0.1

    def __post_init__(self) -> None:
        """Validate ``fix_tol``."""
        if not 0 <= self.fix_tol < 0.5:
            raise ValueError(f"fix_tol must be in [0, 0.5), got {self.fix_tol}.")

    def solve(
        self,
        model: pyo.ConcreteModel,
        problem_class: ProblemClass,
        run_step: StepRunner,
        warmstart: bool,
    ) -> tuple[Any, SolveReport]:
        """Relax, fix and re-solve ``model``; see :meth:`SolveStrategy.solve`.

        The model is left with its original domains and nothing newly fixed,
        with the final solution loaded.
        """
        relaxed_class = {
            ProblemClass.MILP: ProblemClass.LP,
            ProblemClass.MINLP: ProblemClass.NLP,
        }.get(problem_class)
        if relaxed_class is None:
            results, step = run_step(problem_class, "direct", warmstart)
            return results, SolveReport(strategy="relax_and_fix", steps=[step])
        if warmstart:
            _log.warning("RelaxAndFix does not use warm starts; solving cold.")

        binaries = []
        for var in model.component_data_objects(pyo.Var, descend_into=True):
            if var.fixed or not var.is_integer():
                continue
            if var.is_binary():
                binaries.append(var)
            else:
                relaxed_class = problem_class  # general integers stay integer
        domains = [(var, var.domain) for var in binaries]
        fixed: list = []
        steps: list[StepResult] = []
        try:
            for var in binaries:
                var.domain = pyo.UnitInterval
            results, relaxation = run_step(relaxed_class, "relaxation", False)
            steps.append(relaxation)
            for var, domain in domains:
                var.domain = domain
            if _is_infeasible(relaxation.termination):
                return results, SolveReport(strategy="relax_and_fix", steps=steps)
            relaxed = [(var, var.value) for var in binaries]

            ladder = sorted({self.fix_tol, self.fix_tol / 2, self.fix_tol / 4, 0.0})
            for tol in reversed(ladder):
                for var in fixed:
                    var.unfix()
                fixed = []
                for var, value in relaxed:
                    if tol == 0 or value is None or tol <= value <= 1 - tol:
                        continue
                    var.fix(0 if value < tol else 1)
                    fixed.append(var)
                results, step = run_step(
                    problem_class, f"fixed (fix_tol={tol:g})", False
                )
                steps.append(dataclasses.replace(step, n_fixed=len(fixed)))
                if not _is_infeasible(step.termination):
                    break
        finally:
            for var, domain in domains:
                var.domain = domain
            for var in fixed:
                var.unfix()

        optimal = relaxation.termination == pyo.TerminationCondition.optimal
        bound = relaxation.objective if optimal else None
        return results, SolveReport(strategy="relax_and_fix", steps=steps, bound=bound)
