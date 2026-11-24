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

from flexcore.exceptions import FlexSolverError
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


def _is_solved(termination: pyo.TerminationCondition) -> bool:
    """Report whether a solver stopped at a solution it considers final."""
    return termination in (
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.locallyOptimal,
        pyo.TerminationCondition.globallyOptimal,
        pyo.TerminationCondition.feasible,
    )


RELAXED_CLASS = {
    ProblemClass.MILP: ProblemClass.LP,
    ProblemClass.MINLP: ProblemClass.NLP,
}
"""The class each mixed-integer class has once its binaries are relaxed."""

_ROUND_FRACTION = 0.05
"""Share of the free binaries a round fixes when none is within ``fix_tol``."""


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
    """Solve the binary relaxation, fix what it decided, then solve the rest.

    A heuristic: the returned schedule is integer-feasible but not proven
    optimal. ``SolveReport.gap`` measures it against the relaxation's bound.

    Attributes:
        fix_tol: A relaxed binary below ``fix_tol`` is fixed to 0 and one above
            ``1 - fix_tol`` to 1. Must be in ``[0, 0.5)``; ``0`` fixes nothing,
            except in rounds, where each round still fixes the most decided few.
        rounds: Fix the binaries in rounds, re-solving the relaxation after each,
            instead of solving the rest exactly. No mixed-integer solver is
            needed, so this solves a MINLP with an NLP solver alone, but it can
            end without a solution where one exists.

    Raises:
        ValueError: If ``fix_tol`` is outside ``[0, 0.5)``.
    """

    fix_tol: float = 0.1
    rounds: bool = False

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

        Raises:
            FlexSolverError: If ``rounds`` is set and the model has integer
                variables that aren't binary, which it can't fix.
        """
        relaxed_class = RELAXED_CLASS.get(problem_class)
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
        if self.rounds:
            if relaxed_class is problem_class:
                raise FlexSolverError(
                    "RelaxAndFix(rounds=True) fixes binaries only, and this model "
                    "has general integer variables. Install a solver for "
                    f"{problem_class.value} and use RelaxAndFix().",
                    problem_class=problem_class.value,
                )
            return self._solve_in_rounds(model, relaxed_class, run_step, binaries)
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

    def _solve_in_rounds(
        self,
        model: pyo.ConcreteModel,
        relaxed_class: ProblemClass,
        run_step: StepRunner,
        binaries: list,
    ) -> tuple[Any, SolveReport]:
        """Fix the binaries a few at a time, re-solving the relaxation after each.

        Rounding every binary at once is often infeasible. Each round fixes the
        ones already within ``fix_tol`` of 0 or 1 (or the most decided
        ``_ROUND_FRACTION`` if there are none), so the next relaxation can adapt.
        A round with no solution is undone and retried with half as many.

        Args:
            model: The Pyomo model to solve.
            relaxed_class: The model's class with ``binaries`` relaxed.
            run_step: Solves the model once as ``(problem_class, label, warmstart)``.
            binaries: The free binary variables.

        Returns:
            The results of the last step, and the report. If a single binary
            can't be fixed, or a limit stops a step, those results are a failure
            and the model keeps its starting values.
        """
        variables = [
            var
            for var in model.component_data_objects(pyo.Var, descend_into=True)
            if not var.fixed
        ]
        start = [(var, var.value) for var in variables]
        domains = [(var, var.domain) for var in binaries]
        steps: list[StepResult] = []
        bound = None

        def decided(var) -> float:
            """Return how far a relaxed binary is from the nearest of 0 and 1."""
            value = var.value or 0.0  # None if no active expression uses it
            return min(abs(value), abs(1 - value))

        def restore(values) -> None:
            for var, value in values:
                if not var.fixed:
                    var.set_value(value, skip_validation=True)

        solved = False
        try:
            for var in binaries:
                var.domain = pyo.UnitInterval
            results, step = run_step(relaxed_class, "relaxation", False)
            steps.append(step)
            solved = _is_solved(step.termination)
            # A local solver's optimum bounds nothing.
            if solved and relaxed_class is ProblemClass.LP:
                bound = step.objective
            free = list(binaries)
            while solved and free:
                ranked = sorted(free, key=decided)
                batch = [var for var in ranked if decided(var) <= self.fix_tol]
                batch = batch or ranked[: max(1, int(len(free) * _ROUND_FRACTION))]
                relaxed = [(var, var.value) for var in variables if not var.fixed]
                while True:
                    for var in batch:
                        var.fix(0 if (var.value or 0.0) < 0.5 else 1)
                    results, step = run_step(relaxed_class, "round", False)
                    n_fixed = len(binaries) - len(free) + len(batch)
                    steps.append(dataclasses.replace(step, n_fixed=n_fixed))
                    solved = _is_solved(step.termination)
                    if solved:
                        break
                    for var in batch:
                        var.unfix()
                    restore(relaxed)
                    stopped = pyo.TerminationCondition.maxTimeLimit
                    if len(batch) == 1 or step.termination == stopped:
                        break
                    batch = batch[: len(batch) // 2]
                free = [var for var in free if not var.fixed]
        finally:
            for var, domain in domains:
                var.unfix()
                var.domain = domain
            if not solved:
                restore(start)
        return results, SolveReport(strategy="relax_and_fix", steps=steps, bound=bound)
