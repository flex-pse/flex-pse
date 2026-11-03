"""Solver abstraction: model classifier, capability registry, facade, and strategies.

Public API:
    - :func:`classify` and :class:`ProblemClass` -- model classification.
    - :func:`available_solvers` -- installed subset of the capability matrix.
    - :func:`get_solver` and :class:`SolverFacade` -- solver selection facade.
    - :class:`SolverSettings` and :class:`SolverOption` -- solver-neutral tuning.
    - :class:`Direct`, :class:`RelaxAndFix`, :class:`SolveStrategy`,
      :class:`SolveReport`, :class:`StepResult` -- solve strategies.
    - :func:`check_training_range` and :class:`TrainingRangeViolation` -- grey-box
      inputs outside the range their model was trained over.
"""

from flexcore.solvers.classify import ProblemClass, classify
from flexcore.solvers.facade import SolverFacade, get_solver
from flexcore.solvers.grey_box import TrainingRangeViolation, check_training_range
from flexcore.solvers.options import SolverOption, SolverSettings
from flexcore.solvers.registry import available_solvers
from flexcore.solvers.strategies import (
    Direct,
    RelaxAndFix,
    SolveReport,
    SolveStrategy,
    StepResult,
)

__all__ = [
    "ProblemClass",
    "classify",
    "available_solvers",
    "get_solver",
    "SolverFacade",
    "SolverOption",
    "SolverSettings",
    "Direct",
    "RelaxAndFix",
    "SolveStrategy",
    "SolveReport",
    "StepResult",
    "check_training_range",
    "TrainingRangeViolation",
]
