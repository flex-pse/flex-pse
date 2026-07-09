"""Solver abstraction: model classifier, capability registry, and facade.

Public API:
    - :func:`classify` and :class:`ProblemClass` -- model classification.
    - :func:`available_solvers` -- installed subset of the capability matrix.
    - :func:`get_solver` and :class:`SolverFacade` -- solver selection facade.
"""

from flexcore.solvers.classify import ProblemClass, classify
from flexcore.solvers.facade import SolverFacade, get_solver
from flexcore.solvers.registry import available_solvers

__all__ = [
    "ProblemClass",
    "classify",
    "available_solvers",
    "get_solver",
    "SolverFacade",
]
