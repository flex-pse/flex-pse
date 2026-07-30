"""Classify a Pyomo model into a :class:`ProblemClass`.

The classifier walks a model's *active* constraints and objective, inspects
each expression's polynomial degree, and inspects which variables are discrete,
to decide the smallest problem class that describes the model. It never
transforms the model (``plan/01_architecture.md`` §2.2) — it only
reports.
"""

import enum

import pyomo.environ as pyo
from pyomo.core.expr import identify_variables

# Expression-shape ranking, worst-wins when combined across expressions.
_LINEAR = 0
_QUADRATIC = 1
_NONLINEAR = 2


class ProblemClass(enum.Enum):
    """Mathematical program classes that the solver facade reasons about.

    ``LP``/``QP``/``MILP``/``NLP``/``MINLP`` are the classes :func:`classify`
    can return. ``MINLP_OA`` (outer approximation) and ``MINLP_TR`` (trust
    region) are reserved strategy slots — deliberately unimplemented in v0
    and **never** returned by :func:`classify`.
    """

    LP = "LP"
    QP = "QP"
    MILP = "MILP"
    NLP = "NLP"
    MINLP = "MINLP"
    # Reserved strategy slots — documented, deliberately unimplemented:
    MINLP_OA = "MINLP_OA"  # outer approximation, post-v0
    MINLP_TR = "MINLP_TR"  # trust region, post-v0


def _shape(degree: int | None) -> int:
    """Rank a polynomial degree as linear, quadratic, or general nonlinear.

    Args:
        degree: The value returned by ``expr.polynomial_degree()`` — an integer
            degree, or ``None`` for a general (non-polynomial) expression.

    Returns:
        One of ``_LINEAR`` (degree 0 or 1), ``_QUADRATIC`` (degree 2), or
        ``_NONLINEAR`` (``None`` or degree > 2).
    """
    if degree is None:
        return _NONLINEAR
    if degree <= 1:
        return _LINEAR
    if degree == 2:
        return _QUADRATIC
    return _NONLINEAR


def classify(model: pyo.ConcreteModel) -> ProblemClass:
    """Classify a Pyomo model into a :class:`ProblemClass`.

    Only **active** constraints and the active objective are considered;
    inactive constraints, objectives, and blocks are ignored. A variable counts
    as discrete only if it is binary or integer **and not fixed** — so an LP
    with fixed binaries classifies ``LP``. Note that a quadratic *constraint*
    classifies ``NLP``, not ``QP``: only a quadratic *objective* over
    all-linear constraints is ``QP`` (smallest choice consistent with
    the QP capability matrix).

    Args:
        model: A constructed Pyomo model to classify.

    Returns:
        The problem class. Never returns ``MINLP_OA``/``MINLP_TR`` (reserved).

    Example:
        >>> import pyomo.environ as pyo
        >>> from flexcore.solvers import classify, ProblemClass
        >>> m = pyo.ConcreteModel()
        >>> m.x = pyo.Var()
        >>> m.c = pyo.Constraint(expr=m.x <= 1)
        >>> m.obj = pyo.Objective(expr=m.x)
        >>> classify(m) is ProblemClass.LP
        True
    """
    constraint_shape = _LINEAR
    objective_shape = _LINEAR
    has_discrete = False

    def _scan(expr) -> None:
        nonlocal has_discrete
        for var in identify_variables(expr, include_fixed=True):
            if (var.is_binary() or var.is_integer()) and not var.fixed:
                has_discrete = True

    for con in model.component_data_objects(
        pyo.Constraint, active=True, descend_into=True
    ):
        constraint_shape = max(constraint_shape, _shape(con.body.polynomial_degree()))
        _scan(con.body)

    for obj in model.component_data_objects(
        pyo.Objective, active=True, descend_into=True
    ):
        objective_shape = max(objective_shape, _shape(obj.expr.polynomial_degree()))
        _scan(obj.expr)

    all_linear = constraint_shape == _LINEAR and objective_shape == _LINEAR

    if has_discrete:
        return ProblemClass.MILP if all_linear else ProblemClass.MINLP
    if all_linear:
        return ProblemClass.LP
    if constraint_shape == _LINEAR and objective_shape == _QUADRATIC:
        return ProblemClass.QP
    return ProblemClass.NLP
