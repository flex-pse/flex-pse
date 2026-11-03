"""Checks a model can run before it reaches a solver."""

import heapq

import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn


def find_redundant_equalities(model: pyo.ConcreteModel, *, tol: float = 1e-9) -> list:
    """Return the active linear equality rows that earlier rows already imply.

    Nonlinear rows are skipped, so a dependency that runs through one isn't found.
    Fixed variables count as constants, and a row with no free variable is ignored.

    Args:
        model: The model to check.
        tol: Coefficients below this size, after scaling a row, count as zero.

    Returns:
        The redundant constraint data objects, in model order.
    """
    pivots: dict[int, dict[int, float]] = {}
    positions: dict[int, int] = {}
    redundant = []
    for con in model.component_data_objects(
        pyo.Constraint, active=True, descend_into=True
    ):
        if not con.equality:
            continue
        repn = generate_standard_repn(con.body, compute_values=True, quadratic=False)
        if not repn.is_linear() or not repn.linear_vars:
            continue
        scale = max(abs(coef) for coef in repn.linear_coefs)
        row: dict[int, float] = {}
        for coef, var in zip(repn.linear_coefs, repn.linear_vars, strict=True):
            column = positions.setdefault(id(var), len(positions))
            row[column] = row.get(column, 0.0) + coef / scale
        # eliminate on the lowest column until the row is a new pivot or empty
        columns = list(row)
        heapq.heapify(columns)
        while columns:
            column = heapq.heappop(columns)
            value = row.get(column, 0.0)
            if abs(value) < tol:
                row.pop(column, None)
            elif column in pivots:
                pivot = pivots[column]
                factor = value / pivot[column]
                for key, entry in pivot.items():
                    if key not in row:
                        heapq.heappush(columns, key)
                        row[key] = 0.0
                    row[key] -= factor * entry
                del row[column]
            else:
                pivots[column] = row
                break
        if not row:
            redundant.append(con)
    return redundant
