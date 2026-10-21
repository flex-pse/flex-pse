"""Optional UC piece: conditional status implications between two units.

Linear implications on two units' status Binaries: "if x is on then y is on"
or "if x is on then y is off".
"""

from typing import Any

import pyomo.environ as pyo

from flexcore.exceptions import FlexConfigError
from flexops.logic._records import record
from flexops.logic.status import require_binary_status

_ALLOWED_THEN = ("on", "off")


def add_conditional(x_unit: Any, y_unit: Any, *, then: str = "on") -> pyo.Constraint:
    """Attach a conditional status implication between two units, on ``x_unit``.

    * ``then="on"``: ``y_unit.status[t] >= x_unit.status[t]`` (x on implies y on).
    * ``then="off"``: ``y_unit.status[t] <= 1 - x_unit.status[t]`` (x on implies
      y off).

    Args:
        x_unit: The "if" unit; must carry a ``status`` Var. The constraint is
            attached to this unit as ``conditional``.
        y_unit: The "then" unit; must carry a ``status`` Var.
        then: ``"on"`` or ``"off"``, the implied state of ``y_unit`` when
            ``x_unit`` is on.

    Returns:
        The attached ``conditional`` Constraint (on ``x_unit``).

    Raises:
        FlexConfigError: If ``then`` is not ``"on"``/``"off"``, or either unit
            lacks a ``status`` Var, or either counts more than one train
            (``max_count > 1``).
    """
    if then not in _ALLOWED_THEN:
        raise FlexConfigError(
            f"then must be one of {_ALLOWED_THEN!r}, got {then!r}.",
            field="then",
            value=then,
        )
    for label, unit in (("x_unit", x_unit), ("y_unit", y_unit)):
        if not hasattr(unit, "status"):
            raise FlexConfigError(
                f"add_conditional requires {label} ({unit.name!r}) to carry a "
                "status Var; attach one via add_status first.",
                field=label,
                value=unit.name,
            )
    for unit in (x_unit, y_unit):
        require_binary_status(unit, used_by="add_conditional")
    record(
        x_unit.model(),
        "conditional",
        unit=x_unit.getname(fully_qualified=True),
        implies=y_unit.getname(fully_qualified=True),
        state=then,
    )
    x_status, y_status = x_unit.status, y_unit.status
    tb = x_unit._find_time_block()

    @x_unit.Constraint(
        tb.time_index,
        doc=f"Conditional implication (then={then!r}): x on implies y "
        f"{'on' if then == 'on' else 'off'}.",
    )
    def conditional(_b, t):
        if then == "on":
            return y_status[t] >= x_status[t]
        return y_status[t] <= 1 - x_status[t]

    return x_unit.conditional
