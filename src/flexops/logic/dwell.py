"""Optional UC piece: continuous set-point dwell.

Holds a **continuous** process variable (inlet flow, temperature, or any other
time-indexed Var) steady for a minimum duration -- unrelated to the binary
minimum-uptime/downtime logic in
:func:`~flexops.logic.unit_commitment.add_startup_shutdown`, which is a
separate, UC-specific concept. This is a fixed-grid piecewise-constant
constraint: the horizon is partitioned into blocks of ``k`` steps, and the
variable may only change value at block boundaries.
"""

from typing import Any

import pyomo.environ as pyo

from flexcore.exceptions import FlexConfigError
from flexops.logic._records import record
from flexops.logic.status import RollingStateKind, _register_rolling_state


def add_dwell(var: pyo.Var, k: int) -> Any:
    """Hold a continuous, time-indexed Var constant within fixed ``k``-step blocks.

    The horizon is partitioned into blocks starting at ``t = 0, k, 2k, ...``;
    within each block every point is constrained equal to its predecessor, so
    the variable can only change value at a block boundary. The unit is taken
    from ``var.parent_block()`` (the block ``var`` is already attached to).

    Args:
        var: A continuous, time-indexed ``Var`` to hold steady within each
            block (e.g. an inlet flow or temperature set point).
        k: Block length in steps. Must be ``>= 1``.

    Returns:
        The attached ``{var.local_name}_dwell`` Constraint, indexed over every
        ``t`` that is not a block boundary. ``None`` if ``k == 1`` (every
        point is its own block, so no constraint is built).

    Raises:
        FlexConfigError: If ``k < 1``.
    """
    if k < 1:
        raise FlexConfigError(
            f"add_dwell requires k >= 1, got {k}.", field="k", value=k
        )
    unit = var.parent_block()
    record(unit, "dwell", variable=var.local_name, k=k)
    if k == 1:
        return None

    tb = unit._find_time_block()
    idx = [t for t in tb.time_index if t % k != 0]
    name = f"{var.local_name}_dwell"

    def _rule(_b, t):
        return var[t] == var[t - 1]

    unit.add_component(
        name,
        pyo.Constraint(
            idx,
            rule=_rule,
            doc=f"Hold {var.local_name} constant within each {k}-step block.",
        ),
    )
    constraint = unit.find_component(name)
    _register_rolling_state(unit, var, k, RollingStateKind.DWELL)
    return constraint
