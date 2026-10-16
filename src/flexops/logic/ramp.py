"""Optional UC piece: ramp-rate limiting on a time-indexed Var.

Bounds how fast a continuous process variable may change between time
points, either step-to-step or as a net change allowed over a multi-step
window (the "N-minute ramp rate" pattern). The ramped quantity is built as
its own Var with a defining constraint (``tracked[t] == var[t]``, an
identity today) rather than inlining ``var[t]`` directly into the ramp
constraints, so a future extension can redefine that one constraint's
right-hand side (a windowed average, or a surrogate/composite expression of
several Vars) without touching the ramp constraints themselves.
"""

from typing import Any

import pyomo.environ as pyo
from pyomo.environ import units as pyunits

from flexcore.exceptions import FlexConfigError
from flexcore.logger import get_logger
from flexops.core.serialize import units_to_str
from flexops.logic._records import record

_log = get_logger(__name__)
_RAMP_VARS_ATTR = "_flexops_ramp_vars"


def add_ramp_rate(
    var: pyo.Var,
    *,
    name: str | None = None,
    ramp_up: float | None = None,
    ramp_down: float | None = None,
    ramp_rate_units: Any = None,
    window: int = 1,
) -> tuple[pyo.Var, pyo.Constraint | None, pyo.Constraint | None]:
    """Attach a ramp-rate limit to a time-indexed Var.

    Builds a tracked Var (``name``, identical to ``var[t]`` via a defining
    constraint ``<name>_def``) and, for each direction requested, a
    Constraint bounding how much ``<name>`` may change between ``t`` and
    ``t - window``:

    * ``<name>_ramp_up[t]``: ``tracked[t] - tracked[t - window] <= ramp_up
      * ramp_rate_units * window * dt``.
    * ``<name>_ramp_down[t]``: ``tracked[t - window] - tracked[t] <=
      ramp_down * ramp_rate_units * window * dt``.

    Every ``t < window`` is skipped: the horizon carries no pre-horizon
    history to ramp against (mirroring ``add_startup_shutdown``'s untied
    ``t = 0``).

    Attaching more than one ramp to the same ``var`` (e.g. a fast
    step-to-step check plus a slower windowed one) is allowed, but is more
    often an accidental duplicate call, so it is logged as a warning.

    Args:
        var: The time-indexed Var to ramp-limit. The unit block is inferred
            from ``var.parent_block()``.
        name: Component-name stem. Defaults to ``f"{var.local_name}_ramp"``.
        ramp_up: Maximum allowed increase, in ``ramp_rate_units``, over
            ``window`` steps. ``None`` to leave increases unconstrained.
        ramp_down: Maximum allowed decrease, in ``ramp_rate_units``, over
            ``window`` steps. ``None`` to leave decreases unconstrained.
        ramp_rate_units: Pyomo unit expression (e.g. ``pyunits.kW /
            pyunits.hr``) that ``ramp_up``/``ramp_down`` are given in.
            Required whenever either is given.
        window: Number of steps the ramp is measured across. Must be
            ``>= 1``.

    Returns:
        ``(tracked_var, ramp_up_constraint, ramp_down_constraint)``. The
        unused direction's constraint is ``None``.

    Raises:
        FlexConfigError: If ``window < 1``; if both ``ramp_up`` and
            ``ramp_down`` are ``None``; if either is negative; or if either
            is given without ``ramp_rate_units``.
    """
    if window < 1:
        raise FlexConfigError(
            f"add_ramp_rate requires window >= 1, got {window}.",
            field="window",
            value=window,
        )
    if ramp_up is None and ramp_down is None:
        raise FlexConfigError(
            "add_ramp_rate requires at least one of ramp_up/ramp_down.",
            field="ramp_up",
            value=None,
        )
    for field, value in (("ramp_up", ramp_up), ("ramp_down", ramp_down)):
        if value is None:
            continue
        if value < 0:
            raise FlexConfigError(
                f"add_ramp_rate requires {field} >= 0, got {value}.",
                field=field,
                value=value,
            )
        if ramp_rate_units is None:
            raise FlexConfigError(
                f"add_ramp_rate requires ramp_rate_units when {field} is given.",
                field="ramp_rate_units",
                value=None,
            )

    unit = var.parent_block()
    record(
        unit,
        "ramp",
        variable=var.local_name,
        ramp_up=ramp_up,
        ramp_down=ramp_down,
        # get_units writes kW/h where parse_units' result prints kW*h**(-1).
        units=units_to_str(
            pyunits.get_units(ramp_rate_units), where="add_ramp_rate.ramp_rate_units"
        ),
        window=window,
        component_name=name,
    )
    tb = unit._find_time_block()
    var_units = pyunits.get_units(var)
    if name is None:
        name = f"{var.local_name}_ramp"

    ramp_vars = getattr(unit, _RAMP_VARS_ATTR, None)
    if ramp_vars is None:
        ramp_vars = []
        setattr(unit, _RAMP_VARS_ATTR, ramp_vars)
    if any(tracked_var is var for tracked_var in ramp_vars):
        _log.warning(
            "add_ramp_rate: %s already has a ramp rate attached; adding "
            "another as %r.",
            var.name,
            name,
        )
    ramp_vars.append(var)

    unit.add_component(
        name,
        pyo.Var(
            tb.time_index,
            units=var_units,
            doc=f"Ramp-tracked quantity (currently identical to "
            f"{var.local_name}[t]).",
        ),
    )
    tracked = unit.find_component(name)

    def _def_rule(_b, t):
        return tracked[t] == var[t]

    unit.add_component(
        f"{name}_def",
        pyo.Constraint(
            tb.time_index,
            rule=_def_rule,
            doc=f"Defines {name}[t] == {var.local_name}[t].",
        ),
    )

    ramp_index = [t for t in tb.time_index if t >= window]

    ramp_up_constraint = None
    if ramp_up is not None:

        def _ramp_up_rule(_b, t):
            limit = pyunits.convert(
                ramp_up * ramp_rate_units * window * tb.dt, to_units=var_units
            )
            return tracked[t] - tracked[t - window] <= limit

        unit.add_component(
            f"{name}_ramp_up",
            pyo.Constraint(
                ramp_index,
                rule=_ramp_up_rule,
                doc=f"Ramp-up limit ({window}-step window): {name}[t] - "
                f"{name}[t - {window}] <= ramp_up * window * dt.",
            ),
        )
        ramp_up_constraint = unit.find_component(f"{name}_ramp_up")

    ramp_down_constraint = None
    if ramp_down is not None:

        def _ramp_down_rule(_b, t):
            limit = pyunits.convert(
                ramp_down * ramp_rate_units * window * tb.dt, to_units=var_units
            )
            return tracked[t - window] - tracked[t] <= limit

        unit.add_component(
            f"{name}_ramp_down",
            pyo.Constraint(
                ramp_index,
                rule=_ramp_down_rule,
                doc=f"Ramp-down limit ({window}-step window): "
                f"{name}[t - {window}] - {name}[t] <= ramp_down * window * dt.",
            ),
        )
        ramp_down_constraint = unit.find_component(f"{name}_ramp_down")

    return tracked, ramp_up_constraint, ramp_down_constraint
