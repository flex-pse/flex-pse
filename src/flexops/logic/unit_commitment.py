"""Optional UC piece: transition + minimum uptime/downtime logic.

Ties a ``startup[t]``/``shutdown[t]`` binary pair to the rising/falling edges of
a unit's ``status[t]`` binary, and (optionally) enforces a minimum number of
steps the unit must stay up/down once it transitions -- the Rajan-Takriti
rolling-sum formulation. A unit's ``unit_commitment`` config enables this only
when transition bookkeeping (e.g. startup cost, startup-delay chains) or
minimum dwell-time enforcement is needed.

Note: this "minimum uptime/downtime" is a UC-modeling concept (how long a
status must hold once it changes) and is unrelated to the continuous-variable
:func:`~flexops.logic.dwell.add_dwell` (holding a process variable like flow or
temperature steady) -- the two share no code and are conceptually orthogonal.
"""

from typing import Any

import pyomo.environ as pyo

from flexcore.exceptions import FlexConfigError
from flexops.logic._records import record
from flexops.logic.status import (
    RollingStateKind,
    _register_rolling_state,
    _track_binary,
    status_max_count,
)


def add_startup_shutdown(
    unit: Any,
    status_var: pyo.Var,
    min_uptime: int = 1,
    min_downtime: int = 1,
) -> tuple[pyo.Var, pyo.Var]:
    """Attach ``startup[t]``/``shutdown[t]`` binaries and transition logic.

    For each ``t >= 1``, adds:

    * ``transition[t]``: ``status_var[t] - status_var[t-1] == startup[t] -
      shutdown[t]``.
    * ``min_uptime[t]``: ``sum(startup[i] for i in window) <= status_var[t]``,
      window ``= range(max(1, t - min_uptime + 1), t + 1)``. A startup within
      the trailing ``min_uptime`` steps forces the unit on at ``t``.
    * ``min_downtime[t]``: the mirror, ``sum(shutdown[i] for i in window) <=
      n - status_var[t]``, window sized by ``min_downtime``, where ``n`` is the
      unit's ``max_count`` (``1`` for a binary status). With ``n > 1`` the
      trains are interchangeable: a recent startup keeps the count up, so an
      older train may stop while the new one runs.

    At the defaults (``min_uptime=1, min_downtime=1``) each window is the
    single point ``{t}``, so these reduce exactly to the tightening bounds
    ``startup[t] <= status_var[t]`` and ``shutdown[t] <= 1 - status_var[t]``.
    Summing those two implies ``startup[t] + shutdown[t] <= 1`` directly (even
    under a continuous relaxation of ``startup``/``shutdown``), so no separate
    exclusivity constraint is needed.

    No horizon-end truncation is needed -- each window's upper bound is always
    ``t``, which is always in range; only the lower bound is clamped to 1.

    ``t = 0`` is left unconstrained in v0: there is no initial-status Param to
    reference a transition *into* the horizon's first point.

    Args:
        unit: The unit block to attach the transition structure to.
        status_var: The unit's time-indexed ``status`` Var from
            :func:`~flexops.logic.status.add_status`. Binary, or a count when
            the unit was built with ``max_count > 1``.
        min_uptime: Minimum number of steps the unit must stay on after a
            startup. Must be ``>= 1``; ``1`` means no additional requirement
            beyond the instant the unit turns on.
        min_downtime: Minimum number of steps the unit must stay off after a
            shutdown. Must be ``>= 1``, with the same "no-op at 1" meaning.

    Returns:
        The attached ``(startup, shutdown)`` Vars, each indexed over
        ``t >= 1``. Binary, or the number of trains started or stopped (an
        integer in ``[0, max_count]``) for a counted unit.

    Raises:
        FlexConfigError: If ``min_uptime`` or ``min_downtime`` is ``< 1``.
    """
    if min_uptime < 1:
        raise FlexConfigError(
            f"add_startup_shutdown requires min_uptime >= 1, got {min_uptime}.",
            field="min_uptime",
            value=min_uptime,
        )
    if min_downtime < 1:
        raise FlexConfigError(
            f"add_startup_shutdown requires min_downtime >= 1, got {min_downtime}.",
            field="min_downtime",
            value=min_downtime,
        )
    record(unit, "startup_shutdown", min_up=min_uptime, min_down=min_downtime)

    tb = unit._find_time_block()
    active_index = [t for t in tb.time_index if t >= 1]
    train_count = status_max_count(unit)

    if train_count == 1:
        unit.startup = pyo.Var(
            active_index, domain=pyo.Binary, doc="Startup indicator: 1 at a 0->1 edge."
        )
        unit.shutdown = pyo.Var(
            active_index,
            domain=pyo.Binary,
            doc="Shutdown indicator: 1 at a 1->0 edge.",
        )
    else:
        unit.startup = pyo.Var(
            active_index,
            domain=pyo.NonNegativeIntegers,
            bounds=(0, train_count),
            doc="Number of trains started at t.",
        )
        unit.shutdown = pyo.Var(
            active_index,
            domain=pyo.NonNegativeIntegers,
            bounds=(0, train_count),
            doc="Number of trains stopped at t.",
        )
    startup, shutdown = unit.startup, unit.shutdown

    @unit.Constraint(
        active_index,
        doc="status[t]-status[t-1] == startup[t]-shutdown[t]; t=0 is "
        "unconstrained (no initial-status Param in v0).",
    )
    def transition(_b, t):
        return status_var[t] - status_var[t - 1] == startup[t] - shutdown[t]

    # Captured under different names than the constraints below -- the
    # `@unit.Constraint` idiom binds the rule function's own name (e.g.
    # `min_uptime`) as the attribute on `unit`, which would otherwise shadow
    # the `min_uptime`/`min_downtime` parameters inside their own rule bodies.
    uptime_window_size = min_uptime
    downtime_window_size = min_downtime

    @unit.Constraint(
        active_index,
        doc=f"Minimum uptime ({min_uptime} steps): a startup within the "
        "trailing window forces status[t]==1.",
    )
    def min_uptime(_b, t):
        window = range(max(1, t - uptime_window_size + 1), t + 1)
        return sum(startup[i] for i in window) <= status_var[t]

    @unit.Constraint(
        active_index,
        doc=f"Minimum downtime ({min_downtime} steps): a shutdown within the "
        "trailing window forces status[t]==0.",
    )
    def min_downtime(_b, t):
        window = range(max(1, t - downtime_window_size + 1), t + 1)
        return sum(shutdown[i] for i in window) <= train_count - status_var[t]

    if train_count == 1:
        _track_binary(unit, startup)
        _track_binary(unit, shutdown)
    else:
        for transition_var in (startup, shutdown):
            _track_binary(
                unit,
                transition_var,
                discrete=pyo.NonNegativeIntegers,
                relaxed=pyo.NonNegativeReals,
            )

    if uptime_window_size > 1:
        _register_rolling_state(
            unit, startup, uptime_window_size, RollingStateKind.MIN_UPTIME
        )
    if downtime_window_size > 1:
        _register_rolling_state(
            unit, shutdown, downtime_window_size, RollingStateKind.MIN_DOWNTIME
        )

    return startup, shutdown
