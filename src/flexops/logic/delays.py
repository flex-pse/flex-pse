"""Optional UC piece: startup/response delay tied to an upstream unit.

The single-hop primitive behind the chemical-stabilization delay *chains* of
Rao et al. 2024: a unit may not start (or be on) until ``k`` steps after an
upstream unit's status. Full chain templates composing several of these
single-hop primitives are planned future work but not yet implemented.
"""

from typing import Any

import pyomo.environ as pyo

from flexcore.exceptions import FlexConfigError
from flexops.logic._records import record
from flexops.logic.status import (
    RollingStateKind,
    _register_rolling_state,
    require_binary_status,
)


def add_startup_delay(
    unit: Any, upstream: Any, k: int, initially_on: bool = True
) -> pyo.Constraint:
    """Tie ``unit``'s on/off status to an upstream unit's status, delayed by ``k``.

    ``unit`` may be on at ``t`` only if ``upstream`` was on continuously
    across the trailing window ``[t - k, t]`` -- one condition that covers
    both the ``k``-step delay and "upstream off implies downstream off".
    Disaggregated over each lag in that window (``unit.status[t] <=
    upstream.status[t - lag]`` for every ``lag`` in ``0..k``) rather than an
    aggregate sum, because it is tighter under the LP relaxation used by
    :func:`~flexops.logic.status.relax`.

    Args:
        unit: The downstream unit block; must already carry a ``status`` Var
            (via :func:`~flexops.logic.status.add_status`).
        upstream: The upstream unit block whose ``status`` Var gates ``unit``.
        k: Number of steps the downstream start is delayed behind the upstream.
        initially_on: How to handle a window that reaches before ``t = 0``.
            If ``True`` (default), the window clamps to ``[max(t - k, 0), t]``
            -- a unit already running at ``t = 0`` counts as warmed up, the
            right convention for a standalone horizon. If ``False``, the
            window is only checked for ``t >= k`` and ``unit.status[t] == 0``
            is forced for ``t < k`` instead -- the right convention for a
            rolling-horizon window with no assumed prior state.

    Registers ``unit``'s ``status`` Var as rolling-horizon state (trailing
    ``k`` steps) via ``flexops.logic.status._register_rolling_state`` --
    consuming that registry is the rolling-horizon scheduler's job, not built
    here.

    Returns:
        The attached ``startup_delay`` Constraint.

    Raises:
        FlexConfigError: If ``upstream`` has no ``status`` Var, or either unit
            counts more than one train (``max_count > 1``).
    """
    if not hasattr(upstream, "status"):
        raise FlexConfigError(
            f"add_startup_delay requires the upstream unit {upstream.name!r} to "
            "carry a status Var; attach one via add_status first.",
            field="upstream",
            value=upstream.name,
        )
    for member in (unit, upstream):
        require_binary_status(member, used_by="add_startup_delay")
    record(
        unit.model(),
        "startup_delay",
        unit=unit.getname(fully_qualified=True),
        upstream=upstream.getname(fully_qualified=True),
        k=k,
        initially_on=initially_on,
    )
    down_status = unit.status
    up_status = upstream.status
    tb = unit._find_time_block()

    if initially_on:
        index = [(t, lag) for t in tb.time_index for lag in range(min(k, t) + 1)]
    else:
        index = [(t, lag) for t in tb.time_index if t >= k for lag in range(k + 1)]

    @unit.Constraint(
        index,
        doc=f"Startup delay ({k} steps behind {upstream.name}): status[t] <= "
        "upstream.status[t-lag] for lag in the trailing k-step window.",
    )
    def startup_delay(_b, t, lag):
        return down_status[t] <= up_status[t - lag]

    if not initially_on:

        @unit.Constraint(
            [t for t in tb.time_index if t < k],
            doc=f"Startup delay ({k} steps behind {upstream.name}): status[t]==0 "
            "for t<k (no upstream history available).",
        )
        def startup_delay_cold(_b, t):
            return down_status[t] == 0

    _register_rolling_state(unit, down_status, k, RollingStateKind.STARTUP_DELAY)
    return unit.startup_delay
