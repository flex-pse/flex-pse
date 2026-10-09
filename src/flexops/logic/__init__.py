"""Customizable logic constraint layer.

A composable set of optional constraint pieces applied per unit via its
``unit_commitment`` config: :func:`add_status` is the base (present whenever a
unit can be shut off); :func:`add_startup_shutdown` (transition logic plus
built-in minimum uptime/downtime, via its ``min_uptime``/``min_downtime``
args), :func:`add_dwell`, :func:`add_startup_delay`, :func:`add_conditional`,
and :func:`add_bypass` are all optional. :func:`register_parallel_group` is
**model-level** rather than a per-unit method: a unit cannot see its
siblings, so a caller declares the group of interchangeable/hierarchically
related units itself. The group, listed in priority order, is ordered
descending along that list -- its first-listed unit is the first one on -- for
both the units' status and any continuous Vars the caller names. Its
``order_status=False`` mode orders those Vars alone, which is how a group whose
units carry no ``status`` Var is registered.

Note: :func:`add_dwell` is a distinct, unrelated concept from
``add_startup_shutdown``'s minimum uptime/downtime -- it holds a **continuous**
process variable steady, not a unit's on/off status. Any piece that creates
state needing rolling-horizon carry-over (minimum uptime/downtime, dwell, or a
startup delay) registers it via
:func:`~flexops.logic.status._register_rolling_state`; a rolling-horizon
scheduling driver that will consume that registry is planned but not yet
implemented.

:func:`add_ramp_rate` is a further optional piece: it bounds how fast a
continuous Var may change, step-to-step or over a multi-step window.
:func:`add_degradation` is its soft counterpart: it prices wear drivers
(variation, off-design operation, throughput) instead of forbidding them.
"""

from flexops.logic.bypass import add_bypass
from flexops.logic.conditional import add_conditional
from flexops.logic.degeneracy import register_parallel_group
from flexops.logic.degradation import add_degradation
from flexops.logic.delays import add_startup_delay
from flexops.logic.dwell import add_dwell
from flexops.logic.ramp import add_ramp_rate
from flexops.logic.status import add_status, relax, unrelax
from flexops.logic.unit_commitment import add_startup_shutdown

__all__ = [
    "add_status",
    "relax",
    "unrelax",
    "add_startup_shutdown",
    "add_dwell",
    "add_startup_delay",
    "add_conditional",
    "register_parallel_group",
    "add_bypass",
    "add_ramp_rate",
    "add_degradation",
]
