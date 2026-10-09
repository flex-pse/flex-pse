"""UC status base: ``add_status`` + ``relax``/``unrelax``.

``add_status`` is the base unit-commitment piece, present on **every unit that
can be shut off** (a ``Tank`` disables it). It attaches a Binary
``status[t]`` and the two semicontinuous links tying an output variable to that
binary. ``relax``/``unrelax`` toggle the **domain** of every logic-attached
Binary in place -- a first-class domain switch on the live model, never a
rebuild (they touch no constraints, bounds, or fixed values).

Every optional UC piece (startup/shutdown, dwell, ...) appends the binaries it
attaches to a private list on the unit, ``_flexops_logic_binaries``, so a single
``relax``/``unrelax`` call covers all of them.

**Relaxation policy.** The per-unit ``relaxation`` config
(:class:`~flexops.core.ops_block.RelaxationPolicy`, values ``"exact"`` /
``"relaxed"``) both *permit* relaxation in v0, so ``relax`` never refuses.
A hard "never relax" policy is planned future work but not yet implemented:
it would require adding a new ``RelaxationPolicy`` value.
"""

import enum
from typing import Any

import pyomo.environ as pyo

_BINARIES_ATTR = "_flexops_logic_binaries"
_ROLLING_STATE_ATTR = "_flexops_rolling_state"


class RollingStateKind(enum.StrEnum):
    """Why a Var's trailing history matters to the rolling-horizon driver."""

    MIN_UPTIME = "min_uptime"
    MIN_DOWNTIME = "min_downtime"
    DWELL = "dwell"
    STARTUP_DELAY = "startup_delay"
    DEGRADATION = "degradation"


def _register_rolling_state(
    unit: Any, var: pyo.Var, k: int, kind: RollingStateKind
) -> None:
    """Record a Var whose trailing ``k``-step history crosses rolling-horizon windows.

    Registration only: a rolling-horizon scheduling driver that will read a
    solved window's trailing values and fix the next window's initial state
    from them is planned but not yet implemented; that consumption logic is
    out of scope here.

    Args:
        unit: The unit block the Var lives on.
        var: The time-indexed Var whose recent history matters (e.g.
            ``startup``, ``shutdown``, a continuous dwell Var, or a delayed
            unit's ``status``).
        k: The window length driving how much history must be carried
            (min-uptime/downtime steps, dwell block size, or delay steps).
        kind: Why this Var is tracked.
    """
    entries = getattr(unit, _ROLLING_STATE_ATTR, None)
    if entries is None:
        entries = []
        setattr(unit, _ROLLING_STATE_ATTR, entries)
    entries.append({"var": var, "k": k, "kind": kind})


def _track_binary(unit: Any, binary: pyo.Var) -> None:
    """Append ``binary`` to the unit's private logic-binary registry.

    The registry is a plain Python list (never a Pyomo component), created
    lazily on first use so any UC piece may be the first applied to a unit.

    Args:
        unit: The unit block the binary lives on.
        binary: The Binary ``Var`` to track for ``relax``/``unrelax``.
    """
    binaries = getattr(unit, _BINARIES_ATTR, None)
    if binaries is None:
        binaries = []
        setattr(unit, _BINARIES_ATTR, binaries)
    binaries.append(binary)


def _set_domain(binary: pyo.Var, domain: Any) -> None:
    """Set the domain of every data object of a (possibly indexed) Binary Var."""
    for data in binary.values() if binary.is_indexed() else (binary,):
        data.domain = domain


def add_status(
    unit: Any, output_var: pyo.Var, min_output: Any, max_output: Any
) -> pyo.Var:
    """Attach the on/off status binary and its semicontinuous links to a unit.

    Adds ``status[t]`` (Binary, indexed by the unit's time points) plus two
    Constraints linking ``output_var`` to it:

    * ``status_min_link[t]``: ``min_output * status[t] <= output_var[t]``
    * ``status_max_link[t]``: ``output_var[t] <= max_output * status[t]``

    Together they force ``output_var[t] == 0`` when the unit is off
    (``status[t] == 0``) and ``min_output <= output_var[t] <= max_output`` when
    it is on. The status binary is tracked so ``relax``/``unrelax`` cover it.
    Call at most once per unit (v0 uses the fixed component names ``status`` /
    ``status_min_link`` / ``status_max_link``).

    Args:
        unit: The unit block to attach the status structure to.
        output_var: The time-indexed output ``Var`` (e.g. a power draw) gated by
            the status binary.
        min_output: Minimum output when on, a units-carrying quantity matching
            ``output_var`` (e.g. ``0 * pyunits.kW``).
        max_output: Maximum output when on, a units-carrying quantity matching
            ``output_var``.

    Returns:
        The attached Binary ``status`` Var.
    """
    tb = unit._find_time_block()
    unit.status = pyo.Var(
        tb.time_index, domain=pyo.Binary, doc="On/off status binary (1 = on)."
    )
    status = unit.status

    @unit.Constraint(
        tb.time_index,
        doc="Semicontinuous lower link: output_var >= min_output when on, else 0.",
    )
    def status_min_link(_b, t):
        return min_output * status[t] <= output_var[t]

    @unit.Constraint(
        tb.time_index,
        doc="Semicontinuous upper link: output_var <= max_output when on, else 0.",
    )
    def status_max_link(_b, t):
        return output_var[t] <= max_output * status[t]

    _track_binary(unit, status)
    return status


def relax(unit: Any) -> None:
    """Switch every logic-attached Binary on ``unit`` to ``pyo.UnitInterval``.

    A first-class, live-model domain switch (LP relaxation of the unit's
    discrete structure): it changes only variable domains -- never constraints,
    bounds, or fixed values -- so warm starts and references held elsewhere
    (costing, aggregation) survive. A no-op if the unit has no tracked binaries.

    Args:
        unit: The unit whose tracked binaries to relax.
    """
    for binary in getattr(unit, _BINARIES_ATTR, []):
        _set_domain(binary, pyo.UnitInterval)


def unrelax(unit: Any) -> None:
    """Switch every logic-attached Binary on ``unit`` back to ``pyo.Binary``.

    The exact inverse of :func:`relax` (domain-only). A no-op if the unit has no
    tracked binaries.

    Args:
        unit: The unit whose tracked binaries to restore to integrality.
    """
    for binary in getattr(unit, _BINARIES_ATTR, []):
        _set_domain(binary, pyo.Binary)
