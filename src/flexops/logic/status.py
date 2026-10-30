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

from flexcore.exceptions import FlexConfigError
from flexops.core.serialize import to_jsonable
from flexops.logic._records import record, variable_path

_BINARIES_ATTR = "_flexops_logic_binaries"
_MAX_COUNT_ATTR = "_flexops_status_max_count"
_ROLLING_STATE_ATTR = "_flexops_rolling_state"


class RollingStateKind(enum.StrEnum):
    """Why a Var's trailing history matters to the rolling-horizon driver."""

    MIN_UPTIME = "min_uptime"
    MIN_DOWNTIME = "min_downtime"
    DWELL = "dwell"
    STARTUP_DELAY = "startup_delay"
    DEGRADATION = "degradation"


def status_max_count(unit: Any) -> int:
    """Return how many trains ``unit``'s status counts, or 1 for a binary status.

    Args:
        unit: A unit block, with or without ``add_status`` applied.

    Returns:
        The ``max_count`` given to :func:`add_status`, or ``1``.
    """
    return getattr(unit, _MAX_COUNT_ATTR, 1)


def require_binary_status(unit: Any, *, used_by: str) -> None:
    """Raise if ``unit`` has a counted status, for logic that reads status as on/off.

    Args:
        unit: Unit whose status is about to be used.
        used_by: Name of the calling helper, used in the error message.

    Raises:
        FlexConfigError: If ``unit`` was built with ``max_count > 1``.
    """
    if status_max_count(unit) > 1:
        raise FlexConfigError(
            f"{used_by} treats status as on/off, but {unit.name!r} counts up to "
            f"{status_max_count(unit)} trains. Constrain a counted unit through "
            "a resource balance instead, or use max_count=1 and separate units.",
            field="max_count",
            value=status_max_count(unit),
        )


def require_no_intercept(
    unit: Any, relation_name: str, surrogate: Any, max_count: int
) -> None:
    """Raise if a unit counting several trains would carry a constant term.

    Args:
        unit: The unit block the relation is on.
        relation_name: Local name of the relation, used in the message.
        surrogate: The relation's :class:`~flexops.surrogates.base.Surrogate`.
        max_count: Number of trains the unit represents.

    Raises:
        FlexConfigError: If ``max_count > 1`` and ``surrogate.intercept`` is
            nonzero.
    """
    # TODO(#140): scale the intercept by the running count, then remove this guard
    if max_count == 1 or surrogate.intercept == 0.0:
        return
    raise FlexConfigError(
        f"{unit.name!r} counts {max_count} trains, but relation "
        f"{relation_name!r} has a constant term ({surrogate.intercept}). The "
        "constant would be counted once instead of once per running train. Fit "
        "the relation with an intercept of 0, or build one unit per train.",
        field="max_count",
        value=max_count,
    )


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


def _track_binary(
    unit: Any,
    binary: pyo.Var,
    *,
    discrete: Any = pyo.Binary,
    relaxed: Any = pyo.UnitInterval,
) -> None:
    """Append ``binary`` to the unit's private logic-binary registry.

    The registry is a plain Python list (never a Pyomo component), created
    lazily on first use so any UC piece may be the first applied to a unit.

    Args:
        unit: The unit block the binary lives on.
        binary: Discrete ``Var`` to track for ``relax``/``unrelax``.
        discrete: Domain ``unrelax`` restores. ``pyo.Binary`` by default.
        relaxed: Domain ``relax`` sets. ``pyo.UnitInterval`` by default. A
            count passes ``pyo.NonNegativeReals`` and keeps its ``(0, n)``
            bounds.
    """
    binaries = getattr(unit, _BINARIES_ATTR, None)
    if binaries is None:
        binaries = []
        setattr(unit, _BINARIES_ATTR, binaries)
    binaries.append((binary, discrete, relaxed))


def _set_domain(binary: pyo.Var, domain: Any) -> None:
    """Set the domain of every data object of a (possibly indexed) discrete Var."""
    # bounds are separate from the domain, so a count keeps its (0, n) bounds
    for data in binary.values() if binary.is_indexed() else (binary,):
        data.domain = domain


def add_status(
    unit: Any,
    output_var: pyo.Var,
    min_output: Any,
    max_output: Any,
    *,
    max_count: int = 1,
    min_count_on: int | None = None,
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
            ``output_var``. For a counted unit, the limit of one train.
        max_count: Number of identical trains the unit represents. ``1``
            (default) keeps ``status`` binary.
        min_count_on: Fewest trains that can run when any are running, from
            2 to ``max_count``. ``None`` (default) allows any count.

    Returns:
        The attached ``status`` Var. Binary, or an integer in
        ``[0, max_count]`` when ``max_count > 1``.

    Raises:
        FlexConfigError: If ``max_count < 1``, ``min_count_on`` is outside
            ``[2, max_count]``, or ``max_count > 1`` and a swapped relation on
            the unit has a constant term.
    """
    if max_count < 1:
        raise FlexConfigError(
            f"add_status requires max_count >= 1, got {max_count}.",
            field="max_count",
            value=max_count,
        )
    if min_count_on is not None and not 2 <= min_count_on <= max_count:
        raise FlexConfigError(
            f"add_status requires 2 <= min_count_on <= max_count ({max_count}), "
            f"got {min_count_on}. A minimum of 1 is the default and needs no "
            "setting.",
            field="min_count_on",
            value=min_count_on,
        )
    for relation in unit._io_registry.relations:
        if relation.surrogate is not None:
            require_no_intercept(unit, relation.name, relation.surrogate, max_count)
    # leave defaults out so existing specs emit unchanged
    counted_fields: dict[str, int] = {}
    if max_count > 1:
        counted_fields["max_count"] = max_count
    if min_count_on is not None:
        counted_fields["min_count_on"] = min_count_on
    record(
        unit,
        "status",
        variable=variable_path(output_var, unit),
        min_output=to_jsonable(min_output, where="add_status.min_output"),
        max_output=to_jsonable(max_output, where="add_status.max_output"),
        **counted_fields,
    )
    tb = unit._find_time_block()
    if max_count == 1:
        unit.status = pyo.Var(
            tb.time_index, domain=pyo.Binary, doc="On/off status binary (1 = on)."
        )
    else:
        unit.status = pyo.Var(
            tb.time_index,
            domain=pyo.NonNegativeIntegers,
            bounds=(0, max_count),
            doc=f"Number of trains running, an integer in 0..{max_count}.",
        )
    status = unit.status
    setattr(unit, _MAX_COUNT_ATTR, max_count)

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

    if max_count == 1:
        _track_binary(unit, status)
    else:
        # a relaxed count is a real in [0, n], the bounds stay on the Var
        _track_binary(
            unit,
            status,
            discrete=pyo.NonNegativeIntegers,
            relaxed=pyo.NonNegativeReals,
        )

    if min_count_on is not None:
        unit.status_on = pyo.Var(
            tb.time_index,
            domain=pyo.Binary,
            doc="1 when any train is running, else 0.",
        )
        status_on = unit.status_on
        minimum_running = min_count_on

        @unit.Constraint(
            tb.time_index,
            doc="If any train runs, at least min_count_on trains run.",
        )
        def status_on_min(_b, t):
            return minimum_running * status_on[t] <= status[t]

        @unit.Constraint(
            tb.time_index,
            doc="Any train running forces status_on to 1.",
        )
        def status_on_max(_b, t):
            return status[t] <= max_count * status_on[t]

        _track_binary(unit, status_on)
    return status


def relax(unit: Any) -> None:
    """Relax every logic-attached discrete Var on ``unit`` to a continuous domain.

    Args:
        unit: The unit whose tracked binaries to relax.
    """
    for var, _discrete, relaxed in getattr(unit, _BINARIES_ATTR, []):
        _set_domain(var, relaxed)


def unrelax(unit: Any) -> None:
    """Restore integrality of every logic-attached Var on ``unit``, undoing ``relax``.

    Args:
        unit: The unit whose tracked binaries to restore to integrality.
    """
    for var, discrete, _relaxed in getattr(unit, _BINARIES_ATTR, []):
        _set_domain(var, discrete)
