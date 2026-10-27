"""Shared helper code used by the boundary blocks.

:class:`~flexops.unit_models.base.feed.Feed` and
:class:`~flexops.unit_models.base.product.Product` both put a limit on a
time-indexed amount — a withdrawal that must stay under a supply limit, or a
delivery that must stay under a demand limit. Both need the same pieces: a
mutable limit ``Param`` and the ``Constraint`` that enforces it.

There are two ways to set a limit, and they mean different things — not just
two names for the same rule. On the **period** basis
(:func:`add_time_limits`) the limit is a *rate* that must hold in every
single period, like a permitted withdrawal rate or a pump's maximum
throughput. On the **horizon** basis (:func:`add_horizon_limits`) the limit
is a *total amount* over the whole time horizon, like a monthly permit or a
take-or-pay delivery contract, and the optimizer is free to choose how that
total gets spread out over time. A period limit is stricter than a horizon
limit (satisfying the first also satisfies the second), so the caller must
say which one is meant using :class:`LimitBasis`. flex-pse never guesses
this from the units, though it does check that the units match.

The limits are mutable ``Param``\\ s, not ``Var.setlb``/``setub``, for three
reasons: ``set_value`` is the approved way to update a value in place;
Params keep working even after
:meth:`~flexops.core.ops_block.OpsBlockData.set_external_dispatch` fixes the
bounded ``Var``; and the ``Constraint`` that references the ``Param`` carries
a dual, so after solving you can read the shadow price of a resource limit.
"""

import enum
from collections.abc import Sized

import pyomo.environ as pyo
from pyomo.core.base.units_container import UnitsError
from pyomo.environ import units as pyunits

from flexcore.exceptions import FlexConfigError

_SENSES = {
    "min": ("lower", "at least"),
    "max": ("upper", "at most"),
}


class LimitBasis(enum.StrEnum):
    """Whether a boundary limit applies every period or to the horizon total."""

    PERIOD = "period"
    HORIZON = "horizon"


def resolve_basis(value, field: str) -> LimitBasis:
    """Turn a configured basis value into a :class:`LimitBasis`, or raise an error.

    This is called from ``build()`` instead of being set up as a
    ``ConfigValue`` domain, because Pyomo turns every error raised inside a
    domain into a plain ``ValueError`` and throws away the ``field``/``value``
    details we want to keep.

    Args:
        value: The configured basis, as a string or a :class:`LimitBasis`.
        field: The config key it came from, named in the error message.

    Returns:
        The matching :class:`LimitBasis`.

    Raises:
        FlexConfigError: If ``value`` doesn't match any declared basis.
    """
    try:
        return LimitBasis(value)
    except ValueError as exc:
        allowed = ", ".join(repr(basis.value) for basis in LimitBasis)
        raise FlexConfigError(
            f"{field} must be one of {allowed}, got {value!r}.",
            field=field,
            value=value,
        ) from exc


def _convert_limit(bound, units, field: str) -> float:
    """Convert ``bound`` to a plain number in ``units``, or raise an error.

    Args:
        bound: The limit as configured, carrying its own units.
        units: The units the component being built uses.
        field: The config key ``bound`` came from, named in the error message.

    Returns:
        The converted number.

    Raises:
        FlexConfigError: If ``bound`` can't be converted to ``units``. This
            happens when the limit's dimension doesn't match the basis it
            was declared under — for example, a rate given where a total
            was expected.
    """
    try:
        return pyo.value(pyunits.convert(bound, units))
    except UnitsError as exc:
        raise FlexConfigError(
            f"{field}={bound!r} does not convert to {units}, so it cannot bound "
            "the quantity it was given for. Check the limit against the "
            "declared basis: a period-basis limit is a rate (e.g. m**3/hr) and "
            "a horizon-basis limit is a quantity (e.g. m**3).",
            field=field,
            value=bound,
        ) from exc


def _period_limit_values(bound, units, field: str, time_index):
    """Convert a scalar or length-T series limit into Param initial values.

    Args:
        bound: A units-carrying scalar, a sequence of units-carrying values, or
            a time-indexed Pyomo component with units.
        units: The units the limited quantity uses.
        field: The config key ``bound`` came from, named in error messages.
        time_index: The TimeBlock's time set.

    Returns:
        A float for a scalar limit, or ``{t: float}`` for a series.

    Raises:
        FlexConfigError: If a series does not have one value per period, or a
            value does not convert to ``units``.
    """
    is_indexed = getattr(bound, "is_indexed", None)
    if callable(is_indexed):
        if not is_indexed():
            return _convert_limit(bound, units, field)
        values = [bound[i] for i in bound.index_set()]
    elif isinstance(bound, Sized) and not isinstance(bound, (str, bytes)):
        values = list(bound)
    else:
        return _convert_limit(bound, units, field)
    if len(values) != len(time_index):
        raise FlexConfigError(
            f"{field} has {len(values)} values, but the horizon has "
            f"{len(time_index)} time points. A per-period limit is either a "
            "single value or one value per time point.",
            field=field,
            value=bound,
        )
    return {
        t: _convert_limit(value, units, field)
        for t, value in zip(time_index, values, strict=True)
    }


def add_time_limits(
    block,
    quantity,
    name: str,
    *,
    lower=None,
    upper=None,
    fields: tuple[str, str] = ("min", "max"),
) -> None:
    """Add a limit that applies to every period, using mutable Params.

    For each of ``lower``/``upper`` that's given, this builds ``{name}_min``
    / ``{name}_max`` (a mutable ``Param``, one value per time period,
    starting at the given limit and using ``quantity``'s units) and
    ``{name}_min_limit`` / ``{name}_max_limit`` (the ``Constraint`` that
    enforces it).

    A limit given as a series sets each period's value; the Params can also
    be rewritten after they're built, e.g. ``feed.withdrawal_max[t].set_value(v)``.
    If you instead want to force an *exact* value rather than a
    limit, this isn't the right tool: use
    :meth:`~flexops.core.ops_block.OpsBlockData.set_external_dispatch` to fix
    the quantity. For a limit on the total over the whole horizon instead of
    each period, use :func:`add_horizon_limits`.

    Args:
        block: The unit block to build the Params and Constraints on.
        quantity: The time-indexed ``Var``/``Reference`` to limit.
        name: The prefix used for the components this builds.
        lower: A lower limit with units — a scalar, a length-T sequence, or a
            time-indexed Pyomo component — or ``None`` for no lower limit.
            Converted to ``quantity``'s units.
        upper: An upper limit in the same forms as ``lower``, or ``None`` for
            no upper limit.
        fields: The ``(lower, upper)`` config keys these limits came from,
            named in the error if one is rejected.

    Raises:
        FlexConfigError: If a limit doesn't convert to ``quantity``'s units —
            for example, a horizon-total quantity was given where a
            per-period rate was expected — or a series doesn't have one value
            per period.
    """
    tb = block._find_time_block()
    units = pyunits.get_units(quantity[tb.time_index.first()])
    for suffix, bound, field in (("min", lower, fields[0]), ("max", upper, fields[1])):
        if bound is None:
            continue
        sense, prose = _SENSES[suffix]
        param = pyo.Param(
            tb.time_index,
            initialize=_period_limit_values(bound, units, field, tb.time_index),
            mutable=True,
            units=units,
            doc=f"Per-period {sense} limit on {name}; rewrite it with set_value "
            "to move the limit.",
        )
        block.add_component(f"{name}_{suffix}", param)
        limit = block.component(f"{name}_{suffix}")

        def _rule(_b, t, _limit=limit, _suffix=suffix):
            if _suffix == "min":
                return quantity[t] >= _limit[t]
            return quantity[t] <= _limit[t]

        block.add_component(
            f"{name}_{suffix}_limit",
            pyo.Constraint(
                tb.time_index,
                rule=_rule,
                doc=f"{name}[t] is {prose} {name}_{suffix}[t].",
            ),
        )


def add_horizon_limits(
    block,
    quantity,
    name: str,
    *,
    lower=None,
    upper=None,
    fields: tuple[str, str] = ("min", "max"),
) -> None:
    """Add a limit on the total of a time-indexed quantity over the whole horizon.

    This builds a scalar ``{name}_total`` ``Var`` and an equality constraint
    (``eq_{name}_total``) that defines it as
    :math:`\\sum_t quantity[t] \\cdot dt` — the sum of ``quantity`` over
    every period, weighted by each period's length. It then limits *that
    total* with scalar mutable ``{name}_min`` / ``{name}_max`` Params and
    their matching ``_limit`` Constraints. These components use the same
    names as the per-period version — only their shape (scalar vs.
    time-indexed) differs, and only one version is ever built for a given
    quantity.

    The total is a ``Var`` defined by an equality, not a plain
    ``Expression``, so it can be bounded and read back after solving. It is
    deliberately not registered as an input/output variable — it's a
    derived number, not a real stream. Everything else stays the same:
    ``quantity`` is still time-indexed, so costing, plant-level totals, and
    :meth:`~flexops.core.ops_block.OpsBlockData.set_external_dispatch` all
    keep working as before.

    The total uses ``upper``'s units if an upper limit is given, and
    ``lower``'s units otherwise — so a limit given in litres is reported
    back in litres.

    If neither ``lower`` nor ``upper`` is given, this function builds
    nothing: choosing a basis by itself doesn't create a limit.

    Args:
        block: The unit block to build the Var, Params, and Constraints on.
        quantity: The time-indexed ``Var``/``Reference`` whose total is
            being limited.
        name: The prefix used for the components this builds.
        lower: A scalar lower limit on the horizon total, with units (e.g.
            ``m**3``), or ``None``.
        upper: A scalar upper limit on the horizon total, with units, or
            ``None``.
        fields: The ``(lower, upper)`` config keys these limits came from,
            named in the error if one doesn't convert.

    Raises:
        FlexConfigError: If a limit doesn't convert to the horizon total's
            units — for example, a per-period rate was given where a
            horizon total was expected.
    """
    if lower is None and upper is None:
        return
    tb = block._find_time_block()
    basis_bound, basis_field = (
        (upper, fields[1]) if upper is not None else (lower, fields[0])
    )
    total_units = pyunits.get_units(basis_bound)

    # convert() checks the units of the whole rate * dt product, instead of
    # assuming either side's units like the storage equations do. This is
    # also what catches the mistake of giving a rate where a horizon total
    # is expected.
    integral = sum(quantity[t] for t in tb.time_index) * tb.dt
    try:
        total_expr = pyunits.convert(integral, to_units=total_units)
    except UnitsError as exc:
        raise FlexConfigError(
            f"{basis_field}={basis_bound!r} is not a quantity over the horizon: "
            f"{total_units} does not match the total of {name}, which is a rate "
            "integrated over time. A horizon-basis limit is a quantity (e.g. "
            "m**3); pass a rate (e.g. m**3/hr) on the period basis instead.",
            field=basis_field,
            value=basis_bound,
        ) from exc

    block.add_component(
        f"{name}_total",
        pyo.Var(
            initialize=0.0,
            units=total_units,
            doc=f"Total {name} integrated over the whole horizon.",
        ),
    )
    total = block.component(f"{name}_total")
    block.add_component(
        f"eq_{name}_total",
        pyo.Constraint(
            expr=total == total_expr,
            doc=f"Metering: {name}_total is the time integral of {name}[t].",
        ),
    )

    for suffix, bound, field in (("min", lower, fields[0]), ("max", upper, fields[1])):
        if bound is None:
            continue
        sense, prose = _SENSES[suffix]
        block.add_component(
            f"{name}_{suffix}",
            pyo.Param(
                initialize=_convert_limit(bound, total_units, field),
                mutable=True,
                units=total_units,
                doc=f"Horizon {sense} limit on {name}; rewrite it with "
                "set_value to move the bound in place.",
            ),
        )
        limit = block.component(f"{name}_{suffix}")
        expr = total >= limit if suffix == "min" else total <= limit
        block.add_component(
            f"{name}_{suffix}_limit",
            pyo.Constraint(
                expr=expr,
                doc=f"{name}_total is {prose} {name}_{suffix}.",
            ),
        )
