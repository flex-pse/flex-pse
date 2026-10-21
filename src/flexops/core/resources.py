"""Resource balances on a plant or network, built from registered resource terms."""

import pyomo.environ as pyo
from pyomo.core.base.units_container import UnitsError
from pyomo.environ import units as pyunits
from pyomo.network import Arc

from flexcore.exceptions import FlexConfigError
from flexops.core.registration import ResourceTerm, iter_resource_terms
from flexops.core.serialize import to_jsonable
from flexops.core.units import parse_units
from flexops.logic._records import record


def _to_level(value, units, *, field: str):
    """Return ``value`` as a float in ``units``, or ``None``.

    Args:
        value: ``None``, a bare number already in ``units``, or a Pyomo
            expression with units.
        units: Pyomo units to convert to.
        field: Name of the argument, used in the error message.

    Returns:
        The magnitude in ``units``, or ``None`` if ``value`` is ``None``.

    Raises:
        FlexConfigError: If a quantity cannot be converted to ``units``.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    try:
        return pyo.value(pyunits.convert(value, to_units=units))
    except UnitsError as exc:
        raise FlexConfigError(
            f"{field} has units {pyunits.get_units(value)}, which cannot be "
            f"converted to the resource's units {units}. Give {field} in "
            "units of the same dimension.",
            field=field,
            value=str(value),
        ) from exc


def _arc_connected_ids(scope) -> set[int]:
    """Return the ids of every variable on a port an Arc connects.

    Args:
        scope: Block whose model holds the arcs.

    Returns:
        ``id()`` of each Var data object found on a connected port.
    """
    ids: set[int] = set()
    for arc in scope.model().component_data_objects(Arc, descend_into=True):
        # an expanded arc still conserves its flow
        if not (arc.active or arc.expanded_block is not None):
            continue
        for port in (arc.source, arc.destination):
            for var in port.vars.values():
                members = var.values() if var.is_indexed() else (var,)
                ids.update(id(member) for member in members)
    return ids


def _reject_arc_flows(scope, resource: str, terms: list[ResourceTerm]) -> None:
    """Raise if a term's variable is a flow an Arc already conserves.

    Args:
        scope: Plant or network the balance is built on.
        resource: Name of the resource, used in the message.
        terms: Terms about to be summed.

    Raises:
        FlexConfigError: If a term's variable sits on a connected port.
    """
    connected = _arc_connected_ids(scope)
    for term in terms:
        members = term.var.values() if term.var.is_indexed() else (term.var,)
        if any(id(member) in connected for member in members):
            raise FlexConfigError(
                f"Resource {resource!r}: {term.var.name!r} (registered by "
                f"{term.unit.name!r}) is a flow on a port that an Arc "
                "connects, so the Arc already conserves it and balancing it "
                "again would count it twice. Register the metered quantity "
                "instead (a Feed or Product, a power draw, or a variable that "
                "is not on a connected port).",
                field="resource",
                value=resource,
            )


def add_resource_balance(
    scope,
    local_name: str,
    *,
    resource: str,
    units: str,
    stock: bool = False,
    lower=None,
    upper=None,
    initial=None,
):
    """Add a balance on one named resource to a plant or network.

    Args:
        scope: Plant or network to balance.
        local_name: Name of the sub-block added to ``scope``.
        resource: Resource to collect, as units registered it.
        units: Units of ``net`` (a rate) or ``level`` (a quantity), in
            ``parse_units`` syntax.
        stock: True if the resource accumulates over time.
        lower: Lower bound on ``net`` or ``level``, a number in ``units`` or a
            quantity. For a flow with no ``upper`` either, ``None`` means zero.
            Otherwise ``None`` leaves that side open.
        upper: Upper bound, same rules as ``lower``.
        initial: Level at the first time point. Stock only.

    Returns:
        The new sub-block.

    Raises:
        FlexConfigError: If no unit registered ``resource`` (the message lists
            what is registered), a stock has no ``initial`` or a flow has one,
            ``lower`` is above ``upper``, a term cannot be converted to
            ``units``, or a term is a flow on a port an Arc connects.
    """
    if stock and initial is None:
        raise FlexConfigError(
            f"Resource {resource!r} is a stock and needs an initial level.",
            field="initial",
        )
    if not stock and initial is not None:
        raise FlexConfigError(
            f"Resource {resource!r} is a flow and has no initial level. Remove "
            "initial, or declare it a stock.",
            field="initial",
            value=str(initial),
        )
    registered = iter_resource_terms(scope)
    terms = registered.get(resource)
    if not terms:
        raise FlexConfigError(
            f"No unit under {scope.name!r} registered resource {resource!r}. "
            f"Registered resources: {', '.join(registered) or 'none'}.",
            field="resource",
            value=resource,
        )

    tb = scope.time_block
    level_units = parse_units(units)
    lower_value = _to_level(lower, level_units, field="lower")
    upper_value = _to_level(upper, level_units, field="upper")
    if not stock and lower_value is None and upper_value is None:
        lower_value = upper_value = 0.0
    if (
        lower_value is not None
        and upper_value is not None
        and lower_value > upper_value
    ):
        raise FlexConfigError(
            f"Resource {resource!r}: lower ({lower_value}) is above upper "
            f"({upper_value}).",
            field="lower",
            value=lower_value,
        )
    _reject_arc_flows(scope, resource, terms)

    block = pyo.Block()
    scope.add_component(local_name, block)

    def summed(t, *, per_step: bool):
        """Return the terms at ``t`` as one expression in ``level_units``."""
        pieces = []
        for term in terms:
            index = t - term.offset
            # skip terms that land before the horizon starts
            if index not in term.var:
                continue
            raw = term.coeff * term.var[index]
            if per_step:
                raw = tb.dt * raw
            # convert each term on its own so an error names its unit
            try:
                pieces.append(pyunits.convert(raw, to_units=level_units))
            except UnitsError as exc:
                raise FlexConfigError(
                    f"Resource {resource!r}: {term.var.name!r} (registered by "
                    f"{term.unit.name!r}) has units {pyunits.get_units(raw)}, "
                    f"which cannot be converted to the resource's units "
                    f"{units!r}. Change the resource's units, or register the "
                    "variable with a coefficient that fixes the dimension.",
                    field="units",
                    value=units,
                ) from exc
        return sum(pieces) if pieces else 0 * level_units

    if stock:
        block.level = pyo.Var(
            tb.time_index,
            bounds=(lower_value, upper_value),
            initialize=0.0 if lower_value is None else lower_value,
            units=level_units,
            doc=f"Level of resource {resource!r}.",
        )
        block.initial = pyo.Param(
            initialize=_to_level(initial, level_units, field="initial"),
            mutable=True,
            units=level_units,
            doc="Initial level, level[0], registered as rolling horizon state.",
        )
        tb.register_initial_state(block.initial)

        @block.Constraint(doc="level[0] equals initial.")
        def initial_level(b):
            return b.level[0] == b.initial

        @block.Constraint(
            list(tb.time_index)[1:],
            doc="level[t] = level[t-1] + dt * (sum of the terms at t).",
        )
        def balance(b, t):
            return b.level[t] == b.level[t - 1] + summed(t, per_step=True)

    else:
        block.net = pyo.Var(
            tb.time_index,
            bounds=(lower_value, upper_value),
            initialize=0.0,
            units=level_units,
            doc=f"Net contribution to resource {resource!r} (supply minus use).",
        )

        @block.Constraint(
            tb.time_index,
            doc="net[t] equals the sum of the terms at t.",
        )
        def balance(b, t):
            return b.net[t] == summed(t, per_step=False)

    _record_balance(scope, block, resource, units, stock, lower, upper, initial)
    return block


def _record_balance(scope, block, resource, units, stock, lower, upper, initial):
    """Record the call so ``emit_model`` can write it back as a resource element."""
    # leave defaults out so a spec that never set them emits unchanged
    fields = {"name": block.getname(fully_qualified=True), "units": units}
    if resource != block.local_name:
        fields["resource"] = resource
    if stock:
        fields["stock"] = True
    for key, value in (("lower", lower), ("upper", upper), ("initial", initial)):
        if value is not None:
            fields[key] = to_jsonable(value, where=f"{fields['name']}.{key}")
    record(scope.model(), "resource", **fields)
