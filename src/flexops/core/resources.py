"""Resource nodes on a plant or network, built from registered resource terms."""

import enum
from collections.abc import Iterator
from dataclasses import dataclass

import pyomo.environ as pyo
from pyomo.core.base.units_container import UnitsError
from pyomo.environ import units as pyunits
from pyomo.network import Arc

from flexcore.exceptions import FlexConfigError
from flexops.core.registration import ResourceTerm, iter_resource_terms
from flexops.core.units import parse_units
from flexops.logic._records import record


class ResourceKind(enum.StrEnum):
    """How a resource in the graph is conserved."""

    BALANCED = "balanced"
    ARC = "arc"
    OPEN = "open"


@dataclass
class GraphResource:
    """One resource node of a :class:`ResourceGraph`.

    Attributes:
        kind: How the resource is conserved.
        edges: Signed contribution of each task to the resource.
    """

    kind: ResourceKind
    edges: list[ResourceTerm]


@dataclass
class ResourceGraph:
    """The bipartite graph of tasks and resources under a block.

    Attributes:
        tasks: Unit blocks with at least one edge, in model order.
        resources: Resources by name. An Arc appears under its own name.
    """

    tasks: list
    resources: dict[str, GraphResource]


def _live_arcs(block) -> Iterator:
    """Yield each Arc under ``block`` that conserves its flow.

    Args:
        block: Block to search.

    Yields:
        Arc data objects that are active or already expanded.
    """
    for arc in block.component_data_objects(Arc, active=None, descend_into=True):
        if arc.active or arc.expanded_block is not None:
            yield arc


def _arc_connected_ids(scope) -> set[int]:
    """Return the ids of every variable on a port an Arc connects.

    Args:
        scope: Block whose model holds the arcs.

    Returns:
        ``id()`` of each Var data object found on a connected port.
    """
    ids: set[int] = set()
    for arc in _live_arcs(scope.model()):
        for port in arc.ports:
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
                "again would count it twice. Attach the port to the resource "
                "instead of the Arc, or register a variable that is not on a "
                "connected port.",
                field="resource",
                value=resource,
            )


def _edge_label(term: ResourceTerm) -> str:
    """Return the name of a term's edge, unique within one resource."""
    return f"{term.var.name}@{term.offset}" if term.offset else term.var.name


def _term_key(term: ResourceTerm, t: int):
    """Return the index of a term's variable at step ``t``."""
    index = t - term.offset
    return (index, term.phase) if term.phase else index


def _add_shared_state(block, resource: str, terms: list[ResourceTerm]) -> None:
    """Hold every attached port's non-flow states equal to one node variable.

    Args:
        block: The node's block; gets a ``<state>`` Var and ``<state>_equal`` rows.
        resource: Name of the resource.
        terms: Terms of the resource.
    """
    time_index = block.parent_block().time_block.time_index
    states: dict[str, dict[str, object]] = {}
    for unit in dict.fromkeys(term.unit for term in terms):
        registry = unit._io_registry
        for port, attached in registry.attached_ports.items():
            if attached != resource:
                continue
            for state, var in registry.port_states[port].define_state_vars().items():
                if state != "flow_vol_phase":
                    states.setdefault(state, {})[f"{unit.name}.{port}"] = var
    for state, members in states.items():
        first = next(iter(members.values()))
        start = next(iter(time_index))
        block.add_component(
            state,
            pyo.Var(
                time_index,
                initialize=pyo.value(first[start]),
                units=pyunits.get_units(first[start]),
                doc=f"The one {state} of resource {resource!r}.",
            ),
        )
        shared = block.find_component(state)
        block.add_component(
            f"{state}_equal",
            pyo.Constraint(
                list(members),
                time_index,
                rule=lambda _, port, t, _m=members, _v=shared: _m[port][t] == _v[t],
                doc=f"Each attached port's {state} equals the resource's.",
            ),
        )


def add_resource_balance(scope, local_name: str, *, resource: str, units: str):
    """Add a strict node balance on one named resource to a plant or network.

    Args:
        scope: Plant or network to balance.
        local_name: Name of the sub-block added to ``scope``.
        resource: Resource to collect, as units registered it.
        units: Rate units of the balance, in ``parse_units`` syntax.

    Returns:
        The new sub-block, with ``edges``, ``flow[edge, t]`` and ``balance[t]``.

    Raises:
        FlexConfigError: If no unit registered ``resource`` (the message lists
            what is registered), a term cannot be converted to ``units``, or a
            term is a flow on a port an Arc connects.
    """
    registered = iter_resource_terms(scope)
    terms = registered.get(resource)
    if not terms:
        raise FlexConfigError(
            f"No unit under {scope.name!r} registered resource {resource!r}. "
            f"Registered resources: {', '.join(registered) or 'none'}.",
            field="resource",
            value=resource,
        )
    _reject_arc_flows(scope, resource, terms)

    tb = scope.time_block
    rate_units = parse_units(units)
    edges = {_edge_label(term): term for term in terms}

    block = pyo.Block()
    scope.add_component(local_name, block)
    block._flex_balanced_resource = resource
    block.edges = pyo.Set(initialize=list(edges), doc="Edges into this resource.")

    # TODO(#142): make edges Vars so GDP can gate them outside a candidate's disjunct.
    @block.Expression(block.edges, tb.time_index, doc="Signed flow on each edge.")
    def flow(_, edge, t):
        term = edges[edge]
        key = _term_key(term, t)
        # a term before the horizon starts adds nothing
        if key not in term.var:
            return 0 * rate_units
        raw = term.coeff * term.var[key]
        # convert each term on its own so an error names its unit
        try:
            return pyunits.convert(raw, to_units=rate_units)
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

    @block.Constraint(tb.time_index, doc="The edges at t sum to zero.")
    def balance(b, t):
        if all(_term_key(term, t) not in term.var for term in terms):
            return pyo.Constraint.Skip
        return sum(b.flow[edge, t] for edge in b.edges) == 0 * rate_units

    _add_shared_state(block, resource, terms)
    _record_balance(scope, block, resource, units)
    return block


def _record_balance(scope, block, resource, units):
    """Record the call so ``emit_model`` can write it back as a resource element."""
    # leave defaults out so a spec that never set them emits unchanged
    fields = {"name": block.getname(fully_qualified=True), "units": units}
    if resource != block.local_name:
        fields["resource"] = resource
    record(scope.model(), "resource", **fields)


def resource_graph(block) -> ResourceGraph:
    """Return the tasks, resources and edges under ``block``.

    Args:
        block: A plant, network or the whole model.

    Returns:
        The graph. A resource with a node is ``BALANCED``, an Arc is a private
        two-edge ``ARC`` resource, and a registered resource with neither is
        ``OPEN``.
    """
    balanced = {
        candidate._flex_balanced_resource
        for candidate in block.component_data_objects(
            pyo.Block, active=True, descend_into=True
        )
        if hasattr(candidate, "_flex_balanced_resource")
    }
    resources = {
        name: GraphResource(
            ResourceKind.BALANCED if name in balanced else ResourceKind.OPEN, terms
        )
        for name, terms in iter_resource_terms(block).items()
    }
    for arc in _live_arcs(block):
        source, destination = arc.ports
        resources[arc.name] = GraphResource(
            ResourceKind.ARC,
            [
                ResourceTerm(
                    arc.name,
                    port.vars.get("flow_vol_phase"),
                    coeff,
                    0,
                    port.parent_block(),
                )
                for port, coeff in ((source, 1.0), (destination, -1.0))
            ],
        )
    tasks = dict.fromkeys(
        edge.unit for entry in resources.values() for edge in entry.edges
    )
    return ResourceGraph(list(tasks), resources)
