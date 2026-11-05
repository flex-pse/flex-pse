"""Resource blocks: streams reached by Arcs, utilities read from the unit registry."""

import enum
import operator
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import pyomo.environ as pyo
from idaes.core import UnitModelBlockData, declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.core.base.units_container import UnitsError
from pyomo.environ import units as pyunits
from pyomo.network import Arc, Port
from pyomo.util.check_units import check_units_equivalent

from flexcore import nomenclature as nm
from flexcore.config.schema import BalanceType
from flexcore.exceptions import FlexConfigError, FlexResourceWarning
from flexcore.solvers import PRE_SOLVE_CHECKS
from flexops.core.registration import ResourceTerm, iter_resource_terms
from flexops.core.task_block import TaskData, task_of
from flexops.core.time_block import find_time_block
from flexops.core.units import parse_units
from flexops.logic._records import record
from flexops.properties.simple_utility import SimpleUtilityData


class ResourceKind(enum.StrEnum):
    """How a resource in the graph is conserved."""

    STREAM = "stream"
    UTILITY = "utility"
    ARC = "arc"
    OPEN = "open"


class ErrorType(enum.StrEnum):
    """How :func:`verify_resources` reports a one-sided balance."""

    WARN = "warn"
    ERROR = "error"


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
        tasks: Task blocks, or lone units serving as tasks, with at least one edge,
            in model order.
        resources: Resources by name. A stream, a utility or an Arc appears under
            its own block's name, and an open resource under the registered name.
    """

    tasks: list
    resources: dict[str, GraphResource]

    def connections(self, task) -> set[tuple[str, bool]]:
        """Return the resource connections a task presents.

        Args:
            task: An entry of :attr:`tasks`.

        Returns:
            A ``(resource name, whether the task supplies it)`` pair for each
            connection. A task with the same set can replace this one.
        """
        return {
            (name, edge.coeff > 0)
            for name, entry in self.resources.items()
            if entry.kind is not ResourceKind.ARC
            for edge in entry.edges
            if task_of(edge.unit) is task
        }


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


def _arc_conserved(scope, terms: list[ResourceTerm]) -> list[ResourceTerm]:
    """Return the terms whose variable is a flow an Arc already conserves.

    Args:
        scope: Block whose model holds the arcs.
        terms: Terms to check.

    Returns:
        The terms sitting on a connected port or metering the ports of a unit an
        Arc connects.
    """
    connected = _arc_connected_ids(scope)
    connected_units = {
        id(port.parent_block())
        for arc in _live_arcs(scope.model())
        for port in arc.ports
    }
    found = []
    for term in terms:
        members = term.var.values() if term.var.is_indexed() else (term.var,)
        meters_ports = id(term.unit) in connected_units and any(
            record.var is term.var for record in term.unit._io_registry.boundary
        )
        if meters_ports or any(id(member) in connected for member in members):
            found.append(term)
    return found


def _reject_arc_flows(scope, name: str, terms: list[ResourceTerm]) -> None:
    """Raise if a term's variable is a flow an Arc already conserves.

    Args:
        scope: Block the utility collects from.
        name: Name of the resource, used in the message.
        terms: Terms about to be summed.

    Raises:
        FlexConfigError: If any term is conserved by an Arc.
    """
    for term in _arc_conserved(scope, terms):
        raise FlexConfigError(
            f"Utility {name!r}: {term.var.name!r} (registered by "
            f"{term.unit.name!r}) is a flow that an Arc on that unit already "
            "conserves, so balancing it again would count it twice. Use a stream "
            "Resource and an Arc for it, or register a variable that is not "
            "tied to a connected port.",
            field="property_package",
            value=name,
        )


def _edge_label(term: ResourceTerm) -> str:
    """Return the name of a term's edge, unique within one resource."""
    return f"{term.var.name}@{term.offset}" if term.offset else term.var.name


def _port_package(port):
    """Return the property package of the state block behind a port, if it has one."""
    member = next(iter(port.vars.values()))
    first = next(iter(member.values())) if member.is_indexed() else member
    return getattr(first.parent_block(), "params", None)


def _package_class(package):
    """Return a property package's declared class, or ``NoneType`` for none."""
    return type(None) if package is None else package.process_block_class()


def _live_port_arcs(port) -> list:
    """Return the arcs on a port that are active or already expanded."""
    return [
        arc
        for arc in port.arcs(active=None)
        if arc.active or arc.expanded_block is not None
    ]


def _check_units_match(node, port, far, arc) -> None:
    """Raise unless a far port carries each of a resource port's states in its units.

    Args:
        node: The stream resource.
        port: Its inlet or outlet.
        far: The port at the arc's other end.
        arc: The arc.

    Raises:
        FlexConfigError: If a state is missing from the far port or its units differ.
    """
    unit = far.parent_block()
    for name, var in port.vars.items():
        ours = next(iter(var.values())) if var.is_indexed() else var
        other = far.vars.get(name)
        if other is None:
            raise FlexConfigError(
                f"{unit.name!r} has no {name!r} on {far.name!r}, but {arc.name!r} "
                f"joins it to {node.name!r}, whose property package carries it.",
                field="units",
                value=name,
            )
        theirs = next(iter(other.values())) if other.is_indexed() else other
        if not check_units_equivalent(ours, theirs):
            raise FlexConfigError(
                f"{unit.name!r} has {name!r} in {pyunits.get_units(theirs)}, but "
                f"{arc.name!r} joins it to {node.name!r}, which holds it in "
                f"{pyunits.get_units(ours)}. Units must match at a connection.",
                field="units",
                value=name,
            )


def _check_node_port(node, port) -> None:
    """Raise unless a node port's arcs run the right way from matching, active units.

    Args:
        node: The stream resource.
        port: Its inlet or outlet.

    Raises:
        FlexConfigError: If an arc runs the wrong way through the port, joins a
            unit on another package class or with different units, or joins a
            deactivated unit.
    """
    package = node.config.property_package
    for arc in _live_port_arcs(port):
        if (arc.source is port) == (port is node.inlet):
            raise FlexConfigError(
                f"{arc.name!r} runs the wrong way through {port.name!r}: a resource's "
                "inlet is an Arc's destination and its outlet an Arc's source.",
                field="port",
                value=port.name,
            )
        far = arc.source if arc.destination is port else arc.destination
        unit = far.parent_block()
        far_class = _package_class(_port_package(far))
        if far_class is not _package_class(package):
            raise FlexConfigError(
                f"{unit.name!r} uses {far_class.__name__}, but {arc.name!r} joins it "
                f"to {node.name!r}, which uses {_package_class(package).__name__}.",
                field="property_package",
                value=unit.name,
            )
        _check_units_match(node, port, far, arc)
        if not unit.active:
            raise FlexConfigError(
                f"{unit.name!r} is deactivated, but {arc.name!r} still joins it to "
                f"{node.name!r}, so its flow is no longer tied to anything. To "
                "switch a unit off, fix its flow to zero instead.",
                field="port",
                value=unit.name,
            )


def stream_flow_rule(port, name: str, index_set) -> None:
    """Expand a port's flow over its arcs, conserving it across fan-in and fan-out.

    Args:
        port: The port being expanded.
        name: Name of the flow on the port.
        index_set: Index set of the flow.

    Raises:
        FlexConfigError: If the port already has an expanded arc, an undirected arc
            shares a port with another arc or touches a resource, a resource's
            arcs fail its checks, or a resource's flow was fixed by hand.
    """
    arcs = port.arcs(active=True)
    if any(not arc.active for arc in _live_port_arcs(port)):
        raise FlexConfigError(
            f"{port.name!r} already has an expanded arc, and its equations can't take "
            "a new one. Add every arc on a port before expanding.",
            field="port",
            value=port.name,
        )
    node = port.parent_block()
    is_node = isinstance(node, ResourceData)
    if not all(arc.directed for arc in arcs):
        if is_node or len(arcs) > 1:
            raise FlexConfigError(
                f"{port.name!r} has an undirected arc, which conserves flow only "
                "between two ports. Give the arc a source and a destination.",
                field="port",
                value=port.name,
            )
        Port.Equality(port, name, index_set)
        return
    member = port.vars[name]
    if is_node:
        _check_node_port(node, port)
        # open a side closed at zero, off the bound where an interior point stalls
        far = arcs[0].source if arcs[0].destination is port else arcs[0].destination
        for key in index_set:
            if not member[key].fixed or member[key].value != 0:
                raise FlexConfigError(
                    f"{port.name!r}: {member[key].name!r} was changed before arcs were "
                    "expanded. Set a resource's flow through its arcs' units.",
                    field="port",
                    value=port.name,
                )
            member[key].unfix()
            member[key].set_value(far.vars[name][key].value)
    first = member[next(iter(index_set))]
    for arc in arcs:
        one_to_one = (
            len(arc.source.dests(active=True)) == 1
            and len(arc.destination.sources(active=True)) == 1
        )
        block = arc.expanded_block
        if not one_to_one and block.component(name) is None:
            # Pyomo would create this flow without units
            block.add_component(
                name,
                pyo.Var(index_set, domain=first.domain, units=pyunits.get_units(first)),
            )
    Port.Extensive(port, name, index_set, include_splitfrac=False)


def conserve_flow(port, flow_name: str) -> None:
    """Put a port's flow under :func:`stream_flow_rule`.

    Args:
        port: The port.
        flow_name: Name of the flow on the port.
    """
    member = port.vars[flow_name]
    port.remove(flow_name)
    port.add(member, flow_name, rule=stream_flow_rule)


def _balance_relation(balance: BalanceType) -> Callable:
    """Return how a balance relates what enters a resource to what leaves it.

    Args:
        balance: The balance type, other than ``UNBOUNDED``.

    Returns:
        The comparison between the two sides.
    """
    return {
        BalanceType.BALANCED: operator.eq,
        BalanceType.EXPORT: operator.ge,
        BalanceType.IMPORT: operator.le,
    }[balance]


@declare_process_block_class("Resource")
class ResourceData(UnitModelBlockData):
    """A resource node whose balance relates what enters it to what leaves it."""

    CONFIG = UnitModelBlockData.CONFIG()
    CONFIG.get("dynamic").set_default_value(False)
    CONFIG.get("has_holdup").set_default_value(False)
    CONFIG.declare(
        "property_package",
        ConfigValue(
            default=None,
            description="Property block setting the units. A SimpleUtility makes a "
            "utility, collected from the registry; any other a stream, reached by "
            "arcs.",
        ),
    )
    CONFIG.declare(
        "balance",
        ConfigValue(
            default=BalanceType.BALANCED,
            domain=BalanceType,
            description="How what enters relates to what leaves at each step.",
        ),
    )

    @property
    def utility_name(self) -> str | None:
        """The name units register this utility under, or ``None`` for a stream."""
        package = self.config.property_package
        return (
            package.config.resource if isinstance(package, SimpleUtilityData) else None
        )

    def build(self) -> None:
        """Build a stream's ports and balance, or a utility's edges and balance.

        Raises:
            FlexConfigError: If the property package is missing, the block is
                indexed, the package has no flow to balance, or a utility can't
                collect its terms.
        """
        super().build()
        package = self.config.property_package
        if package is None or self.parent_component().is_indexed():
            raise FlexConfigError(
                f"Resource {self.name!r} needs a property_package, and can't be "
                "indexed: a spec has no form for that.",
                field="property_package",
            )
        if self.utility_name is None:
            self._build_stream()
        else:
            self._build_utility()
        record(
            self.model(),
            "resource",
            name=self.getname(fully_qualified=True),
            property_package=package.getname(fully_qualified=True),
            balance=self.config.balance.value,
        )

    def _build_stream(self) -> None:
        """Build the ports, the flow balance and one value of each intensive state."""
        package = self.config.property_package
        time_index = find_time_block(self.model()).time_index
        self.inlet_state = package.build_state_block(time_index=time_index)
        self.outlet_state = package.build_state_block(time_index=time_index)
        flow_name = package.get_flow_basis_var_name()
        inlet_vars = self.inlet_state.define_state_vars()
        outlet_vars = self.outlet_state.define_state_vars()
        for port_name, add in (
            ("inlet", self.add_inlet_port),
            ("outlet", self.add_outlet_port),
        ):
            state = self.find_component(f"{port_name}_state")
            conserve_flow(add(name=port_name, block=state), flow_name)
        inlet_vars[flow_name].fix(0.0)
        outlet_vars[flow_name].fix(0.0)
        balance = self.config.balance
        for name, inlet_var in inlet_vars.items():
            is_flow = name == flow_name
            if is_flow and balance is BalanceType.UNBOUNDED:
                continue
            outlet_var = outlet_vars[name]
            relation = _balance_relation(balance) if is_flow else operator.eq

            def _relate(_, *index, _in=inlet_var, _out=outlet_var, _rel=relation):
                key = index[0] if len(index) == 1 else index
                return _rel(_in[key], _out[key])

            self.add_component(
                "balance" if is_flow else f"{name}_equal",
                pyo.Constraint(
                    inlet_var.index_set(),
                    rule=_relate,
                    doc=f"How the {name} entering the resource relates to what leaves.",
                ),
            )

    def _build_utility(self) -> None:
        """Build the edges and balance from what units under the parent registered.

        Raises:
            FlexConfigError: If this repeats another utility, no unit registered
                the resource, two terms share a label, a term cannot be converted to
                the package's units, or a term is a flow an Arc connects.
        """
        scope = self.parent_block()
        name = self.utility_name
        for sibling in scope.component_data_objects(pyo.Block, descend_into=False):
            if (
                sibling is not self
                and isinstance(sibling, ResourceData)
                and sibling.utility_name == name
            ):
                raise FlexConfigError(
                    f"{sibling.name!r} already balances {name!r} here, so "
                    f"{self.name!r} would only repeat it.",
                    field="property_package",
                    value=name,
                )
        registered = iter_resource_terms(scope)
        terms = registered.get(name)
        if not terms:
            raise FlexConfigError(
                f"No unit under {scope.name!r} registered {name!r}. "
                f"Registered resources: {', '.join(registered) or 'none'}.",
                field="property_package",
                value=name,
            )
        _reject_arc_flows(scope, name, terms)

        time_index = find_time_block(self.model()).time_index
        units = self.config.property_package.config.units
        rate_units = parse_units(units)
        edges = {_edge_label(term): term for term in terms}
        if len(edges) != len(terms):
            labels = [_edge_label(term) for term in terms]
            raise FlexConfigError(
                f"Utility {name!r}: {sorted(set(labels))} holds a variable "
                "registered twice, which would drop one term. Register it once with "
                "the combined coefficient.",
                field="property_package",
                value=name,
            )
        self.edges = pyo.Set(initialize=list(edges), doc="Edges into this resource.")

        # TODO(#142): make edges Vars so GDP can gate them outside a disjunct.
        @self.Expression(self.edges, time_index, doc="Signed flow on each edge.")
        def flow(_, edge, t):
            term = edges[edge]
            key = t - term.offset
            # a term before the horizon starts adds nothing
            if key not in term.var:
                return 0 * rate_units
            raw = term.coeff * term.var[key]
            # convert each term on its own so an error names its unit
            try:
                return pyunits.convert(raw, to_units=rate_units)
            except UnitsError as exc:
                raise FlexConfigError(
                    f"Utility {name!r}: {term.var.name!r} (registered by "
                    f"{term.unit.name!r}) has units {pyunits.get_units(raw)}, which "
                    f"cannot be converted to the package's units {units!r}. Change the "
                    "package's units, or register a variable of that dimension.",
                    field="units",
                    value=units,
                ) from exc

        if self.config.balance is BalanceType.UNBOUNDED:
            return
        relation = _balance_relation(self.config.balance)

        @self.Constraint(time_index, doc="How the edges at t relate to zero.")
        def balance(b, t):
            if all(t - term.offset not in term.var for term in terms):
                return pyo.Constraint.Skip
            return relation(sum(b.flow[edge, t] for edge in b.edges), 0 * rate_units)


def _report_one_sided(message: str, error_type: ErrorType, resource: str) -> None:
    """Warn or raise about a balance that forces its flows to zero.

    Args:
        message: What's wrong.
        error_type: How to report it.
        resource: Name of the resource, for the error.

    Raises:
        FlexConfigError: If ``error_type`` is ``ERROR``.
    """
    if error_type is ErrorType.ERROR:
        raise FlexConfigError(message, field="property_package", value=resource)
    warnings.warn(message, FlexResourceWarning, stacklevel=3)


def _missing_side(balance: BalanceType, has_supply: bool, has_consumer: bool) -> str:
    """Return the side whose absence forces a balance's flows to zero.

    Args:
        balance: The balance type.
        has_supply: Whether anything can supply the resource.
        has_consumer: Whether anything can consume the resource.

    Returns:
        ``"no import"``, ``"no consumer"``, or an empty string if nothing is forced.
    """
    if not has_supply and balance in (BalanceType.BALANCED, BalanceType.EXPORT):
        return "no import"
    if not has_consumer and balance in (BalanceType.BALANCED, BalanceType.IMPORT):
        return "no consumer"
    return ""


def check_stream_arcs(block) -> None:
    """Check every arc on the stream resources under ``block``.

    Args:
        block: A plant, network or the whole model.

    Raises:
        FlexConfigError: If an arc runs the wrong way through a resource's port, or
            joins an incompatible package or a deactivated unit.
    """
    for candidate in block.component_data_objects(
        pyo.Block, active=True, descend_into=True
    ):
        if isinstance(candidate, ResourceData) and candidate.utility_name is None:
            _check_node_port(candidate, candidate.inlet)
            _check_node_port(candidate, candidate.outlet)


def _verify_stream(node, error_type: ErrorType) -> None:
    """Report a stream whose arcs only reach one side.

    Args:
        node: The stream resource.
        error_type: How to report it.

    Raises:
        FlexConfigError: If a side has no arcs and ``error_type`` is ``ERROR``.
    """
    has_supply = bool(_live_port_arcs(node.inlet))
    has_consumer = bool(_live_port_arcs(node.outlet))
    missing = _missing_side(node.config.balance, has_supply, has_consumer)
    if missing and (has_supply or has_consumer):
        _report_one_sided(
            f"Stream {node.name!r} has {missing}, so its balance forces its flow to "
            "zero. Connect an arc to its other port, or change its balance type.",
            error_type,
            node.name,
        )


def _only_supplies(term: ResourceTerm) -> bool:
    """Return whether a term's bounds and sign mean it can never consume."""
    if term.coeff > 0:
        return all(var.lb is not None and var.lb >= 0 for var in term.var.values())
    return all(var.ub is not None and var.ub <= 0 for var in term.var.values())


def _verify_electricity(node, terms: list[ResourceTerm], error_type: ErrorType) -> None:
    """Check an electricity utility, whose coefficients alone can't show direction.

    Args:
        node: The utility resource.
        terms: Its terms.
        error_type: How to report a balance that forces its flows to zero.

    Raises:
        FlexConfigError: If a unit beside it already holds the power balance, or the
            balance has no supply or no consumer and ``error_type`` is ``ERROR``.
    """
    for block in node.parent_block().component_data_objects(
        pyo.Block, active=True, descend_into=True
    ):
        if getattr(block, "closes_power_balance", False):
            raise FlexConfigError(
                f"{block.name!r} is a GridConnection, which already holds this "
                f"plant's power balance, so remove {node.name!r}.",
                field="property_package",
                value=node.utility_name,
            )
    has_supply = any(
        term.coeff > 0 or getattr(term.unit, "supplies_power", False) for term in terms
    )
    has_consumer = len(terms) > 1 and not all(_only_supplies(term) for term in terms)
    missing = _missing_side(node.config.balance, has_supply, has_consumer)
    if missing:
        _report_one_sided(
            f"Electricity utility {node.name!r} has {missing}, so its balance forces "
            "its flows to zero. Add what's missing, change its balance type, or remove "
            "the utility. A custom unit that can supply power sets "
            "supplies_power = True.",
            error_type,
            node.utility_name,
        )


def _verify_utility(node, error_type: ErrorType) -> None:
    """Check a utility's edges against what its units register now.

    Args:
        node: The utility resource.
        error_type: How to report a one-signed balance.

    Raises:
        FlexConfigError: If the registered terms differ from the utility's edges,
            a term is a flow an Arc conserves, a unit beside an electricity utility
            already holds the power balance, or every edge has one sign and
            ``error_type`` is ``ERROR``.
    """
    name = node.utility_name
    scope = node.parent_block()
    terms = iter_resource_terms(scope).get(name, [])
    current = {_edge_label(term) for term in terms}
    built = set(node.edges)
    if current != built:
        raise FlexConfigError(
            f"Utility {node.name!r} is out of date. Registered or reactivated since "
            f"it was built: {sorted(current - built)}. Deactivated since: "
            f"{sorted(built - current)}. Build a utility after its units, and fix a "
            "unit's flow to zero to switch it off.",
            field="property_package",
            value=name,
        )
    _reject_arc_flows(scope, name, terms)
    if name == nm.ELECTRICITY_RESOURCE:
        _verify_electricity(node, terms, error_type)
        return
    missing = _missing_side(
        node.config.balance,
        any(term.coeff > 0 for term in terms),
        any(term.coeff < 0 for term in terms),
    )
    if missing:
        _report_one_sided(
            f"Utility {node.name!r} has {missing}, so its balance forces its flows "
            "to zero. Add a unit that registers the other direction, or change its "
            "balance type.",
            error_type,
            name,
        )


def _power_is_accounted_for(unit) -> bool:
    """Return whether anything accounts for what on-site power supplies.

    Args:
        unit: A unit that supplies power.

    Returns:
        Whether a block holding that unit also holds a ``GridConnection`` or an
        electricity utility, or costing on the model bills electricity.
    """
    model_blocks = unit.model().component_data_objects(pyo.Block, descend_into=True)
    if any(getattr(block, "bills_electricity", False) for block in model_blocks):
        return True
    parent = unit.parent_block()
    while parent is not None:
        for block in parent.component_data_objects(
            pyo.Block, active=True, descend_into=False
        ):
            if getattr(block, "closes_power_balance", False) or (
                isinstance(block, ResourceData)
                and block.utility_name == nm.ELECTRICITY_RESOURCE
            ):
                return True
        parent = parent.parent_block()
    return False


def verify_resources(block, *, error_type: ErrorType | str = ErrorType.WARN) -> None:
    """Check every resource under ``block`` against the model as it is now.

    Args:
        block: A plant, network or the whole model.
        error_type: How to report a one-sided balance, or on-site power nothing
            accounts for: ``"warn"`` or ``"error"``.

    Raises:
        FlexConfigError: If ``error_type`` is unknown, a stream's arc runs the wrong
            way, joins an incompatible package or a deactivated unit, a utility is
            out of date, counts a flow an Arc conserves, or sits beside a
            ``GridConnection`` on electricity, or, with ``error_type`` ``"error"``,
            a balance is one-sided or nothing accounts for on-site power.

    Warns:
        FlexResourceWarning: If ``error_type`` is ``"warn"`` and a balance is
            one-sided or nothing accounts for on-site power.
    """
    try:
        error_type = ErrorType(error_type)
    except ValueError as exc:
        raise FlexConfigError(
            f"error_type must be 'warn' or 'error', got {error_type!r}.",
            field="error_type",
            value=error_type,
        ) from exc
    check_stream_arcs(block)
    for candidate in block.component_data_objects(
        pyo.Block, active=True, descend_into=True
    ):
        if isinstance(candidate, ResourceData):
            if candidate.utility_name is None:
                _verify_stream(candidate, error_type)
            else:
                _verify_utility(candidate, error_type)
        elif getattr(
            candidate, "supplies_power", False
        ) and not _power_is_accounted_for(candidate):
            _report_one_sided(
                f"{candidate.name!r} is on-site power or storage that nothing accounts "
                "for, so what it supplies disappears. Add a GridConnection, a Resource "
                "of ELECTRICITY, or a FlexCosting that bills electricity.",
                error_type,
                nm.ELECTRICITY_RESOURCE,
            )


def _verify_before_solve(model) -> None:
    """Run :func:`verify_resources` strictly, and revalidate a validated RTN."""
    verify_resources(model, error_type=ErrorType.ERROR)
    if getattr(model, "rtn_validated", False):
        validate_rtn(model)


PRE_SOLVE_CHECKS.append(_verify_before_solve)


def _inside_one_task(first, second) -> bool:
    """Return whether two blocks both sit in the same :class:`Task`."""
    task = task_of(first)
    return isinstance(task, TaskData) and task is task_of(second)


def _is_under(candidate, block) -> bool:
    """Return whether ``candidate`` is ``block`` or sits somewhere below it."""
    while candidate is not None:
        if candidate is block:
            return True
        candidate = candidate.parent_block()
    return False


def _arc_flow(arc, port):
    """Return the variable carrying an arc's flow, seen from one of its ports.

    Args:
        arc: The arc.
        port: The port on the arc to read a one-to-one flow from.

    Returns:
        The arc's own flow if expansion made one, or the port's flow if the arc is
        the only one on both its ports, or ``None`` if it isn't known yet.
    """
    package = _port_package(port)
    name = None if package is None else package.get_flow_basis_var_name()
    block = arc.expanded_block
    own = None if block is None or name is None else block.component(name)
    if own is not None:
        return own
    shared = any(len(end.arcs(active=None)) > 1 for end in arc.ports)
    return None if shared or name is None else port.vars.get(name)


def resource_graph(block) -> ResourceGraph:
    """Return the tasks, resources and edges under ``block``.

    Args:
        block: A plant, network or the whole model.

    Returns:
        The graph. A stream's edges are the arcs that reach units or resources
        under ``block``, wherever the stream sits. A utility's are the terms it
        collected, an Arc between tasks is a private two-edge ``ARC`` resource, an
        Arc inside a task isn't in the graph, and terms no utility collects are
        ``OPEN`` under their registered name.
    """
    nodes = [
        candidate
        for candidate in block.component_data_objects(
            pyo.Block, active=True, descend_into=True
        )
        if isinstance(candidate, ResourceData)
    ]
    resources = {}
    collected = set()
    for node in nodes:
        name = node.utility_name
        if name is None:
            resources[node.name] = GraphResource(ResourceKind.STREAM, [])
        else:
            terms = iter_resource_terms(node.parent_block()).get(name, [])
            resources[node.name] = GraphResource(ResourceKind.UTILITY, terms)
            collected.update((name, _edge_label(term)) for term in terms)
    for name, terms in iter_resource_terms(block).items():
        conserved = {id(term) for term in _arc_conserved(block, terms)}
        rest = [
            term
            for term in terms
            if (name, _edge_label(term)) not in collected and id(term) not in conserved
        ]
        if rest:
            resources[name] = GraphResource(ResourceKind.OPEN, rest)
    for arc in _live_arcs(block.model()):
        source, destination = arc.ports
        ends = ((source, 1.0), (destination, -1.0))
        if not any(_is_under(port.parent_block(), block) for port, _ in ends):
            continue
        owners = [port.parent_block() for port, _ in ends]
        if _inside_one_task(*owners):
            continue
        if not any(isinstance(owner, ResourceData) for owner in owners):
            resources[arc.name] = GraphResource(
                ResourceKind.ARC,
                [
                    ResourceTerm(
                        arc.name, _arc_flow(arc, port), coeff, 0, port.parent_block()
                    )
                    for port, coeff in ends
                ],
            )
            continue
        for port, coeff in ends:
            owner = port.parent_block()
            if isinstance(owner, ResourceData):
                far = source if port is destination else destination
                # what leaves a port's far end enters the resource, and vice versa
                resources.setdefault(
                    owner.name, GraphResource(ResourceKind.STREAM, [])
                ).edges.append(
                    ResourceTerm(
                        owner.name, _arc_flow(arc, far), -coeff, 0, far.parent_block()
                    )
                )
    tasks = dict.fromkeys(
        task_of(edge.unit)
        for entry in resources.values()
        for edge in entry.edges
        if not isinstance(edge.unit, ResourceData)
    )
    return ResourceGraph(list(tasks), resources)


def validate_rtn(model) -> None:
    """Check a model is a pure resource-task network and record that on it.

    Args:
        model: The whole model. Sets ``model.rtn_validated`` to the outcome.

    Raises:
        FlexConfigError: If an Arc joins two resources, or two units not grouped in
            one task, or a resource sits inside a task. The message lists them all.
    """
    model.rtn_validated = False
    problems = [
        f"{block.name!r} is a resource inside a task; move it to the plant."
        for block in model.component_data_objects(pyo.Block, descend_into=True)
        if isinstance(block, ResourceData) and isinstance(task_of(block), TaskData)
    ]
    for arc in _live_arcs(model):
        first, second = (port.parent_block() for port in arc.ports)
        at_resource = isinstance(first, ResourceData), isinstance(second, ResourceData)
        if all(at_resource):
            problems.append(
                f"{arc.name!r} joins two resources directly; represent the bypass as "
                "a task."
            )
        elif not any(at_resource) and not _inside_one_task(first, second):
            problems.append(
                f"{arc.name!r} joins {first.name!r} and {second.name!r} directly; "
                "group the units in a task, or connect them through a resource."
            )
    if problems:
        raise FlexConfigError(
            "Not a pure resource-task network:\n- " + "\n- ".join(problems),
            field="rtn",
        )
    model.rtn_validated = True
