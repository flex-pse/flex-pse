"""The ordered build stages build_model runs, and apply_spec."""

import re
from dataclasses import dataclass, field
from pathlib import Path

import pyomo.environ as pyo
from pydantic import TypeAdapter
from pyomo.network import Arc, Port

from flexcore import nomenclature as nm
from flexcore.config.io import resolve_source_path, resolve_sources
from flexcore.config.schema import SurrogateSpec, SurrogateType, UnitConfig
from flexcore.config.spec import (
    AUTO,
    KINDS,
    PACKAGE_REF,
    Connection,
    CostingElement,
    Element,
    FlowsheetSpec,
    SurrogateElement,
    UnitElement,
    element_target,
    find_unit,
    parent_name,
)
from flexcore.exceptions import FlexConfigError
from flexops import logic
from flexops.core.extensions import resolve_extension
from flexops.core.fixing import FixReport, apply_state
from flexops.core.network_block import NetworkBlock, NetworkBlockData
from flexops.core.ops_block import OpsBlockData
from flexops.core.plant_block import PlantBlock, PlantBlockData
from flexops.core.resources import add_resource_balance
from flexops.core.time_block import TimeBlock
from flexops.core.units import parse_units
from flexops.costing import FlexCosting
from flexops.properties import PROPERTY_PACKAGES
from flexops.surrogates import surrogate_from_spec

LOGIC_ORDER: tuple[str, ...] = (
    "status",
    "bypass",
    "dwell",
    "startup_shutdown",
    "startup_delay",
    "conditional",
    "parallel_group",
)
"""Order the logic stage builds kinds in, status first and cross-unit logic last."""

ENERGY_RELATION = f"{nm.POWER_ELECTRICAL}_relation"
"""Name of the relation holding a unit's electrical energy relationship."""

INTENSITY_PARAMETER = nm.INTENSITY_VARS[nm.PowerKind.ELECTRICAL]
"""Name of the registered parameter a constant-intensity relationship fixes."""


@dataclass
class BuildContext:
    """What the build stages share.

    Attributes:
        base_dir: The spec file's directory, for relative source paths.
        expand_arcs: Whether the topology stage expands arcs.
        members: Each unit element's name mapped to its block data: one for a
            scalar unit, one per index for an indexed unit.
    """

    base_dir: Path | None
    expand_arcs: bool = False
    members: dict[str, list[OpsBlockData]] = field(default_factory=dict)


def parse_quantity(value, *, strict: bool = True):
    """Turn a persisted units-carrying value into a Pyomo expression.

    Args:
        value: Either a ``{"value": ..., "units": ...}`` mapping (a list
            ``value`` gives a list of quantities), a list of such mappings, a
            ``"<number> <units>"`` string (the form ``TimeConfig.time_step``
            uses), or any other value, which is returned unchanged.
        strict: Whether a string with no numeric magnitude is an error. Pass
            ``False`` where a plain string is itself a legal value — a
            construction option naming an enum member (``"polarization"``) is
            not a botched quantity — and it is returned unchanged instead.

    Returns:
        A units-carrying Pyomo expression, or ``value`` itself.

    Raises:
        FlexConfigError: If ``strict`` and a quantity string has no numeric
            magnitude.
    """
    if isinstance(value, dict) and set(value) == {"value", "units"}:
        units = parse_units(value["units"])
        if isinstance(value["value"], list):
            return [item * units for item in value["value"]]
        return value["value"] * units
    if (
        isinstance(value, list)
        and value
        and all(
            isinstance(item, dict) and set(item) == {"value", "units"} for item in value
        )
    ):
        return [parse_quantity(item) for item in value]
    if isinstance(value, str):
        magnitude, _, units = value.strip().partition(" ")
        try:
            return float(magnitude) * parse_units(units)
        except ValueError as exc:
            if not strict:
                return value
            raise FlexConfigError(
                f"Could not read {value!r} as a quantity; write it as a number "
                "and its units, e.g. '15 min'.",
                value=value,
            ) from exc
    return value


def elements_of(spec: FlowsheetSpec, *kinds: str) -> list:
    """Return the spec's elements of the given kinds in build order.

    Args:
        spec: The flat spec.
        kinds: Element kinds to return.

    Returns:
        The elements sorted by name depth, then name, then what they target;
        the order of the file is never used.
    """

    def build_order(element):
        target = element_target(element)
        name = "" if target or element.kind == "time" else element.name
        return (name.count("."), name, target)

    return sorted(spec.of_kind(*kinds), key=build_order)


def _attach(model, name: str, component) -> None:
    """Add ``component`` under the last segment of ``name`` on its parent block."""
    parent = parent_name(name)
    block = model.find_component(parent) if parent else model
    block.add_component(name.rpartition(".")[2], component)


def _members(model, ctx: BuildContext, unit_name: str) -> list[OpsBlockData]:
    """Return the block data of a unit element, looking it up on the model if needed."""
    if unit_name not in ctx.members:
        unit = model.find_component(unit_name)
        if unit is None:
            raise FlexConfigError(
                f"Unit {unit_name!r} is not on the model. Name a unit element "
                "that was built.",
                field="unit",
                value=unit_name,
            )
        ctx.members[unit_name] = list(unit.values()) if unit.is_indexed() else [unit]
    return ctx.members[unit_name]


def stage_declare(model, spec: FlowsheetSpec, ctx: BuildContext) -> None:
    """Add the TimeBlock, property packages and unprocessed costing blocks.

    Args:
        model: The empty model to populate.
        spec: The validated spec.
        ctx: The shared build context.

    Raises:
        FlexConfigError: If a property package class is unknown.
    """
    (time,) = spec.of_kind("time")
    model.time_block = TimeBlock(
        start_date=time.start_date,
        end_date=time.end_date,
        time_step=parse_quantity(time.time_step),
    )
    for element in elements_of(spec, "property_package"):
        package_class = PROPERTY_PACKAGES.get(element.property_class)
        if package_class is None:
            raise FlexConfigError(
                f"Unknown property_class {element.property_class!r}. Known property "
                f"packages: {', '.join(sorted(PROPERTY_PACKAGES))}.",
                field=f"{element.name}.property_class",
                value=element.property_class,
            )
        options = {
            key: parse_quantity(value, strict=False)
            for key, value in element.options.items()
        }
        model.add_component(element.name, package_class(**options))
    for element in elements_of(spec, "costing"):
        model.add_component(element.name, _build_costing(model, element, ctx))


def stage_topology(model, spec: FlowsheetSpec, ctx: BuildContext) -> None:
    """Build networks, plants, sets, units and connections, expanding arcs if asked.

    Args:
        model: The model with its declared blocks.
        spec: The validated spec.
        ctx: The shared build context; its ``units`` and ``members`` are filled.

    Raises:
        FlexConfigError: If a connection endpoint is not a port.
    """
    for element in elements_of(spec, "network"):
        _attach(model, element.name, NetworkBlock(time_block=model.time_block))
    for element in elements_of(spec, "plant"):
        _attach(model, element.name, PlantBlock(time_block=model.time_block))
    for element in elements_of(spec, "set"):
        _attach(model, element.name, pyo.Set(initialize=element.values, ordered=True))
    units = elements_of(spec, "unit")
    for element in units:
        _build_unit(model, spec, element, ctx)
    for element in units:
        for connection in sorted(element.connections, key=lambda c: (c.port, c.to)):
            _build_connection(model, element.name, connection, ctx)
    if ctx.expand_arcs:
        pyo.TransformationFactory("network.expand_arcs").apply_to(model)


def stage_surrogates(model, spec: FlowsheetSpec, ctx: BuildContext) -> None:
    """Swap each surrogate element's relation into its unit.

    Args:
        model: The model with built units.
        spec: The validated spec.
        ctx: The shared build context.
    """
    for element in elements_of(spec, "surrogate"):
        _apply_surrogate(model, element, ctx)


def stage_ramping(model, spec: FlowsheetSpec, ctx: BuildContext) -> None:
    """Add each ramp element's limit to every member of its unit.

    Args:
        model: The model with built units.
        spec: The validated spec.
        ctx: The shared build context.

    Raises:
        FlexConfigError: If a ramp's variable is not on its unit.
    """
    for element in elements_of(spec, "ramp"):
        for member in _members(model, ctx, element.unit):
            logic.add_ramp_rate(
                _variable(member, element),
                name=element.component_name,
                ramp_up=element.ramp_up,
                ramp_down=element.ramp_down,
                ramp_rate_units=parse_units(element.units),
                window=element.window,
            )


def stage_logic(model, spec: FlowsheetSpec, ctx: BuildContext) -> None:
    """Build the logic elements, kind by kind in ``LOGIC_ORDER``.

    A unit name covers every member of an indexed unit; ``'name[i]'`` covers one.

    Args:
        model: The model with built units.
        spec: The validated spec.
        ctx: The shared build context.

    Raises:
        FlexConfigError: If a variable is not on its unit, two indexed units
            paired by a cross-unit element have different member counts, a
            ``status`` targets a unit that manages its own, or logic that needs
            a status names a unit without one.
    """
    for kind in LOGIC_ORDER:
        for element in elements_of(spec, kind):
            if kind == "parallel_group":
                units = [m for u in element.units for m in _members(model, ctx, u)]
                if element.order_status:
                    for unit in units:
                        _require_status(unit, kind)
                logic.register_parallel_group(
                    units,
                    variables=element.variables,
                    order_status=element.order_status,
                )
            elif kind in ("conditional", "startup_delay"):
                other = element.implies if kind == "conditional" else element.upstream
                for unit, partner in _pairs(model, ctx, element.unit, other):
                    _require_status(unit, kind)
                    _require_status(partner, kind)
                    if kind == "conditional":
                        logic.add_conditional(unit, partner, then=element.state)
                    else:
                        logic.add_startup_delay(
                            unit, partner, element.k, element.initially_on
                        )
            else:
                for member in _members(model, ctx, element.unit):
                    _add_unit_logic(member, element)


def _variable(member, element):
    """Return the variable an element names on ``member``, or raise naming it."""
    var = member.find_component(element.variable)
    if var is None:
        raise FlexConfigError(
            f"{element.kind} names variable {element.variable!r}, which is not on "
            f"{member.name!r}.",
            field="variable",
            value=element.variable,
        )
    return var


def _limit(value):
    """Return a spec limit (a number or a ``Quantity``) as a Pyomo value."""
    return value if isinstance(value, float) else value.value * parse_units(value.units)


def _optional_limit(value):
    """Return a spec limit as a Pyomo value, or ``None`` if it's unset."""
    return None if value is None else _limit(value)


def _require_status(member, kind: str) -> None:
    """Raise if ``member`` has no status for logic of ``kind`` to use."""
    if member.find_component("status") is None:
        raise FlexConfigError(f"{kind} on {member.name!r} needs a status element.")


def _add_unit_logic(member, element) -> None:
    """Build one unit-level logic element on one unit member."""
    if element.kind == "status":
        if not member.allows_status_element:
            raise FlexConfigError(
                f"status on {member.name!r}: this unit manages its own status; "
                "set unit_commitment.status on the unit instead."
            )
        logic.add_status(
            member,
            _variable(member, element),
            _limit(element.min_output),
            _limit(element.max_output),
            max_count=element.max_count,
            min_count_on=element.min_count_on,
        )
    elif element.kind == "bypass":
        logic.add_bypass(member, _variable(member, element), _limit(element.max))
    elif element.kind == "dwell":
        logic.add_dwell(_variable(member, element), element.k)
    else:
        _require_status(member, element.kind)
        logic.add_startup_shutdown(
            member, member.status, element.min_up, element.min_down
        )


def _pairs(model, ctx: BuildContext, first: str, second: str) -> list[tuple]:
    """Pair the members of two unit references for a cross-unit element.

    Args:
        model: The built model.
        ctx: The shared build context.
        first: The reference the constraint is built on.
        second: The other reference.

    Returns:
        Each member of ``first`` with the single member of ``second``, or with
        the member of ``second`` at the same position.

    Raises:
        FlexConfigError: If both have several members, in different numbers.
    """
    firsts, seconds = _members(model, ctx, first), _members(model, ctx, second)
    if len(seconds) == 1:
        return [(unit, seconds[0]) for unit in firsts]
    if len(firsts) != len(seconds):
        raise FlexConfigError(
            f"{first!r} has {len(firsts)} members and {second!r} has "
            f"{len(seconds)}; pair them member by member, e.g. '{first}[0]'.",
            field="unit",
            value=first,
        )
    return list(zip(firsts, seconds, strict=True))


def stage_resources(model, spec: FlowsheetSpec, ctx: BuildContext) -> None:
    """Add a balance for each resource element to its plant or network.

    Args:
        model: The model with built units and logic.
        spec: The validated spec.
        ctx: The shared build context.

    Raises:
        FlexConfigError: If the parent is not a plant or network, the resource
            has no registered terms, or its units do not fit its terms.
    """
    for element in elements_of(spec, "resource"):
        parent = model.find_component(parent_name(element.name))
        if not isinstance(parent, (PlantBlockData, NetworkBlockData)):
            raise FlexConfigError(
                f"Resource {element.name!r} needs a plant or network as its "
                f"parent, but {parent_name(element.name)!r} is not one.",
                field="name",
                value=element.name,
            )
        local_name = element.name.rpartition(".")[2]
        add_resource_balance(
            parent,
            local_name,
            resource=element.resource or local_name,
            units=element.units,
            stock=element.stock,
            lower=_optional_limit(element.lower),
            upper=_optional_limit(element.upper),
            initial=_optional_limit(element.initial),
        )


def stage_state(model, spec: FlowsheetSpec, ctx: BuildContext) -> None:
    """Apply the fix, unfix and dispatch elements together, all or nothing.

    The :class:`~flexops.core.fixing.FixReport` is kept as
    ``model._flex_fix_report``.

    Args:
        model: The model with built units and logic.
        spec: The validated spec; its fix_on_violation and fix_tolerance are the
            defaults.
        ctx: The shared build context.

    Raises:
        FlexConfigError: If an element doesn't fit the model, two elements
            conflict, or a value violates bounds or domain under ``"error"``.
    """
    model._flex_fix_report = apply_state(
        model,
        elements_of(spec, "fix", "unfix", "dispatch"),
        base_dir=ctx.base_dir,
        on_violation=spec.fix_on_violation,
        tolerance=spec.fix_tolerance,
    )


def stage_extensions(model, spec: FlowsheetSpec, ctx: BuildContext) -> None:
    """Call each extension element's function on the model, sorted by name.

    Args:
        model: The model with units, logic and dispatch built.
        spec: The validated spec.
        ctx: The shared build context.

    Raises:
        FlexConfigError: If an extension can't be imported.
    """
    for element in sorted(spec.of_kind("extension"), key=lambda e: e.name):
        fn = resolve_extension(element.entry_point)
        # A decorated extension records itself; keep the spec's name for it.
        pinned = {"_name": element.name} if getattr(fn, "flex_extension", False) else {}
        fn(model, **element.options, **pinned)


def stage_costing(model, spec: FlowsheetSpec, ctx: BuildContext) -> None:
    """Process every costing block and add the objective, if the spec has one.

    Args:
        model: The model with every cost term already registered.
        spec: The validated spec.
        ctx: The shared build context.

    Raises:
        FlexConfigError: If the objective is ``'extension'`` but the model doesn't
            hold exactly one active Objective.
    """
    for element in elements_of(spec, "costing"):
        model.find_component(element.name).cost_process()
    for element in elements_of(spec, "objective"):
        if element.expression == "extension":
            active = list(model.component_data_objects(pyo.Objective, active=True))
            if len(active) != 1:
                raise FlexConfigError(
                    "objective expression 'extension' needs an extension that adds "
                    f"an Objective; the model holds {len(active)} active objectives.",
                    field="expression",
                    value="extension",
                )
            continue
        costing = model.find_component(
            spec.of_kind("costing")[0].name
            if element.costing_package == AUTO
            else element.costing_package
        )
        sense = pyo.minimize if element.sense == "minimize" else pyo.maximize
        _attach(
            model,
            element.name,
            pyo.Objective(expr=costing.aggregate_operating_cost, sense=sense),
        )


STAGES = (
    stage_declare,
    stage_topology,
    stage_surrogates,
    stage_ramping,
    stage_logic,
    stage_resources,
    stage_state,
    stage_extensions,
    stage_costing,
)
"""Build order. Changing it is a schema-contract change (see CONTRIBUTING.md)."""


def apply_relation_spec(
    unit, spec: SurrogateSpec, relation_name: str = ENERGY_RELATION
) -> tuple[bool, dict[str, float]]:
    """Write a relationship spec into a live unit as ``relation_name``.

    A ``constant_intensity`` spec fixes the unit's intensity parameter; any
    richer form swaps the named relation in place.

    Args:
        unit: The built unit to mutate.
        spec: The relationship to attach.
        relation_name: The relation a richer spec replaces.

    Returns:
        ``(swapped, fixed values)``: whether the Constraint was swapped, and the
        parameters that were fixed.

    Raises:
        FlexConfigError: If a ``constant_intensity`` spec carries no
            coefficient, or the unit does not register one as a regressable
            process parameter.
    """
    registry = unit._io_registry
    if spec.surrogate_type is not SurrogateType.CONSTANT_INTENSITY:
        surrogate_block = unit.swap_relation(relation_name, surrogate_from_spec(spec))
        coefficients = getattr(surrogate_block, "coefficients", None)
        if coefficients is not None and hasattr(coefficients, "items"):
            coef_names = {name for name, _ in coefficients.items()}
            already_registered = any(
                p.relation_name == relation_name and p.name in coef_names
                for p in registry.parameters
            )
            if not already_registered:
                unit.register_surrogate_coefficients(relation_name)
            for coef_name, coef_value in spec.data["coefficients"].items():
                var = coefficients[coef_name]
                var.set_value(coef_value)
                var.fix()
        return True, dict(spec.data.get("coefficients", {}))

    given = spec.data.get("coefficients", {})
    if INTENSITY_PARAMETER not in given:
        raise FlexConfigError(
            f"A 'constant_intensity' relationship must carry its coefficient "
            f"under data['coefficients'][{INTENSITY_PARAMETER!r}]; got "
            f"{sorted(given)}.",
            field="data",
            value=sorted(given),
        )
    coefficient = given[INTENSITY_PARAMETER]
    regressable = {
        record.name: record.param
        for record in registry.parameters
        if record.regressable
    }
    if INTENSITY_PARAMETER not in regressable:
        raise FlexConfigError(
            f"A 'constant_intensity' relationship determines "
            f"{INTENSITY_PARAMETER!r}, which {unit.name!r} does not register as a "
            f"regressable process parameter (it registers "
            f"{sorted(regressable)}). Supply this unit's relationship through "
            "surrogates= instead.",
            field=INTENSITY_PARAMETER,
            value=unit.name,
        )

    unit.update_parameters({INTENSITY_PARAMETER: coefficient})
    parameter = regressable[INTENSITY_PARAMETER]
    if parameter.is_variable_type():
        parameter.fix()
    return False, {INTENSITY_PARAMETER: coefficient}


def _resolve_source(source, base_dir):
    """Resolve every file path in a source of any shape, passing tags through.

    Args:
        source: A path or tag, a list of them, a mapping of utility to one, or None.
        base_dir: The config file's directory, or None.

    Returns:
        ``source`` with the same shape and each ``.json``/``.csv`` path resolved.
    """
    if isinstance(source, list):
        return [_resolve_source(item, base_dir) for item in source]
    if isinstance(source, dict):
        return {key: _resolve_source(item, base_dir) for key, item in source.items()}
    # A source may name a historian tag rather than a file; only resolve files.
    if source is None or not source.lower().endswith((".json", ".csv")):
        return source
    return str(resolve_source_path(source, base_dir))


def _build_costing(model, element: CostingElement, ctx: BuildContext):
    """Build the FlexCosting block a costing element describes.

    Args:
        model: The model being built (supplies the TimeBlock).
        element: The validated costing element.
        ctx: The shared build context, resolving file paths.

    Returns:
        The constructible ``FlexCosting`` block.
    """

    def quantities(prices):
        return {
            name: parse_quantity(
                {
                    "value": resolve_sources(price.value, ctx.base_dir),
                    "units": price.units,
                }
            )
            for name, price in (prices or {}).items()
        }

    import_prices = quantities(element.energy_prices)
    export_prices = quantities(element.export_prices)
    return FlexCosting(
        time_block=model.time_block,
        tariff_file=_resolve_source(element.tariff_source, ctx.base_dir),
        energy_prices=import_prices or None,
        export_prices=export_prices or None,
        decomposition_type=element.decomposition_type,
        currency=element.currency,
        dr_event_file=_resolve_source(
            None if element.dr is None else element.dr.events_source, ctx.base_dir
        ),
        consumption_estimate=element.consumption_estimate,
        fixed_operating_cost=element.fixed_operating_cost,
        prorate_monthly_charges=element.prorate_monthly_charges,
        lifetime_years=element.lifetime_years,
        discount_rate=element.discount_rate,
        interest_rate=element.interest_rate,
    )


def _build_unit(model, spec: FlowsheetSpec, element: UnitElement, ctx: BuildContext):
    """Build one unit element onto its parent and record it in the context."""
    runtime = {}
    for key, kind, choice in (
        ("property_package", "property_package", element.property_package),
        ("costing_package", "costing", element.costing_package),
    ):
        if choice is not None:
            name = spec.of_kind(kind)[0].name if choice == AUTO else choice
            runtime[key] = model.find_component(name)
    unit_config = UnitConfig(
        unit_model_class=element.unit_model_class,
        construction_options=_with_packages(model, element.construction_options),
        io_variables=element.io_variables,
        unit_commitment=element.unit_commitment,
    )
    index_set = None if element.index is None else model.find_component(element.index)
    block = OpsBlockData.build_from_config(unit_config, index_set=index_set, **runtime)
    _attach(model, element.name, block)
    block = model.find_component(element.name)
    ctx.members[element.name] = list(block.values()) if index_set else [block]


def _with_packages(model, value):
    """Replace each ``{"$package": name}`` in ``value`` with that package block."""
    if isinstance(value, dict):
        if set(value) == {PACKAGE_REF}:
            return model.find_component(value[PACKAGE_REF])
        return {key: _with_packages(model, item) for key, item in value.items()}
    if isinstance(value, list):
        return [_with_packages(model, item) for item in value]
    return value


def _port(model, ctx: BuildContext, unit_name: str, path: str):
    """Return the port at ``path``, or raise listing the unit's ports."""
    port = model.find_component(path)
    if port is None:
        unit = _members(model, ctx, unit_name)[0]
        ports = unit.component_objects(Port, descend_into=False)
        raise FlexConfigError(
            f"Unit {unit_name!r} has no port at {path!r}. Available ports: "
            f"{sorted(p.local_name for p in ports)}. Check the connection's port, "
            "or the unit's unit_model_class.",
            field="port",
            value=path,
        )
    return port


def _build_connection(
    model, unit_name: str, conn: Connection, ctx: BuildContext
) -> None:
    """Build the arc a unit's connection describes.

    Args:
        model: The model with every unit built.
        unit_name: Element name of the connection's source unit.
        conn: The connection.
        ctx: The shared build context.

    Raises:
        FlexConfigError: If a port is missing, or the arc's name is taken.
    """
    source_unit = model.find_component(unit_name)
    if conn.index is not None:
        index_set = model.find_component(conn.index)
    else:
        index_set = source_unit.index_set() if source_unit.is_indexed() else None
    destination_unit, destination_port = find_unit(ctx.members, conn.to)
    shared = []
    for a, b in zip(
        unit_name.split(".")[:-1], destination_unit.split(".")[:-1], strict=False
    ):
        if a != b:
            break
        shared.append(a)
    parent = ".".join(shared)
    block = model.find_component(parent) if parent else model
    start = len(parent) + 1 if parent else 0
    derived = (
        f"{unit_name[start:]}_{conn.port}_to_{destination_unit[start:]}"
        f"_{destination_port}"
    )
    # Drop indices and {i} so the derived name is a plain Pyomo name.
    name = conn.name or re.sub(r"\[[^\]]*\]|\{i\}", "", derived).replace(".", "_")
    if block.find_component(name) is not None:
        raise FlexConfigError(
            f"Cannot name the arc {name!r} on {parent or 'the model'!r}: that name "
            f"is taken. Set an explicit 'name' on the connection {conn.port!r} -> "
            f"{conn.to!r}.",
            field="name",
            value=name,
        )
    source_template = f"{unit_name}.{conn.port}"
    if source_unit.is_indexed():
        source_template = f"{unit_name}[{{i}}].{conn.port}"
    endpoints = {
        member: (
            _port(model, ctx, unit_name, source_template.replace("{i}", str(member))),
            _port(model, ctx, destination_unit, conn.to.replace("{i}", str(member))),
        )
        for member in ([None] if index_set is None else list(index_set))
    }
    if index_set is None:
        source, destination = endpoints[None]
        if conn.directed:
            arc = Arc(source=source, destination=destination, doc=conn.doc)
        else:
            arc = Arc(ports=(source, destination), doc=conn.doc)
    else:
        arc = Arc(
            index_set,
            rule=lambda b, i: (
                {"source": endpoints[i][0], "destination": endpoints[i][1]}
                if conn.directed
                else endpoints[i]
            ),
            doc=conn.doc,
        )
    block.add_component(name, arc)


def _apply_surrogate(model, element: SurrogateElement, ctx: BuildContext) -> None:
    """Swap a surrogate element's relation into every block of its unit.

    Args:
        model: The built model.
        element: The surrogate element.
        ctx: The shared build context.

    Raises:
        FlexConfigError: If the unit is missing or the data is invalid.
    """
    relation = SurrogateSpec(
        surrogate_type=element.surrogate_type,
        data=resolve_sources(element.data, ctx.base_dir),
        provenance=element.provenance,
    )
    for member in _members(model, ctx, element.unit):
        apply_relation_spec(member, relation, relation_name=element.relation)


def apply_spec(
    model,
    elements,
    *,
    on_violation: str | None = None,
    tolerance: float | None = None,
) -> FixReport:
    """Apply mutable spec elements to a built model, without rebuilding it.

    Fix, unfix and dispatch elements are checked and applied as one batch. Their
    conflicts with earlier state are not errors: a later apply is an update.

    Args:
        model: The built model, updated in place.
        elements: Element models or dicts, validated as spec elements.
        on_violation: Fix policy; omitted means the built spec's
            fix_on_violation, or ``"error"``.
        tolerance: Fix bounds tolerance; omitted means the built spec's
            fix_tolerance, or ``1e-9``.

    Returns:
        The :class:`~flexops.core.fixing.FixReport` of the fix, unfix and
        dispatch elements.

    Raises:
        FlexConfigError: If any element's kind is immutable (checked before
            anything is applied), a unit or variable is not on the model, or an
            element fails to apply.
    """
    parsed = TypeAdapter(list[Element]).validate_python(elements)
    for element in parsed:
        if not KINDS[element.kind][1]:
            raise FlexConfigError(
                f"'{element.kind}' elements are immutable; change them in the spec "
                "and rebuild with build_model",
                field="kind",
                value=element.kind,
            )
    spec = getattr(model, "_flex_spec", None)
    if on_violation is None:
        on_violation = "error" if spec is None else spec.fix_on_violation
    if tolerance is None:
        tolerance = 1e-9 if spec is None else spec.fix_tolerance
    ctx = BuildContext(base_dir=None)
    for element in parsed:
        if element.kind == "surrogate":
            _apply_surrogate(model, element, ctx)
    state = [e for e in parsed if e.kind != "surrogate"]
    report = apply_state(model, state, on_violation=on_violation, tolerance=tolerance)
    if not hasattr(model, "_flex_applied"):
        model._flex_applied = []
    for element in parsed:
        if element.kind in ("fix", "unfix"):
            model._flex_applied = [
                e
                for e in model._flex_applied
                if e.kind not in ("fix", "unfix") or e.variable != element.variable
            ]
        model._flex_applied.append(element)
    return report
