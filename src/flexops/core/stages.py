"""The ordered build stages build_model runs, and apply_stages/apply_spec."""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pyomo.environ as pyo
from pydantic import TypeAdapter
from pyomo.network import Arc, Port

from flexcore import nomenclature as nm
from flexcore.config.io import load_spec, resolve_source_path, resolve_sources
from flexcore.config.schema import SurrogateSpec, SurrogateType, UnitConfig
from flexcore.config.spec import (
    AUTO,
    KINDS,
    PACKAGE_REF,
    Connection,
    CostingElement,
    DispatchElement,
    Element,
    FlowsheetSpec,
    SurrogateElement,
    UnitElement,
    element_target,
    find_unit,
    parent_name,
)
from flexcore.exceptions import FlexConfigError
from flexops.core.network_block import NetworkBlock
from flexops.core.ops_block import OpsBlockData
from flexops.core.plant_block import PlantBlock
from flexops.core.time_block import TimeBlock
from flexops.core.units import parse_units
from flexops.costing import FlexCosting
from flexops.properties import PROPERTY_PACKAGES
from flexops.surrogates import surrogate_from_spec

STAGES: tuple[str, ...] = (
    "declare",
    "topology",
    "surrogates",
    "degradation",
    "ramping",
    "logic",
    "state",
    "extensions",
    "costing",
)
"""Build order. Changing it is a schema-contract change (see CONTRIBUTING.md)."""

POST_TOPOLOGY_STAGES = STAGES[2:]
"""Stages that can run on an already-built model."""

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
        units: Each unit element's name mapped to its built block (the indexed
            component for an indexed unit).
        members: Each unit element's name mapped to its block data: one for a
            scalar unit, one per index for an indexed unit.
    """

    base_dir: Path | None
    expand_arcs: bool = False
    units: dict[str, OpsBlockData] = field(default_factory=dict)
    members: dict[str, list[OpsBlockData]] = field(default_factory=dict)


def parse_quantity(value, *, strict: bool = True):
    """Turn a persisted units-carrying value into a Pyomo expression.

    Args:
        value: Either a ``{"value": ..., "units": ...}`` mapping (a list
            ``value`` gives a list of quantities), a
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


def _package(model, spec: FlowsheetSpec, choice: str | None, kind: str):
    """Return the package component a unit's ``choice`` names, or None."""
    if choice is None:
        return None
    return model.find_component(
        spec.of_kind(kind)[0].name if choice == AUTO else choice
    )


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


def stage_degradation(model, spec: FlowsheetSpec, ctx: BuildContext) -> None:
    """Reserved: builds degradation terms from the spec."""


def stage_ramping(model, spec: FlowsheetSpec, ctx: BuildContext) -> None:
    """Reserved: builds ramp limits from the spec (planned for schema 0.1.x)."""


def stage_logic(model, spec: FlowsheetSpec, ctx: BuildContext) -> None:
    """Reserved: builds status, startup, shutdown logic (planned for 0.1.x)."""


def stage_state(model, spec: FlowsheetSpec, ctx: BuildContext) -> None:
    """Apply each dispatch element to its unit.

    Args:
        model: The model with built units.
        spec: The validated spec.
        ctx: The shared build context.

    Raises:
        FlexConfigError: If a dispatch variable or source is invalid.
    """
    for element in elements_of(spec, "dispatch"):
        _apply_dispatch(model, element, ctx)


def stage_extensions(model, spec: FlowsheetSpec, ctx: BuildContext) -> None:
    """Reserved: runs user-supplied constraint builders (planned for 0.1.x)."""


def stage_costing(model, spec: FlowsheetSpec, ctx: BuildContext) -> None:
    """Process every costing block and add the objective, if the spec has one.

    Args:
        model: The model with every cost term already registered.
        spec: The validated spec.
        ctx: The shared build context.
    """
    for element in elements_of(spec, "costing"):
        model.find_component(element.name).cost_process()
    for element in elements_of(spec, "objective"):
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


STAGE_FUNCTIONS = {name: globals()[f"stage_{name}"] for name in STAGES}


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


def _resolve_units(model, spec: FlowsheetSpec) -> dict[str, OpsBlockData]:
    """Map each spec unit element name to its block on ``model``.

    Args:
        model: The model holding the units.
        spec: The spec naming them.

    Returns:
        Unit element name to built block.

    Raises:
        FlexConfigError: If a unit name does not resolve on the model.
    """
    units = {}
    for element in spec.of_kind("unit"):
        unit = model.find_component(element.name)
        if unit is None:
            raise FlexConfigError(
                f"Spec unit {element.name!r} is not on the model.",
                field=element.name,
                value=element.name,
            )
        units[element.name] = unit
    return units


def apply_stages(model, config, stages: Sequence[str] = POST_TOPOLOGY_STAGES) -> None:
    """Run build stages on a model that already exists, without rebuilding it.

    The model must have been built by ``build_model`` (it carries
    ``model._flex_build_context``), or have the same component layout the
    spec describes. Stages always run in ``STAGES`` order, whatever order
    they are passed in.

    Args:
        model: The built model to update in place.
        config: Anything :func:`~flexcore.config.io.load_spec` accepts.
        stages: The stage names to run.

    Raises:
        FlexConfigError: If ``stages`` names ``declare`` or ``topology`` (those
            build the model, they can't be re-run on it); if a name is not in
            ``STAGES``; if a spec unit is not on the model; or if ``costing`` is
            requested on a model whose costing block has already been processed.
    """
    spec = load_spec(config)
    unknown = sorted(set(stages) - set(STAGES))
    if unknown:
        raise FlexConfigError(
            f"Unknown stage(s) {unknown}. Known stages: {', '.join(STAGES)}.",
            field="stages",
            value=unknown,
        )
    building = sorted(set(stages) & set(STAGES[:2]))
    if building:
        raise FlexConfigError(
            f"Stage(s) {building} build the model and cannot be re-run on an "
            f"existing one; apply_stages accepts {', '.join(POST_TOPOLOGY_STAGES)}.",
            field="stages",
            value=building,
        )
    if "costing" in stages and any(
        model.find_component(element.name).find_component("aggregate_operating_cost")
        is not None
        for element in spec.of_kind("costing")
    ):
        raise FlexConfigError(
            "Stage 'costing' already ran on this model; running cost_process() "
            "twice would double-register costs.",
            field="stages",
            value="costing",
        )
    ctx = getattr(model, "_flex_build_context", None)
    if ctx is None:
        ctx = BuildContext(base_dir=spec._base_dir, units=_resolve_units(model, spec))
    for name in STAGES:
        if name in stages:
            STAGE_FUNCTIONS[name](model, spec, ctx)


def _resolve_source(source, base_dir):
    """Resolve a file path against the config directory, passing tags through.

    Args:
        source: A path or tag as written in the config, or None.
        base_dir: The config file's directory, or None.

    Returns:
        The resolved path string, or ``source`` unchanged when it is None or not
        a ``.json``/``.csv`` file name.
    """
    # A source may name a historian tag rather than a file; only resolve files.
    if source is None or not source.lower().endswith((".json", ".csv")):
        return source
    return str(resolve_source_path(source, base_dir))


def _resolve_tariff_source(source, base_dir):
    """Resolve every file path in a ``tariff_source`` of any shape.

    Args:
        source: A string, list of strings, mapping of utility to string, or None.
        base_dir: The config file's directory, or None.

    Returns:
        ``source`` with the same shape and each file path resolved.
    """
    if isinstance(source, list):
        return [_resolve_source(item, base_dir) for item in source]
    if isinstance(source, dict):
        return {key: _resolve_source(item, base_dir) for key, item in source.items()}
    return _resolve_source(source, base_dir)


def _build_costing(model, element: CostingElement, ctx: BuildContext):
    """Build the FlexCosting block a costing element describes.

    Args:
        model: The model being built (supplies the TimeBlock).
        element: The validated costing element.
        ctx: The shared build context, resolving file paths.

    Returns:
        The constructible ``FlexCosting`` block.
    """
    prices = {
        name: parse_quantity(
            {
                "value": resolve_sources(price.value, ctx.base_dir),
                "units": price.units,
            }
        )
        for name, price in (element.energy_prices or {}).items()
    }
    return FlexCosting(
        time_block=model.time_block,
        tariff_file=_resolve_tariff_source(element.tariff_source, ctx.base_dir),
        energy_prices=prices or None,
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
            runtime[key] = _package(model, spec, choice, kind)
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
    ctx.units[element.name] = block
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


def _slug(text: str) -> str:
    """Drop ``[...]`` and ``{i}`` from a path and join its segments with ``_``."""
    return re.sub(r"\[[^\]]*\]|\{i\}", "", text).replace(".", "_")


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


def _fill(template: str, member) -> str:
    """Put an index member into a ``{i}`` template; None leaves it unchanged."""
    return template if member is None else template.replace("{i}", str(member))


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
    source_unit = ctx.units[unit_name]
    if conn.index is not None:
        index_set = model.find_component(conn.index)
    else:
        index_set = source_unit.index_set() if source_unit.is_indexed() else None
    destination_unit, destination_port = find_unit(ctx.units, conn.to)
    parent = _common_parent(unit_name, destination_unit)
    block = model.find_component(parent) if parent else model
    start = len(parent) + 1 if parent else 0
    name = conn.name or _slug(
        f"{unit_name[start:]}_{conn.port}_to_{destination_unit[start:]}"
        f"_{destination_port}"
    )
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
            _port(model, ctx, unit_name, _fill(source_template, member)),
            _port(model, ctx, destination_unit, _fill(conn.to, member)),
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


def _common_parent(first: str, second: str) -> str:
    """Return the deepest dotted path that is a parent of both unit names."""
    shared = []
    for a, b in zip(
        parent_name(first).split("."), parent_name(second).split("."), strict=False
    ):
        if a != b:
            break
        shared.append(a)
    return ".".join(shared)


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


def _apply_dispatch(model, element: DispatchElement, ctx: BuildContext) -> None:
    """Set and fix a variable of every block of a unit to a dispatch series.

    Args:
        model: The built model.
        element: The dispatch element.
        ctx: The shared build context.

    Raises:
        FlexConfigError: If the unit or variable is missing, or the values cannot
            be read as a time-indexed series.
    """
    members = _members(model, ctx, element.unit)
    for member in members:
        if member.find_component(element.variable) is None:
            raise FlexConfigError(
                f"dispatch names variable {element.variable!r}, which is not on "
                f"{member.name!r}.",
                field="variable",
                value=element.variable,
            )
    raw = resolve_sources(element.values, ctx.base_dir)
    # JSON keys are always strings; integer time indices come back as "0".
    series = (
        dict(enumerate(raw))
        if isinstance(raw, list)
        else {int(k) if k.lstrip("-").isdigit() else k: v for k, v in raw.items()}
    )
    for member in members:
        member.set_external_dispatch(
            member.find_component(element.variable), series, fix=element.fix
        )


APPLIERS = {"surrogate": _apply_surrogate, "dispatch": _apply_dispatch}
"""Mutable element kind -> function applying one such element to a model."""


def apply_spec(model, elements) -> None:
    """Apply mutable spec elements to a built model, without rebuilding it.

    Args:
        model: The built model, updated in place.
        elements: Element models or dicts, validated as spec elements.

    Raises:
        FlexConfigError: If any element's kind is immutable (checked before
            anything is applied), a unit is not on the model, or an element
            fails to apply.
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
    ctx = BuildContext(base_dir=None)
    if not hasattr(model, "_flex_applied"):
        model._flex_applied = []
    for element in sorted(parsed, key=lambda e: list(APPLIERS).index(e.kind)):
        APPLIERS[element.kind](model, element, ctx)
        model._flex_applied.append(element)
