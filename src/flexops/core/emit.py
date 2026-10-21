"""emit_model: read the flat spec a live model was, or could have been, built from."""

import gzip
import json
import math
import re
import warnings
from pathlib import Path

import pyomo.environ as pyo
from dateutil.relativedelta import relativedelta
from pydantic import TypeAdapter
from pyomo.core.base.componentuid import ComponentUID
from pyomo.environ import units as pyunits
from pyomo.network import Arc

from flexcore.config.io import resolve_sources
from flexcore.config.spec import (
    AUTO,
    SCHEMA_VERSION,
    Connection,
    CostingElement,
    DispatchElement,
    Element,
    ExtensionElement,
    FixElement,
    FlowsheetSpec,
    NetworkElement,
    ObjectiveElement,
    PlantElement,
    PropertyPackageElement,
    SetElement,
    SourcedPrice,
    SourceRef,
    SurrogateElement,
    TimeElement,
    UnfixElement,
    UnitElement,
    element_target,
    find_unit,
)
from flexcore.exceptions import FlexConfigError, FlexEmitWarning
from flexops import unit_models
from flexops.core.build import build_model
from flexops.core.compare import model_differences
from flexops.core.fixing import element_points, time_of, variable_points
from flexops.core.network_block import NetworkBlock
from flexops.core.ops_block import OpsBlockData
from flexops.core.plant_block import PlantBlock
from flexops.core.serialize import to_jsonable, units_to_str
from flexops.core.time_block import find_time_block
from flexops.costing import FlexCosting
from flexops.properties import PROPERTY_PACKAGES


def emit_model(
    model,
    *,
    data_dir: Path | None = None,
    inline_limit: int = 2000,
    check: bool = True,
    relative_to: Path | None = None,
) -> FlowsheetSpec:
    """Read the flat spec a live model was (or could have been) built from.

    Args:
        model: The built Pyomo model.
        data_dir: Directory to write lists longer than ``inline_limit`` to, as
            ``.json.gz`` files the spec references with ``$source``.
        inline_limit: Longest list kept inside the spec.
        check: Rebuild the spec and warn about every difference from ``model``.
        relative_to: Directory the spec will be saved in; file paths under it
            are written relative to it.

    Returns:
        The validated :class:`~flexcore.config.spec.FlowsheetSpec`.

    Raises:
        FlexConfigError: If the model holds something the spec can't express
            and that would change the model if left out (an option with no
            spec form, indexed members that differ, an arc no template fits).

    Warns:
        FlexEmitWarning: For anything left out of the spec: a custom surrogate,
            an objective no costing block or extension made, a long list with
            no ``data_dir``, a costing block that
            has not run ``cost_process()``, fixes that can't be found because
            the model doesn't rebuild, and every difference the check finds.
    """
    packages, costings = {}, {}
    for block in model.component_objects(pyo.Block, descend_into=False):
        if isinstance(block, FlexCosting):
            costings[block.local_name] = block
        elif any(isinstance(block, cls) for cls in PROPERTY_PACKAGES.values()):
            packages[block.local_name] = block
    for name, block in costings.items():
        if block.find_component("aggregate_operating_cost") is None:
            warnings.warn(
                f"{name} has not run cost_process(); build_model always runs it, "
                "so the rebuilt model will also hold its cost terms",
                FlexEmitWarning,
                stacklevel=2,
            )
    elements = [_time_element(find_time_block(model))]
    elements += [_package_element(n, b) for n, b in packages.items()]
    elements += [_costing_element(b, relative_to) for b in costings.values()]
    sets: dict[str, SetElement] = {}
    units: dict[str, list] = {}
    _structure(model, packages, costings, elements, sets, units)
    unit_elements = {el.name: el for el in elements if el.kind == "unit"}
    _connections(model, unit_elements, units, sets)
    elements += sets.values()
    elements += _mutable_elements(model, units)
    elements += _logic_elements(model, units)
    extensions = [
        ExtensionElement(kind="extension", **record)
        for record in model.__dict__.get("_flex_extensions", [])
    ]
    elements += extensions
    elements += _objectives(model, costings, bool(extensions))
    base_dir = relative_to or (None if data_dir is None else Path(data_dir).parent)
    elements += _state_elements(model, elements, base_dir)
    _offload(elements, data_dir, inline_limit, base_dir)
    spec = FlowsheetSpec(
        schema_version=SCHEMA_VERSION, name=model.name, elements=elements
    )
    spec._base_dir = None if base_dir is None else Path(base_dir)
    if check:
        _check(model, spec)
    return spec


def _package_element(name: str, block) -> PropertyPackageElement:
    """Describe a property package block as the spec element that builds it.

    Args:
        name: The package's name on the model root.
        block: The package block.

    Returns:
        The :class:`~flexcore.config.spec.PropertyPackageElement`.

    Raises:
        FlexConfigError: If an option has no spec form.
    """
    property_class = next(
        key for key, cls in PROPERTY_PACKAGES.items() if isinstance(block, cls)
    )
    options = {
        value.name(): to_jsonable(value.value(), where=f"{name}.{value.name()}")
        for value in block.config.user_values()
    }
    return PropertyPackageElement(
        kind="property_package",
        name=name,
        property_class=property_class,
        options=options,
    )


def _time_element(time_block) -> TimeElement:
    """Describe a TimeBlock as the spec's time element.

    Args:
        time_block: The model's TimeBlock.

    Returns:
        The :class:`~flexcore.config.spec.TimeElement`.

    Raises:
        FlexConfigError: If ``max_length`` is not the default (the spec has no
            field for it) or the time step's units can't be written.
    """
    config = time_block.config
    if config.max_length != relativedelta(months=1):
        raise FlexConfigError(
            f"{time_block.name}.max_length is not the default one month; the spec "
            "has no field for it.",
            field="max_length",
            value=config.max_length,
        )
    step = config.time_step
    units = units_to_str(pyunits.get_units(step), where="time_block.time_step")
    return TimeElement(
        kind="time",
        start_date=_iso(config.start_date),
        end_date=_iso(config.end_date),
        time_step=f"{pyo.value(step):g} {units}",
    )


def _iso(value) -> str:
    """Return a date as an ISO string, passing strings through."""
    return value if isinstance(value, str) else value.isoformat()


def _costing_element(block, relative_to) -> CostingElement:
    """Describe a FlexCosting block as the spec element that builds it.

    Args:
        block: The costing block.
        relative_to: Directory to write file paths under it relative to, or None.

    Returns:
        The :class:`~flexcore.config.spec.CostingElement`.

    Raises:
        FlexConfigError: If a tariff was passed as an object, or a price has
            no units.
    """
    config = block.config
    if config.tariff is not None:
        raise FlexConfigError(
            "a tariff passed as an object can't be written to a spec; pass "
            "tariff_file",
            field="tariff",
        )
    prices = {"energy_prices": {}, "export_prices": {}}
    for field, entries in prices.items():
        for name, price in (config[field] or {}).items():
            data = to_jsonable(price, where=f"{field}[{name!r}]")
            if isinstance(data, list) and all(isinstance(d, dict) for d in data):
                units = {d["units"] for d in data}
                if len(units) == 1:
                    data = {"value": [d["value"] for d in data], "units": units.pop()}
            if not isinstance(data, dict):
                raise FlexConfigError(
                    f"{field}[{name!r}] needs units to be written to a spec",
                    field=field,
                    value=name,
                )
            entries[name] = SourcedPrice(**data)
    tariff = config.tariff_file
    if isinstance(tariff, (list, tuple)):
        tariff = [_relative_path(item, relative_to) for item in tariff]
    elif isinstance(tariff, dict):
        tariff = {k: _relative_path(v, relative_to) for k, v in tariff.items()}
    else:
        tariff = _relative_path(tariff, relative_to)
    return CostingElement(
        kind="costing",
        name=block.local_name,
        tariff_source=tariff,
        energy_prices=prices["energy_prices"] or None,
        export_prices=prices["export_prices"] or None,
        decomposition_type=config.decomposition_type,
        currency=config.currency,
        dr=(
            None
            if config.demand_response is None
            else config.demand_response.model_dump(mode="json")
        ),
        consumption_estimate=config.consumption_estimate,
        fixed_operating_cost=config.fixed_operating_cost,
        prorate_monthly_charges=config.prorate_monthly_charges,
        lifetime_years=config.lifetime_years,
        discount_rate=config.discount_rate,
        interest_rate=config.interest_rate,
    )


def _relative_path(path, relative_to):
    """Return ``path`` relative to ``relative_to`` if it lies under it.

    Args:
        path: A file path, or None.
        relative_to: A directory, or None to leave ``path`` as it is.

    Returns:
        A POSIX-style relative path string, or ``path`` unchanged (as a string)
        when it is None, not absolute, or outside ``relative_to``.
    """
    if path is None:
        return None
    if relative_to is None or not Path(path).is_absolute():
        return str(path)
    try:
        return Path(path).resolve().relative_to(Path(relative_to).resolve()).as_posix()
    except ValueError:
        return str(path)


def _unit_element(unit, name: str, packages: dict, costings: dict) -> UnitElement:
    """Describe a unit as the spec element that builds it, without connections.

    Args:
        unit: The unit block (one member of an indexed unit).
        name: The unit element's name (the indexed component's for a member).
        packages: Property-package element names mapped to their blocks.
        costings: Costing element names mapped to their blocks.

    Returns:
        The :class:`~flexcore.config.spec.UnitElement`.

    Raises:
        FlexConfigError: If an option has no spec form, the class is not a
            flexops unit model, or the unit's package or costing block is not
            among those given.
    """
    # Set by the framework or build_from_config, or written as their own fields.
    skip = {
        "dynamic",
        "has_holdup",
        "property_package",
        "property_package_args",
        "costing_package",
        "flexops_config",
        "unit_commitment",
        "external_dispatch",
    }
    given = {value.name(): value.value() for value in unit.config.user_values()}
    options = {
        key: to_jsonable(value, where=f"{name}.{key}", packages=packages)
        for key, value in given.items()
        if key not in skip
    }

    def choose(block, named: dict, field: str):
        if block is None:
            return None
        matches = [key for key, other in named.items() if other is block]
        if not matches:
            raise FlexConfigError(
                f"{name}.{field} is a block that is not on the model root, so "
                "it can't be written to a spec.",
                field=f"{name}.{field}",
            )
        return AUTO if len(named) == 1 else matches[0]

    io_variables = [
        {
            "name": record.name,
            "role": record.role,
            "units": units_to_str(
                pyunits.get_units(next(iter(record.var.values()))),
                where=f"{name}.{record.name}",
            ),
            "tag_hint": record.tag_hint,
            "time_indexed": record.time_indexed,
        }
        for record in unit._io_registry.io_variables
    ]
    return UnitElement(
        kind="unit",
        name=name,
        unit_model_class=unit_model_class_name(unit),
        construction_options=options,
        property_package=choose(
            unit.config.property_package, packages, "property_package"
        ),
        costing_package=choose(
            unit.config.costing_package, costings, "costing_package"
        ),
        io_variables=io_variables,
        unit_commitment=given.get("unit_commitment"),
    )


def unit_model_class_name(unit_or_class) -> str:
    """Return the flexops.unit_models name of a unit or unit-model class.

    Args:
        unit_or_class: A built unit or a unit-model class.

    Returns:
        The first name in the type's MRO, minus a ``Data`` suffix, that
        ``flexops.unit_models.__all__`` lists.

    Raises:
        FlexConfigError: If nothing in the MRO is a flexops unit model.
    """
    declared = unit_or_class if isinstance(unit_or_class, type) else type(unit_or_class)
    for cls in declared.__mro__:
        name = cls.__name__.removesuffix("Data")
        if name in unit_models.__all__:
            return name
    raise FlexConfigError(
        f"{unit_or_class!r} is not a flexops unit model, so it can't be written "
        "to a spec.",
        field="unit_model_class",
        value=unit_or_class,
    )


def _set_element(owner, sets: dict) -> str:
    """Record the set element building ``owner``'s index set and return its name.

    A set that is not a component of its block (Pyomo's implicit set for an
    index given as a list) is named ``<owner>_index``.
    """
    index_set = owner.index_set()
    name = owner.getname(fully_qualified=True)
    if index_set.dimen != 1:
        raise FlexConfigError(
            f"{name}: multi-dimensional unit index sets aren't supported",
            field=name,
        )
    parent = index_set.parent_block()
    if parent is not None and parent.component(index_set.local_name) is index_set:
        name = index_set.getname(fully_qualified=True)
    else:
        name = f"{name}_index"
    sets[name] = SetElement(kind="set", name=name, values=list(index_set))
    return name


def _first_difference(a, b, path: str = ""):
    """Return ``(path, a value, b value)`` where two JSON values first differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            found = _first_difference(
                a.get(key), b.get(key), f"{path}.{key}" if path else key
            )
            if found:
                return found
        return None
    return None if a == b else (path, a, b)


def _shared(unit: str, labelled: list[tuple[str, object]]):
    """Return the value every member of an indexed unit shares.

    Args:
        unit: The unit element's name, for the error.
        labelled: ``(member name, JSON value)`` per member.

    Returns:
        The first member's value.

    Raises:
        FlexConfigError: If any member's value differs from the first's.
    """
    (first_label, first), *rest = labelled
    for label, value in rest:
        found = _first_difference(first, value)
        if found:
            path, a, b = found
            raise FlexConfigError(
                f"{first_label} and {label} differ in {path} ({a} vs {b}); build "
                "them as separate units to configure them differently",
                field=unit,
            )
    return first


def _structure(block, packages, costings, elements, sets, units) -> None:
    """Append network, plant and unit elements found under ``block``, depth first.

    Args:
        block: The block to walk (the model, a network or a plant).
        packages: Package element names mapped to their blocks.
        costings: Costing element names mapped to their blocks.
        elements: List the elements are appended to.
        sets: Set elements by name, filled for indexed units.
        units: Unit element names mapped to their member blocks, filled.
    """
    for component in block.component_objects(pyo.Block, descend_into=False):
        name = component.getname(fully_qualified=True)
        members = list(component.values())
        if isinstance(component, NetworkBlock):
            elements.append(NetworkElement(kind="network", name=name))
            _structure(component, packages, costings, elements, sets, units)
        elif isinstance(component, PlantBlock):
            elements.append(PlantElement(kind="plant", name=name))
            _structure(component, packages, costings, elements, sets, units)
        elif members and all(isinstance(m, OpsBlockData) for m in members):
            described = [_unit_element(m, name, packages, costings) for m in members]
            _shared(
                name,
                [
                    (m.getname(fully_qualified=True), d.model_dump(mode="json"))
                    for m, d in zip(members, described, strict=True)
                ],
            )
            element = described[0]
            if component.is_indexed():
                element.index = _set_element(component, sets)
            elements.append(element)
            units[name] = members


def _template(rest: str, member) -> str:
    """Return a path tail with every occurrence of ``member`` made ``{i}``."""
    return rest if member is None else rest.replace(str(member), "{i}")


def _connections(model, unit_elements: dict, units: dict, sets: dict) -> None:
    """Add a connection to its source unit element for every Arc on the model.

    Args:
        model: The model.
        unit_elements: Unit elements by name; their connections are filled.
        units: Unit element names mapped to their member blocks.
        sets: Set elements by name, filled for arcs over their own set.

    Raises:
        FlexConfigError: If an arc's ends are not unit ports, or no ``{i}``
            template reproduces every member of an indexed arc.
    """
    for arc in model.component_objects(Arc, active=None, descend_into=True):
        found = set()
        for member, data in arc.items() if arc.is_indexed() else [(None, arc)]:
            label = data.getname(fully_qualified=True)
            src, dst = (data.source, data.destination) if data.directed else data.ports
            src_path = src.getname(fully_qualified=True)
            dst_path = dst.getname(fully_qualified=True)
            source = find_unit(unit_elements, src_path)
            destination = find_unit(unit_elements, dst_path)
            if source is None or destination is None:
                raise FlexConfigError(
                    f"Arc {label} joins {src_path} and {dst_path}; both ends must be "
                    "ports of units to be written to a spec.",
                    field=label,
                )
            unit, rest = source
            port = _template(rest, member)
            if unit_elements[unit].index is not None:
                if member is None or not port.startswith("[{i}]."):
                    raise FlexConfigError(
                        f"Arc {label} leaves {src_path}; an arc from an indexed "
                        "unit must be indexed over the same members to be written "
                        "to a spec.",
                        field=label,
                    )
                port = port.removeprefix("[{i}].")
            end, end_rest = destination
            to = end + ("" if end_rest.startswith("[") else ".")
            to += _template(end_rest, member)
            found.add((unit, port, to))
            if len(found) > 1:
                raise FlexConfigError(
                    f"Arc {label} doesn't match the '{{i}}' template of the arc's "
                    f"other members ({sorted(found)}); rename the ports so the index "
                    "appears only where it varies.",
                    field=label,
                )
        ((unit, port, to),) = found
        index = None
        if arc.is_indexed():
            source_index = units[unit][0].parent_component()
            if not source_index.is_indexed() or (
                source_index.index_set() is not arc.index_set()
            ):
                index = _set_element(arc, sets)
        unit_elements[unit].connections.append(
            Connection(
                port=port,
                to=to,
                name=arc.local_name,
                doc=arc.doc,
                index=index,
                directed=next(iter(arc.values())).directed,
            )
        )


def _mutable_elements(model, units: dict) -> list:
    """Return the surrogate and dispatch elements, one per target.

    Elements recorded by build_model and apply_spec come first, the latest one
    winning; what the live units hold then replaces them.

    Args:
        model: The model.
        units: Unit element names mapped to their member blocks.

    Returns:
        The surrogate and dispatch elements.
    """
    spec = getattr(model, "_flex_spec", None)
    recorded = [] if spec is None else spec.of_kind("surrogate", "dispatch")
    applied = getattr(model, "_flex_applied", [])
    targets = {}
    for element in [*recorded, *applied]:
        if element.kind in ("surrogate", "dispatch"):
            targets[(element.kind, *element_target(element))] = element
    for unit, members in units.items():
        for element in [*_surrogates(unit, members), *_dispatches(unit, members)]:
            targets[(element.kind, *element_target(element))] = element
    return list(targets.values())


def _surrogates(unit: str, members: list) -> list[SurrogateElement]:
    """Return the surrogate elements of a unit's active swapped relations."""
    per_member = []
    for member in members:
        specs = {}
        for record in member._io_registry.relations:
            if record.fitted is None or not record.fitted.active:
                continue
            if record.spec is None:
                warnings.warn(
                    f"{unit}.{record.name} was swapped with a custom Surrogate "
                    "object; it can't be written to a spec",
                    FlexEmitWarning,
                    stacklevel=4,
                )
                continue
            specs[record.name] = record.spec.model_dump(mode="json")
        per_member.append((member.getname(fully_qualified=True), specs))
    return [
        SurrogateElement(
            kind="surrogate",
            unit=unit,
            relation=relation,
            surrogate_type=spec["surrogate_type"],
            data=spec["data"],
            provenance=spec["provenance"],
        )
        for relation, spec in _shared(unit, per_member).items()
    ]


def _dispatches(unit: str, members: list) -> list[DispatchElement]:
    """Return the dispatch elements of a unit's externally dispatched variables."""
    per_member = [(m.getname(fully_qualified=True), m._flex_dispatch) for m in members]
    return [
        DispatchElement(kind="dispatch", unit=unit, variable=variable, **record)
        for variable, record in _shared(unit, per_member).items()
    ]


def _logic_elements(model, units: dict) -> list:
    """Return the logic elements recorded by logic-helper calls.

    Records that every member of an indexed unit shares become one element on
    the unit; otherwise each member gets its own.

    Args:
        model: The model, holding cross-unit records.
        units: Unit element names mapped to their member blocks.

    Returns:
        The validated logic elements.
    """
    found = []
    for unit, members in units.items():
        records = [m.__dict__.get("_flex_logic", []) for m in members]
        for kind in dict.fromkeys(r.kind for rs in records for r in rs):
            fields = [[r.fields for r in rs if r.kind == kind] for rs in records]
            if all(f == fields[0] for f in fields):
                found += [{"kind": kind, "unit": unit, **f} for f in fields[0]]
                continue
            for member, member_fields in zip(members, fields, strict=True):
                name = member.getname(fully_qualified=True)
                found += [{"kind": kind, "unit": name, **f} for f in member_fields]
    found += [
        {"kind": r.kind, **r.fields} for r in model.__dict__.get("_flex_logic", [])
    ]
    return TypeAdapter(list[Element]).validate_python(found)


def _state_elements(model, elements: list, base_dir) -> list:
    """Return the fix and unfix elements that bring a rebuild to the model's state.

    Recorded fix and unfix elements the live model still agrees with come first;
    then every point fixed or unfixed differently from a rebuild of ``elements``
    plus those becomes a new element.

    Args:
        model: The model.
        elements: Every other element emitted so far.
        base_dir: Directory the elements' file paths are relative to.

    Returns:
        The fix and unfix elements.

    Warns:
        FlexEmitWarning: If the rebuild fails, so code-made fixes can't be found.
    """
    spec = getattr(model, "_flex_spec", None)
    recorded = (
        []
        if spec is None
        else [(e, spec._base_dir) for e in spec.of_kind("fix", "unfix")]
    )
    recorded += [
        (e, None)
        for e in getattr(model, "_flex_applied", [])
        if e.kind in ("fix", "unfix")
    ]
    kept = [
        (
            e
            if e.kind == "unfix"
            else e.model_copy(update={"value": resolve_sources(e.value, base)})
        )
        for e, base in recorded
        if _agrees(model, e, base)
    ]
    pristine_spec = FlowsheetSpec(
        schema_version=SCHEMA_VERSION, name=model.name, elements=[*elements, *kept]
    )
    pristine_spec._base_dir = None if base_dir is None else Path(base_dir)
    try:
        pristine = build_model(pristine_spec, expand_arcs=_arcs_expanded(model))
    except Exception as exc:
        warnings.warn(
            f"fixes and unfixes made in code can't be found: rebuilding the spec "
            f"failed ({exc})",
            FlexEmitWarning,
            stacklevel=3,
        )
        return kept
    return kept + _code_state(model, pristine)


def _agrees(model, element, base_dir) -> bool:
    """Return whether every point of a recorded fix or unfix element still holds."""
    try:
        points = element_points(model, element, base_dir)
    except FlexConfigError:
        return False
    return all(
        (
            not point.fixed
            if value is None
            else point.fixed and math.isclose(point.value, value, rel_tol=1e-12)
        )
        for point, value in points
    )


def _code_state(model, pristine) -> list:
    """Return elements for points fixed or unfixed differently from ``pristine``."""
    rebuilt = {
        v.name: v for v in pristine.component_data_objects(pyo.Var, descend_into=True)
    }
    time_index = find_time_block(model).time_index
    fixes, unfixes = {}, {}
    for point in model.component_data_objects(pyo.Var, descend_into=True):
        other = rebuilt.get(point.name)
        if other is None:
            continue
        if point.fixed and point.value is not None:
            if not other.fixed or not math.isclose(
                point.value, other.value, rel_tol=1e-12
            ):
                fixes[id(point)] = point
        elif not point.fixed and other.fixed:
            unfixes[id(point)] = point
    found = [
        *_fix_groups(model, fixes.values(), time_index),
        *_unfix_groups(model, unfixes.values()),
    ]
    return _merge_members(model, found)


def _path(component, model) -> str:
    """Return a component's ComponentUID path from the model root."""
    return str(ComponentUID(component, context=model))


def _fix_groups(model, points, time_index) -> list[dict]:
    """Group fixed points into fix element fields, one per variable and non-time index.

    Each group also carries ``base``, its variable's path, for merging.
    """
    groups: dict[tuple, list] = {}
    for point in points:
        index = point.index()
        timed = time_of(point, time_index) is not None
        rest = (index[1:] if isinstance(index, tuple) else ()) if timed else None
        groups.setdefault((id(point.parent_component()), rest), []).append(point)
    found = []
    for (_, rest), members in groups.items():
        var = members[0].parent_component()
        base = _path(var, model)
        if rest is not None:
            values = {time_of(p, time_index): float(p.value) for p in members}
            times = sorted(values)
            series = [values[t] for t in times]
            at = None if times == list(time_index) else times
            value = series[0] if at is None and len(set(series)) == 1 else series
            path = base
            if rest:
                # The point's path with its time position (the first index) made *.
                inner = _path(members[0], model)[len(base) + 1 : -1].split(",")
                path = f"{base}[{','.join(['*', *inner[1:]])}]"
            found.append({"variable": path, "value": value, "at": at, "base": base})
        elif len(members) == len(var) and len({p.value for p in members}) == 1:
            value = float(members[0].value)
            found.append({"variable": base, "value": value, "at": None, "base": base})
        else:
            for p in members:
                fields = {"variable": _path(p, model), "value": float(p.value)}
                found.append({**fields, "at": None, "base": base})
    return [{"kind": "fix", **fields} for fields in found]


def _unfix_groups(model, points) -> list[dict]:
    """Group unfixed points into unfix element fields: a whole variable, or each point.

    Each group also carries ``base``, its variable's path, for merging.
    """
    groups: dict[int, list] = {}
    for point in points:
        groups.setdefault(id(point.parent_component()), []).append(point)
    found = []
    for members in groups.values():
        var = members[0].parent_component()
        base = _path(var, model)
        whole = len(members) == len(var)
        paths = [base] if whole else [_path(p, model) for p in members]
        found += [{"kind": "unfix", "variable": p, "base": base} for p in paths]
    return found


def _merge_members(model, found: list[dict]) -> list:
    """Merge elements that differ only in block indices into one ``*`` element.

    Args:
        model: The model.
        found: Fix and unfix element fields, each with its variable's ``base``.

    Returns:
        The validated elements, sorted by kind and variable.
    """
    groups: dict[tuple, list[dict]] = {}
    for fields in found:
        base = fields.pop("base")
        # Every block index in the variable's path becomes '*', e.g. ro[0] -> ro[*].
        stars = re.sub(
            r"\[([^\]]*)\]",
            lambda g: "[" + ",".join(["*"] * len(g[1].split(","))) + "]",
            base,
        )
        wildcard = stars + fields["variable"][len(base) :]
        value, at = json.dumps(fields.get("value")), json.dumps(fields.get("at"))
        groups.setdefault((fields["kind"], wildcard, value, at), []).append(fields)
    merged = []
    for (_, wildcard, _, _), members in groups.items():
        covered = {
            id(p) for f in members for p in variable_points(model, f["variable"])
        }
        everything = {id(p) for p in variable_points(model, wildcard)}
        if len(members) > 1 and covered == everything:
            members = [{**members[0], "variable": wildcard}]
        merged += members
    merged.sort(key=lambda f: (f["kind"], f["variable"]))
    return [
        FixElement(**f) if f["kind"] == "fix" else UnfixElement(**f) for f in merged
    ]


def _objectives(model, costings: dict, extensions: bool) -> list[ObjectiveElement]:
    """Return the objective, warning about any no costing block or extension made.

    Args:
        model: The model.
        costings: Costing element names mapped to their blocks.
        extensions: Whether the model has extensions, which may have added a
            custom objective.

    Returns:
        The objective elements.
    """
    found = []
    for objective in model.component_data_objects(
        pyo.Objective, active=True, descend_into=True
    ):
        costing = next(
            (
                name
                for name, block in costings.items()
                if objective.parent_block() is model
                and objective.expr is block.find_component("aggregate_operating_cost")
            ),
            None,
        )
        if costing is None and extensions:
            found.append(
                ObjectiveElement(
                    kind="objective",
                    name=objective.local_name,
                    expression="extension",
                )
            )
            continue
        if costing is None:
            warnings.warn(
                f"Objective {objective.name} is not a costing block's "
                "aggregate_operating_cost; it can't be written to a spec yet",
                FlexEmitWarning,
                stacklevel=3,
            )
            continue
        found.append(
            ObjectiveElement(
                kind="objective",
                name=objective.local_name,
                costing_package="auto" if len(costings) == 1 else costing,
                sense="minimize" if objective.sense == pyo.minimize else "maximize",
            )
        )
    return found


def _slug(text: str) -> str:
    """Replace every character but letters, digits, ``_`` and ``-`` with ``_``."""
    return re.sub(r"[^\w-]", "_", text)


def _offload(elements: list, data_dir, inline_limit: int, base_dir) -> None:
    """Move every list longer than ``inline_limit`` into a ``.json.gz`` file.

    Args:
        elements: The emitted elements; changed in place.
        data_dir: Directory for the files, or None to keep lists inline.
        inline_limit: Longest list kept inline.
        base_dir: Directory the written ``$source`` paths are relative to.

    Warns:
        FlexEmitWarning: For a long list when ``data_dir`` is None.
    """

    def write(owner: str, field: str, values: list):
        if len(values) <= inline_limit:
            return None
        if data_dir is None:
            warnings.warn(
                f"{owner}.{field} has {len(values)} values; pass data_dir to "
                "write it to a file beside the spec",
                FlexEmitWarning,
                stacklevel=4,
            )
            return None
        path = Path(data_dir) / f"{_slug(owner)}.{field}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(json.dumps(values).encode(), mtime=0))
        return _relative_path(path, base_dir)

    def walk(owner: str, field: str, value):
        if isinstance(value, dict):
            return {k: walk(owner, f"{field}.{k}", v) for k, v in value.items()}
        if isinstance(value, list):
            source = write(owner, field, value)
            return value if source is None else {"$source": source}
        return value

    for element in elements:
        if element.kind == "costing":
            for field in ("energy_prices", "export_prices"):
                for carrier, price in (getattr(element, field) or {}).items():
                    if isinstance(price.value, list):
                        source = write(element.name, f"{field}.{carrier}", price.value)
                        if source is not None:
                            price.value = SourceRef(source=source)
            history = None if element.dr is None else element.dr.historical_power
            if history is not None and isinstance(history.values_kW, list):
                source = write(element.name, "dr.historical_power", history.values_kW)
                if source is not None:
                    history.values_kW = SourceRef(source=source)
        elif element.kind == "fix" and isinstance(element.value, list):
            source = write(element.variable, "value", element.value)
            if source is not None:
                element.value = SourceRef(source=source)
        elif element.kind == "dispatch" and isinstance(element.values, list):
            source = write(".".join(element_target(element)), "values", element.values)
            if source is not None:
                element.values = SourceRef(source=source)
        elif element.kind == "surrogate":
            element.data = walk(".".join(element_target(element)), "data", element.data)


def _arcs_expanded(model) -> bool:
    """Return whether the model's arcs have been expanded (any arc is inactive)."""
    return any(
        not arc.active
        for arc in model.component_data_objects(Arc, active=None, descend_into=True)
    )


def _check(model, spec: FlowsheetSpec) -> None:
    """Rebuild ``spec`` and warn once about every difference from ``model``."""
    try:
        rebuilt = build_model(spec, expand_arcs=_arcs_expanded(model))
    except Exception as exc:
        warnings.warn(
            f"emit_model's check was skipped: rebuilding the spec failed ({exc})",
            FlexEmitWarning,
            stacklevel=3,
        )
        return
    diffs = model_differences(model, rebuilt)
    if diffs:
        warnings.warn(
            f"The spec rebuilds a model with {len(diffs)} difference(s); these "
            "parts can't be written to a spec yet:\n" + "\n".join(diffs[:30]),
            FlexEmitWarning,
            stacklevel=3,
        )
