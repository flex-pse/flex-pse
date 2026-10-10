"""emit_model: read the flat spec a live model was, or could have been, built from."""

import gzip
import json
import re
import warnings
from pathlib import Path

import pyomo.environ as pyo
from pyomo.network import Arc

from flexcore.config.spec import (
    SCHEMA_VERSION,
    Connection,
    DispatchElement,
    FlowsheetSpec,
    NetworkElement,
    ObjectiveElement,
    PlantElement,
    PropertyPackageElement,
    SetElement,
    SourceRef,
    SurrogateElement,
    element_target,
    find_unit,
)
from flexcore.exceptions import FlexConfigError, FlexEmitWarning
from flexops.core.build import build_model
from flexops.core.compare import model_differences
from flexops.core.network_block import NetworkBlock
from flexops.core.ops_block import OpsBlockData
from flexops.core.plant_block import PlantBlock
from flexops.core.serialize import relative_path, to_jsonable
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
        FlexEmitWarning: For anything left out of the spec: a custom surrogate
            or objective, a long list with no ``data_dir``, a costing block that
            has not run ``cost_process()``, and every difference the check
            finds.
    """
    packages, costings = _root_blocks(model)
    for name, block in costings.items():
        if block.find_component("aggregate_operating_cost") is None:
            warnings.warn(
                f"{name} has not run cost_process(); build_model always runs it, "
                "so the rebuilt model will also hold its cost terms",
                FlexEmitWarning,
                stacklevel=2,
            )
    elements = [find_time_block(model).to_element()]
    elements += [property_package_element(n, b) for n, b in packages.items()]
    elements += [b.to_element(relative_to=relative_to) for b in costings.values()]
    sets: dict[str, SetElement] = {}
    units: dict[str, list] = {}
    _structure(model, packages, costings, elements, sets, units)
    unit_elements = {el.name: el for el in elements if el.kind == "unit"}
    _connections(model, unit_elements, units, sets)
    elements += sets.values()
    elements += _mutable_elements(model, units)
    elements += _objectives(model, costings)
    base_dir = relative_to or (None if data_dir is None else Path(data_dir).parent)
    _offload(elements, data_dir, inline_limit, base_dir)
    spec = FlowsheetSpec(
        schema_version=SCHEMA_VERSION, name=model.name, elements=elements
    )
    spec._base_dir = None if base_dir is None else Path(base_dir)
    if check:
        _check(model, spec)
    return spec


def property_package_element(name: str, block) -> PropertyPackageElement:
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


def _root_blocks(model) -> tuple[dict, dict]:
    """Return the model root's property packages and costing blocks by name."""
    packages, costings = {}, {}
    for block in model.component_objects(pyo.Block, descend_into=False):
        if isinstance(block, FlexCosting):
            costings[block.local_name] = block
        elif any(isinstance(block, cls) for cls in PROPERTY_PACKAGES.values()):
            packages[block.local_name] = block
    return packages, costings


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
            described = [
                m.to_unit_element(name, packages=packages, costings=costings)
                for m in members
            ]
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
    targets = {}
    for element in [*recorded, *getattr(model, "_flex_applied", [])]:
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


def _objectives(model, costings: dict) -> list[ObjectiveElement]:
    """Return the cost objective, warning about every other active objective."""
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
    """Replace ``.``, ``[`` and ``]`` with ``_`` for a file name."""
    return re.sub(r"[.\[\]]", "_", text)


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
        return relative_path(path, base_dir)

    def walk(owner: str, field: str, value):
        if isinstance(value, dict):
            return {k: walk(owner, f"{field}.{k}", v) for k, v in value.items()}
        if isinstance(value, list):
            source = write(owner, field, value)
            return value if source is None else {"$source": source}
        return value

    for element in elements:
        if element.kind == "costing":
            for carrier, price in (element.energy_prices or {}).items():
                if isinstance(price.value, list):
                    source = write(
                        element.name, f"energy_prices.{carrier}", price.value
                    )
                    if source is not None:
                        price.value = SourceRef(source=source)
        elif element.kind == "dispatch" and isinstance(element.values, list):
            source = write(".".join(element_target(element)), "values", element.values)
            if source is not None:
                element.values = SourceRef(source=source)
        elif element.kind == "surrogate":
            element.data = walk(".".join(element_target(element)), "data", element.data)


def _check(model, spec: FlowsheetSpec) -> None:
    """Rebuild ``spec`` and warn once about every difference from ``model``."""
    expand_arcs = any(
        not arc.active
        for arc in model.component_data_objects(Arc, active=None, descend_into=True)
    )
    try:
        rebuilt = build_model(spec, expand_arcs=expand_arcs)
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
