"""Fix and unfix model variables from fix, unfix and dispatch spec elements.

The state stage plans every point first, checks conflicts and bounds, and only
then changes the model, so it is all-or-nothing and order-agnostic.
"""

import difflib
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pyomo.environ as pyo
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.units_container import UnitsError
from pyomo.environ import units as pyunits

from flexcore.config.io import resolve_sources
from flexcore.exceptions import FlexConfigError, FlexFixWarning
from flexcore.logger import get_logger
from flexops.core.time_block import find_time_block
from flexops.core.units import parse_units


@dataclass(frozen=True)
class FixViolation:
    """A fix value outside its variable's bounds or domain that was fixed or skipped.

    Attributes:
        variable: Name of the point, e.g. ``"plant.pump.flow_in[3]"``.
        value: The value written, in the variable's units.
        lb: The variable's lower bound, or None.
        ub: The variable's upper bound, or None.
        kind: ``"below_lb"``, ``"above_ub"`` or ``"domain"``.
        action: ``"fixed"`` (fixed anyway) or ``"skipped"`` (left free).
        element: The element's name, or ``"<kind> <variable>"`` when it has none.
    """

    variable: str
    value: float
    lb: float | None
    ub: float | None
    kind: Literal["below_lb", "above_ub", "domain"]
    action: Literal["fixed", "skipped"]
    element: str


@dataclass
class FixReport:
    """What the state stage changed, kept on a built model as ``_flex_fix_report``.

    Attributes:
        fixed: Names of the points fixed by fix elements and fixing dispatches.
        unfixed: Names of the points unfixed.
        skipped: Names of the points left free because their value violated.
        violations: Every violation that was fixed anyway or skipped.
    """

    fixed: list[str] = field(default_factory=list)
    unfixed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    violations: list[FixViolation] = field(default_factory=list)


def element_label(element) -> str:
    """Return an element's name, or ``"<kind> <target>"`` when it has none."""
    if element.name:
        return element.name
    if element.kind == "dispatch":
        return f"dispatch {element.unit}.{element.variable}"
    return f"{element.kind} {element.variable}"


def variable_points(model, path: str) -> list:
    """Return every variable point a ComponentUID path matches on ``model``.

    Args:
        model: The model the path is relative to.
        path: A ComponentUID string such as ``'plant.ro[*].status'``.

    Returns:
        The matched ``VarData``, an indexed variable contributing all its points.

    Raises:
        FlexConfigError: If the path is malformed, matches nothing (naming the
            closest variables), or matches something that is not a variable.
    """
    try:
        matches = list(ComponentUID(path).list_components(model))
    except Exception as exc:
        raise FlexConfigError(
            f"{path!r} is not a valid component path ({type(exc).__name__}); write "
            "it like 'plant.pump.flow_in[3]' or 'plant.ro[*].status'.",
            field="variable",
            value=path,
        ) from exc
    if not matches:
        names = [c.name for c in model.component_objects(pyo.Var, descend_into=True)]
        close = difflib.get_close_matches(path, names, n=3)
        hint = f" Did you mean {', '.join(close)}?" if close else ""
        raise FlexConfigError(
            f"{path!r} matches no component in the model.{hint}",
            field="variable",
            value=path,
        )
    points = {}
    for match in matches:
        if match.ctype is pyo.Param:
            raise FlexConfigError(
                f"{path!r} matches {match.name}, a Param; Params are set, not fixed.",
                field="variable",
                value=path,
            )
        if match.ctype is not pyo.Var:
            raise FlexConfigError(
                f"{path!r} matches {match.name}, a {type(match).__name__}, which is "
                "not a variable.",
                field="variable",
                value=path,
            )
        for point in match.values() if match.is_indexed() else [match]:
            points[id(point)] = point
    return list(points.values())


def time_of(point, time_index):
    """Return a point's time index, or None if its variable isn't indexed by time first.

    Args:
        point: A ``VarData``.
        time_index: The model's TimeBlock ``time_index``, or None.

    Returns:
        The time index of the point, or None.
    """
    var = point.parent_component()
    if time_index is None or not var.is_indexed():
        return None
    if next(iter(var.index_set().subsets())) is not time_index:
        return None
    index = point.index()
    return index[0] if isinstance(index, tuple) else index


def _fix_values(model, element, base_dir) -> list[tuple]:
    """Return ``(point, value)`` for every point a fix element sets."""
    label = element_label(element)
    points = variable_points(model, element.variable)
    try:
        time_index = find_time_block(model).time_index
    except FlexConfigError:
        time_index = None
    raw = resolve_sources(element.value, base_dir)
    if isinstance(raw, dict):
        try:
            raw = {int(key): value for key, value in raw.items()}
        except ValueError as exc:
            raise FlexConfigError(
                f"{label}: mapping keys must be integer time indices, got "
                f"{sorted(raw)[:5]}.",
                field="value",
            ) from exc
    times = {id(p): time_of(p, time_index) for p in points}
    if (isinstance(raw, list | dict) or element.at is not None) and None in set(
        times.values()
    ):
        raise FlexConfigError(
            f"{label}: {element.variable!r} has no time index, so give one number "
            "and no at.",
            field="value",
        )
    if element.at is not None:
        unknown = sorted(set(element.at) - set(time_index))
        if unknown:
            raise FlexConfigError(
                f"{label}: at names time indices {unknown} outside the horizon.",
                field="at",
                value=unknown,
            )
        points = [p for p in points if times[id(p)] in element.at]
    if isinstance(raw, list):
        keys = list(time_index) if element.at is None else element.at
        if len(raw) != len(keys):
            per = "time point" if element.at is None else "entry of at"
            raise FlexConfigError(
                f"{label}: expected {len(keys)} values (one per {per}), got "
                f"{len(raw)}.",
                field="value",
            )
        raw = dict(zip(keys, raw, strict=True))
    if isinstance(raw, dict):
        missing = sorted({times[id(p)] for p in points} - set(raw))
        if missing:
            raise FlexConfigError(
                f"{label}: the values are missing time indices {missing}.",
                field="value",
                value=missing,
            )
        values = [(p, float(raw[times[id(p)]])) for p in points]
    else:
        values = [(p, float(raw)) for p in points]
    if element.units is None:
        return values
    return [(p, _convert(p, v, element.units, label)) for p, v in values]


def _convert(point, value: float, units: str, label: str) -> float:
    """Convert ``value`` from ``units`` to the point's own units."""
    target = pyunits.get_units(point)
    if target is None or str(target) == "dimensionless":
        raise FlexConfigError(
            f"{label}: {point.name} has no units; remove units from the element.",
            field="units",
            value=units,
        )
    try:
        return pyunits.convert_value(
            value, from_units=parse_units(units), to_units=target
        )
    except UnitsError as exc:
        raise FlexConfigError(
            f"{label}: can't convert {units!r} to {point.name}'s units {target}.",
            field="units",
            value=units,
        ) from exc


def _dispatch_values(model, element, base_dir) -> list[tuple]:
    """Return ``(point, value)`` over the horizon for each member of a dispatch unit."""
    unit = model.find_component(element.unit)
    if unit is None:
        raise FlexConfigError(
            f"Unit {element.unit!r} is not on the model. Name a unit element "
            "that was built.",
            field="unit",
            value=element.unit,
        )
    raw = resolve_sources(element.values, base_dir)
    # JSON keys are always strings; integer time indices come back as "0".
    series = (
        dict(enumerate(raw))
        if isinstance(raw, list)
        else {int(k) if k.lstrip("-").isdigit() else k: v for k, v in raw.items()}
    )
    values = []
    for member in unit.values() if unit.is_indexed() else [unit]:
        var = member.find_component(element.variable)
        if var is None or not var.is_indexed():
            raise FlexConfigError(
                f"dispatch names variable {element.variable!r}, which is not on "
                f"{member.name!r} as a time-indexed variable.",
                field="variable",
                value=element.variable,
            )
        time_block = member._find_time_block()
        resolved = member._resolve_dispatch_series(series, time_block)
        if set(resolved) != set(time_block.time_index):
            raise FlexConfigError(
                f"{element_label(element)}: expected one value per time point "
                f"({time_block.n_points}), got indices {sorted(resolved)}.",
                field="values",
            )
        values += [(var[t], resolved[t]) for t in time_block.time_index]
    return values


def element_points(model, element, base_dir: Path | None = None) -> list[tuple]:
    """Return ``(point, value)`` for every point a state element touches.

    Args:
        model: The model.
        element: A fix, unfix or dispatch element.
        base_dir: Directory ``$source`` paths resolve against.

    Returns:
        Each ``VarData`` with the value it is set to (None for unfix).

    Raises:
        FlexConfigError: If the path or values don't fit the model.
    """
    if element.kind == "unfix":
        return [(p, None) for p in variable_points(model, element.variable)]
    if element.kind == "dispatch":
        return _dispatch_values(model, element, base_dir)
    return _fix_values(model, element, base_dir)


def _violation(point, value: float, tolerance: float):
    """Return the kind of violation a value is for a point, or None."""
    if value not in point.domain:
        return "domain"
    lb, ub = point.lb, point.ub
    if lb is not None and value < lb - tolerance:
        return "below_lb"
    if ub is not None and value > ub + tolerance:
        return "above_ub"
    return None


def _first(lines: list[str], limit: int = 10) -> str:
    """Join the first ``limit`` lines, noting how many more there are."""
    more = f"\n  ... and {len(lines) - limit} more" if len(lines) > limit else ""
    return "\n  " + "\n  ".join(lines[:limit]) + more


def apply_state(
    model,
    elements: list,
    *,
    base_dir: Path | None = None,
    on_violation: Literal["error", "warn", "skip"] = "error",
    tolerance: float = 1e-9,
) -> FixReport:
    """Apply fix, unfix and dispatch elements together, all or nothing.

    Args:
        model: The built model, changed in place.
        elements: Validated fix, unfix and dispatch elements, in any order.
        base_dir: Directory ``$source`` paths resolve against.
        on_violation: Policy for fix elements that don't set their own, and for
            dispatches.
        tolerance: Bounds tolerance for elements that don't set their own.

    Returns:
        The :class:`FixReport`.

    Raises:
        FlexConfigError: If an element doesn't fit the model, two elements set a
            point differently, or a value violates bounds or domain under
            ``"error"`` (or domain under ``"warn"``); nothing is changed then.

    Warns:
        FlexFixWarning: Once per element whose violating points were fixed
            anyway or skipped.
    """
    planned = [(el, element_points(model, el, base_dir)) for el in elements]
    entries: dict[int, list] = {}
    for element, points in planned:
        for point, value in points:
            entries.setdefault(id(point), []).append((point, element, value))

    conflicts = []
    for found in entries.values():
        values = [value for _, _, value in found]
        unfixing = [v is None for v in values]
        if (any(unfixing) and not all(unfixing)) or not all(
            math.isclose(v, values[0], rel_tol=1e-12) for v in values if v is not None
        ):
            sets = "; ".join(
                f"{element_label(el)} {'unfixes' if v is None else f'sets {v}'}"
                for _, el, v in found
            )
            conflicts.append(f"{found[0][0].name}: {sets}")
    if conflicts:
        raise FlexConfigError(
            f"{len(conflicts)} point(s) conflict: the spec is order-agnostic, so a "
            "point may be fixed to one value or unfixed, never both:"
            + _first(sorted(conflicts)),
            field="variable",
        )

    errors, violations, skipped = [], [], set()
    for found in entries.values():
        for point, element, value in found:
            if value is None:
                continue
            # Only fix elements carry their own policy and tolerance.
            own = element if element.kind == "fix" else None
            policy = (own and own.on_violation) or on_violation
            tol = tolerance if own is None or own.tolerance is None else own.tolerance
            kind = _violation(point, value, tol)
            if kind is None:
                continue
            label = element_label(element)
            if policy == "error" or (policy == "warn" and kind == "domain"):
                errors.append(
                    f"{point.name} = {value} (lb {point.lb}, ub {point.ub}): {kind}, "
                    f"from {label}"
                )
                continue
            action = "skipped" if policy == "skip" else "fixed"
            if action == "skipped":
                skipped.add(id(point))
            violations.append(
                FixViolation(point.name, value, point.lb, point.ub, kind, action, label)
            )
    if errors:
        raise FlexConfigError(
            f"{len(errors)} value(s) are outside their variable's bounds or domain, "
            "so nothing was fixed. Correct the values, or set on_violation to "
            "'warn' or 'skip' on purpose:" + _first(sorted(errors)),
            field="value",
        )

    report = FixReport(
        violations=sorted(violations, key=lambda v: (v.element, v.variable))
    )
    for key, found in entries.items():
        point, element, value = found[0]
        if key in skipped:
            report.skipped.append(point.name)
        elif value is None:
            point.unfix()
            report.unfixed.append(point.name)
        elif all(el.kind == "dispatch" and not el.fix for _, el, _ in found):
            point.set_value(value, skip_validation=True)
        else:
            # Checked above; skip Pyomo's own out-of-bounds log line.
            point.fix(value, skip_validation=True)
            report.fixed.append(point.name)
    for element, points in planned:
        if element.kind == "dispatch":
            _record_dispatch(model, element, points)
    for name in ("fixed", "unfixed", "skipped"):
        getattr(report, name).sort()
    _warn(report.violations)
    return report


def _record_dispatch(model, element, points: list[tuple]) -> None:
    """Record a dispatch on each unit member, as ``set_external_dispatch`` does."""
    unit = model.find_component(element.unit)
    members = list(unit.values()) if unit.is_indexed() else [unit]
    per_member = len(points) // len(members)
    for i, member in enumerate(members):
        member._flex_dispatch[element.variable] = {
            "values": [v for _, v in points[i * per_member : (i + 1) * per_member]],
            "fix": element.fix,
        }


def _warn(violations: list[FixViolation]) -> None:
    """Warn once per element, and log every violating point."""
    logger = get_logger(__name__)
    by_element: dict[str, list[FixViolation]] = {}
    for violation in violations:
        by_element.setdefault(violation.element, []).append(violation)
        logger.warning(
            f"{violation.element}: {violation.variable} = {violation.value} is "
            f"{violation.kind} (lb {violation.lb}, ub {violation.ub}); "
            f"{violation.action}"
        )
    for element, found in by_element.items():
        lines = [f"{v.variable} = {v.value} ({v.kind}): {v.action}" for v in found]
        warnings.warn(
            f"{element}: {len(found)} value(s) outside bounds or domain were "
            "fixed anyway or skipped (left free):" + _first(lines),
            FlexFixWarning,
            stacklevel=3,
        )
