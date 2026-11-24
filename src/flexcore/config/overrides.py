"""Apply element-level overrides to a flat spec, e.g. the scenarios of one base."""

import copy
from collections.abc import Iterable, Mapping

from pydantic import ValidationError

from flexcore.config.io import _format_validation_error, load_spec
from flexcore.config.spec import TARGET_FIELDS, FlowsheetSpec
from flexcore.exceptions import FlexConfigError

REMOVE = "$remove"
"""Key of an override that drops the element it matches."""


def deep_merge(target: dict, patch: Mapping) -> None:
    """Merge ``patch`` into ``target`` in place, recursing into nested dicts.

    Args:
        target: The dictionary to change.
        patch: The values to merge in. A value that isn't a dict replaces the
            target's, and so does a ``$source`` reference on either side.
    """
    for key, value in patch.items():
        current = target.get(key)
        if (
            isinstance(value, Mapping)
            and isinstance(current, dict)
            and "$source" not in value
            and "$source" not in current
        ):
            deep_merge(current, value)
        else:
            target[key] = copy.deepcopy(value)


def _keys(element: Mapping) -> list[tuple]:
    """Return what an element can be matched by: its name, then what it targets."""
    kind, name = element.get("kind"), element.get("name")
    keys = []
    if name is not None:
        keys.append(("name", name))
    elif kind in ("time", "objective"):
        keys.append(("kind", kind))
    fields = TARGET_FIELDS.get(kind, ())
    if fields and all(element.get(field) is not None for field in fields):
        keys.append((kind, *(str(element[field]) for field in fields)))
    return keys


def _match(elements: list[dict], override: Mapping) -> dict | None:
    """Return the one element ``override`` matches, or None.

    Raises:
        FlexConfigError: If the override identifies no element, or more than one.
    """
    keys = _keys(override)
    if not keys:
        raise FlexConfigError(
            "An override needs a 'name', or a 'kind' with the fields that say what "
            f"it sets (e.g. 'kind': 'fix' and 'variable'); got {dict(override)}.",
            field="overrides",
        )
    kind = override.get("kind")
    matches = [
        el for el in elements if keys[0] in _keys(el) and kind in (None, el["kind"])
    ]
    for key in keys[1:]:  # e.g. the same penalty name on two units
        if len(matches) > 1:
            matches = [el for el in matches if key in _keys(el)]
    if len(matches) > 1:
        raise FlexConfigError(
            f"The override {dict(override)} matches {len(matches)} elements. Add "
            "its 'kind' and what it sets, or name the element in the base spec.",
            field="overrides",
        )
    return matches[0] if matches else None


def apply_overrides(
    spec: FlowsheetSpec | Mapping, overrides: Iterable[Mapping]
) -> FlowsheetSpec:
    """Return a copy of a spec with element-level overrides applied in order.

    Each override is matched to an element by ``name``, by ``kind`` alone for the
    time element and an unnamed objective, or otherwise by ``kind`` and what the
    element sets (a ``fix`` by its ``variable``, a ``status`` by its ``unit``):

    - a match is deep-merged: dicts merge, anything else is replaced;
    - ``"$remove": true`` drops the match instead;
    - no match appends the override as a new element.

    Args:
        spec: The base spec, or a mapping to load one from; not changed.
        overrides: The overrides, as plain mappings.

    Returns:
        The validated spec, resolving file paths as ``spec`` does.

    Raises:
        FlexConfigError: If an override identifies no element or several, removes
            one that isn't there, adds one without a ``kind``, or leaves the spec
            invalid.

    Example:
        >>> scenario = apply_overrides(base, [
        ...     {"name": "pv", "construction_options": {"curtailable": True}},
        ...     {"kind": "fix", "variable": "pump.speed", "$remove": True},
        ... ])  # doctest: +SKIP
    """
    base = load_spec(spec)
    data = base.model_dump(mode="json", by_alias=True, exclude_unset=True)
    elements = data["elements"]
    for override in overrides:
        patch = {key: value for key, value in override.items() if key != REMOVE}
        element = _match(elements, patch)
        if override.get(REMOVE, False):
            if element is None:
                raise FlexConfigError(
                    f"Can't remove {patch}: no such element.", field="overrides"
                )
            elements.remove(element)
        elif element is not None:
            deep_merge(element, patch)
        elif "kind" not in patch:
            raise FlexConfigError(
                f"No element matches {patch}, and a new one needs a 'kind'.",
                field="overrides",
            )
        else:
            elements.append(copy.deepcopy(patch))
    try:
        result = FlowsheetSpec.model_validate(data)
    except ValidationError as exc:
        raise FlexConfigError(_format_validation_error(exc)) from exc
    result._base_dir = base._base_dir
    return result
