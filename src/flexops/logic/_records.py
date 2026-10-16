"""Records of logic-helper calls, so emit_model can write them back as elements."""

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

RECORDING_SUPPRESSED = contextvars.ContextVar(
    "flex_suppress_logic_records", default=False
)
"""True while logic helpers are called internally, so they leave no record."""


@dataclass
class LogicRecord:
    """One logic-helper call, as the fields of the spec element that rebuilds it.

    Attributes:
        kind: The spec element kind.
        owner: The unit (unit-level kinds) or the model (cross-unit kinds).
        fields: The element's fields as JSON data: variable paths relative to
            the unit, units as full component names.
    """

    kind: str
    owner: Any
    fields: dict


@contextmanager
def suppress_logic_records():
    """Stop logic helpers recording while the block runs (for internal calls)."""
    token = RECORDING_SUPPRESSED.set(True)
    try:
        yield
    finally:
        RECORDING_SUPPRESSED.reset(token)


def record(owner, kind: str, **fields) -> None:
    """Append a :class:`LogicRecord` to ``owner._flex_logic`` unless suppressed.

    Args:
        owner: The unit or model the record belongs on.
        kind: The spec element kind.
        **fields: The element's fields, already JSON data.
    """
    if not RECORDING_SUPPRESSED.get():
        owner.__dict__.setdefault("_flex_logic", []).append(
            LogicRecord(kind, owner, fields)
        )


def variable_path(var, unit) -> str:
    """Return ``var``'s path relative to ``unit``, or its full name if it isn't on it.

    Args:
        var: The variable a helper was given.
        unit: The unit the helper builds on.

    Returns:
        The relative path; a variable outside ``unit`` keeps its full name, which
        emit_model's rebuild check then reports.
    """
    try:
        return var.getname(fully_qualified=True, relative_to=unit)
    except RuntimeError:
        return var.getname(fully_qualified=True)
