"""Turn live construction values and Pyomo units into plain spec data."""

import datetime
import enum
import math
from collections.abc import Mapping

import pyomo.environ as pyo
from pydantic import BaseModel
from pyomo.core.expr.numvalue import NumericValue
from pyomo.environ import units as pyunits

from flexcore.config.spec import PACKAGE_REF
from flexcore.exceptions import FlexConfigError
from flexops.core.units import parse_units


def units_to_str(units, *, where: str = "a value") -> str:
    """Write Pyomo units as a string that parse_units reads back exactly.

    Args:
        units: A Pyomo units expression.
        where: What the units belong to, named in the error.

    Returns:
        The units with ``^`` exponents, e.g. ``'kWh/m^3'``.

    Raises:
        FlexConfigError: If the string does not parse back to the same units.
    """
    text = str(units).replace("**", "^")
    try:
        factor = pyunits.convert_value(
            1.0, from_units=parse_units(text), to_units=units
        )
    except Exception as exc:
        raise FlexConfigError(
            f"Units {units} of {where} can't be written as a spec string yet",
            field=where,
        ) from exc
    if not math.isclose(factor, 1.0, rel_tol=1e-12):
        raise FlexConfigError(
            f"Units {units} of {where} can't be written as a spec string yet",
            field=where,
        )
    return text


def _quantity(value, units, where: str):
    """Return ``value`` alone if ``units`` are dimensionless, else value and units."""
    if units is None or units == pyunits.dimensionless:
        return value
    return {"value": value, "units": units_to_str(units, where=where)}


def to_jsonable(value, *, where: str, packages: Mapping | None = None):
    """Convert a construction-option value to plain JSON data.

    Args:
        value: The live value.
        where: The option's name, used in errors.
        packages: Package element names mapped to their blocks; a value that is
            one of these blocks becomes ``{"$package": name}``.

    Returns:
        JSON data a spec can hold.

    Raises:
        FlexConfigError: If the value, or anything inside it, has no spec form.
    """
    for name, block in (packages or {}).items():
        if value is block:
            return {PACKAGE_REF: name}
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise FlexConfigError(
                f"{where} is {value}, which has no spec form", field=where
            )
        return value
    if getattr(value, "is_indexed", None) and value.is_indexed():
        members = [value[i] for i in value.index_set()]
        units = pyunits.get_units(members[0]) if members else None
        return _quantity([pyo.value(v) for v in members], units, where)
    if isinstance(value, NumericValue):
        return _quantity(pyo.value(value), pyunits.get_units(value), where)
    if isinstance(value, (tuple, list)):
        return [
            to_jsonable(item, where=f"{where}[{i}]", packages=packages)
            for i, item in enumerate(value)
        ]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {
            key: to_jsonable(item, where=f"{where}.{key}", packages=packages)
            for key, item in value.items()
        }
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    raise FlexConfigError(
        f"{where} holds a {type(value).__name__}, which has no spec form",
        field=where,
    )
