"""Parse the compact units strings a persisted config or surrogate data uses.

A leaf module (imports only ``flexcore`` and Pyomo) so both ``build.py`` and
``flexops.surrogates`` can parse a units string without a cross-package
import cycle.
"""

import re

from pyomo.environ import units as pyunits

from flexcore.exceptions import FlexConfigError
from flexops.costing import currency_units

_UNIT_TOKEN = re.compile(r"^([A-Za-z$]+)(?:\*\*|\^)?(-?\d+)?$")


def parse_units(text: str):
    """Parse a units string into a Pyomo units expression.

    Handles the compact forms persisted configs use: ``"min"``, ``"m^3/hr"``,
    ``"kWh/m^3"``, ``"USD/kWh"``, and the ``"kg/m^2/s"`` and ``"1/s"`` forms Pyomo
    prints — ``*``-separated factors, ``^``/``**`` exponents, each factor after a
    ``/`` divides, and a bare ``1`` is dimensionless. A token Pyomo does not know
    is registered as a currency (so ``"USD"`` works without the costing block
    existing yet).

    Args:
        text: The units string.

    Returns:
        The corresponding Pyomo units expression.

    Raises:
        FlexConfigError: If a token is not a parsable unit name and exponent.
    """
    numerator, *denominators = text.strip().split("/")
    result = 1
    for side, factors in [(1, numerator), *((-1, d) for d in denominators)]:
        for token in factors.split("*") if factors.strip() else []:
            token = token.strip()
            if not token or token == "1":
                continue
            match = _UNIT_TOKEN.match(token)
            if match is None:
                raise FlexConfigError(
                    f"Could not parse {token!r} in units string {text!r}. Write "
                    "units as '*'-separated factors divided by '/', e.g. "
                    "'kg/m^2/s'.",
                    field="units",
                    value=text,
                )
            name, exponent = match.group(1), int(match.group(2) or 1)
            unit = getattr(pyunits, name, None)
            if unit is None:
                unit = currency_units(name)
            result = result * unit ** (side * exponent)
    return result


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
