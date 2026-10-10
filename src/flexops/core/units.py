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
