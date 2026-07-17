"""Canonical power-variable names for flex-pse (project standard, §4).

Every unit model registers at least one power draw through
:meth:`flexops.core.ops_block.OpsBlockData.register_power`, using one of the
names defined here. Both are **powers in kW** despite the domain word "energy":
:data:`POWER_ELECTRICAL` is a unit's electrical draw, :data:`POWER_THERMAL` its
heat/gas-driven duty. FlexCosting aggregates these into a kW time series and
hands it to EECO, which derives kWh internally from the time step (kW only ever
crosses the costing boundary).

The names live here as constants so a typo becomes an import error rather than a
silently-unaggregated variable. Do not hand-type the string values anywhere
else, and never introduce a variable named bare ``power``, ``energy``, or
``work`` (``plan/00_conventions.md`` §2, ``plan/01_architecture.md`` §4).
"""

import enum

POWER_ELECTRICAL = "power_electrical"
"""str: name of a unit's electrical draw Var (kW)."""

POWER_THERMAL = "power_thermal"
"""str: name of a unit's thermal/gas-driven duty Var (kW)."""


class PowerKind(enum.StrEnum):
    """The kinds of power draw a unit can declare.

    Each member's value is the ``kind`` string accepted by ``register_power``
    and ``declare_power`` and stored on an
    :class:`flexops.core.registration.PowerRecord`.
    """

    ELECTRICAL = "electrical"
    THERMAL = "thermal"
