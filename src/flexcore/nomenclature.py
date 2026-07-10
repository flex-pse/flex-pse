"""Canonical energy-variable names for flex-pse (project standard, §4).

Every unit model registers at least one energy draw through
:meth:`flexops.core.ops_block.OpsBlockData.register_energy`, using one of the
names defined here. Both are **powers in kW** despite the domain word "energy":
:data:`ELECTRICAL_POWER` is a unit's electrical draw, :data:`THERMAL_POWER` its
heat/gas-driven duty. FlexCosting aggregates these into a kW time series and
hands it to EECO, which derives kWh internally from the time step (kW only ever
crosses the costing boundary).

The names live here as constants so a typo becomes an import error rather than a
silently-unaggregated variable. Do not hand-type the string values anywhere
else, and never introduce a variable named bare ``power``, ``energy``, or
``work`` (``plan/00_conventions.md`` §2, ``plan/01_architecture.md`` §4).
"""

import enum

ELECTRICAL_POWER = "electrical_power"
"""str: name of a unit's electrical draw Var (kW)."""

THERMAL_POWER = "thermal_power"
"""str: name of a unit's thermal/gas-driven duty Var (kW)."""


class EnergyKind(enum.StrEnum):
    """The kinds of energy draw a unit can declare.

    Each member's value is the ``kind`` string accepted by ``register_energy``
    and ``declare_energy`` and stored on an
    :class:`flexops.core.registration.EnergyRecord`.
    """

    ELECTRICAL = "electrical"
    THERMAL = "thermal"
