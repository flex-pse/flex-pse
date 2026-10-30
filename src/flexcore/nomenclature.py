"""Canonical energy-variable names for flex-pse.

Every unit model registers at least one power draw through
:meth:`flexops.core.ops_block.OpsBlockData.register_power`, using one of the
power names defined here. Both are **powers in kW** despite the domain word
"energy": :data:`POWER_ELECTRICAL` is a unit's electrical draw and
:data:`POWER_THERMAL` its heat duty (at a registered temperature). FlexCosting
aggregates them into per-carrier kW time series and hands the electrical one to
EECO, which derives kWh internally from the time step.

**Fuel is not a power.** A combustible fuel is metered and billed on **volume**,
so a unit that burns fuel registers its :data:`FUEL_USAGE` flow — a volumetric
rate in m³/hr — through
:meth:`flexops.core.ops_block.OpsBlockData.register_fuel_usage`, and FlexCosting
hands that volume series to EECO. flex-pse assumes **no** heating value anywhere;
if a tariff prices gas on an energy basis, converting it is EECO's job (it
applies its own fuel heating-value assumption).

The names live here as constants so a typo becomes an import error rather than a
silently-unaggregated variable. Do not hand-type the string values anywhere
else, and never introduce a variable named bare ``power``, ``energy``, or
``work``.
"""

import enum

POWER_ELECTRICAL = "power_electrical"
"""str: name of a unit's electrical draw Var (kW)."""

POWER_THERMAL = "power_thermal"
"""str: name of a unit's thermal duty Var (kW), registered at a temperature."""

ELECTRICITY_RESOURCE = "electricity"
"""str: resource name electrical power is collected under. A draw consumes it,
an export supplies it."""

FUEL_USAGE = "fuel_usage"
"""str: name of a fuel-usage Var — a volumetric flow in m³/hr, **not** a power.
A unit burning one named fuel uses ``f"{FUEL_USAGE}_{fuel_name}"`` (e.g.
``fuel_usage_natural_gas``); a facility-level series is the bare name."""


class PowerKind(enum.StrEnum):
    """The kinds of power draw a unit can declare.

    Each member's value is the ``kind`` string accepted by ``register_power``
    and ``declare_power`` and stored on an
    :class:`flexops.core.registration.PowerRecord`. Fuel is absent by design —
    it is a volumetric flow (:data:`FUEL_USAGE`), not a power.
    """

    ELECTRICAL = "electrical"
    THERMAL = "thermal"


POWER_VARS = {
    PowerKind.ELECTRICAL: (POWER_ELECTRICAL, "Electrical draw of the unit"),
    PowerKind.THERMAL: (POWER_THERMAL, "Thermal/gas-driven duty of the unit"),
}
"""dict: per-``PowerKind`` ``(var name, doc)`` for a unit's own power Var,
consumed by :meth:`flexops.core.ops_block.OpsBlockData.declare_power`."""

INTENSITY_VARS = {
    PowerKind.ELECTRICAL: "energy_intensity",
    PowerKind.THERMAL: "thermal_intensity",
}
"""dict: per-``PowerKind`` name of a unit's constant-intensity Param."""

TOTAL_FEED = "total_feed"
"""str: name of the feed-flow aggregation Expression, indexed (resource, t)."""

TOTAL_PRODUCT = "total_product"
"""str: name of the product-flow aggregation Expression, indexed (product, t)."""

TOTAL_POWER_VARS = {
    PowerKind.ELECTRICAL: (
        "total_electrical_power",
        "Sum of the child units' power_electrical (kW).",
    ),
    PowerKind.THERMAL: (
        "total_thermal_power",
        "Sum of the child units' power_thermal (kW).",
    ),
}
"""dict: per-``PowerKind`` ``(Expression name, doc)`` for a plant's or
network's power-total aggregation, indexed over time."""

TOTAL_FUEL_USAGE = "total_fuel_usage"
"""str: name of the fuel-usage aggregation Expression, indexed (fuel_name, t)."""
