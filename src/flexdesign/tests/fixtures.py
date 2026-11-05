"""Design studies shared by the flexdesign tests, prorated to one day per month."""

from flexcore.config.design import DesignSpec
from flexcore.config.spec import SCHEMA_VERSION


def _base_elements(costing: dict, *, battery: bool) -> list[dict]:
    """Return a pump load, solar and optionally a battery, with one costing element."""
    elements = [
        {
            "kind": "time",
            "start_date": "2025-01-01",
            "end_date": "2025-01-02",
            "time_step": "1 hr",
        },
        {
            "kind": "property_package",
            "name": "properties",
            "property_class": "SimpleAqueousFlow",
            "options": {
                "has_pressure": True,
                "density": {"value": 1000.0, "units": "kg/m^3"},
            },
        },
        costing,
        {
            "kind": "unit",
            "name": "pump",
            "unit_model_class": "Pump",
            "construction_options": {"power_relation": "hydraulic", "efficiency": 0.75},
        },
        {
            "kind": "unit",
            "name": "pv",
            "unit_model_class": "GenericRenewables",
            "property_package": None,
            "construction_options": {
                "capacity": {"value": 500.0, "units": "kW"},
                "capacity_factor": [0.5] * 24,
            },
        },
        {"kind": "fix", "variable": "pump.delta_pressure", "value": 300000.0},
        {"kind": "fix", "variable": "pump.inlet_state.pressure", "value": 100000.0},
        {
            "kind": "fix",
            "name": "load",
            "variable": "pump.inlet_state.flow_vol_phase[*,Liq]",
            "value": 30.0,
        },
        {"kind": "objective"},
    ]
    if battery:
        elements.append(
            {
                "kind": "unit",
                "name": "battery",
                "unit_model_class": "BatteryModel",
                "property_package": None,
                "construction_options": {
                    "capacity": {"value": 5000.0, "units": "kWh"},
                    "power_charge_max": {"value": 2000.0, "units": "kW"},
                    "power_discharge_max": {"value": 2000.0, "units": "kW"},
                    "eta_charge": 0.95,
                    "eta_discharge": 0.95,
                    "soc_min": 0.0,
                    "soc_max": 1.0,
                    "soh": 1.0,
                    "cyclic": True,
                },
            }
        )
    return elements


def native_costing(rate: float = 0.07) -> dict:
    """Return a costing element with a flat native price and no export credit."""
    return {
        "kind": "costing",
        "name": "costing",
        "energy_prices": {"electrical": {"value": 0.10, "units": "USD/kWh"}},
        "export_prices": {"electrical": {"value": 0.0, "units": "USD/kWh"}},
        "discount_rate": rate,
    }


def pump_battery_solar_study(
    *,
    energy_cost: float = 300.0,
    power_cost: float = 200.0,
    solar_cost: float = 1200.0,
    solar_upper: float = 1500.0,
    costing: dict | None = None,
    **overrides,
) -> DesignSpec:
    """Return a 10-year study with 2 modeled years and 2 representative months."""
    sun = [0.0] * 6 + [0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.6, 0.4, 0.2]
    sun += [0.0] * 6
    price = [0.08] * 16 + [0.30] * 5 + [0.08] * 3
    changes = []
    for scale, month in ((0.5, 1), (1.0, 7)):
        shaped = [scale * value for value in sun]
        changes.append(
            {
                "element": "pv",
                "months": [month],
                "values": {"construction_options": {"capacity_factor": shaped}},
            }
        )
    if costing is None:
        costing = native_costing()
        for month in (1, 7):
            changes.append(
                {
                    "element": "costing",
                    "months": [month],
                    "values": {
                        "energy_prices": {"electrical": {"value": price}},
                        "export_prices": {"electrical": {"value": [0.0] * 24}},
                    },
                }
            )
    changes.append({"element": "load", "years": [2031], "values": {"value": 60.0}})
    fields = {
        "schema_version": "0.1.0",
        "name": "pump_battery_solar",
        "base": {
            "schema_version": SCHEMA_VERSION,
            "name": "base",
            "elements": _base_elements(costing, battery=True),
        },
        "years": [{"year": 2026, "represents": 5}, {"year": 2031, "represents": 5}],
        "months": [
            {"month": 1, "weight": 6.0, "days": 1},
            {"month": 7, "weight": 6.0, "days": 1},
        ],
        "changes": changes,
        "capacities": [
            {
                "name": "battery_energy",
                "variable": "battery.capacity",
                "capital_cost": energy_cost,
                "lifetime_years": 15,
                "upper": 1500.0,
            },
            {
                "name": "battery_power",
                "variable": "battery.power_capacity",
                "capital_cost": power_cost,
                "lifetime_years": 15,
                "upper": 1500.0,
            },
            {
                "name": "solar",
                "variable": "pv.capacity",
                "capital_cost": solar_cost,
                "lifetime_years": 25,
                "upper": solar_upper,
            },
        ],
    }
    return DesignSpec.model_validate({**fields, **overrides})


def solar_only_study(
    *,
    years: list[dict],
    weights: tuple[float, float],
    changes: list[dict],
    capital_cost: float,
    lifetime_years: float,
    rate: float,
    lower: float = 0.0,
    days: int | None = None,
) -> DesignSpec:
    """Return a pump and solar study over January and July with flat prices."""
    return DesignSpec.model_validate(
        {
            "schema_version": "0.1.0",
            "name": "solar_only",
            "base": {
                "schema_version": SCHEMA_VERSION,
                "name": "base",
                "elements": _base_elements(native_costing(rate), battery=False),
            },
            "years": years,
            "months": [
                {"month": 1, "weight": weights[0], "days": days},
                {"month": 7, "weight": weights[1], "days": days},
            ],
            "changes": changes,
            "capacities": [
                {
                    "name": "solar",
                    "variable": "pv.capacity",
                    "capital_cost": capital_cost,
                    "lifetime_years": lifetime_years,
                    "lower": lower,
                    "upper": 500.0,
                }
            ],
        }
    )
