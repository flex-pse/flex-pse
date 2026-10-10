"""A Pump swapped to a multilinear surrogate, and a Battery under external dispatch."""

import pyomo.environ as pyo
from pyomo.environ import units as pyunits

import flexops as fo
from flexops.costing import currency_units

EXPAND_ARCS = False

SURROGATE = {
    "kind": "surrogate",
    "unit": "site.pump",
    "relation": "power_electrical_relation",
    "surrogate_type": "multilinear",
    "data": {
        "input_variables": {"flow_out": "m^3/hr"},
        "output_variables": {"power_electrical": "kW"},
        "coefficients": {"flow_out": 0.8, "intercept": 1.5},
    },
    "provenance": {"source": "datasheet"},
}


def build() -> pyo.ConcreteModel:
    """Build the model in code."""
    m = pyo.ConcreteModel(name="surrogate_and_dispatch")
    m.time_block = fo.TimeBlock(
        start_date="2025-01-01", end_date="2025-01-01T06:00", time_step=1 * pyunits.hr
    )
    m.properties = fo.SimpleAqueousFlow()
    m.costing = fo.FlexCosting(
        time_block=m.time_block,
        energy_prices={"electrical": 0.1 * currency_units("USD") / pyunits.kWh},
    )
    m.site = fo.PlantBlock(time_block=m.time_block)
    m.site.pump = fo.Pump(property_package=m.properties, costing_package=m.costing)
    m.site.battery = fo.BatteryModel(
        capacity=10 * pyunits.kWh, costing_package=m.costing
    )
    fo.apply_spec(m, [SURROGATE])
    charge = [0.0, 1.0, 2.0, 2.0, 0.5, 0.0]
    m.site.battery.set_external_dispatch(
        m.site.battery.power_charge, dict(enumerate(charge))
    )
    m.costing.cost_process()
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
    return m
