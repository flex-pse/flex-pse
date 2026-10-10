"""An aqueous Pump and a gas Combustor, each on its own property package."""

import pyomo.environ as pyo
from pyomo.environ import units as pyunits

import flexops as fo
from flexops.costing import currency_units

EXPAND_ARCS = False


def build() -> pyo.ConcreteModel:
    """Build the model in code."""
    m = pyo.ConcreteModel(name="two_packages")
    m.time_block = fo.TimeBlock(
        start_date="2025-01-01", end_date="2025-01-02", time_step=1 * pyunits.hr
    )
    m.water = fo.SimpleAqueousFlow(has_pressure=True)
    m.gas = fo.SimpleGasFlow()
    m.costing = fo.FlexCosting(
        time_block=m.time_block,
        energy_prices={"electrical": 0.1 * currency_units("USD") / pyunits.kWh},
    )
    m.site = fo.PlantBlock(time_block=m.time_block)
    m.site.pump = fo.Pump(property_package=m.water, costing_package=m.costing)
    m.site.combustor = fo.Combustor(property_package=m.gas, inlet_names=("fuel",))
    m.costing.cost_process()
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
    return m
