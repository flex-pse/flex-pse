"""One electricity price per time point over 3,000 points."""

import pyomo.environ as pyo
from pyomo.environ import units as pyunits

import flexops as fo
from flexops.costing import currency_units

EXPAND_ARCS = False

N_POINTS = 3000


def build() -> pyo.ConcreteModel:
    """Build the model in code."""
    m = pyo.ConcreteModel(name="long_prices")
    m.time_block = fo.TimeBlock(
        start_date="2025-01-01", end_date="2025-01-21T20:00", time_step=10 * pyunits.min
    )
    price_units = currency_units("USD") / pyunits.kWh
    prices = [(0.08 + 0.04 * (t % 144 >= 96)) * price_units for t in range(N_POINTS)]
    m.properties = fo.SimpleAqueousFlow()
    m.costing = fo.FlexCosting(
        time_block=m.time_block, energy_prices={"electrical": prices}
    )
    m.site = fo.PlantBlock(time_block=m.time_block)
    m.site.plant = fo.ConstantEnergyIntensityModel(
        property_package=m.properties,
        energy_intensity=0.5 * pyunits.kWh / pyunits.m**3,
        costing_package=m.costing,
    )
    m.costing.cost_process()
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
    return m
