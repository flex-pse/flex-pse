"""The frozen API script's model, priced with energy_prices instead of a tariff."""

import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.network import Arc

import flexops as fo
from flexops.costing import currency_units

EXPAND_ARCS = True


def build() -> pyo.ConcreteModel:
    """Build the model in code."""
    m = pyo.ConcreteModel(name="waterfacility")
    m.time_block = fo.TimeBlock(
        start_date="2025-01-01", end_date="2025-01-30", time_step=15 * pyunits.min
    )
    m.properties = fo.SimpleAqueousFlow()
    m.costing = fo.FlexCosting(
        time_block=m.time_block,
        energy_prices={"electrical": 0.12 * currency_units("USD") / pyunits.kWh},
    )
    m.waterfacility = fo.PlantBlock(time_block=m.time_block)
    m.waterfacility.tank = fo.Tank(property_package=m.properties)
    m.waterfacility.plant = fo.ConstantEnergyIntensityModel(
        property_package=m.properties,
        energy_intensity=0.5 * pyunits.kWh / pyunits.m**3,
        costing_package=m.costing,
    )
    m.waterfacility.tank_to_plant = Arc(
        source=m.waterfacility.tank.outlet, destination=m.waterfacility.plant.inlet
    )
    m.waterfacility.battery = fo.BatteryModel(
        capacity=1 * pyunits.kWh, costing_package=m.costing
    )
    m.costing.cost_process()
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
    return m
