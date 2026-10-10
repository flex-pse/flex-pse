"""A network of two plants joined by one inter-plant arc."""

import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.network import Arc

import flexops as fo
from flexops.costing import currency_units

EXPAND_ARCS = False


def build() -> pyo.ConcreteModel:
    """Build the model in code."""
    m = pyo.ConcreteModel(name="network")
    m.time_block = fo.TimeBlock(
        start_date="2025-01-01", end_date="2025-01-02", time_step=1 * pyunits.hr
    )
    m.properties = fo.SimpleAqueousFlow()
    m.costing = fo.FlexCosting(
        time_block=m.time_block,
        energy_prices={"electrical": 0.1 * currency_units("USD") / pyunits.kWh},
    )
    m.net = fo.NetworkBlock(time_block=m.time_block)
    m.net.a = fo.PlantBlock(time_block=m.time_block)
    m.net.b = fo.PlantBlock(time_block=m.time_block)
    m.net.a.tank = fo.Tank(property_package=m.properties)
    m.net.a.pump = fo.ConstantEnergyIntensityModel(
        property_package=m.properties,
        energy_intensity=0.4 * pyunits.kWh / pyunits.m**3,
        costing_package=m.costing,
    )
    m.net.b.tank = fo.Tank(property_package=m.properties)
    m.net.a.tank_to_pump = Arc(
        source=m.net.a.tank.outlet, destination=m.net.a.pump.inlet
    )
    m.net.a_to_b = Arc(
        source=m.net.a.pump.outlet,
        destination=m.net.b.tank.inlet,
        doc="Transfer main from plant a to plant b.",
    )
    m.costing.cost_process()
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
    return m
