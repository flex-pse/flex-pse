"""Units directly on the model with two arcs, the pump-scheduling layout."""

import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.network import Arc

import flexops as fo
from flexops.costing import currency_units

EXPAND_ARCS = True


def build() -> pyo.ConcreteModel:
    """Build the model in code."""
    m = pyo.ConcreteModel(name="pump_scheduling")
    m.time_block = fo.TimeBlock(
        start_date="2025-07-08", end_date="2025-07-09", time_step=1 * pyunits.hr
    )
    m.properties = fo.SimpleAqueousFlow(
        has_pressure=True, density=998.0 * pyunits.kg / pyunits.m**3
    )
    m.costing = fo.FlexCosting(
        time_block=m.time_block,
        energy_prices={"electrical": 0.2 * currency_units("USD") / pyunits.kWh},
    )
    m.feed_pump = fo.Pump(
        property_package=m.properties,
        power_relation="hydraulic",
        efficiency=0.7,
        costing_package=m.costing,
    )
    m.tank = fo.Tank(
        property_package=m.properties,
        max_volume=500.0 * pyunits.m**3,
        initial_volume=250.0 * pyunits.m**3,
        level_min=0.1,
        level_max=0.95,
    )
    m.product_pump = fo.Pump(
        property_package=m.properties,
        power_relation="hydraulic",
        efficiency=0.75,
        costing_package=m.costing,
    )
    m.feed_to_tank = Arc(source=m.feed_pump.outlet, destination=m.tank.inlet)
    m.tank_to_product = Arc(source=m.tank.outlet, destination=m.product_pump.inlet)
    m.battery = fo.BatteryModel(
        capacity=40.0 * pyunits.kWh,
        power_charge_max=10.0 * pyunits.kW,
        costing_package=m.costing,
    )
    m.costing.cost_process()
    m.objective = pyo.Objective(
        expr=m.costing.aggregate_operating_cost, sense=pyo.minimize
    )
    return m
