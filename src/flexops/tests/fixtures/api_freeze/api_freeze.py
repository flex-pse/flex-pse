# The frozen flex-pse public API, verbatim from PLAN.md §2.
#
# BREAKING-CHANGE TRIPWIRE. From milestone M09 on, any pull request that has to
# edit this file to keep src/flexops/tests/test_api_freeze.py green is making a
# breaking change and must say so in its description. Do not "improve" it.
#
# The same tripwire covers api_freeze_config.json, this script's config-driven
# twin: it must keep describing the same model, and the test holds the two to
# the same solved objective. That notice lives here because JSON has no comment
# syntax and the config schema forbids undocumented keys.
#
# It runs with the working directory set to where its data fixtures live
# (the sibling data/ directory), since it loads them by bare filename. Its arcs
# are not expanded here: applying pyo.TransformationFactory("network.expand_arcs")
# is the caller's explicit step, as is the solve.
import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.network import Arc
import flexops as fo

m = pyo.ConcreteModel()
m.time_block = fo.TimeBlock(
    start_date="2025-01-01", end_date="2025-01-30", time_step=15 * pyunits.min
)
m.properties = fo.SimpleAqueousFlow()
m.costing = fo.FlexCosting(
    time_block=m.time_block,
    tariff_file="tariff.json",
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
