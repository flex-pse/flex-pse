"""Indexed trains joined by directed, undirected and fan-in indexed arcs."""

import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.network import Arc

import flexops as fo
from flexops.costing import currency_units

EXPAND_ARCS = True


def build() -> pyo.ConcreteModel:
    """Build the model in code."""
    m = pyo.ConcreteModel(name="trains")
    m.time_block = fo.TimeBlock(
        start_date="2025-01-01", end_date="2025-01-02", time_step=1 * pyunits.hr
    )
    m.properties = fo.SimpleAqueousFlow()
    m.costing = fo.FlexCosting(
        time_block=m.time_block,
        energy_prices={"electrical": 0.1 * currency_units("USD") / pyunits.kWh},
    )
    m.p = fo.PlantBlock(time_block=m.time_block)
    p = m.p
    p.trains = pyo.Set(initialize=[0, 1, 2], ordered=True)
    p.train = fo.ConstantEnergyIntensityModel(
        p.trains,
        property_package=m.properties,
        energy_intensity=0.3 * pyunits.kWh / pyunits.m**3,
        costing_package=m.costing,
    )
    p.split = fo.Splitter(
        p.trains, property_package=m.properties, outlet_names=("a", "b")
    )
    p.out = fo.Product(p.trains, property_package=m.properties, inlet_names=("a",))
    p.mix = fo.Mixer(property_package=m.properties, inlet_names=("t0", "t1", "t2"))
    p.train_to_split = Arc(
        p.trains,
        rule=lambda b, i: {
            "source": b.train[i].outlet,
            "destination": b.split[i].inlet,
        },
    )
    p.split_to_out = Arc(
        p.trains, rule=lambda b, i: (b.split[i].outlet_a, b.out[i].inlet_a)
    )
    p.split_to_mix = Arc(
        p.trains,
        rule=lambda b, i: {
            "source": b.split[i].outlet_b,
            "destination": b.mix.find_component(f"inlet_t{i}"),
        },
    )
    m.costing.cost_process()
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
    return m
