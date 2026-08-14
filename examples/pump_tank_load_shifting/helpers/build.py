"""Build and solve the pump + tank (+ battery) model from an :class:`ExampleConfig`.

Mirrors ``flexops.tests.costing.test_load_shifting_component``'s headline
result: minimizing ``FlexCosting`` operating cost under a time-of-use tariff
shifts pumping (and battery discharge) out of the peak window. The model
itself -- TimeBlock, properties, costing, plant, units, and arcs -- is built
entirely by :func:`flexops.core.build.build_model` from ``config.model``.
Only the facility's fixed draw, the pump's flow cap, and its unit-commitment
relaxation are not expressible in a persisted config today (nothing in the
build path consumes ``io_variables``, and LP relaxation is a runtime-only
domain switch, see :mod:`flexops.logic.status`) and are applied here,
post-build.
"""

import flexops as fo
import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.opt import assert_optimal_termination

from flexcore.solvers import get_solver
from flexops.core.build import parse_quantity
from flexops.logic import add_startup_shutdown, add_status, relax

from .config import ExampleConfig


def build_model(config: ExampleConfig) -> pyo.ConcreteModel:
    """Build the unsolved pump (+ tank + battery) model described by ``config``.

    Args:
        config: The validated example config.

    Returns:
        The unsolved ``ConcreteModel``, with arcs expanded.
    """
    m = fo.build_model(config.model)
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)
    plant = m.find_component(config.model.plant.name)

    draw = pyo.value(
        pyunits.convert(parse_quantity(config.facility_draw), pyunits.m**3 / pyunits.hr)
    )
    max_flow = pyo.value(
        pyunits.convert(parse_quantity(config.pump_max_flow), pyunits.m**3 / pyunits.hr)
    )
    for t in m.time_block.time_index:
        plant.tank.outlet_state.flow_vol_phase[t, "Liq"].fix(draw)
        pump_flow = plant.pump.inlet_state.flow_vol_phase[t, "Liq"]
        pump_flow.setlb(0.0)
        pump_flow.setub(max_flow)

    uc = plant.pump.config.unit_commitment
    if uc.status:
        # Constant-intensity relation (power = energy_intensity * flow): a
        # flow bound converts directly to a power bound. min_on_power covers
        # the fixed facility draw so "always on" stays feasible.
        # energy_intensity is already a fixed Var on the built Pump, in
        # kWh/m^3 (OpsBlockData.add_constant_intensity_relation) -- read it
        # back rather than re-parsing config.
        energy_intensity = pyo.value(plant.pump.energy_intensity)
        status = add_status(
            plant.pump,
            plant.pump.power_electrical,
            energy_intensity * draw * pyunits.kW,
            energy_intensity * max_flow * pyunits.kW,
        )
        if uc.startup_shutdown:
            add_startup_shutdown(
                plant.pump, status, min_uptime=uc.min_up, min_downtime=uc.min_down
            )
        if config.pump_relax:
            # First-class LP relaxation: same UC structure, domain switched
            # Binary -> UnitInterval, no rebuild.
            relax(plant.pump)

    last = list(m.time_block.time_index)[-1]
    m.terminal = pyo.Constraint(
        expr=plant.tank.volume[last] >= plant.tank.initial_volume
    )
    if "battery" in config.model.plant.units:
        # Sustainable arbitrage: don't let the optimizer dump all stored
        # energy for a one-time credit at the horizon end.
        m.battery_terminal = pyo.Constraint(
            expr=plant.battery.charge[last] >= plant.battery.charge_init
        )
    return m


def solve_model(model: pyo.ConcreteModel):
    """Solve ``model`` with HiGHS and assert optimal termination.

    Args:
        model: The built (unsolved) model.

    Returns:
        The Pyomo results object.
    """
    # A 1% MIP gap keeps an exact (binary) pump-UC solve interactive-scale;
    # harmless (ignored) for the LP/relaxed cases.
    results = get_solver(model=model, prefer="highs").solve(
        model, options={"mip_rel_gap": 0.01}
    )
    assert_optimal_termination(results)
    return results
