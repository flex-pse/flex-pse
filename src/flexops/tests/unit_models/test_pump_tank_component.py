"""Component-tier integration test: Pump -> Arc -> Tank LP system."""

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits
from pyomo.network import Arc
from pyomo.opt import assert_optimal_termination

from flexcore.solvers import get_solver
from flexops.core.time_block import TimeBlock
from flexops.properties.simple_aqueous import SimpleAqueousFlow
from flexops.unit_models import Pump, Tank


@pytest.mark.component
@pytest.mark.needs_highs
def test_pump_fills_tank_lp():
    """Minimizing pumped power over a 24-hour horizon respects the tank mass balance."""
    m = pyo.ConcreteModel()
    m.time_block = TimeBlock(
        start_date="2025-01-01", end_date="2025-01-02", time_step=1 * pyunits.hr
    )
    m.properties = SimpleAqueousFlow()
    m.pump = Pump(property_package=m.properties)
    m.tank = Tank(
        property_package=m.properties,
        max_volume=1000 * pyunits.m**3,
        initial_volume=200 * pyunits.m**3,
    )
    m.arc = Arc(source=m.pump.outlet, destination=m.tank.inlet)
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    demand = 50.0
    for t in m.time_block.time_index:
        m.tank.outlet_state.flow_vol_phase[t, "Liq"].fix(demand)

    @m.Objective(sense=pyo.minimize)
    def total_power(b):
        return sum(b.pump.power_electrical[t] for t in b.time_block.time_index)

    solver = get_solver(model=m)
    results = solver.solve(m)
    assert_optimal_termination(results)

    n = m.time_block.n_points
    # Backward differencing: only flows at t=1..n-1 appear in a holdup
    # constraint (t=0's flow has no constraint tying it to volume, since
    # volume[0] is fixed by the initial condition, not a difference equation).
    total_in = sum(
        pyo.value(m.pump.inlet_state.flow_vol_phase[t, "Liq"]) for t in range(1, n)
    )
    total_demand = demand * (n - 1)
    initial_volume = pyo.value(m.tank.volume[0])
    final_volume = pyo.value(m.tank.volume[n - 1])
    assert total_in == pytest.approx(
        total_demand + (final_volume - initial_volume), rel=1e-6
    )
