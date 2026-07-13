"""Component-tier system solve: Pump → Arc → StorageTank over 24 hourly steps."""

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits
from pyomo.network import Arc

from flexcore.exceptions import FlexSolverError
from flexops import SimpleAqueousFlow, TimeBlock
from flexops.unit_models import Pump, StorageTank


@pytest.mark.component
@pytest.mark.needs_highs
def test_pump_fills_tank_lp():
    """Minimize pump energy while the tank meets a fixed 50 m³/hr demand.

    Asserts optimal termination and the mass-balance identity: total pumped
    volume equals total demand plus the initial→final holdup change. Flows at
    the last time point enter no holdup constraint (the difference equation
    stops at N-2), so the sums run over t = 0..N-2.
    """
    m = pyo.ConcreteModel()
    m.time_block = TimeBlock(
        start_date="2025-01-01",
        end_date="2025-01-02",
        time_step=1 * pyunits.hr,
    )
    m.properties = SimpleAqueousFlow(fixed_density=True)
    m.pump = Pump(property_package=m.properties)
    m.tank = StorageTank(
        property_package=m.properties,
        max_volume=1000.0,
        initial_volume=200.0,
    )
    m.pump_to_tank = Arc(source=m.pump.outlet, destination=m.tank.inlet)
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    t_index = m.time_block.time_index
    assert len(t_index) == 24
    for t in t_index:
        m.tank.flow_out[t].fix(50.0)
    m.obj = pyo.Objective(expr=sum(m.pump.electrical_power[t] for t in t_index))

    try:
        from flexcore.solvers import get_solver

        solver = get_solver(model=m)
    except (ImportError, FlexSolverError):
        pytest.skip(
            "flexcore.solvers.get_solver not available (M05 may land in parallel)"
        )
    results = solver.solve(m)
    assert pyo.check_optimal_termination(results)

    balanced = list(t_index)[:-1]
    dt_hr = pyo.value(pyunits.convert(m.time_block.dt, pyunits.hr))
    pumped = dt_hr * sum(pyo.value(m.tank.flow_in[t]) for t in balanced)
    demanded = dt_hr * sum(pyo.value(m.tank.flow_out[t]) for t in balanced)
    holdup_change = pyo.value(m.tank.V[len(t_index) - 1]) - pyo.value(m.tank.V[0])
    assert pumped == pytest.approx(demanded + holdup_change, rel=1e-6)
