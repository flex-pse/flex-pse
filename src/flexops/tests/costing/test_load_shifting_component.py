"""Component-tier end-to-end test: pump+tank LP shifts load off-peak.

The headline economic result. A pump feeds a storage tank against the demo
tariff; minimizing the FlexCosting operating cost pushes all pumping out of the
peak window. Each test solves a small LP with HiGHS in well under 10 s.
"""

from pathlib import Path

import numpy as np
import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits
from pyomo.network import Arc
from pyomo.opt import assert_optimal_termination

from flexops.core.time_block import TimeBlock
from flexops.costing import FlexCosting, evaluate_cost, load_tariff
from flexops.properties.simple_aqueous import SimpleAqueousFlow
from flexops.unit_models import Pump, Tank

_FIXTURES = Path(__file__).parent.parent / "fixtures"
_TARIFF_JSON = _FIXTURES / "tariff_tou_demo.json"

# Peak window is 16:00-21:00 (hour_end exclusive) -> hours 16..20.
_PEAK_HOURS = (16, 17, 18, 19, 20)

# Regression baseline: the optimal operating cost ($) of the headline LP under
# the demo tariff, recorded from a verified HiGHS run (2026-07-28).
# Changing this is a deliberate diff.
#
# Re-recorded on 2026-07-28 (was 1467.388888888889, recorded 2026-07-23) when
# monthly-charge prorating became the default: this horizon is 24 h of a 31-day
# month, so the $150/month customer charge and the $21.50 + $19.00 per kW monthly
# demand charges are now billed at 24/744 of their monthly amount instead of in
# full. The energy charge is unaffected.
_EXPECTED_OBJECTIVE = 147.4964157706093


def _build_headline(tariff=None) -> pyo.ConcreteModel:
    """Pump -> Arc -> Tank + FlexCosting; objective = operating cost.

    Args:
        tariff: An EECO tariff object to pass as ``tariff=``; when ``None`` the
            demo tariff file is loaded via ``tariff_file=``.
    """
    m = pyo.ConcreteModel()
    m.time_block = TimeBlock(
        start_date="2025-07-08", end_date="2025-07-09", time_step=1 * pyunits.hr
    )
    m.properties = SimpleAqueousFlow(fixed_density=True)
    if tariff is None:
        m.costing = FlexCosting(time_block=m.time_block, tariff_file=str(_TARIFF_JSON))
    else:
        m.costing = FlexCosting(time_block=m.time_block, tariff=tariff)

    m.pump = Pump(
        property_package=m.properties,
        energy_intensity=0.5 * pyunits.kWh / pyunits.m**3,
        costing_package=m.costing,
    )
    m.tank = Tank(
        property_package=m.properties,
        max_volume=1000 * pyunits.m**3,
        initial_volume=200 * pyunits.m**3,
    )
    m.arc = Arc(source=m.pump.outlet, destination=m.tank.inlet)
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    for t in m.time_block.time_index:
        m.tank.outlet_state.flow_vol_phase[t, "Liq"].fix(100.0)
        pump_flow = m.pump.inlet_state.flow_vol_phase[t, "Liq"]
        pump_flow.setlb(0.0)
        pump_flow.setub(300.0)

    m.costing.cost_process()
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
    # Terminal condition: end at least as full as we started (else the LP just
    # drains the tank and never pumps).
    m.terminal = pyo.Constraint(expr=m.tank.volume[23] >= 200.0)
    return m


@pytest.mark.component
@pytest.mark.needs_highs
def test_load_shifting_headline():
    """The optimal LP pushes all pumping out of the peak window."""
    from flexcore.solvers import get_solver

    m = _build_headline()
    results = get_solver(model=m, prefer="highs").solve(m)
    assert_optimal_termination(results)

    peak_pumping = sum(
        pyo.value(m.pump.inlet_state.flow_vol_phase[t, "Liq"]) for t in _PEAK_HOURS
    )
    assert peak_pumping == pytest.approx(0.0, abs=1e-6)
    assert pyo.value(m.objective) == pytest.approx(_EXPECTED_OBJECTIVE, rel=1e-6)


@pytest.mark.component
@pytest.mark.needs_highs
def test_report_cost_post_hoc():
    """report_cost is an independent EECO post-hoc recomputation on realized power."""
    from flexcore.solvers import get_solver

    m = _build_headline()
    get_solver(model=m, prefer="highs").solve(m)

    report = m.costing.report_cost(m)

    tb = m.time_block
    dt_hours = pyo.value(pyunits.convert(tb.dt, pyunits.hr))
    realized = np.array(
        [pyo.value(m.costing.aggregate_electrical_power[t]) for t in tb.time_index]
    )
    independent = evaluate_cost(
        realized, load_tariff(_TARIFF_JSON), dt_hours, time_index=tb.datetime_index
    )

    # The reported bill is recomputed post-hoc, never read off the objective.
    assert report.operating.electricity == pytest.approx(independent, rel=1e-9)
    assert report.operating.fuel == pytest.approx(0.0)
    assert report.operating.fixed == pytest.approx(0.0)
    assert report.operating.dr_revenue == pytest.approx(0.0)
    assert report.capital.by_component == {}
    assert report.capital.total == pytest.approx(0.0)
    assert report.operating.total == pytest.approx(report.operating.electricity)
    assert report.total == pytest.approx(report.operating.total)
    # On this short horizon the convex relaxation is tight, so the reported bill
    # coincides with the relaxed objective; the reporting rule is encoded by
    # the independent recomputation above, not by trusting the objective. The two
    # diverge once the tiered surcharge is reached.
    assert report.operating.total == pytest.approx(pyo.value(m.objective), rel=1e-6)


@pytest.mark.component
@pytest.mark.needs_highs
def test_demand_charge_reduces_peak():
    """Removing the demand charges raises the optimal peak aggregate power."""
    from flexcore.solvers import get_solver

    def _peak(tariff) -> float:
        m = _build_headline(tariff=tariff)
        get_solver(model=m, prefer="highs").solve(m)
        return max(
            pyo.value(m.costing.aggregate_electrical_power[t])
            for t in m.time_block.time_index
        )

    full = load_tariff(_TARIFF_JSON)
    no_demand = full[full["type"] != "demand"].reset_index(drop=True)

    peak_with = _peak(full)
    peak_without = _peak(no_demand)
    # Demand charges flatten the profile -> a strictly lower peak.
    assert peak_with < peak_without - 1e-6
