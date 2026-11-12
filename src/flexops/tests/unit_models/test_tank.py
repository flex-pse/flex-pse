"""Harness-driven and hand tests for Tank(SISOBlock)."""

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits
from pyomo.opt import assert_optimal_termination

from flexcore.config.schema import UnitCommitmentConfig
from flexcore.solvers import get_solver
from flexops import build_model, emit_model
from flexops.core.compare import assert_models_equivalent
from flexops.testing import UnitModelTestHarness, dummy_time_block
from flexops.unit_models import Pump, Tank


class TestTank(UnitModelTestHarness):
    """Fixing both flows and the initial volume determines the holdup trajectory."""

    expected_dof = 0

    def configure(self):
        m = dummy_time_block(4)
        m.unit = Tank(
            property_package=m.properties,
            max_volume=1000 * pyunits.m**3,
            initial_volume=200 * pyunits.m**3,
        )
        return m, m.unit


def _tank(n: int = 4, **kwargs):
    """Build a fresh Tank on an ``n``-point dummy_time_block."""
    m = dummy_time_block(n)
    m.unit = Tank(
        property_package=m.properties,
        max_volume=1000 * pyunits.m**3,
        initial_volume=200 * pyunits.m**3,
        **kwargs,
    )
    return m, m.unit


@pytest.mark.unit
def test_mass_balance_by_hand():
    """Holdup-constraint bodies evaluate to 0 on a hand-computed trajectory."""
    m, unit = _tank(4)

    # Backward differencing: volume[t] = volume[t-1] + dt*(flow_in[t] - flow_out[t]),
    # with volume[-1] = initial_volume (200), so step 0's flows count too.
    flow_in = [100.0, 100.0, 0.0, 0.0]
    flow_out = [50.0, 50.0, 50.0, 50.0]
    volumes = [212.5, 225.0, 212.5, 200.0]

    for t in m.time_block.time_index:
        unit.inlet_state.flow_vol_phase[t, "Liq"].set_value(flow_in[t])
        unit.outlet_state.flow_vol_phase[t, "Liq"].set_value(flow_out[t])
        unit.volume[t].set_value(volumes[t])

    assert len(unit.holdup) == 4
    for t in unit.holdup:
        assert pyo.value(unit.holdup[t].body) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_capacity_bounded_by_max_volume():
    """capacity is bounded (min_volume, max_volume): the chosen size
    <= the maximum possible."""
    _, unit = _tank(4)
    assert unit.capacity.lb == pytest.approx(0.0)
    assert unit.capacity.ub == pytest.approx(1000.0)
    assert pyo.value(unit.capacity) == pytest.approx(1000.0)
    assert unit.capacity.fixed


@pytest.mark.unit
def test_level_bounds():
    """level is bounded (level_min, level_max); level_definition
    ties it to volume/capacity."""
    m, unit = _tank(4, level_min=0.1, level_max=0.9)

    for t in m.time_block.time_index:
        assert unit.level[t].lb == pytest.approx(0.1)
        assert unit.level[t].ub == pytest.approx(0.9)

    # capacity fixed at 1000 (default); volume=500 => level=0.5 satisfies the
    # defining constraint body exactly (no solver).
    unit.volume[0].set_value(500.0)
    unit.level[0].set_value(0.5)
    assert pyo.value(unit.level_definition[0].body) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_tank_logic_disabled():
    """A tank never builds a status var/UC constraints, even if UC is requested."""
    _, tank_default = _tank(4)
    assert not hasattr(tank_default, "status")
    assert tank_default.config.unit_commitment.status is False

    _, tank_uc_on = _tank(4, unit_commitment=UnitCommitmentConfig(status=True))
    assert not hasattr(tank_uc_on, "status")
    assert tank_uc_on.config.unit_commitment.status is False


@pytest.mark.unit
def test_pump_does_not_disable_logic():
    """Contrast with Pump, which leaves unit_commitment as configured."""
    m = dummy_time_block(3)
    m.unit = Pump(
        property_package=m.properties,
        unit_commitment=UnitCommitmentConfig(status=True),
    )
    assert m.unit.config.unit_commitment.status is True


@pytest.mark.unit
def test_fix_initial_state_registers_initial_state_or_frees_start():
    """initial_volume is a fixed Var registered as initial state, unless
    fix_initial_state=False, which leaves it free and unregistered."""
    m, unit = _tank(4)
    assert isinstance(unit.initial_volume, pyo.Var)
    assert unit.initial_volume.fixed
    assert pyo.value(unit.initial_volume) == pytest.approx(200.0)
    assert unit.initial_volume in m.time_block.initial_state_params
    assert not hasattr(unit, "initial_volume_eq")

    m_free, unit_free = _tank(4, fix_initial_state=False)
    assert not unit_free.initial_volume.fixed
    assert pyo.value(unit_free.initial_volume) == pytest.approx(200.0)
    assert not m_free.time_block.initial_state_params


@pytest.mark.unit
def test_initial_volume_can_be_swept_in_place():
    """update_parameters sets the fixed initial_volume Var without a rebuild."""
    _, unit = _tank(4)
    holdup_0 = unit.holdup[0]
    unit.update_parameters({"initial_volume": 350.0})
    assert unit.initial_volume.fixed
    assert pyo.value(unit.initial_volume) == pytest.approx(350.0)
    assert unit.holdup[0] is holdup_0


@pytest.mark.unit
def test_free_start_stays_in_fill_window_of_chosen_capacity():
    """initial_volume obeys level_min/level_max of capacity, like volume[t]."""
    _, unit = _tank(4, fix_initial_state=False, level_min=0.1, level_max=0.9)
    unit.capacity.fix(500.0)
    con = unit.initial_volume_limit
    for value, ok in [(40.0, False), (50.0, True), (450.0, True), (460.0, False)]:
        unit.initial_volume.set_value(value)
        body = pyo.value(con.body)
        inside = pyo.value(con.lower) - 1e-9 <= body <= pyo.value(con.upper) + 1e-9
        assert inside is ok, value


@pytest.mark.unit
def test_cyclic_tank_requires_final_volume_at_least_start():
    """cyclic=True adds volume[N-1] >= initial_volume, fixed or free; the
    default builds no such row."""
    _, unit_default = _tank(4)
    assert not hasattr(unit_default, "cyclic_volume")

    def satisfied(con):
        body = pyo.value(con.body)
        lower, upper = pyo.value(con.lower), pyo.value(con.upper)
        return (lower is None or body >= lower - 1e-9) and (
            upper is None or body <= upper + 1e-9
        )

    m, unit = _tank(4, cyclic=True)
    assert unit.initial_volume in m.time_block.initial_state_params
    unit.volume[0].set_value(100.0)  # volume[0] is not the start any more
    unit.volume[3].set_value(150.0)
    assert not satisfied(unit.cyclic_volume)  # 150 < initial_volume (200)
    unit.volume[3].set_value(200.0)
    assert satisfied(unit.cyclic_volume)

    _, unit_free = _tank(4, cyclic=True, fix_initial_state=False)
    unit_free.initial_volume.set_value(300.0)
    unit_free.volume[3].set_value(250.0)
    assert not satisfied(unit_free.cyclic_volume)
    unit_free.volume[3].set_value(300.0)
    assert satisfied(unit_free.cyclic_volume)


@pytest.mark.unit
def test_tank_storage_options_survive_emit_and_rebuild():
    """cyclic and fix_initial_state round-trip through emit_model and build_model."""
    m, _ = _tank(4, cyclic=True, fix_initial_state=False)
    rebuilt = build_model(emit_model(m))
    assert rebuilt.unit.config.cyclic is True
    assert rebuilt.unit.config.fix_initial_state is False
    assert_models_equivalent(m, rebuilt)


@pytest.mark.component
@pytest.mark.needs_highs
@pytest.mark.parametrize("cyclic", [False, True])
@pytest.mark.parametrize("fix_initial_state", [True, False])
def test_every_step_of_flow_is_in_the_balance(fix_initial_state, cyclic):
    """start + dt * (sum(in) - sum(out)) == volume[N-1]: step 0's flows count.

    Before this was fixed, holdup skipped t = 0, so a steady 50 m^3/hr demand
    drew 12.5 m^3 out of a 200 m^3 tank in the first step without lowering it.
    """
    m, unit = _tank(4, fix_initial_state=fix_initial_state, cyclic=cyclic)
    tb = m.time_block
    for t in tb.time_index:
        unit.outlet_state.flow_vol_phase[t, "Liq"].fix(50.0)
    unit.inlet_state.flow_vol_phase[:, "Liq"].setlb(0)
    m.obj = pyo.Objective(
        expr=sum(unit.inlet_state.flow_vol_phase[t, "Liq"] for t in tb.time_index)
    )
    assert_optimal_termination(get_solver(model=m).solve(m))

    dt = pyo.value(pyunits.convert(tb.dt, pyunits.hr))
    inflow = sum(
        pyo.value(unit.inlet_state.flow_vol_phase[t, "Liq"]) for t in tb.time_index
    )
    outflow = sum(
        pyo.value(unit.outlet_state.flow_vol_phase[t, "Liq"]) for t in tb.time_index
    )
    start = pyo.value(unit.initial_volume)
    end = pyo.value(unit.volume[tb.n_points - 1])
    assert start + dt * (inflow - outflow) == pytest.approx(end, abs=1e-6)
    assert outflow * dt == pytest.approx(50.0)  # 4 steps * 0.25 h * 50 m^3/hr

    if fix_initial_state and not cyclic:
        # Nothing forces inflow: the tank drains 200 -> 150 over four steps.
        assert inflow == pytest.approx(0.0, abs=1e-6)
        volumes = [pyo.value(unit.volume[t]) for t in tb.time_index]
        assert volumes == pytest.approx([187.5, 175.0, 162.5, 150.0])
    if cyclic:
        # Ending at least at the start means every m^3 out was pumped in.
        assert inflow == pytest.approx(outflow, abs=1e-6)
