"""Harness-driven and hand tests for Tank(SISOBlock)."""

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits

from flexcore.config.schema import UnitCommitmentConfig
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

    # Backward differencing: volume[t] = volume[t-1] + dt*(flow_in[t] - flow_out[t]).
    flow_in = [100.0, 100.0, 0.0, 0.0]
    flow_out = [50.0, 50.0, 50.0, 50.0]
    volumes = [200.0, 212.5, 200.0, 187.5]

    for t in m.time_block.time_index:
        unit.inlet_state.flow_vol_phase[t, "Liq"].set_value(flow_in[t])
        unit.outlet_state.flow_vol_phase[t, "Liq"].set_value(flow_out[t])
        unit.volume[t].set_value(volumes[t])

    assert len(unit.holdup) == 3
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
def test_fix_initial_state_pins_volume_and_registers_initial_state():
    """volume[0] is pinned to initial_volume unless fix_initial_state=False."""
    m, unit = _tank(4)
    assert unit.initial_volume in m.time_block.initial_state_params
    assert hasattr(unit, "initial_volume_eq")

    m_free, unit_free = _tank(4, fix_initial_state=False)
    assert unit_free.initial_volume not in m_free.time_block.initial_state_params
    assert not hasattr(unit_free, "initial_volume_eq")


@pytest.mark.unit
def test_cyclic_tank_requires_final_volume_at_least_start():
    """cyclic=True adds volume[N-1] >= volume[0]; the default builds no such row."""
    _, unit_default = _tank(4)
    assert not hasattr(unit_default, "cyclic_volume")

    m, unit = _tank(4, cyclic=True)
    assert unit.initial_volume in m.time_block.initial_state_params
    unit.volume[0].set_value(300.0)
    unit.volume[3].set_value(250.0)
    assert pyo.value(unit.cyclic_volume.body) > pyo.value(unit.cyclic_volume.upper)
    unit.volume[3].set_value(300.0)
    assert pyo.value(unit.cyclic_volume.body) <= pyo.value(unit.cyclic_volume.upper)


@pytest.mark.unit
def test_tank_storage_options_survive_emit_and_rebuild():
    """cyclic and fix_initial_state round-trip through emit_model and build_model."""
    m, _ = _tank(4, cyclic=True, fix_initial_state=False)
    rebuilt = build_model(emit_model(m))
    assert rebuilt.unit.config.cyclic is True
    assert rebuilt.unit.config.fix_initial_state is False
    assert_models_equivalent(m, rebuilt)
