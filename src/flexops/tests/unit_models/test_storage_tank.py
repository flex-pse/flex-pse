"""Harness-driven and hand tests for StorageTank(SISOBlock) (M04, R6)."""

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits

from flexcore.config.schema import UnitCommitmentConfig
from flexops.testing import UnitModelTestHarness, dummy_time_block
from flexops.unit_models import Pump, StorageTank


class TestStorageTank(UnitModelTestHarness):
    """Fixing both flows and the initial volume determines the holdup trajectory."""

    expected_dof = 0

    def configure(self):
        m = dummy_time_block(4)
        m.unit = StorageTank(
            property_package=m.properties,
            max_volume=1000 * pyunits.m**3,
            initial_volume=200 * pyunits.m**3,
        )
        return m, m.unit


def _tank(n: int = 4, **kwargs):
    """Build a fresh StorageTank on an ``n``-point dummy_time_block."""
    m = dummy_time_block(n)
    m.unit = StorageTank(
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
