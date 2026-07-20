"""Unit tests for the SISOBlock IO-topology base (M04, architecture §3.4)."""

import pyomo.environ as pyo
import pytest
from pyomo.network import Port

from flexcore import nomenclature as nm
from flexops.testing import dummy_time_block
from flexops.unit_models.base import SISOBlock


def _build_siso(n: int = 3):
    """Build a bare SISOBlock on a fresh ``dummy_time_block(n)``."""
    m = dummy_time_block(n)
    m.unit = SISOBlock(property_package=m.properties)
    return m, m.unit


@pytest.mark.unit
def test_siso_ports_and_mass_balance():
    """Inlet/outlet ports exist with flow_vol_phase; mass balance indexed by t."""
    m, unit = _build_siso(3)

    assert isinstance(unit.inlet, Port)
    assert isinstance(unit.outlet, Port)
    assert "flow_vol_phase" in unit.inlet.vars
    assert "flow_vol_phase" in unit.outlet.vars

    assert len(unit.mass_balance) == m.time_block.n_points

    profile = {0: 10.0, 1: 20.0, 2: 30.0}
    for t, flow in profile.items():
        unit.inlet_state.flow_vol_phase[t, "Liq"].set_value(flow)
        unit.outlet_state.flow_vol_phase[t, "Liq"].set_value(flow)

    for t in m.time_block.time_index:
        assert pyo.value(unit.mass_balance[t].body) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_siso_registers_no_energy():
    """A bare SISOBlock declares no power_electrical/power_thermal (base contract)."""
    _, unit = _build_siso(3)
    assert not hasattr(unit, nm.POWER_ELECTRICAL)
    assert not hasattr(unit, nm.POWER_THERMAL)
    assert unit._io_registry.power == []
