"""Unit tests for the ``SISOBlock`` IO-topology base (ports + mass balance)."""

import pyomo.environ as pyo
import pytest
from pyomo.network import Port

from flexcore.nomenclature import ELECTRICAL_POWER, THERMAL_POWER
from flexops.testing import dummy_time_block
from flexops.unit_models.base import SISOBlock


def _bare_siso(n: int = 3) -> pyo.ConcreteModel:
    """A dummy model carrying one bare ``SISOBlock``."""
    m = dummy_time_block(n)
    m.unit = SISOBlock(property_package=m.properties)
    return m


@pytest.mark.unit
def test_siso_ports_and_mass_balance():
    """Inlet/outlet ports expose flow_vol; the mass balance holds pointwise.

    Constraint-body point check (testing doc §5): fix a known inlet profile,
    set the outlet equal, and every per-``t`` mass-balance body evaluates to 0
    — no solver involved.
    """
    m = _bare_siso(3)
    unit = m.unit
    t_index = m.time_block.time_index

    assert isinstance(unit.inlet, Port)
    assert isinstance(unit.outlet, Port)
    assert hasattr(unit.inlet, "flow_vol")
    assert hasattr(unit.outlet, "flow_vol")

    assert len(unit.mass_balance) == len(t_index) == 3

    profile = {0: 10.0, 1: 25.0, 2: 0.0}
    for t in t_index:
        unit.inlet.flow_vol[t].fix(profile[t])
        unit.outlet.flow_vol[t].set_value(profile[t])
    for t in t_index:
        assert pyo.value(unit.mass_balance[t].body) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_siso_flow_vol_handle():
    """The convenience ``flow_vol`` handle references the inlet state's flow."""
    m = _bare_siso(3)
    m.unit.inlet.flow_vol[0].set_value(42.0)
    assert pyo.value(m.unit.flow_vol[0]) == pytest.approx(42.0)


@pytest.mark.unit
def test_siso_registers_no_energy():
    """A bare SISOBlock declares no power draw (energy is a subclass concern)."""
    m = _bare_siso(3)
    assert not hasattr(m.unit, ELECTRICAL_POWER)
    assert not hasattr(m.unit, THERMAL_POWER)
    assert m.unit._io_registry.power == []
