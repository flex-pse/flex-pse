"""Unit tests for the SISOBlock IO-topology base (M04, architecture §3.4)."""

import pandas as pd
import pyomo.environ as pyo
import pytest
from idaes.core.util.model_statistics import degrees_of_freedom
from pyomo.environ import units as pyunits
from pyomo.network import Port
from pyomo.util.check_units import assert_units_consistent

from flexcore import nomenclature as nm
from flexops.core.time_block import TimeBlock
from flexops.properties.simple_aqueous import SimpleAqueousFlow
from flexops.testing import dummy_time_block
from flexops.unit_models.base import SISOBlock


def _build_siso(n: int = 3, **kwargs):
    """Build a bare SISOBlock on a fresh ``dummy_time_block(n)``."""
    m = dummy_time_block(n)
    m.unit = SISOBlock(property_package=m.properties, **kwargs)
    return m, m.unit


def _build_siso_with_options(n: int = 3, *, allow_bypass=None):
    """Build an ``n``-point SISOBlock with pressure/temperature enabled."""
    m = pyo.ConcreteModel()
    start = pd.Timestamp("2025-01-01")
    m.time_block = TimeBlock(
        start_date=start,
        end_date=start + n * pd.Timedelta(minutes=15),
        time_step=15 * pyunits.min,
    )
    m.properties = SimpleAqueousFlow(has_pressure=True, has_temperature=True)
    kwargs = {} if allow_bypass is None else {"allow_bypass": allow_bypass}
    m.unit = SISOBlock(property_package=m.properties, **kwargs)
    return m, m.unit


@pytest.mark.unit
def test_siso_ports_and_mass_balance():
    """Inlet/outlet ports exist with flow_vol_phase; the flow bypass is indexed by t."""
    m, unit = _build_siso(3)

    assert isinstance(unit.inlet, Port)
    assert isinstance(unit.outlet, Port)
    assert "flow_vol_phase" in unit.inlet.vars
    assert "flow_vol_phase" in unit.outlet.vars

    assert len(unit.bypass_flow_vol_phase_eq) == m.time_block.n_points

    profile = {0: 10.0, 1: 20.0, 2: 30.0}
    for t, flow in profile.items():
        unit.inlet_state.flow_vol_phase[t, "Liq"].set_value(flow)
        unit.outlet_state.flow_vol_phase[t, "Liq"].set_value(flow)

    for t in m.time_block.time_index:
        assert pyo.value(unit.bypass_flow_vol_phase_eq[t, "Liq"].body) == pytest.approx(
            0.0, abs=1e-9
        )


@pytest.mark.unit
def test_siso_registers_no_energy():
    """A bare SISOBlock declares no power_electrical/power_thermal (base contract)."""
    _, unit = _build_siso(3)
    assert not hasattr(unit, nm.POWER_ELECTRICAL)
    assert not hasattr(unit, nm.POWER_THERMAL)
    assert unit._io_registry.power == []


@pytest.mark.unit
def test_siso_bypasses_all_state_vars():
    """allow_bypass=True (the SISO default) passes every non-fixed state var through."""
    m, unit = _build_siso_with_options(3)

    assert unit.config.allow_bypass is True
    assert hasattr(unit, "bypass_flow_vol_phase_eq")
    assert hasattr(unit, "bypass_pressure_eq")
    assert hasattr(unit, "bypass_temperature_eq")
    # dens_mass is fixed under fixed_density (the property default), so no
    # redundant bypass equality is built for it.
    assert not hasattr(unit, "bypass_dens_mass_eq")

    for t in m.time_block.time_index:
        unit.inlet_state.flow_vol_phase[t, "Liq"].set_value(5.0)
        unit.outlet_state.flow_vol_phase[t, "Liq"].set_value(5.0)
        unit.inlet_state.pressure[t].set_value(200000.0)
        unit.outlet_state.pressure[t].set_value(200000.0)
        unit.inlet_state.temperature[t].set_value(310.0)
        unit.outlet_state.temperature[t].set_value(310.0)
        assert pyo.value(unit.bypass_pressure_eq[t].body) == pytest.approx(
            0.0, abs=1e-9
        )
        assert pyo.value(unit.bypass_temperature_eq[t].body) == pytest.approx(
            0.0, abs=1e-9
        )


@pytest.mark.unit
def test_siso_units_consistent_with_options():
    """A SISOBlock with pressure/temperature enabled stays units-consistent."""
    _, unit = _build_siso_with_options(3)
    assert_units_consistent(unit)


@pytest.mark.unit
def test_siso_allow_bypass_false_leaves_dof():
    """allow_bypass=False builds no bypass constraints; outlet states are left free."""
    m, unit = _build_siso_with_options(3, allow_bypass=False)

    assert not hasattr(unit, "bypass_flow_vol_phase_eq")
    assert not hasattr(unit, "bypass_pressure_eq")
    assert not hasattr(unit, "bypass_temperature_eq")

    for t in m.time_block.time_index:
        unit.inlet_state.flow_vol_phase[t, "Liq"].fix(5.0)
        unit.inlet_state.pressure[t].fix(200000.0)
        unit.inlet_state.temperature[t].fix(310.0)

    # IDAES's degrees_of_freedom counts only unfixed vars appearing in
    # active constraints; with none built, it is trivially 0 -- the
    # meaningful check is that the outlet states are genuinely unconstrained
    # (a developer must wire the relationship by hand).
    assert degrees_of_freedom(m) == 0
    for t in m.time_block.time_index:
        assert not unit.outlet_state.flow_vol_phase[t, "Liq"].fixed
        assert not unit.outlet_state.pressure[t].fixed
        assert not unit.outlet_state.temperature[t].fixed
