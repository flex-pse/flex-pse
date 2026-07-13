"""Tests for ``StorageTank``: harness class, hand mass balance, logic disable."""

import pyomo.environ as pyo
import pytest

from flexcore.config.schema import UnitCommitmentConfig
from flexops.testing import UnitModelTestHarness, dummy_time_block
from flexops.unit_models import Pump, StorageTank

_FLOW_IN = [100.0, 100.0, 0.0, 0.0]
_FLOW_OUT = [50.0, 50.0, 50.0, 50.0]
# Hand-computed from V[0]=200 with dt=0.25 h, backwards difference:
# V[t] = V[t-1] + 0.25*(in[t] - out[t]).
_TRAJECTORY = [200.0, 212.5, 200.0, 187.5]


def _tank_model(**tank_kwargs) -> pyo.ConcreteModel:
    """A 4-point dummy model carrying one StorageTank."""
    m = dummy_time_block(4)
    m.unit = StorageTank(
        property_package=m.properties,
        max_volume=1000.0,
        initial_volume=200.0,
        **tank_kwargs,
    )
    return m


class TestStorageTank(UnitModelTestHarness):
    """A StorageTank on a 4-point dummy model (dt = 0.25 h).

    With the flow profiles below and ``V[0] = 200`` m³, the backwards-difference
    holdup trajectory is 200, 212.5, 200, 187.5 m³.
    """

    expected_dof = 0
    expected_solution = {
        "V[0]": 200.0,
        "V[1]": 212.5,
        "V[2]": 200.0,
        "V[3]": 187.5,
    }

    def configure(self):
        """Build a 4-point dummy model with one StorageTank; fix nothing."""
        m = _tank_model()
        for t in m.time_block.time_index:
            m.unit.flow_in[t].set_value(_FLOW_IN[t])
            m.unit.flow_out[t].set_value(_FLOW_OUT[t])
        return m, m.unit


@pytest.mark.unit
def test_mass_balance_by_hand():
    """Every holdup-constraint body evaluates to 0 on the hand trajectory.

    Constraint-body point check (testing doc §5): no solver. Also pins the
    off-by-one contract — exactly N-1 = 3 backwards-difference holdup
    constraints on 4 points, indexed t = 1..N-1 (none at t = 0, which would
    reference V[-1]).
    """
    m = _tank_model()
    unit = m.unit
    t_index = m.time_block.time_index

    assert len(unit.holdup_balance) == 3

    for t in t_index:
        unit.flow_in[t].fix(_FLOW_IN[t])
        unit.flow_out[t].fix(_FLOW_OUT[t])
        unit.V[t].set_value(_TRAJECTORY[t])
    for t in list(t_index)[1:]:
        assert pyo.value(unit.holdup_balance[t].body) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_tank_replaces_siso_mass_balance():
    """The tank has no SISO pass-through balance; holdup replaces it."""
    m = _tank_model()
    assert m.unit.component("mass_balance") is None


@pytest.mark.unit
def test_tank_initial_state_registered():
    """``initial_volume`` is rolling-horizon state and a non-regressable param."""
    m = _tank_model()
    assert m.unit.initial_volume in m.time_block.initial_state_params
    records = {r.name: r for r in m.unit._io_registry.parameters}
    assert records["initial_volume"].regressable is False


@pytest.mark.unit
def test_tank_capacity_fixed_by_default():
    """``capacity`` is a design Var fixed to max_volume; not registered as IO."""
    m = _tank_model()
    assert m.unit.capacity.fixed
    assert pyo.value(m.unit.capacity) == pytest.approx(1000.0)
    assert len(m.unit.capacity_limit) == 4
    io_names = {r.name for r in m.unit._io_registry.io_variables}
    assert "capacity" not in io_names


@pytest.mark.unit
def test_tank_logic_disabled():
    """The canonical R6 check: a tank never has on/off status.

    Both with no ``unit_commitment`` config and with one explicitly passed on,
    the built tank has no ``status`` Var, no Binary Vars at all, and its stored
    UC config is forced off — a tank with an on/off binary is a modeling bug.
    The Pump, by contrast, keeps the base capability.
    """
    uc_on = UnitCommitmentConfig(status=True, startup_shutdown=True)
    for kwargs in ({}, {"unit_commitment": uc_on}):
        m = _tank_model(**kwargs)
        assert m.unit.component("status") is None
        binaries = [
            v
            for v in m.unit.component_data_objects(pyo.Var, descend_into=True)
            if v.is_binary()
        ]
        assert binaries == []
        assert m.unit.config.unit_commitment.status is False
        assert m.unit.config.unit_commitment.startup_shutdown is False

    m = dummy_time_block(3)
    m.pump = Pump(property_package=m.properties)
    assert m.pump.config.unit_commitment.status is True
