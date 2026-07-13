"""Tests for OpsBlock: the base class of every flex-pse unit model.

Defines a throwaway ``DummyOps`` unit in this module and exercises the
registration API, the base-provided power Var, model-wide registry discovery,
the external-dispatch hook, and the in-place ``update_parameters`` helper.
"""

import pyomo.environ as pyo
import pytest
from idaes.core import declare_process_block_class
from idaes.core.util.model_statistics import degrees_of_freedom
from pyomo.environ import units as pyunits
from pyomo.network import Port
from pyomo.util.check_units import assert_units_consistent

from flexcore.config.schema import UnitConfig
from flexcore.exceptions import FlexConfigError
from flexcore.nomenclature import ELECTRICAL_POWER
from flexops.core.ops_block import OpsBlockData
from flexops.core.registration import (
    IOVariableRecord,
    ParameterRecord,
    PowerRecord,
    iter_io_registry,
)
from flexops.core.time_block import TimeBlock


@declare_process_block_class("DummyOps")
class DummyOpsData(OpsBlockData):
    """A minimal unit exercising the OpsBlock registration API."""

    def build(self):
        super().build()
        tb = self._find_time_block()
        self.flow_in = pyo.Var(
            tb.time_index,
            initialize=1.0,
            units=pyunits.m**3 / pyunits.hr,
            doc="Inlet volumetric flow",
        )
        self.flow_out = pyo.Var(
            tb.time_index,
            initialize=1.0,
            units=pyunits.m**3 / pyunits.hr,
            doc="Outlet volumetric flow",
        )
        self.energy_intensity = pyo.Param(
            initialize=0.5,
            mutable=True,
            units=pyunits.kWh / pyunits.m**3,
            doc="Electrical energy intensity",
        )
        self.register_io_variable(self.flow_in, role="input")
        self.register_io_variable(self.flow_out, role="output")
        self.register_process_parameter(self.energy_intensity, regressable=True)
        power = self.declare_power("electrical")

        self.inlet = Port(initialize={"flow_vol": self.flow_in}, doc="Inlet port")
        self.outlet = Port(initialize={"flow_vol": self.flow_out}, doc="Outlet port")

        @self.Constraint(tb.time_index, doc="Mass balance: 10% loss")
        def mass_balance(b, t):
            return b.flow_out[t] == 0.9 * b.flow_in[t]

        @self.Constraint(tb.time_index, doc="Electrical energy relationship")
        def energy_eq(b, t):
            return power[t] == pyunits.convert(
                b.energy_intensity * b.flow_in[t], pyunits.kW
            )


# ``declare_process_block_class`` injects the constructible ``DummyOps`` wrapper
# into this module's namespace at runtime; bind the name explicitly so static
# tools (ruff) resolve it.
DummyOps = globals()["DummyOps"]


def _model(n_points: int = 4):
    """Build a ConcreteModel with a TimeBlock of ``n_points`` points."""
    m = pyo.ConcreteModel()
    end_hour = n_points // 4
    m.time_block = TimeBlock(
        start_date="2025-01-01",
        end_date=f"2025-01-01T0{end_hour}:00",
        time_step=15 * pyunits.min,
    )
    return m


@pytest.fixture
def dummy_model():
    """A ConcreteModel with a 4-point TimeBlock and one DummyOps unit."""
    m = _model(4)
    m.unit = DummyOps()
    return m


@pytest.mark.unit
def test_dummy_ops_builds(dummy_model):
    """electrical_power exists, indexed by time_index, carrying kW."""
    unit = dummy_model.unit
    power = getattr(unit, ELECTRICAL_POWER)
    assert power.is_indexed()
    assert set(power.index_set()) == set(dummy_model.time_block.time_index)
    assert pyunits.get_units(power[0]) == pyunits.kW


@pytest.mark.unit
def test_registration_records(dummy_model):
    """The registry captures the two IO vars, the parameter, and the power var."""
    reg = dummy_model.unit._io_registry
    assert len(reg.io_variables) == 2
    for rec in reg.io_variables:
        assert isinstance(rec, IOVariableRecord)
        assert rec.role in ("input", "output")
        assert rec.time_indexed is True
        assert rec.units
    assert {r.role for r in reg.io_variables} == {"input", "output"}

    assert len(reg.parameters) == 1
    assert isinstance(reg.parameters[0], ParameterRecord)
    assert reg.parameters[0].regressable is True

    assert len(reg.power) == 1
    assert isinstance(reg.power[0], PowerRecord)
    assert reg.power[0].kind == "electrical"
    assert reg.power[0].name == ELECTRICAL_POWER


@pytest.mark.unit
def test_iter_io_registry_finds_dummy(dummy_model):
    """Model-wide discovery yields exactly the DummyOps block."""
    pairs = list(iter_io_registry(dummy_model))
    assert len(pairs) == 1
    block, reg = pairs[0]
    assert block is dummy_model.unit
    assert len(reg.io_variables) == 2


@pytest.mark.unit
def test_units_consistent(dummy_model):
    """The unit's constraints are dimensionally consistent."""
    assert_units_consistent(dummy_model.unit)


@pytest.mark.unit
def test_dof_zero_when_inputs_fixed(dummy_model):
    """Fixing the input flow at every time point determines the model."""
    for t in dummy_model.time_block.time_index:
        dummy_model.unit.flow_in[t].fix(2.0)
    assert degrees_of_freedom(dummy_model) == 0


@pytest.mark.unit
def test_bad_role_raises(dummy_model):
    """An unknown IO role is a config error."""
    with pytest.raises(FlexConfigError):
        dummy_model.unit.register_io_variable(dummy_model.unit.flow_in, role="both")


@pytest.mark.unit
def test_bad_kind_raises(dummy_model):
    """An unknown power kind is a config error."""
    with pytest.raises(FlexConfigError):
        dummy_model.unit.register_power(dummy_model.unit.flow_in, kind="kinetic")


@pytest.mark.unit
def test_no_time_block_raises():
    """A unit built on a TimeBlock-less model errors clearly."""
    m = pyo.ConcreteModel()
    with pytest.raises(FlexConfigError):
        m.unit = DummyOps()


@pytest.mark.unit
def test_from_config_not_implemented():
    """Config-driven construction is deferred to M09."""
    cfg = UnitConfig(unit_model_class="DummyOps")
    with pytest.raises(NotImplementedError, match="M09"):
        OpsBlockData.from_config(cfg)


@pytest.mark.unit
def test_set_external_dispatch_removes_dof(dummy_model):
    """Dispatching a free controllable var fixes it and drops n_points DOF."""
    tb = dummy_model.time_block
    dof_before = degrees_of_freedom(dummy_model)
    series = {i: 1.5 + i for i in tb.time_index}
    dummy_model.unit.set_external_dispatch(dummy_model.unit.flow_in, series)
    for t in tb.time_index:
        assert dummy_model.unit.flow_in[t].fixed is True
        assert pyo.value(dummy_model.unit.flow_in[t]) == pytest.approx(series[t])
    assert degrees_of_freedom(dummy_model) == dof_before - tb.n_points


@pytest.mark.unit
def test_set_external_dispatch_by_timestamp(dummy_model):
    """A timestamp-keyed series is aligned via the TimeBlock's index_of."""
    tb = dummy_model.time_block
    series = {tb.timestamp_of(i): float(i) for i in tb.time_index}
    dummy_model.unit.set_external_dispatch(dummy_model.unit.flow_in, series)
    for t in tb.time_index:
        assert pyo.value(dummy_model.unit.flow_in[t]) == pytest.approx(float(t))


@pytest.mark.unit
def test_set_external_dispatch_misaligned_raises(dummy_model):
    """A series that does not cover every time point is a config error."""
    short = {0: 1.0, 1: 2.0}
    with pytest.raises(FlexConfigError):
        dummy_model.unit.set_external_dispatch(dummy_model.unit.flow_in, short)


@pytest.mark.unit
def test_set_external_dispatch_unindexed_raises(dummy_model):
    """Dispatching an unindexed var is a config error."""
    dummy_model.unit.scalar = pyo.Var(units=pyunits.kW, doc="scalar")
    with pytest.raises(FlexConfigError):
        dummy_model.unit.set_external_dispatch(dummy_model.unit.scalar, {0: 1.0})


@pytest.mark.unit
def test_update_parameters_in_place(dummy_model):
    """update_parameters mutates the live Param; existing constraints see it.

    The energy_eq constraint built at construction time must reflect the new
    parameter value without any component being deleted or rebuilt (the
    flex-pse no-delete update path).
    """
    unit = dummy_model.unit
    constraint = unit.energy_eq[0]
    unit.flow_in[0].fix(2.0)
    body_before = pyo.value(constraint.body)

    unit.update_parameters({"energy_intensity": 1.0})

    assert pyo.value(unit.energy_intensity) == pytest.approx(1.0)
    # Same constraint object, new residual: no rebuild happened. The body is
    # power - intensity*flow, so +0.5 kWh/m^3 at 2 m^3/hr lowers it by 1 kW.
    assert unit.energy_eq[0] is constraint
    assert pyo.value(constraint.body) == pytest.approx(body_before - 1.0)


@pytest.mark.unit
def test_update_parameters_unknown_name_raises(dummy_model):
    """Updating a name that is not a registered parameter is a config error."""
    with pytest.raises(FlexConfigError):
        dummy_model.unit.update_parameters({"not_registered": 1.0})
