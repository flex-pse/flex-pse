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
from pyomo.util.check_units import assert_units_consistent

from flexcore import nomenclature as nm
from flexcore.config.schema import UnitConfig
from flexcore.exceptions import FlexConfigError
from flexops.core.ops_block import OpsBlockData, RelaxationPolicy
from flexops.core.registration import (
    IOVariableRecord,
    ParameterRecord,
    PowerRecord,
    iter_io_registry,
)
from flexops.core.time_block import TimeBlock
from flexops.properties.simple_aqueous import SimpleAqueousFlow


@declare_process_block_class("DummyOps")
class DummyOpsData(OpsBlockData):
    """A minimal unit exercising the OpsBlock registration API.

    Builds its inlet/outlet state blocks and ports from the configured
    ``property_package`` via :meth:`~OpsBlockData.add_stream_ports`, then keys
    its electrical energy to the volumetric outlet flow.
    """

    def build(self):
        super().build()
        tb = self._find_time_block()
        self.add_stream_ports()
        self.energy_intensity = pyo.Param(
            initialize=0.5,
            mutable=True,
            units=pyunits.kWh / pyunits.m**3,
            doc="Electrical energy per unit outlet flow",
        )
        self.register_process_parameter(self.energy_intensity, regressable=True)
        power = self.declare_power(nm.PowerKind.ELECTRICAL)

        @self.Constraint(tb.time_index, doc="Mass balance: 10% loss")
        def mass_balance(b, t):
            return (
                b.outlet_state.flow_vol_phase[t, "Liq"]
                == 0.9 * b.inlet_state.flow_vol_phase[t, "Liq"]
            )

        @self.Constraint(tb.time_index, doc="Electrical energy per unit outlet flow")
        def energy_eq(b, t):
            return power[t] == pyunits.convert(
                b.energy_intensity * b.outlet_state.flow_vol_phase[t, "Liq"],
                pyunits.kW,
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
    m.props = SimpleAqueousFlow()
    return m


@pytest.fixture
def dummy_model():
    """A ConcreteModel with a 4-point TimeBlock and one DummyOps unit."""
    m = _model(4)
    m.unit = DummyOps(property_package=m.props)
    return m


@pytest.mark.unit
def test_dummy_ops_builds(dummy_model):
    """power_electrical exists, indexed by time_index, carrying kW."""
    unit = dummy_model.unit
    power = getattr(unit, nm.POWER_ELECTRICAL)
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
    assert reg.power[0].name == nm.POWER_ELECTRICAL


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
        dummy_model.unit.inlet_state.flow_vol_phase[t, "Liq"].fix(2.0)
    assert degrees_of_freedom(dummy_model) == 0


@pytest.mark.unit
def test_bad_role_raises(dummy_model):
    """An unknown IO role is a config error."""
    with pytest.raises(FlexConfigError):
        dummy_model.unit.register_io_variable(
            dummy_model.unit.inlet_state.flow_vol_phase, role="both"
        )


@pytest.mark.unit
def test_bad_kind_raises(dummy_model):
    """A power kind that is not a PowerKind member is a config error."""
    with pytest.raises(FlexConfigError):
        dummy_model.unit.register_power(
            dummy_model.unit.inlet_state.flow_vol_phase, kind="kinetic"
        )


@pytest.mark.unit
def test_no_time_block_raises():
    """A unit built on a TimeBlock-less model errors clearly."""
    m = pyo.ConcreteModel()
    m.props = SimpleAqueousFlow()
    with pytest.raises(FlexConfigError):
        m.unit = DummyOps(property_package=m.props)


@pytest.mark.unit
def test_build_from_config_not_implemented():
    """Config-driven construction is deferred to M09."""
    cfg = UnitConfig(unit_model_class="DummyOps")
    with pytest.raises(NotImplementedError, match="M09"):
        OpsBlockData.build_from_config(cfg)


@pytest.mark.unit
def test_set_external_dispatch_removes_dof(dummy_model):
    """Dispatching a free controllable var fixes it and drops n_points DOF."""
    tb = dummy_model.time_block
    power = getattr(dummy_model.unit, nm.POWER_ELECTRICAL)
    dof_before = degrees_of_freedom(dummy_model)
    series = {i: 1.5 + i for i in tb.time_index}
    dummy_model.unit.set_external_dispatch(power, series)
    for t in tb.time_index:
        assert power[t].fixed is True
        assert pyo.value(power[t]) == pytest.approx(series[t])
    assert degrees_of_freedom(dummy_model) == dof_before - tb.n_points


@pytest.mark.unit
def test_set_external_dispatch_by_timestamp(dummy_model):
    """A timestamp-keyed series is aligned via the TimeBlock's index_of."""
    tb = dummy_model.time_block
    power = getattr(dummy_model.unit, nm.POWER_ELECTRICAL)
    series = {tb.timestamp_of(i): float(i) for i in tb.time_index}
    dummy_model.unit.set_external_dispatch(power, series)
    for t in tb.time_index:
        assert pyo.value(power[t]) == pytest.approx(float(t))


@pytest.mark.unit
def test_set_external_dispatch_misaligned_raises(dummy_model):
    """A series that does not cover every time point is a config error."""
    short = {0: 1.0, 1: 2.0}
    power = getattr(dummy_model.unit, nm.POWER_ELECTRICAL)
    with pytest.raises(FlexConfigError):
        dummy_model.unit.set_external_dispatch(power, short)


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
    unit.outlet_state.flow_vol_phase[0, "Liq"].fix(2.0)
    body_before = pyo.value(constraint.body)

    unit.update_parameters({"energy_intensity": 1.0})

    assert pyo.value(unit.energy_intensity) == pytest.approx(1.0)
    # Same constraint object, new residual: no rebuild happened. The body is
    # power - intensity*outlet flow, so +0.5 kWh/m^3 at 2 m^3/hr lowers it by
    # 1 kW.
    assert unit.energy_eq[0] is constraint
    assert pyo.value(constraint.body) == pytest.approx(body_before - 1.0)


@pytest.mark.unit
def test_update_parameters_unknown_name_raises(dummy_model):
    """Updating a name that is not a registered parameter is a config error."""
    with pytest.raises(FlexConfigError):
        dummy_model.unit.update_parameters({"not_registered": 1.0})


@pytest.mark.unit
def test_update_parameters_with_units(dummy_model):
    """A unit-carrying value updates the Param in its declared units."""
    unit = dummy_model.unit
    unit.update_parameters({"energy_intensity": 0.8 * pyunits.kWh / pyunits.m**3})
    assert pyo.value(unit.energy_intensity) == pytest.approx(0.8)


@pytest.mark.unit
def test_register_process_parameter_not_regressable(dummy_model):
    """regressable=False is recorded so FlexParameterize will not fit it."""
    unit = dummy_model.unit
    unit.design_capacity = pyo.Param(
        initialize=10.0, mutable=True, units=pyunits.m**3 / pyunits.hr
    )
    unit.register_process_parameter(unit.design_capacity, regressable=False)
    record = unit._io_registry.parameters[-1]
    assert record.name == "design_capacity"
    assert record.regressable is False


@pytest.mark.unit
def test_register_power_rejects_string(dummy_model):
    """register_power requires a PowerKind; even a valid-value string raises."""
    unit = dummy_model.unit
    with pytest.raises(FlexConfigError):
        unit.register_power(getattr(unit, nm.POWER_ELECTRICAL), kind="electrical")


@pytest.mark.unit
def test_declare_power_thermal(dummy_model):
    """declare_power(PowerKind.THERMAL) builds and registers power_thermal in kW."""
    var = dummy_model.unit.declare_power(nm.PowerKind.THERMAL)
    assert var is getattr(dummy_model.unit, nm.POWER_THERMAL)
    assert pyunits.get_units(var[0]) == pyunits.kW
    assert dummy_model.unit._io_registry.power[-1].kind == "thermal"


@pytest.mark.unit
def test_declare_power_bad_kind_raises(dummy_model):
    """declare_power without a PowerKind is a config error."""
    with pytest.raises(FlexConfigError):
        dummy_model.unit.declare_power("kinetic")


@pytest.mark.unit
def test_flexops_config_rejects_raw_dict():
    """The flexops_config slot rejects a raw dict (never an unvalidated dict)."""
    m = _model(4)
    with pytest.raises(ValueError, match="never pass a raw dict"):
        m.unit = DummyOps(flexops_config={"unit_model_class": "DummyOps"})


@pytest.mark.unit
def test_flexops_config_accepts_unit_config():
    """A validated UnitConfig is stored on the config block as-is."""
    m = _model(4)
    cfg = UnitConfig(unit_model_class="DummyOps")
    m.unit = DummyOps(flexops_config=cfg, property_package=m.props)
    assert m.unit.config.flexops_config is cfg


@pytest.mark.unit
def test_unit_commitment_rejects_raw_dict():
    """The unit_commitment slot rejects anything but a UnitCommitmentConfig."""
    m = _model(4)
    with pytest.raises(ValueError, match="UnitCommitmentConfig"):
        m.unit = DummyOps(unit_commitment={"status": True})


@pytest.mark.unit
def test_unit_commitment_none_coerces_to_defaults():
    """unit_commitment=None coerces to an all-defaults UnitCommitmentConfig."""
    from flexcore.config.schema import UnitCommitmentConfig

    m = _model(4)
    m.unit = DummyOps(unit_commitment=None, property_package=m.props)
    assert m.unit.config.unit_commitment == UnitCommitmentConfig()


@pytest.mark.unit
def test_external_dispatch_slot_rejects_raw_dict():
    """The external_dispatch slot rejects anything but an ExternalDispatchSpec."""
    m = _model(4)
    with pytest.raises(ValueError, match="ExternalDispatchSpec"):
        m.unit = DummyOps(external_dispatch={"variable": "x", "source": "s.csv"})


@pytest.mark.unit
def test_relaxation_invalid_value_raises():
    """An unknown relaxation policy is a config error naming the choices."""
    m = _model(4)
    with pytest.raises(ValueError, match="'exact', 'relaxed'"):
        m.unit = DummyOps(relaxation="bogus")


@pytest.mark.unit
def test_relaxation_valid_value_stored():
    """A valid relaxation string coerces to the RelaxationPolicy enum."""
    m = _model(4)
    m.unit = DummyOps(relaxation="relaxed", property_package=m.props)
    assert m.unit.config.relaxation is RelaxationPolicy.RELAXED


@pytest.mark.unit
def test_multiple_time_blocks_raises():
    """A model with two TimeBlocks errors clearly at unit build."""
    m = _model(4)
    m.time_block_2 = TimeBlock(
        start_date="2025-01-01",
        end_date="2025-01-01T01:00",
        time_step=15 * pyunits.min,
    )
    with pytest.raises(FlexConfigError, match="found 2"):
        m.unit = DummyOps(property_package=m.props)


@pytest.mark.unit
def test_set_external_dispatch_without_fixing(dummy_model):
    """fix=False sets the trajectory but leaves the degrees of freedom."""
    tb = dummy_model.time_block
    power = getattr(dummy_model.unit, nm.POWER_ELECTRICAL)
    dof_before = degrees_of_freedom(dummy_model)
    series = {i: 2.0 for i in tb.time_index}
    dummy_model.unit.set_external_dispatch(power, series, fix=False)
    for t in tb.time_index:
        assert power[t].fixed is False
        assert pyo.value(power[t]) == pytest.approx(2.0)
    assert degrees_of_freedom(dummy_model) == dof_before


@pytest.mark.unit
def test_set_external_dispatch_non_mapping_raises(dummy_model):
    """A series without items() (e.g. a bare list) is a config error."""
    power = getattr(dummy_model.unit, nm.POWER_ELECTRICAL)
    with pytest.raises(FlexConfigError, match="mapping or pandas Series"):
        dummy_model.unit.set_external_dispatch(power, [1.0, 2.0, 3.0, 4.0])


@pytest.mark.unit
def test_set_external_dispatch_out_of_range_index_raises(dummy_model):
    """An integer key outside [0, n_points) is a config error."""
    power = getattr(dummy_model.unit, nm.POWER_ELECTRICAL)
    with pytest.raises(FlexConfigError, match="out of range"):
        dummy_model.unit.set_external_dispatch(power, {99: 1.0})
