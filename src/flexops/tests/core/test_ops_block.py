"""Tests for OpsBlock: the base class of every flex-pse unit model.

Defines a throwaway ``DummyOps`` unit in this module and exercises the
registration API, the base-provided power Var, model-wide registry discovery,
the external-dispatch hook, and the in-place ``update_parameters`` helper.
"""

import math

import pyomo.environ as pyo
import pytest
from idaes.core import declare_process_block_class
from idaes.core.util.model_statistics import degrees_of_freedom
from pyomo.environ import units as pyunits
from pyomo.network import Port
from pyomo.util.check_units import assert_units_consistent

from flexcore import nomenclature as nm
from flexcore.config.schema import ExternalDispatchSpec, SurrogateSpec, UnitConfig
from flexcore.exceptions import FlexConfigError
from flexops.core import ops_block
from flexops.core.ops_block import (
    OpsBlock,
    OpsBlockData,
    RelaxationPolicy,
    _costing_package_domain,
)
from flexops.core.registration import (
    FuelUsageRecord,
    IOVariableRecord,
    ParameterRecord,
    PowerRecord,
    iter_io_registry,
    iter_swapped_relations,
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
def test_build_from_config_rejects_a_class_outside_the_library():
    """Only a flexops unit-model class can be named; the error lists the options.

    ``DummyOps`` is this test module's own OpsBlock subclass, not part of
    ``flexops.unit_models`` -- config-driven construction resolves the class
    name against the shipped library (see
    ``src/flexops/tests/core/test_build_from_config.py`` for the built path).
    """
    cfg = UnitConfig(unit_model_class="DummyOps")
    with pytest.raises(FlexConfigError, match="Unknown unit_model_class"):
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
def test_declare_process_parameter(dummy_model):
    """declare_process_parameter builds a fixed scalar Var and registers it."""
    unit = dummy_model.unit
    var = unit.declare_process_parameter(
        "efficiency",
        0.75,
        pyunits.dimensionless,
        "Some efficiency.",
        bounds=(0.0, 1.0),
    )
    assert var is unit.efficiency
    assert var.fixed
    assert var.value == pytest.approx(0.75)
    assert var.bounds == (0.0, 1.0)
    assert var.doc == "Some efficiency."
    assert not var.is_indexed()
    record = unit._io_registry.parameters[-1]
    assert record.name == "efficiency"
    assert record.regressable is True


@pytest.mark.unit
def test_declare_process_parameter_converts_units(dummy_model):
    """A units-carrying value is converted into the Var's declared units."""
    var = dummy_model.unit.declare_process_parameter(
        "holdup", 1500 * pyunits.L, pyunits.m**3, "A holdup volume."
    )
    assert str(pyunits.get_units(var)) == "m**3"
    assert var.value == pytest.approx(1.5)


@pytest.mark.unit
def test_declare_process_parameter_accepts_a_bare_number(dummy_model):
    """A bare number is taken to be in the Var's declared units already."""
    var = dummy_model.unit.declare_process_parameter(
        "holdup", 1.5, pyunits.m**3, "A holdup volume."
    )
    assert var.value == pytest.approx(1.5)


@pytest.mark.unit
def test_declare_process_parameter_not_regressable(dummy_model):
    """regressable=False is forwarded so FlexParameterize will not fit it."""
    dummy_model.unit.declare_process_parameter(
        "n_cells", 250.0, pyunits.dimensionless, "A count.", regressable=False
    )
    assert dummy_model.unit._io_registry.parameters[-1].regressable is False


@pytest.mark.unit
def test_declare_process_parameter_is_updatable_in_place(dummy_model):
    """The declared Var is a registered parameter, so update_parameters reaches it."""
    unit = dummy_model.unit
    unit.declare_process_parameter(
        "efficiency", 0.75, pyunits.dimensionless, "Some efficiency."
    )
    unit.update_parameters({"efficiency": 0.9})
    assert unit.efficiency.value == pytest.approx(0.9)
    assert unit.efficiency.fixed


@pytest.mark.unit
def test_register_power_rejects_string(dummy_model):
    """register_power requires a PowerKind; even a valid-value string raises."""
    unit = dummy_model.unit
    with pytest.raises(FlexConfigError):
        unit.register_power(getattr(unit, nm.POWER_ELECTRICAL), kind="electrical")


@pytest.mark.unit
def test_declare_power_thermal(dummy_model):
    """declare_power(PowerKind.THERMAL, temperature=...) builds power_thermal in kW."""
    var = dummy_model.unit.declare_power(
        nm.PowerKind.THERMAL, temperature=350 * pyunits.K
    )
    assert var is getattr(dummy_model.unit, nm.POWER_THERMAL)
    assert pyunits.get_units(var[0]) == pyunits.kW
    record = dummy_model.unit._io_registry.power[-1]
    assert record.kind == "thermal"
    assert pyunits.get_units(record.temperature) == pyunits.K


@pytest.mark.unit
def test_declare_power_thermal_requires_temperature(dummy_model):
    """A thermal draw without a temperature is a config error."""
    with pytest.raises(FlexConfigError, match="temperature"):
        dummy_model.unit.declare_power(nm.PowerKind.THERMAL)


@pytest.mark.unit
def test_declare_power_takes_no_fuel_name(dummy_model):
    """Fuel is a volumetric flow, not a PowerKind: declare_power has no fuel_name."""
    with pytest.raises(TypeError):
        dummy_model.unit.declare_power(nm.PowerKind.ELECTRICAL, fuel_name="natural_gas")
    assert not hasattr(nm.PowerKind, "FUEL")


@pytest.mark.unit
def test_register_fuel_usage(dummy_model):
    """register_fuel_usage records a volumetric fuel flow under its fuel name."""
    unit = dummy_model.unit
    usage = pyo.Var(
        dummy_model.time_block.time_index,
        initialize=0.0,
        units=pyunits.m**3 / pyunits.hr,
    )
    unit.add_component(f"{nm.FUEL_USAGE}_natural_gas", usage)
    unit.register_fuel_usage(usage, fuel_name="natural_gas")

    record = unit._io_registry.fuel[-1]
    assert isinstance(record, FuelUsageRecord)
    assert record.var is usage
    assert record.name == f"{nm.FUEL_USAGE}_natural_gas"
    assert record.fuel_name == "natural_gas"


@pytest.mark.unit
def test_register_fuel_usage_requires_fuel_name(dummy_model):
    """A fuel usage flow with no fuel name is a config error."""
    unit = dummy_model.unit
    usage = pyo.Var(
        dummy_model.time_block.time_index,
        initialize=0.0,
        units=pyunits.m**3 / pyunits.hr,
    )
    unit.add_component("gas_flow", usage)
    with pytest.raises(FlexConfigError, match="fuel_name"):
        unit.register_fuel_usage(usage, fuel_name="")


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


@pytest.mark.unit
def test_add_stream_ports_requires_property_package():
    """add_stream_ports on a unit with no property_package is a config error."""
    m = _model(4)
    m.unit = OpsBlock()
    with pytest.raises(FlexConfigError, match="property_package"):
        m.unit.add_stream_ports()


@pytest.mark.unit
def test_check_power_metadata_electrical_rejects_temperature(dummy_model):
    """A non-thermal power draw takes no temperature."""
    with pytest.raises(FlexConfigError, match="takes no temperature"):
        dummy_model.unit.declare_power(
            nm.PowerKind.ELECTRICAL, temperature=350 * pyunits.K
        )


@pytest.mark.unit
def test_costing_package_domain_accepts_duck_typed_package_and_forwards():
    """A costing_package exposing register_unit_power is accepted and used."""

    class _StubCosting:
        def __init__(self):
            self.calls = []

        def register_unit_power(self, unit, var, kind):
            self.calls.append((unit, var, kind))

    m = _model(4)
    costing = _StubCosting()
    m.unit = DummyOps(property_package=m.props, costing_package=costing)
    assert m.unit.config.costing_package is costing
    power = getattr(m.unit, nm.POWER_ELECTRICAL)
    assert (m.unit, power, nm.PowerKind.ELECTRICAL) in costing.calls


@pytest.mark.unit
def test_external_dispatch_slot_accepts_valid_spec():
    """A validated ExternalDispatchSpec is stored on the config as-is."""
    m = _model(4)
    spec = ExternalDispatchSpec(variable="power_electrical", source="x.json")
    m.unit = DummyOps(property_package=m.props, external_dispatch=spec)
    assert m.unit.config.external_dispatch is spec


@pytest.mark.unit
def test_costing_package_domain_rejects_non_duck_typed_value():
    """A costing_package with no register_unit_power is a config error."""
    with pytest.raises(FlexConfigError, match="register_unit_power"):
        _costing_package_domain("not-a-costing-package")


@pytest.mark.unit
def test_pass_through_noop_when_not_allowed():
    """allow_pass_through=False (the default) builds no constraints."""
    m = _model(4)
    m.unit = OpsBlock(property_package=m.props)
    m.unit.add_stream_ports()
    before = list(m.unit.component_objects(pyo.Constraint))
    m.unit.add_pass_through_constraints(m.unit.inlet, m.unit.outlet)
    assert list(m.unit.component_objects(pyo.Constraint)) == before


@pytest.mark.unit
def test_pass_through_builds_equality_constraints():
    """allow_pass_through=True links every non-fixed inlet state var to the outlet."""
    m = _model(4)
    m.unit = OpsBlock(property_package=m.props, allow_pass_through=True)
    m.unit.add_stream_ports()
    m.unit.add_pass_through_constraints(m.unit.inlet, m.unit.outlet)
    constraint = m.unit.pass_through_flow_vol_phase_eq
    for t in m.time_block.time_index:
        m.unit.inlet_state.flow_vol_phase[t, "Liq"].fix(2.0)
        m.unit.outlet_state.flow_vol_phase[t, "Liq"].set_value(2.0)
        assert pyo.value(constraint[t, "Liq"].body) == pytest.approx(0.0)


@pytest.mark.unit
def test_pass_through_skips_a_fully_fixed_state_var():
    """A state var already fixed at every index gets no redundant equality."""
    m = pyo.ConcreteModel()
    m.time_block = TimeBlock(
        start_date="2025-01-01",
        end_date="2025-01-01T01:00",
        time_step=15 * pyunits.min,
    )
    m.props = SimpleAqueousFlow(has_pressure=True)
    m.unit = OpsBlock(property_package=m.props, allow_pass_through=True)
    m.unit.add_stream_ports()
    for t in m.time_block.time_index:
        m.unit.inlet_state.pressure[t].fix(101325.0)

    m.unit.add_pass_through_constraints(m.unit.inlet, m.unit.outlet)

    assert m.unit.find_component("pass_through_pressure_eq") is None
    assert m.unit.find_component("pass_through_flow_vol_phase_eq") is not None


@pytest.mark.unit
def test_pass_through_requires_stream_port_state_blocks():
    """A port not built by add_stream_ports has no sibling state block."""
    m = _model(4)
    m.unit = OpsBlock(property_package=m.props, allow_pass_through=True)
    m.unit.bare_port = Port(initialize={})
    with pytest.raises(FlexConfigError, match="add_stream_ports"):
        m.unit.add_pass_through_constraints(m.unit.bare_port, m.unit.bare_port)


@pytest.mark.unit
def test_pass_through_unknown_exclude_var_raises():
    """An exclude_vars name that is not a state variable is a config error."""
    m = _model(4)
    m.unit = OpsBlock(property_package=m.props, allow_pass_through=True)
    m.unit.add_stream_ports()
    with pytest.raises(FlexConfigError, match="not state variables"):
        m.unit.add_pass_through_constraints(
            m.unit.inlet, m.unit.outlet, exclude_vars=("not_a_state_var",)
        )


@pytest.mark.unit
def test_swap_relation_unregistered_relation_raises():
    """Swapping a relation that was never registered is a config error."""
    m = _model(4)
    m.unit = OpsBlock(property_package=m.props)
    with pytest.raises(FlexConfigError, match="not a registered relation"):
        m.unit.swap_relation(
            "power_electrical_relation", SurrogateSpec(functional_form="linear")
        )


@pytest.mark.unit
def test_swap_relation_unknown_input_variable_raises():
    """A fitted input variable not found on the unit is a config error."""
    m = _model(4)
    m.unit = OpsBlock(property_package=m.props)
    m.unit.add_stream_ports()
    flow = pyo.Reference(m.unit.outlet_state.flow_vol_phase[:, "Liq"])
    m.unit.add_constant_intensity_relation(
        flow, intensity=0.5 * pyunits.kWh / pyunits.m**3
    )
    with pytest.raises(FlexConfigError, match="not a variable"):
        m.unit.swap_relation(
            "power_electrical_relation",
            SurrogateSpec(functional_form="linear", input_variables=["nope"]),
        )


@pytest.mark.unit
def test_swap_relation_builds_linear_fit():
    """A linear surrogate deactivates the old relation and adds the fitted one."""
    m = _model(4)
    m.unit = OpsBlock(property_package=m.props)
    m.unit.add_stream_ports()
    flow = pyo.Reference(m.unit.outlet_state.flow_vol_phase[:, "Liq"])
    m.unit.add_constant_intensity_relation(
        flow, intensity=0.5 * pyunits.kWh / pyunits.m**3
    )
    old_relation = m.unit.power_electrical_relation

    m.unit.swap_relation(
        "power_electrical_relation",
        SurrogateSpec(
            functional_form="linear",
            input_variables=["power_electrical"],
            coefficients={"power_electrical": 2.0, "intercept": 1.0},
        ),
    )

    assert m.unit.power_electrical_relation is old_relation
    assert old_relation[0].active is False
    fitted = m.unit.power_electrical_relation_fitted
    m.unit.power_electrical[0].fix(3.0)
    # power - (intercept + 2*power) = 3.0 - (1.0 + 6.0) == -4.0
    assert pyo.value(fitted[0].body) == pytest.approx(-4.0)


def _unit_with_relation(has_pressure: bool = False):
    """Build a bare unit carrying ``flow_out`` and a constant-intensity relation."""
    m = _model(4)
    if has_pressure:
        m.pressure_props = SimpleAqueousFlow(has_pressure=True)
    m.unit = OpsBlock(
        property_package=m.pressure_props if has_pressure else m.props,
        allow_pass_through=False,
    )
    m.unit.add_stream_ports()
    m.unit.add_component(
        "flow_out", pyo.Reference(m.unit.outlet_state.flow_vol_phase[:, "Liq"])
    )
    m.unit.add_constant_intensity_relation(
        m.unit.flow_out, intensity=0.5 * pyunits.kWh / pyunits.m**3
    )
    return m, m.unit


@pytest.mark.unit
def test_swap_relation_builds_quadratic_fit():
    """A squared term is a coefficient key, not a new builder."""
    _, unit = _unit_with_relation()

    unit.swap_relation(
        "power_electrical_relation",
        SurrogateSpec(
            functional_form="quadratic",
            input_variables=["flow_out"],
            coefficients={"intercept": 1.0, "flow_out": 2.0, "flow_out^2": 0.5},
        ),
    )

    unit.flow_out[0].set_value(3.0)
    unit.power_electrical[0].fix(0.0)
    # power - (1.0 + 2*3 + 0.5*9) == -11.5
    assert pyo.value(unit.power_electrical_relation_fitted[0].body) == pytest.approx(
        -11.5
    )


@pytest.mark.unit
def test_swap_relation_builds_expanded_bilinear_fit():
    """Outlet flow, outlet pressure, and their cross term in one relationship."""
    _, unit = _unit_with_relation(has_pressure=True)

    unit.swap_relation(
        "power_electrical_relation",
        SurrogateSpec(
            functional_form="bilinear",
            input_variables=["flow_out", "outlet_state.pressure"],
            coefficients={
                "intercept": 1.0,
                "flow_out": 2.0,
                "outlet_state.pressure": 1e-3,
                "flow_out*outlet_state.pressure": 5e-4,
            },
        ),
    )

    unit.flow_out[0].set_value(3.0)
    unit.outlet_state.pressure[0].set_value(2.0e5)
    unit.power_electrical[0].fix(0.0)
    # power - (1 + 2*3 + 1e-3*2e5 + 5e-4*3*2e5) == -(1 + 6 + 200 + 300)
    assert pyo.value(unit.power_electrical_relation_fitted[0].body) == pytest.approx(
        -507.0
    )


@pytest.mark.unit
def test_swap_relation_unknown_functional_form_raises():
    """A form with no registered builder names itself and the known forms."""
    _, unit = _unit_with_relation()
    with pytest.raises(FlexConfigError, match="vendor_curve_v3"):
        unit.swap_relation(
            "power_electrical_relation",
            SurrogateSpec(functional_form="vendor_curve_v3"),
        )


@pytest.mark.unit
def test_swap_relation_rejects_a_term_above_the_forms_degree():
    """A squared term under functional_form='linear' is a mislabelled spec."""
    _, unit = _unit_with_relation()
    with pytest.raises(FlexConfigError, match=r"flow_out\^2"):
        unit.swap_relation(
            "power_electrical_relation",
            SurrogateSpec(
                functional_form="linear",
                input_variables=["flow_out"],
                coefficients={"flow_out^2": 1.0},
            ),
        )


@pytest.mark.unit
def test_swap_relation_unknown_coefficient_factor_raises():
    """A coefficient naming a factor that is not on the unit is a config error."""
    _, unit = _unit_with_relation()
    with pytest.raises(FlexConfigError, match="not a variable"):
        unit.swap_relation(
            "power_electrical_relation",
            SurrogateSpec(
                functional_form="bilinear",
                input_variables=["flow_out"],
                coefficients={"flow_out*nope": 1.0},
            ),
        )


def _flow_relation_unit():
    """A bare unit carrying one registered, swappable non-power relation.

    Its target (``flow_out``, m^3/hr) proves ``swap_relation`` is not
    power-specific: nothing here is named or unit-carrying like a
    ``power_<kind>_relation``.
    """
    m = _model(4)
    m.unit = OpsBlock(property_package=m.props, allow_pass_through=False)
    m.unit.add_stream_ports()
    m.unit.add_component(
        "flow_out", pyo.Reference(m.unit.outlet_state.flow_vol_phase[:, "Liq"])
    )
    flow_out = m.unit.flow_out

    @m.unit.Constraint(m.time_block.time_index)
    def flow_relation(b, t):
        return flow_out[t] == 10.0

    m.unit.register_relation(m.unit.flow_relation, target=flow_out)
    return m, m.unit


@pytest.mark.unit
def test_register_relation_records_its_target():
    """register_relation records the constraint, its name, and its target."""
    _, unit = _unit_with_relation()
    records = {record.name: record for record in unit._io_registry.relations}
    assert "power_electrical_relation" in records
    record = records["power_electrical_relation"]
    assert record.target is unit.power_electrical
    assert record.target_name == "power_electrical"
    assert record.fitted is None


@pytest.mark.unit
def test_register_relation_rejects_a_multidimensional_target():
    """A target indexed over more than time is out of scope for this milestone."""
    m = _model(4)
    m.unit = OpsBlock(property_package=m.props, allow_pass_through=False)
    m.unit.add_stream_ports()
    m.unit.multi = pyo.Var(m.time_block.time_index, ["a", "b"], initialize=0.0)

    @m.unit.Constraint(m.time_block.time_index, ["a", "b"])
    def multi_relation(b, t, comp):
        return b.multi[t, comp] == 0.0

    with pytest.raises(FlexConfigError, match="M10b"):
        m.unit.register_relation(m.unit.multi_relation, target=m.unit.multi)


@pytest.mark.unit
def test_swap_relation_replaces_a_registered_relation():
    """Swapping a non-power relation deactivates the old, activates the new."""
    m, unit = _flow_relation_unit()
    old = unit.flow_relation

    unit.swap_relation(
        "flow_relation",
        SurrogateSpec(
            functional_form="linear",
            input_variables=["flow_out"],
            coefficients={"flow_out": 2.0, "intercept": 1.0},
        ),
    )

    fitted = unit.find_component("flow_relation_fitted")
    assert all(not old[t].active for t in m.time_block.time_index)
    assert fitted is not None
    assert all(fitted[t].active for t in m.time_block.time_index)


@pytest.mark.unit
def test_swap_relation_takes_units_from_its_target():
    """The fitted constraint carries the target's own units, not kW."""
    _, unit = _flow_relation_unit()

    unit.swap_relation(
        "flow_relation",
        SurrogateSpec(
            functional_form="linear",
            input_variables=["flow_out"],
            coefficients={"flow_out": 2.0, "intercept": 1.0},
        ),
    )

    fitted = unit.flow_relation_fitted
    # A stray kW hardcode would make this m^3/hr == m^3/hr + kW: inconsistent.
    assert_units_consistent(fitted)
    unit.flow_out[0].set_value(5.0)
    # flow_out - (intercept + 2*flow_out) == 5 - (1 + 10) == -6.0
    assert pyo.value(fitted[0].body) == pytest.approx(-6.0)


@pytest.mark.unit
def test_swap_relation_unknown_name_lists_registered_relations():
    """An unregistered relation name is refused, listing what is registered."""
    _, unit = _unit_with_relation()
    with pytest.raises(FlexConfigError, match="power_electrical_relation"):
        unit.swap_relation("nope", SurrogateSpec(functional_form="linear"))


@pytest.mark.unit
def test_conservation_constraints_are_not_registered():
    """A pass-through mass balance is never swappable: it was never registered."""
    m = _model(4)
    m.unit = OpsBlock(property_package=m.props, allow_pass_through=True)
    m.unit.add_stream_ports()
    m.unit.add_pass_through_constraints(m.unit.inlet, m.unit.outlet)

    registered = {record.name for record in m.unit._io_registry.relations}
    assert "pass_through_flow_vol_phase_eq" not in registered
    with pytest.raises(FlexConfigError, match="not a registered relation"):
        m.unit.swap_relation(
            "pass_through_flow_vol_phase_eq", SurrogateSpec(functional_form="linear")
        )


@pytest.mark.unit
def test_swap_relation_accepts_a_non_polynomial_builder(monkeypatch):
    """The builder registry is not limited to polynomials.

    Any Pyomo-expressible function works -- here a softplus (ICNN-style)
    forward pass, verified against a hand computation.
    """

    def _softplus_body(unit, surrogate, target):
        q = unit.find_component("flow_out")
        c = surrogate.coefficients

        def body(t):
            x = q[t] / pyunits.get_units(q[t])
            return c["wz"] * pyo.log(1 + pyo.exp(c["w"] * x + c["b"])) + c["c"]

        return body

    monkeypatch.setitem(ops_block._RELATION_BUILDERS, "softplus", _softplus_body)
    _, unit = _unit_with_relation()

    unit.swap_relation(
        "power_electrical_relation",
        SurrogateSpec(
            functional_form="softplus",
            input_variables=["flow_out"],
            coefficients={"w": 0.3, "b": -1.0, "wz": 2.0, "c": 5.0},
        ),
    )

    unit.flow_out[0].set_value(10.0)
    unit.power_electrical[0].set_value(0.0)
    expected = 2.0 * math.log(1 + math.exp(0.3 * 10 - 1.0)) + 5.0
    assert pyo.value(unit.power_electrical_relation_fitted[0].body) == pytest.approx(
        -expected
    )


@pytest.mark.unit
def test_swap_relation_skips_indices_a_body_declines(monkeypatch):
    """A body returning Constraint.Skip omits that index from the fitted relation.

    This is what lets a lagged/state-space form skip the horizon points where
    its lag does not exist, rather than raising a KeyError.
    """

    def _lagged_body(unit, surrogate, target):
        def body(t):
            return pyo.Constraint.Skip if t < 1 else 2.0

        return body

    monkeypatch.setitem(ops_block._RELATION_BUILDERS, "lagged", _lagged_body)
    _, unit = _unit_with_relation()

    unit.swap_relation(
        "power_electrical_relation", SurrogateSpec(functional_form="lagged")
    )

    fitted = unit.power_electrical_relation_fitted
    assert 0 not in fitted
    assert 1 in fitted


@pytest.mark.unit
def test_reswapping_deactivates_a_builders_auxiliary_constraints(monkeypatch):
    """A second swap deactivates whatever the first builder's own attached.

    Without this, a ReLU big-M or ARIMA-innovations builder's auxiliary
    equality would stay active alongside the new one, double-counting.
    """

    def _aux_body(unit, surrogate, target):
        tag = surrogate.coefficients["tag"]
        tb = unit.model().time_block
        suffix = f"_{tag:.0f}"
        unit.add_component(f"aux_z{suffix}", pyo.Var(tb.time_index, initialize=0.0))
        z = unit.find_component(f"aux_z{suffix}")
        unit.add_component(
            f"aux_eq{suffix}",
            pyo.Constraint(tb.time_index, rule=lambda b, t: z[t] == tag),
        )
        return lambda t: z[t]

    monkeypatch.setitem(ops_block._RELATION_BUILDERS, "aux", _aux_body)
    _, unit = _unit_with_relation()

    unit.swap_relation(
        "power_electrical_relation",
        SurrogateSpec(functional_form="aux", coefficients={"tag": 1.0}),
    )
    first_eq = unit.aux_eq_1
    assert first_eq[0].active

    unit.swap_relation(
        "power_electrical_relation",
        SurrogateSpec(functional_form="aux", coefficients={"tag": 2.0}),
    )

    assert not first_eq[0].active
    assert unit.aux_eq_2[0].active


@pytest.mark.unit
def test_iter_swapped_relations_reports_only_swapped_relations():
    """Only a relation with a fitted replacement is yielded; an untouched
    model yields nothing."""
    m = _model(4)
    for name in ("unit_a", "unit_b"):
        m.add_component(name, OpsBlock(property_package=m.props))
        unit = m.find_component(name)
        unit.add_stream_ports()
        unit.add_component(
            "flow_out", pyo.Reference(unit.outlet_state.flow_vol_phase[:, "Liq"])
        )
        unit.add_constant_intensity_relation(
            unit.flow_out, intensity=0.5 * pyunits.kWh / pyunits.m**3
        )

    assert list(iter_swapped_relations(m)) == []

    m.unit_a.swap_relation(
        "power_electrical_relation",
        SurrogateSpec(
            functional_form="linear",
            input_variables=["flow_out"],
            coefficients={"flow_out": 1.0},
        ),
    )

    swapped = list(iter_swapped_relations(m))
    assert len(swapped) == 1
    block, record = swapped[0]
    assert block is m.unit_a
    assert record.name == "power_electrical_relation"
    assert record.fitted is m.unit_a.power_electrical_relation_fitted


@pytest.mark.unit
def test_add_constant_intensity_relation_records_its_basis():
    """The registry records which flow the intensity is metered against."""
    _, unit = _unit_with_relation()
    _, registry = next(iter_io_registry(unit))
    assert registry.intensity_basis[nm.PowerKind.ELECTRICAL] == "flow_out"


@pytest.mark.unit
def test_add_constant_intensity_relation_auto_swaps_from_surrogate():
    """A non-constant-intensity SurrogateSpec triggers the fit at construction."""
    m = _model(4)
    cfg = UnitConfig(
        unit_model_class="OpsBlock",
        surrogate=SurrogateSpec(
            functional_form="linear", input_variables=["power_electrical"]
        ),
    )
    m.unit = OpsBlock(property_package=m.props, flexops_config=cfg)
    m.unit.add_stream_ports()
    flow = pyo.Reference(m.unit.outlet_state.flow_vol_phase[:, "Liq"])
    m.unit.add_constant_intensity_relation(
        flow, intensity=0.5 * pyunits.kWh / pyunits.m**3
    )
    assert m.unit.power_electrical_relation[0].active is False
    assert m.unit.find_component("power_electrical_relation_fitted") is not None
