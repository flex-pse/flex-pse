"""Tests for OpsBlock: the base class of every flex-pse unit model.

Defines a throwaway ``DummyOps`` unit in this module and exercises the
registration API, the base-provided power Var, model-wide registry discovery,
the external-dispatch hook, and the in-place ``update_parameters`` helper.
"""

import math
from typing import Any

import pyomo.environ as pyo
import pytest
from idaes.core import declare_process_block_class
from idaes.core.util.model_statistics import degrees_of_freedom
from pyomo.environ import units as pyunits
from pyomo.network import Port
from pyomo.util.check_units import assert_units_consistent

from flexcore import nomenclature as nm
from flexcore.config.schema import (
    ExternalDispatchSpec,
    SurrogateSpec,
    SurrogateType,
    UnitConfig,
)
from flexcore.exceptions import FlexConfigError
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
from flexops.surrogates import MultilinearSurrogate
from flexops.surrogates.base import Surrogate


def _multilinear(coefficients, input_variables=None, output_variables=None):
    """A MultilinearSurrogate over ``coefficients``, with sensible defaults.

    Args:
        coefficients: The ``data['coefficients']`` mapping.
        input_variables: ``{name: units}``; defaults to every non-intercept
            factor named in ``coefficients``, in m^3/hr.
        output_variables: ``{name: units}``; defaults to
            ``{"power_electrical": "kW"}``.
    """
    if input_variables is None:
        names = {
            factor
            for key in coefficients
            if key != "intercept"
            for factor in key.split("*")
        }
        input_variables = {name: "m^3/hr" for name in names or {"flow_out"}}
    return MultilinearSurrogate(
        {
            "input_variables": input_variables,
            "output_variables": output_variables or {"power_electrical": "kW"},
            "coefficients": coefficients,
        }
    )


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
def test_update_parameters_surrogate_coefficients():
    """update_parameters changes registered surrogate coefficient Vars in place.

    After swapping a multilinear surrogate and registering its coefficients,
    update_parameters must mutate the live VarData entries. The fitted
    constraint body must see the new values without any rebuild.
    """
    _, unit = _flow_relation_unit()
    unit.swap_relation(
        "flow_relation",
        _multilinear(
            {"flow_out": 2.0, "intercept": 1.0},
            output_variables={"flow_out": "m^3/hr"},
        ),
    )
    unit.flow_out[0].set_value(3.0)
    fitted = unit.surrogate_flow.fitted
    body_before = pyo.value(fitted[0].body)

    unit.update_parameters({"intercept": 5.0, "flow_out": 0.5})

    assert pyo.value(unit.surrogate_flow.coefficients["intercept"]) == pytest.approx(
        5.0
    )
    assert pyo.value(unit.surrogate_flow.coefficients["flow_out"]) == pytest.approx(0.5)
    assert fitted[0] is fitted[0]
    assert pyo.value(fitted[0].body) == pytest.approx(3.0 - (5.0 + 0.5 * 3.0))
    assert pyo.value(fitted[0].body) != pytest.approx(body_before)


@pytest.mark.unit
def test_update_parameters_surrogate_coefficients_isolation():
    """Updating one surrogate's coefficients does not affect a deactivated
    surrogate that shares the same coefficient names.

    Verifies that coefficient Vars are unique per surrogate block, that
    relation records preserve their surrogate block histories, and that the
    active surrogate's coefficients are the ones reachable via the parameter
    registry.
    """
    _, unit = _two_flow_relation_unit()

    # Swap both relations with multilinear surrogates that share coefficient names.
    unit.swap_relation(
        "flow_relation",
        _multilinear(
            {"flow_out": 2.0, "intercept": 1.0},
            output_variables={"flow_out": "m^3/hr"},
        ),
    )

    unit.swap_relation(
        "secondary_flow_relation",
        _multilinear(
            {"flow_out": 2.0, "intercept": 1.0},
            input_variables={"flow_out": "m^3/hr"},
            output_variables={"flow_in": "m^3/hr"},
        ),
    )

    # Swap the second relation again so its first surrogate is deactivated.
    unit.swap_relation(
        "secondary_flow_relation",
        _multilinear(
            {"flow_out": 3.0, "intercept": 4.0},
            input_variables={"flow_out": "m^3/hr"},
            output_variables={"flow_in": "m^3/hr"},
        ),
    )

    # Re-register relation 1's coefficients so they are the active ones.
    unit.switch_surrogate_block("surrogate_flow")

    # Snapshot the deactivated first surrogate for relation 2.
    block_deactivated = unit.find_component("surrogate_secondary_flow")
    assert block_deactivated is not None
    deactivated_coefs_before = {
        name: pyo.value(var) for name, var in block_deactivated.coefficients.items()
    }

    # Update parameters on the active surrogate.
    unit.update_parameters({"intercept": 5.0, "flow_out": 0.5})

    # Active surrogate's coefficients changed.
    assert pyo.value(unit.surrogate_flow.coefficients["intercept"]) == pytest.approx(
        5.0
    )
    assert pyo.value(unit.surrogate_flow.coefficients["flow_out"]) == pytest.approx(0.5)

    # Deactivated surrogate's coefficients are untouched.
    for name, var in block_deactivated.coefficients.items():
        assert pyo.value(var) == pytest.approx(deactivated_coefs_before[name])

    # Relation records are preserved with full surrogate block histories.
    records = {r.name: r for r in unit._io_registry.relations}
    assert "flow_relation" in records
    assert "secondary_flow_relation" in records
    assert len(records["flow_relation"].surrogate_blocks) == 1
    assert len(records["secondary_flow_relation"].surrogate_blocks) == 2

    # Each surrogate block owns its own unique coefficient Vars.
    block_active = unit.surrogate_flow
    block_active_2 = unit.find_component("surrogate_secondary_flow_1")
    assert block_active_2 is not None
    coef_ids_1 = {id(var) for _, var in block_active.coefficients.items()}
    coef_ids_deactivated = {
        id(var) for _, var in block_deactivated.coefficients.items()
    }
    coef_ids_active_2 = {id(var) for _, var in block_active_2.coefficients.items()}
    assert coef_ids_1.isdisjoint(coef_ids_deactivated)
    assert coef_ids_1.isdisjoint(coef_ids_active_2)
    assert coef_ids_deactivated.isdisjoint(coef_ids_active_2)


@pytest.mark.unit
def test_register_surrogate_coefficients_preserves_other_relation_params():
    """Registering coefficients for one relation does not drop another relation's
    parameters from the unit registry when both surrogates share coefficient
    names.
    """
    _, unit = _two_flow_relation_unit()

    unit.swap_relation(
        "flow_relation",
        _multilinear(
            {"flow_out": 2.0, "intercept": 1.0},
            output_variables={"flow_out": "m^3/hr"},
        ),
    )

    unit.swap_relation(
        "secondary_flow_relation",
        _multilinear(
            {"flow_out": 2.0, "intercept": 1.0},
            input_variables={"flow_out": "m^3/hr"},
            output_variables={"flow_in": "m^3/hr"},
        ),
    )

    by_relation: dict[str, list] = {}
    for p in unit._io_registry.parameters:
        by_relation.setdefault(p.relation_name or "<unit>", []).append(p.name)

    assert by_relation["flow_relation"] == ["flow_out", "intercept"]
    assert by_relation["secondary_flow_relation"] == ["flow_out", "intercept"]

    coef_map = {p.name: p for p in unit._io_registry.parameters}
    secondary = unit.surrogate_secondary_flow.coefficients
    assert coef_map["intercept"].param is secondary["intercept"]
    assert coef_map["flow_out"].param is secondary["flow_out"]

    flow_block = unit.surrogate_flow
    flow_coef_ids = {id(var) for _, var in flow_block.coefficients.items()}
    assert id(coef_map["intercept"].param) not in flow_coef_ids


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
            "power_electrical_relation", _multilinear({"flow_out": 1.0})
        )


@pytest.mark.unit
def test_swap_relation_unknown_input_variable_raises():
    """A declared input variable not found on the unit is a config error."""
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
            _multilinear({"intercept": 1.0}, input_variables={"nope": "m^3/hr"}),
        )


def _unit_with_relation():
    """Build a bare unit carrying ``flow_out`` and a constant-intensity relation."""
    m = _model(4)
    m.unit = OpsBlock(property_package=m.props, allow_pass_through=False)
    m.unit.add_stream_ports()
    m.unit.add_component(
        "flow_out", pyo.Reference(m.unit.outlet_state.flow_vol_phase[:, "Liq"])
    )
    m.unit.add_constant_intensity_relation(
        m.unit.flow_out, intensity=0.5 * pyunits.kWh / pyunits.m**3
    )
    return m, m.unit


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


def _two_flow_relation_unit():
    """A bare unit carrying two registered, swappable non-power relations.

    Both relations use ``flow_out`` as their surrogate's input variable, so
    their multilinear surrogates share the same coefficient names. This lets
    us verify that ``update_parameters`` on one surrogate does not mutate the
    coefficient Vars of a deactivated surrogate on the same unit.
    """
    m = _model(4)
    m.unit = OpsBlock(property_package=m.props, allow_pass_through=False)
    m.unit.add_stream_ports()
    m.unit.add_component(
        "flow_out", pyo.Reference(m.unit.outlet_state.flow_vol_phase[:, "Liq"])
    )
    m.unit.add_component(
        "flow_in", pyo.Reference(m.unit.inlet_state.flow_vol_phase[:, "Liq"])
    )
    flow_out = m.unit.flow_out
    flow_in = m.unit.flow_in

    @m.unit.Constraint(m.time_block.time_index)
    def flow_relation(b, t):
        return flow_out[t] == 10.0

    @m.unit.Constraint(m.time_block.time_index)
    def secondary_flow_relation(b, t):
        return flow_in[t] == 5.0

    m.unit.register_relation(m.unit.flow_relation, target=flow_out)
    m.unit.register_relation(m.unit.secondary_flow_relation, target=flow_in)
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
        _multilinear(
            {"flow_out": 2.0, "intercept": 1.0},
            output_variables={"flow_out": "m^3/hr"},
        ),
    )

    fitted = unit.surrogate_flow.fitted
    assert all(not old[t].active for t in m.time_block.time_index)
    assert fitted is not None
    assert all(fitted[t].active for t in m.time_block.time_index)


@pytest.mark.unit
def test_swap_relation_auto_registers_coefficients():
    """swap_relation registers the surrogate's coefficients by default."""
    _, unit = _flow_relation_unit()
    unit.swap_relation(
        "flow_relation",
        _multilinear(
            {"flow_out": 2.0, "intercept": 1.0},
            output_variables={"flow_out": "m^3/hr"},
        ),
    )

    param_names = {p.name for p in unit._io_registry.parameters}
    assert "intercept" in param_names
    assert "flow_out" in param_names
    coef_map = {p.name: p for p in unit._io_registry.parameters}
    assert coef_map["intercept"].param is unit.surrogate_flow.coefficients["intercept"]
    assert coef_map["flow_out"].param is unit.surrogate_flow.coefficients["flow_out"]


@pytest.mark.unit
def test_swap_relation_auto_register_opt_out_skips_registration():
    """auto_register_coefficients=False leaves the registry untouched."""
    _, unit = _flow_relation_unit()
    unit.swap_relation(
        "flow_relation",
        _multilinear(
            {"flow_out": 2.0, "intercept": 1.0},
            output_variables={"flow_out": "m^3/hr"},
        ),
        auto_register_coefficients=False,
    )

    assert not unit._io_registry.parameters
    assert unit.surrogate_flow.coefficients["intercept"].value == pytest.approx(1.0)


@pytest.mark.unit
def test_swap_relation_takes_units_from_its_target():
    """The fitted constraint carries the target's own units, not the surrogate's."""
    _, unit = _flow_relation_unit()

    unit.swap_relation(
        "flow_relation",
        _multilinear(
            {"flow_out": 2.0, "intercept": 1.0},
            output_variables={"flow_out": "m^3/hr"},
        ),
    )

    fitted = unit.surrogate_flow.fitted
    # A stray kW hardcode would make this m^3/hr == m^3/hr + kW: inconsistent.
    assert_units_consistent(fitted)
    unit.flow_out[0].set_value(5.0)
    # flow_out - (intercept + 2*flow_out) == 5 - (1 + 10) == -6.0
    assert pyo.value(fitted[0].body) == pytest.approx(-6.0)


@pytest.mark.unit
def test_swap_relation_reads_a_coefficient_in_its_declared_basis():
    """A coefficient fitted in one unit basis attaches to a model in another.

    ``flow_out`` is m^3/hr on the model; declaring the surrogate's own input
    basis as m^3/s means each factor is converted before the coefficient
    multiplies it, not read in whatever units the model happens to carry.
    """
    _, unit = _flow_relation_unit()

    unit.swap_relation(
        "flow_relation",
        _multilinear(
            {"flow_out": 1.0, "intercept": 0.0},
            input_variables={"flow_out": "m^3/s"},
            output_variables={"flow_out": "m^3/hr"},
        ),
    )

    unit.flow_out[0].set_value(36.0)
    # 36 m^3/hr -> 0.01 m^3/s; 1.0 * 0.01 == 0.01 m^3/hr (output already
    # matches the target, so no further conversion). Naively reading 36.0 as
    # if it were already in m^3/s (no conversion) would give 36.0, not 0.01.
    fitted = unit.surrogate_flow.fitted
    assert pyo.value(fitted[0].body) == pytest.approx(36.0 - 0.01)


@pytest.mark.unit
def test_swap_relation_unknown_name_lists_registered_relations():
    """An unregistered relation name is refused, listing what is registered."""
    _, unit = _unit_with_relation()
    with pytest.raises(FlexConfigError, match="power_electrical_relation"):
        unit.swap_relation("nope", _multilinear({"intercept": 1.0}))


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
            "pass_through_flow_vol_phase_eq", _multilinear({"intercept": 1.0})
        )


@pytest.mark.unit
def test_swap_relation_accepts_a_non_multilinear_surrogate():
    """swap_relation is not limited to MultilinearSurrogate.

    Any Surrogate subclass works -- here a softplus (ICNN-style) forward
    pass, verified against a hand computation.
    """

    class _SoftplusSurrogate(Surrogate):
        def _validate(self):
            pass

        @property
        def input_variables(self):
            return {"flow_out": "m^3/hr"}

        @property
        def output_variables(self):
            return {"power_electrical": "kW"}

        def build(self, unit, target) -> tuple[None, Any]:
            q = unit.resolve_variable("flow_out", field="input_variables")
            c = self.data

            def body(t):
                x = q[t] / pyunits.get_units(q[t])
                value = c["wz"] * pyo.log(1 + pyo.exp(c["w"] * x + c["b"])) + c["c"]
                return value * pyunits.get_units(target[t])

            return None, body

    _, unit = _unit_with_relation()

    unit.swap_relation(
        "power_electrical_relation",
        _SoftplusSurrogate({"w": 0.3, "b": -1.0, "wz": 2.0, "c": 5.0}),
    )

    unit.flow_out[0].set_value(10.0)
    unit.power_electrical[0].set_value(0.0)
    expected = 2.0 * math.log(1 + math.exp(0.3 * 10 - 1.0)) + 5.0
    assert pyo.value(unit.power_electrical_relation_fitted[0].body) == pytest.approx(
        -expected
    )


class _LaggedSurrogate(Surrogate):
    """A surrogate whose body skips the horizon point where its lag is undefined."""

    def _validate(self):
        pass

    @property
    def input_variables(self):
        return {}

    @property
    def output_variables(self):
        return {"power_electrical": "kW"}

    def build(self, unit, target) -> tuple[None, Any]:
        def body(t):
            return pyo.Constraint.Skip if t < 1 else 2.0 * pyunits.get_units(target[t])

        return None, body


@pytest.mark.unit
def test_swap_relation_skips_indices_a_body_declines():
    """A body returning Constraint.Skip omits that index from the fitted relation.

    This is what lets a lagged/state-space form skip the horizon points where
    its lag does not exist, rather than raising a KeyError.
    """
    _, unit = _unit_with_relation()

    unit.swap_relation("power_electrical_relation", _LaggedSurrogate({}))

    fitted = unit.power_electrical_relation_fitted
    assert 0 not in fitted
    assert 1 in fitted


@pytest.mark.unit
def test_reswapping_deactivates_a_builders_auxiliary_constraints():
    """A second swap deactivates whatever the first builder's own attached.

    Without this, a ReLU big-M or ARIMA-innovations builder's auxiliary
    equality would stay active alongside the new one, double-counting.
    """

    class _AuxSurrogate(Surrogate):
        def _validate(self):
            pass

        @property
        def input_variables(self):
            return {}

        @property
        def output_variables(self):
            return {"power_electrical": "kW"}

        def build(self, unit, target) -> tuple[None, Any]:
            tag = self.data["tag"]
            tb = unit.model().time_block
            suffix = f"_{tag:.0f}"
            unit.add_component(f"aux_z{suffix}", pyo.Var(tb.time_index, initialize=0.0))
            z = unit.find_component(f"aux_z{suffix}")
            unit.add_component(
                f"aux_eq{suffix}",
                pyo.Constraint(tb.time_index, rule=lambda b, t: z[t] == tag),
            )

            def body(t):
                return z[t] * pyunits.get_units(target[t])

            return None, body

    _, unit = _unit_with_relation()

    unit.swap_relation("power_electrical_relation", _AuxSurrogate({"tag": 1.0}))
    first_eq = unit.aux_eq_1
    assert first_eq[0].active

    unit.swap_relation("power_electrical_relation", _AuxSurrogate({"tag": 2.0}))

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

    m.unit_a.swap_relation("power_electrical_relation", _multilinear({"flow_out": 1.0}))

    swapped = list(iter_swapped_relations(m))
    assert len(swapped) == 1
    block, record = swapped[0]
    assert block is m.unit_a
    assert record.name == "power_electrical_relation"
    assert record.fitted is m.unit_a.surrogate_power_electrical.fitted


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
            surrogate_type=SurrogateType.MULTILINEAR,
            data={
                "input_variables": {"power_electrical": "kW"},
                "output_variables": {"power_electrical": "kW"},
                "coefficients": {"power_electrical": 1.0},
            },
        ),
    )
    m.unit = OpsBlock(property_package=m.props, flexops_config=cfg)
    m.unit.add_stream_ports()
    flow = pyo.Reference(m.unit.outlet_state.flow_vol_phase[:, "Liq"])
    m.unit.add_constant_intensity_relation(
        flow, intensity=0.5 * pyunits.kWh / pyunits.m**3
    )
    assert m.unit.power_electrical_relation[0].active is False
    assert m.unit.surrogate_power_electrical.fitted is not None


@pytest.mark.unit
def test_list_surrogate_blocks_empty_before_swap():
    """No surrogate has been built, so the history is empty."""
    _, unit = _unit_with_relation()
    assert unit.list_surrogate_blocks("power_electrical_relation") == []


@pytest.mark.unit
def test_list_surrogate_blocks_after_swap():
    """A single swap produces one entry in the history."""
    _, unit = _flow_relation_unit()
    unit.swap_relation(
        "flow_relation",
        _multilinear(
            {"flow_out": 1.0, "intercept": 0.0},
            output_variables={"flow_out": "m^3/hr"},
        ),
    )
    names = unit.list_surrogate_blocks("flow_relation")
    assert names == ["surrogate_flow"]


@pytest.mark.unit
def test_list_surrogate_blocks_after_multiple_swaps():
    """Every swap is recorded; the list is oldest first."""
    _, unit = _flow_relation_unit()
    unit.swap_relation(
        "flow_relation",
        _multilinear(
            {"flow_out": 1.0, "intercept": 0.0},
            output_variables={"flow_out": "m^3/hr"},
        ),
    )
    unit.swap_relation(
        "flow_relation",
        _multilinear(
            {"flow_out": 2.0, "intercept": 0.0},
            output_variables={"flow_out": "m^3/hr"},
        ),
    )
    names = unit.list_surrogate_blocks("flow_relation")
    assert names == ["surrogate_flow", "surrogate_flow_1"]


@pytest.mark.unit
def test_list_surrogate_blocks_unknown_relation_raises():
    """An unregistered relation name raises FlexConfigError."""
    _, unit = _unit_with_relation()
    with pytest.raises(FlexConfigError, match="nope"):
        unit.list_surrogate_blocks("nope")


@pytest.mark.unit
def test_current_surrogate_block_none_before_swap():
    """No surrogate is active before any swap."""
    _, unit = _unit_with_relation()
    assert unit.current_surrogate_block("power_electrical_relation") is None


@pytest.mark.unit
def test_current_surrogate_block_after_swap():
    """After a swap, current returns the active block's local name."""
    _, unit = _flow_relation_unit()
    unit.swap_relation(
        "flow_relation",
        _multilinear(
            {"flow_out": 1.0, "intercept": 0.0},
            output_variables={"flow_out": "m^3/hr"},
        ),
    )
    assert unit.current_surrogate_block("flow_relation") == "surrogate_flow"


@pytest.mark.unit
def test_current_surrogate_block_unknown_relation_raises():
    """An unregistered relation name raises FlexConfigError."""
    _, unit = _unit_with_relation()
    with pytest.raises(FlexConfigError, match="nope"):
        unit.current_surrogate_block("nope")


@pytest.mark.unit
def test_list_surrogate_blocks_without_relation_returns_all():
    """Calling list_surrogate_blocks() with no relation returns every block."""
    _, unit = _flow_relation_unit()
    unit.swap_relation(
        "flow_relation",
        _multilinear(
            {"flow_out": 1.0, "intercept": 0.0},
            output_variables={"flow_out": "m^3/hr"},
        ),
    )
    unit.swap_relation(
        "flow_relation",
        _multilinear(
            {"flow_out": 2.0, "intercept": 0.0},
            output_variables={"flow_out": "m^3/hr"},
        ),
    )
    assert unit.list_surrogate_blocks() == ["surrogate_flow", "surrogate_flow_1"]


@pytest.mark.unit
def test_list_surrogate_blocks_without_relation_returns_empty_when_none():
    """Calling list_surrogate_blocks() with no surrogates returns []."""
    _, unit = _flow_relation_unit()
    assert unit.list_surrogate_blocks() == []


@pytest.mark.unit
def test_current_surrogate_block_without_relation_returns_dict():
    """Calling current_surrogate_block() with no relation returns active blocks."""
    _, unit = _flow_relation_unit()
    unit.swap_relation(
        "flow_relation",
        _multilinear(
            {"flow_out": 1.0, "intercept": 0.0},
            output_variables={"flow_out": "m^3/hr"},
        ),
    )
    assert unit.current_surrogate_block() == {"flow_relation": "surrogate_flow"}


@pytest.mark.unit
def test_current_surrogate_block_without_relation_returns_none_when_none_active():
    """Calling current_surrogate_block() with no active surrogates returns None."""
    _, unit = _flow_relation_unit()
    assert unit.current_surrogate_block() is None


@pytest.mark.unit
def test_switch_surrogate_block_reactivates_previous():
    """Switching back to a previously built block reactivates it."""
    _, unit = _flow_relation_unit()
    unit.swap_relation(
        "flow_relation",
        _multilinear(
            {"flow_out": 1.0, "intercept": 0.0},
            output_variables={"flow_out": "m^3/hr"},
        ),
    )
    unit.swap_relation(
        "flow_relation",
        _multilinear(
            {"flow_out": 2.0, "intercept": 0.0},
            output_variables={"flow_out": "m^3/hr"},
        ),
    )
    assert unit.current_surrogate_block("flow_relation") == "surrogate_flow_1"

    unit.switch_surrogate_block("surrogate_flow")
    assert unit.current_surrogate_block("flow_relation") == "surrogate_flow"


@pytest.mark.unit
def test_switch_surrogate_block_unknown_name_raises():
    """A block name that does not exist on the unit raises FlexConfigError."""
    _, unit = _flow_relation_unit()
    with pytest.raises(FlexConfigError, match="not_found"):
        unit.switch_surrogate_block("not_found")


@pytest.mark.unit
def test_switch_surrogate_block_non_surrogate_name_raises():
    """A non-surrogate component name raises FlexConfigError."""
    _, unit = _flow_relation_unit()
    with pytest.raises(FlexConfigError, match="flow_relation"):
        unit.switch_surrogate_block("flow_relation")


@pytest.mark.unit
def test_switch_surrogate_block_activates_fitted_constraint():
    """Switching to a block reactivates the block and its fitted Constraint.

    ``block.activate()`` alone does not re-activate a Constraint that was
    explicitly deactivated when the block was last switched away from, so
    ``switch_surrogate_block`` must activate the fitted Constraint explicitly.
    """
    _, unit = _flow_relation_unit()
    unit.swap_relation(
        "flow_relation",
        _multilinear(
            {"flow_out": 1.0, "intercept": 0.0},
            output_variables={"flow_out": "m^3/hr"},
        ),
    )
    unit.swap_relation(
        "flow_relation",
        _multilinear(
            {"flow_out": 2.0, "intercept": 0.0},
            output_variables={"flow_out": "m^3/hr"},
        ),
    )

    block = unit.surrogate_flow
    fitted = block.fitted
    assert not block.active
    assert not fitted.active

    unit.switch_surrogate_block("surrogate_flow")

    assert block.active
    assert fitted.active


@pytest.mark.unit
def test_switch_surrogate_block_among_three_preserves_activation():
    """Three surrogates can be built and switched between freely.

    After each ``swap_relation`` only the newest block is active. After
    each ``switch_surrogate_block`` exactly that block and its fitted
    Constraint are active; every previously built surrogate block and
    fitted Constraint is inactive.
    """
    _, unit = _flow_relation_unit()

    specs = [
        _multilinear(
            {"flow_out": 1.0, "intercept": 0.0},
            output_variables={"flow_out": "m^3/hr"},
        ),
        _multilinear(
            {"flow_out": 2.0, "intercept": 0.0},
            output_variables={"flow_out": "m^3/hr"},
        ),
        _multilinear(
            {"flow_out": 3.0, "intercept": 0.0},
            output_variables={"flow_out": "m^3/hr"},
        ),
    ]

    expected_names = ["surrogate_flow", "surrogate_flow_1", "surrogate_flow_2"]

    for spec in specs:
        unit.swap_relation("flow_relation", spec)

    all_blocks = [unit.find_component(name) for name in expected_names]
    all_fitted = [
        all_blocks[0].find_component("fitted"),
        all_blocks[1].find_component("fitted_2"),
        all_blocks[2].find_component("fitted_3"),
    ]

    assert unit.current_surrogate_block("flow_relation") == "surrogate_flow_2"
    for block, fitted, name in zip(all_blocks, all_fitted, expected_names, strict=True):
        if name == "surrogate_flow_2":
            assert block.active
            assert fitted.active
        else:
            assert not block.active
            assert not fitted.active

    for expected_name, expected_block, _expected_fitted in zip(
        expected_names, all_blocks, all_fitted, strict=True
    ):
        unit.switch_surrogate_block(expected_name)
        assert unit.current_surrogate_block("flow_relation") == expected_name
        for block, fitted in zip(all_blocks, all_fitted, strict=True):
            if block is expected_block:
                assert block.active
                assert fitted.active
            else:
                assert not block.active
                assert not fitted.active


@pytest.mark.unit
def test_unfix_surrogate_coefficients_named_relation():
    """Passing a relation_name only unfixes that relation's coefficients."""
    m, unit = _flow_relation_unit()
    unit.swap_relation(
        "flow_relation",
        _multilinear(
            {"flow_out": 1.0, "intercept": 0.0},
            output_variables={"flow_out": "m^3/hr"},
        ),
    )

    for _, var in unit.surrogate_flow.coefficients.items():
        var.fix()

    assert all(var.is_fixed() for _, var in unit.surrogate_flow.coefficients.items())

    unit.unfix_surrogate_coefficients("flow_relation")

    assert all(
        not var.is_fixed() for _, var in unit.surrogate_flow.coefficients.items()
    )


@pytest.mark.unit
def test_unfix_surrogate_coefficients_unknown_relation_raises():
    """An unregistered relation name raises FlexConfigError."""
    _, unit = _unit_with_relation()
    with pytest.raises(FlexConfigError, match="nope"):
        unit.unfix_surrogate_coefficients("nope")


@pytest.mark.unit
def test_fix_surrogate_coefficients_named_relation():
    """Passing a relation_name only fixes that relation's coefficients."""
    m, unit = _flow_relation_unit()
    unit.swap_relation(
        "flow_relation",
        _multilinear(
            {"flow_out": 1.0, "intercept": 0.0},
            output_variables={"flow_out": "m^3/hr"},
        ),
    )

    for _, var in unit.surrogate_flow.coefficients.items():
        var.unfix()

    assert all(
        not var.is_fixed() for _, var in unit.surrogate_flow.coefficients.items()
    )

    unit.fix_surrogate_coefficients("flow_relation")

    assert all(var.is_fixed() for _, var in unit.surrogate_flow.coefficients.items())


@pytest.mark.unit
def test_fix_surrogate_coefficients_unknown_relation_raises():
    """An unregistered relation name raises FlexConfigError."""
    _, unit = _unit_with_relation()
    with pytest.raises(FlexConfigError, match="nope"):
        unit.fix_surrogate_coefficients("nope")


class _NoCoefficientsSurrogate(Surrogate):
    """A surrogate whose block carries no ``coefficients`` attribute."""

    surrogate_type = SurrogateType.MULTILINEAR

    def _validate(self):
        pass

    @property
    def input_variables(self):
        return {}

    @property
    def output_variables(self):
        return {"flow_out": "m^3/hr"}

    def build(self, unit, target):
        block = pyo.Block(concrete=True)
        output_units = pyunits.get_units(target[0])
        return block, lambda t: 1.0 * output_units


class _DispatchSurrogate(Surrogate):
    """Surrogate double exposing configurable OpsBlock dispatch helpers."""

    surrogate_type = SurrogateType.MULTILINEAR

    def _validate(self):
        pass

    @property
    def input_variables(self):
        return {}

    @property
    def output_variables(self):
        return {"flow_out": "m^3/hr"}

    def __init__(self, *, spec=False, objective=None):
        super().__init__({})
        self._has_spec = spec
        self._objective = objective

    def build(self, unit, target):
        block = pyo.Block(concrete=True)
        output_units = pyunits.get_units(target[0])
        if self._objective == "regression":
            block.get_regression_objective = lambda **kwargs: kwargs.get("value", 1.0)
        elif self._objective == "fallback":
            block.get_objective = lambda **kwargs: kwargs.get("value", 2.0)
        if self._has_spec:
            self.get_surrogate_spec = lambda block, target: {
                "block": block,
                "target": target,
            }
        return block, lambda t: 1.0 * output_units


@pytest.mark.unit
def test_get_surrogate_spec_dispatches_named_and_default_relation():
    """get_surrogate_spec delegates for an explicit or first active relation."""
    _, unit = _flow_relation_unit()
    unit.swap_relation("flow_relation", _DispatchSurrogate(spec=True))

    named = unit.get_surrogate_spec("flow_relation")
    default = unit.get_surrogate_spec()

    assert named["target"] is unit.flow_out
    assert set(named) == {"block", "target"}
    assert default["target"] is unit.flow_out


@pytest.mark.unit
def test_get_surrogate_spec_rejects_invalid_or_unsupported_relations():
    """get_surrogate_spec reports lookup, activation, and capability errors."""
    _, unit = _flow_relation_unit()
    with pytest.raises(FlexConfigError, match="not_registered"):
        unit.get_surrogate_spec("not_registered")
    with pytest.raises(FlexConfigError, match="no active surrogate blocks"):
        unit.get_surrogate_spec()

    unit.swap_relation("flow_relation", _NoCoefficientsSurrogate({}))
    with pytest.raises(FlexConfigError, match="does not implement get_surrogate_spec"):
        unit.get_surrogate_spec()


@pytest.mark.unit
def test_get_surrogate_spec_rejects_registered_relation_without_surrogate():
    """An explicitly selected registered relation must have an active surrogate."""
    _, unit = _flow_relation_unit()
    with pytest.raises(FlexConfigError, match="has no active surrogate block"):
        unit.get_surrogate_spec("flow_relation")


@pytest.mark.unit
def test_get_surrogate_objective_dispatches_helpers_and_default_relation():
    """get_surrogate_objective prefers regression then fallback helpers."""
    _, unit = _flow_relation_unit()
    unit.swap_relation("flow_relation", _DispatchSurrogate(objective="regression"))
    assert unit.get_surrogate_objective(value=3.0) == pytest.approx(3.0)

    _, fallback_unit = _flow_relation_unit()
    fallback_unit.swap_relation(
        "flow_relation", _DispatchSurrogate(objective="fallback")
    )
    assert fallback_unit.get_surrogate_objective(value=4.0) == pytest.approx(4.0)


@pytest.mark.unit
def test_get_surrogate_objective_rejects_invalid_or_unsupported_relations():
    """get_surrogate_objective reports lookup, activation, and capability errors."""
    _, unit = _flow_relation_unit()
    with pytest.raises(FlexConfigError, match="not_registered"):
        unit.get_surrogate_objective("not_registered")
    with pytest.raises(FlexConfigError, match="no active surrogate blocks"):
        unit.get_surrogate_objective()
    with pytest.raises(FlexConfigError, match="has no active surrogate block"):
        unit.get_surrogate_objective("flow_relation")

    unit.swap_relation("flow_relation", _NoCoefficientsSurrogate({}))
    with pytest.raises(FlexConfigError, match="does not expose a usable objective"):
        unit.get_surrogate_objective()


@pytest.mark.unit
def test_register_surrogate_coefficients_unknown_relation_raises():
    """An unregistered relation name raises FlexConfigError."""
    _, unit = _unit_with_relation()
    with pytest.raises(FlexConfigError, match="nope"):
        unit.register_surrogate_coefficients("nope")


@pytest.mark.unit
def test_register_surrogate_coefficients_no_surrogate_block_raises():
    """Registering coefficients on a relation with no swapped surrogate raises."""
    m, unit = _flow_relation_unit()
    with pytest.raises(FlexConfigError, match="has no surrogate block"):
        unit.register_surrogate_coefficients("flow_relation")


@pytest.mark.unit
def test_register_surrogate_coefficients_no_coefficients_attribute_raises():
    """A surrogate block without a ``coefficients`` attribute raises."""
    m, unit = _flow_relation_unit()
    unit.swap_relation("flow_relation", _NoCoefficientsSurrogate({}))
    with pytest.raises(FlexConfigError, match="has no 'coefficients'"):
        unit.register_surrogate_coefficients("flow_relation")


@pytest.mark.unit
def test_unfix_surrogate_coefficients_none_no_active_blocks_raises():
    """Calling unfix with relation_name=None and no surrogates raises."""
    _, unit = _flow_relation_unit()
    with pytest.raises(FlexConfigError, match="has no active surrogate blocks"):
        unit.unfix_surrogate_coefficients()


@pytest.mark.unit
def test_fix_surrogate_coefficients_none_no_active_blocks_raises():
    """Calling fix with relation_name=None and no surrogates raises."""
    _, unit = _flow_relation_unit()
    with pytest.raises(FlexConfigError, match="has no active surrogate blocks"):
        unit.fix_surrogate_coefficients()


@pytest.mark.unit
def test_unfix_surrogate_coefficients_named_relation_no_surrogate_block_raises():
    """A registered relation with no swapped surrogate raises."""
    m, unit = _flow_relation_unit()
    with pytest.raises(FlexConfigError, match="has no active surrogate block"):
        unit.unfix_surrogate_coefficients("flow_relation")


@pytest.mark.unit
def test_fix_surrogate_coefficients_named_relation_no_surrogate_block_raises():
    """A registered relation with no swapped surrogate raises."""
    m, unit = _flow_relation_unit()
    with pytest.raises(FlexConfigError, match="has no active surrogate block"):
        unit.fix_surrogate_coefficients("flow_relation")


@pytest.mark.unit
def test_unfix_surrogate_coefficients_none_skips_relations_without_surrogate():
    """When relation_name is None, relations without surrogates are skipped."""
    m, unit = _flow_relation_unit()
    unit.add_component(
        "second_flow_relation",
        pyo.Constraint(
            m.time_block.time_index,
            rule=lambda b, t: unit.flow_out[t] == 10.0,
        ),
    )
    unit.register_relation(unit.second_flow_relation, target=unit.flow_out)
    unit.swap_relation(
        "flow_relation",
        _multilinear(
            {"flow_out": 1.0, "intercept": 0.0},
            output_variables={"flow_out": "m^3/hr"},
        ),
    )
    for _, var in unit.surrogate_flow.coefficients.items():
        var.fix()

    unit.unfix_surrogate_coefficients()

    assert all(
        not var.is_fixed() for _, var in unit.surrogate_flow.coefficients.items()
    )


@pytest.mark.unit
def test_fix_surrogate_coefficients_none_skips_relations_without_surrogate():
    """When relation_name is None, relations without surrogates are skipped."""
    m, unit = _flow_relation_unit()
    unit.add_component(
        "second_flow_relation",
        pyo.Constraint(
            m.time_block.time_index,
            rule=lambda b, t: unit.flow_out[t] == 10.0,
        ),
    )
    unit.register_relation(unit.second_flow_relation, target=unit.flow_out)
    unit.swap_relation(
        "flow_relation",
        _multilinear(
            {"flow_out": 1.0, "intercept": 0.0},
            output_variables={"flow_out": "m^3/hr"},
        ),
    )
    for _, var in unit.surrogate_flow.coefficients.items():
        var.unfix()

    unit.fix_surrogate_coefficients()

    assert all(var.is_fixed() for _, var in unit.surrogate_flow.coefficients.items())
