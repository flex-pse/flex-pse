"""Combustor(OpsBlockData): multi-inlet gas mixing, electrical export (§3.2, §3.4)."""

import pyomo.environ as pyo
import pytest
from idaes.core import LiquidPhase, declare_process_block_class
from pyomo.environ import units as pyunits
from pyomo.network import Port

from flexcore import nomenclature as nm
from flexcore.config.schema import SurrogateSpec, SurrogateType, UnitConfig
from flexcore.exceptions import FlexConfigError
from flexops.properties.simple_gas import SimpleGasFlowData
from flexops.surrogates import MultilinearSurrogate
from flexops.testing import UnitModelTestHarness, dummy_gas_time_block
from flexops.unit_models import Combustor
from flexops.unit_models.powergeneration.combustor import CombustorPowerRelation

_DIGESTER_HV = 6.0 * pyunits.kWh / pyunits.m**3
_NATURAL_GAS_HV = 10.5 * pyunits.kWh / pyunits.m**3
_EFFICIENCY = 0.35
_INTERCEPT = 1.0
_SLOPE = 2.0


@declare_process_block_class("_TwoPhaseGasFlow")
class _TwoPhaseGasFlowData(SimpleGasFlowData):
    """Test-only stub: SimpleGasFlow plus a second phase.

    Exists only to exercise Combustor's single-phase guard -- no other
    two-phase property package exists in the repository.
    """

    def build(self) -> None:
        super().build()
        self.Liq = LiquidPhase()


# declare_process_block_class injects the constructible wrapper into this
# module's namespace at runtime; bind it explicitly (as simple_gas.py does)
# so static tools resolve the forward reference used below.
_TwoPhaseGasFlow = globals()["_TwoPhaseGasFlow"]


def _combustor(n: int = 3, **kwargs):
    """Build a Combustor on an ``n``-point ``dummy_gas_time_block``."""
    m = dummy_gas_time_block(n)
    m.unit = Combustor(property_package=m.properties, **kwargs)
    return m, m.unit


def _fix(unit, name: str, t, value: float) -> None:
    """Fix a named state var on inlet/outlet state block ``name`` at time ``t``."""
    getattr(unit, name)[t].fix(value)


def _surrogate_spec() -> SurrogateSpec:
    """A multilinear relationship for ``power_generated``, in magnitude space."""
    return SurrogateSpec(
        surrogate_type=SurrogateType.MULTILINEAR,
        data={
            "input_variables": {"flow_in_fuel": "m^3/hr"},
            "output_variables": {"power_generated": "kW"},
            "coefficients": {"intercept": _INTERCEPT, "flow_in_fuel": _SLOPE},
        },
    )


def _surrogate_config() -> UnitConfig:
    """A ``UnitConfig`` carrying :func:`_surrogate_spec`, for a config-time swap."""
    return UnitConfig(unit_model_class="Combustor", surrogate=_surrogate_spec())


def _multilinear_surrogate() -> MultilinearSurrogate:
    """The built surrogate, for a direct ``swap_relation`` call."""
    return MultilinearSurrogate(_surrogate_spec().data)


class TestCombustorConstantIntensity(UnitModelTestHarness):
    """Single ``fuel`` inlet, no heating values: the constant-intensity relation.

    With the default ``energy_intensity=2.0 kWh/m^3`` and the inlet's
    ``flow_vol_phase`` at its construction-time initial value of 1.0 m^3/hr,
    ``power_electrical[t] == -(2.0 * 1.0) == -2.0`` kW.
    """

    expected_dof = 0
    expected_solution = {"power_electrical[0]": -2.0}

    def configure(self):
        return _combustor(3, inlet_names=("fuel",))


class TestCombustorHeatingValue(UnitModelTestHarness):
    """Two named inlets, both with heating values: the heating-value relation.

    Both inlet flows sit at their construction-time initial value of 1.0
    m^3/hr, so ``power_electrical[t] == -efficiency * (HV_digester + HV_natural)
    == -0.35 * (6.0 + 10.5) == -5.775`` kW.
    """

    expected_dof = 0
    expected_solution = {"power_electrical[0]": -5.775}

    def configure(self):
        return _combustor(
            3,
            inlet_names=("digester_gas", "natural_gas"),
            heating_values={
                "digester_gas": _DIGESTER_HV,
                "natural_gas": _NATURAL_GAS_HV,
            },
            efficiency=_EFFICIENCY,
        )


# -- topology and balances -------------------------------------------------


@pytest.mark.unit
def test_combustor_builds_one_port_per_inlet_name():
    """Ports are named ``inlet_<name>`` per ``inlet_names``, plus one ``outlet``."""
    _, unit = _combustor(3, inlet_names=("digester_gas", "natural_gas", "air"))

    ports = {p.local_name for p in unit.component_objects(Port, descend_into=False)}
    assert ports == {"inlet_digester_gas", "inlet_natural_gas", "inlet_air", "outlet"}
    for port in ports:
        assert len(list(getattr(unit, port).values())) > 0
    assert unit.find_component("inlet") is None


@pytest.mark.unit
def test_combustor_mixing_mass_balance_body():
    """The flue-gas flow is the fuel sum scaled up by the entrained air."""
    ratio = 9.5
    _, unit = _combustor(3, inlet_names=("a", "b", "c"), air_to_fuel_ratio=ratio)
    fuel_sum = 1.0 + 2.0 + 3.0
    for t in range(3):
        _fix(unit, "flow_in_a", t, 1.0)
        _fix(unit, "flow_in_b", t, 2.0)
        _fix(unit, "flow_in_c", t, 3.0)
        _fix(unit, "flow_out", t, (1 + ratio) * fuel_sum)
        assert pyo.value(unit.mixing_mass_balance[t].body) == pytest.approx(
            0.0, abs=1e-9
        )
        unit.flow_out[t].fix(fuel_sum)
        assert pyo.value(unit.mixing_mass_balance[t].body) == pytest.approx(
            fuel_sum - (1 + ratio) * fuel_sum, abs=1e-9
        )


@pytest.mark.unit
def test_combustor_arbitrary_inlet_count():
    """Four heating-value inlets build, are units-consistent, and all sum."""
    from pyomo.util.check_units import assert_units_consistent

    _, unit = _combustor(
        3,
        inlet_names=("a", "b", "c", "d"),
        heating_values={
            "a": 5.0 * pyunits.kWh / pyunits.m**3,
            "b": 6.0 * pyunits.kWh / pyunits.m**3,
            "c": 7.0 * pyunits.kWh / pyunits.m**3,
            "d": 8.0 * pyunits.kWh / pyunits.m**3,
        },
    )
    assert_units_consistent(unit)
    for t in range(3):
        for name, val in zip(("a", "b", "c", "d"), (1.0, 2.0, 3.0, 4.0), strict=True):
            _fix(unit, f"flow_in_{name}", t, val)
        ratio = pyo.value(unit.air_to_fuel_ratio)
        unit.flow_out[t].fix((1 + ratio) * 10.0)
        assert pyo.value(unit.mixing_mass_balance[t].body) == pytest.approx(
            0.0, abs=1e-9
        )


@pytest.mark.unit
def test_combustor_registers_only_the_reference_inlet_intensive_states():
    """Every inlet's flow is an input; only the reference inlet's other states are."""
    _, unit = _combustor(3, inlet_names=("digester_gas", "natural_gas"))
    inputs = {
        rec.var.parent_block().local_name + "." + rec.name
        for rec in unit._io_registry.io_variables
        if rec.role == "input"
    }
    assert "inlet_digester_gas_state.flow_vol_phase" in inputs
    assert "inlet_natural_gas_state.flow_vol_phase" in inputs
    for state_var in ("pressure", "temperature"):
        assert f"inlet_digester_gas_state.{state_var}" in inputs
        assert f"inlet_natural_gas_state.{state_var}" not in inputs

    outputs = {
        rec.var.parent_block().local_name + "." + rec.name
        for rec in unit._io_registry.io_variables
        if rec.role == "output"
    }
    for state_var in ("pressure", "temperature", "flow_vol_phase"):
        assert f"outlet_state.{state_var}" in outputs


# -- outlet state -----------------------------------------------------------


@pytest.mark.unit
def test_combustor_outlet_temperature_is_the_flue_gas_temperature():
    """``outlet_temperature_eq`` holds the outlet at ``flue_gas_temperature``."""
    _, unit = _combustor(3, inlet_names=("fuel",), flue_gas_temperature=800 * pyunits.K)
    assert unit.find_component("pass_through_temperature_eq") is None
    for t in range(3):
        unit.outlet_state.temperature[t].fix(800.0)
        assert pyo.value(unit.outlet_temperature_eq[t].body) == pytest.approx(
            0.0, abs=1e-9
        )
        unit.outlet_state.temperature[t].fix(750.0)
        # Pyomo orders a scalar Var (flue_gas_temperature) before an indexed
        # one (outlet_state.temperature[t]) in an equality's body regardless
        # of the write order in the rule, so only the magnitude is asserted.
        assert abs(pyo.value(unit.outlet_temperature_eq[t].body)) == pytest.approx(
            50.0, abs=1e-9
        )


@pytest.mark.unit
def test_combustor_air_to_fuel_ratio_is_a_fixed_regressable_parameter():
    """The ratio is a fixed, registered, non-negative scalar Var."""
    _, unit = _combustor(3, inlet_names=("fuel",), air_to_fuel_ratio=8.0)

    assert unit.air_to_fuel_ratio.fixed
    assert pyo.value(unit.air_to_fuel_ratio) == pytest.approx(8.0)
    assert unit.air_to_fuel_ratio.bounds == (0.0, None)
    registered = {rec.name: rec.regressable for rec in unit._io_registry.parameters}
    assert registered["air_to_fuel_ratio"] is True


@pytest.mark.unit
def test_combustor_zero_air_to_fuel_ratio_recovers_the_plain_inlet_sum():
    """A zero ratio entrains no air, leaving flue volume == the fuel sum."""
    _, unit = _combustor(3, inlet_names=("a", "b"), air_to_fuel_ratio=0.0)
    for t in range(3):
        _fix(unit, "flow_in_a", t, 2.0)
        _fix(unit, "flow_in_b", t, 3.0)
        _fix(unit, "flow_out", t, 5.0)
        assert pyo.value(unit.mixing_mass_balance[t].body) == pytest.approx(
            0.0, abs=1e-9
        )


@pytest.mark.unit
def test_combustor_rejects_a_negative_air_to_fuel_ratio():
    """A negative air-to-gas ratio is not a physical IC design."""
    with pytest.raises(ValueError, match="air_to_fuel_ratio"):
        _combustor(3, inlet_names=("fuel",), air_to_fuel_ratio=-1.0)


@pytest.mark.unit
def test_combustor_passes_pressure_through():
    """``pass_through_pressure_eq`` exists exactly once and holds by hand."""
    _, unit = _combustor(3, inlet_names=("fuel",))
    assert unit.find_component("pass_through_pressure_eq") is not None
    for t in range(3):
        unit.inlet_fuel_state.pressure[t].fix(101325.0)
        unit.outlet_state.pressure[t].fix(101325.0)
        assert pyo.value(unit.pass_through_pressure_eq[t].body) == pytest.approx(
            0.0, abs=1e-9
        )


# -- relation selection and gating ------------------------------------------


@pytest.mark.unit
def test_combustor_defaults_to_constant_intensity():
    """No ``heating_values`` given -> ``CONSTANT_INTENSITY``; ``efficiency`` absent."""
    _, unit = _combustor(3, inlet_names=("fuel",))
    assert unit._power_relation is CombustorPowerRelation.CONSTANT_INTENSITY
    assert unit.find_component("energy_intensity") is not None
    assert unit.find_component("efficiency") is None


@pytest.mark.unit
def test_combustor_selects_heating_value_when_every_inlet_has_one():
    """A heating value for every inlet selects ``HEATING_VALUE``, not intensity."""
    _, unit = _combustor(
        3,
        inlet_names=("digester_gas", "natural_gas"),
        heating_values={"digester_gas": _DIGESTER_HV, "natural_gas": _NATURAL_GAS_HV},
    )
    assert unit._power_relation is CombustorPowerRelation.HEATING_VALUE
    for name in ("digester_gas", "natural_gas"):
        var = unit.find_component(f"heating_value_{name}")
        assert var is not None
        assert var.fixed
        assert str(pyunits.get_units(var)) == str(pyunits.kWh / pyunits.m**3)
    assert unit.find_component("energy_intensity") is None


@pytest.mark.unit
def test_combustor_heating_values_cover_utility_fuel_sources():
    """Heating values must cover both inlets and utility fuel sources."""
    _, unit = _combustor(
        3,
        inlet_names=("digester_gas",),
        utility_fuel_source="natural_gas",
        heating_values={
            "digester_gas": _DIGESTER_HV,
            "natural_gas": _NATURAL_GAS_HV,
        },
    )
    assert unit._power_relation is CombustorPowerRelation.HEATING_VALUE
    assert unit.find_component("heating_value_digester_gas") is not None
    assert unit.find_component("heating_value_natural_gas") is not None


@pytest.mark.unit
def test_combustor_rejects_partial_heating_values_with_utility_fuel():
    """Partial heating values raise when utility fuel sources are present."""
    with pytest.raises(FlexConfigError) as excinfo:
        _combustor(
            3,
            inlet_names=("digester_gas",),
            utility_fuel_source="natural_gas",
            heating_values={"digester_gas": _DIGESTER_HV},
        )
    assert excinfo.value.field == "heating_values"


@pytest.mark.unit
@pytest.mark.parametrize(
    "heating_values",
    [
        {"digester_gas": _DIGESTER_HV},
        {"digester_gas": _DIGESTER_HV, "natural_gas": _NATURAL_GAS_HV, "coal": 1.0},
    ],
)
def test_combustor_rejects_a_bad_heating_value_mapping(heating_values):
    """A partial mapping, or one naming an unknown inlet, is rejected by field."""
    with pytest.raises(FlexConfigError) as excinfo:
        _combustor(
            3,
            inlet_names=("digester_gas", "natural_gas"),
            heating_values=heating_values,
        )
    assert excinfo.value.field == "heating_values"


@pytest.mark.unit
def test_combustor_rejects_efficiency_under_constant_intensity():
    """``efficiency`` set with no ``heating_values`` is a no-op the config rejects."""
    with pytest.raises(FlexConfigError) as excinfo:
        _combustor(3, inlet_names=("fuel",), efficiency=0.5)
    assert excinfo.value.field == "efficiency"


@pytest.mark.unit
def test_combustor_rejects_energy_intensity_under_heating_value():
    """``energy_intensity`` set alongside a full ``heating_values`` map is rejected."""
    with pytest.raises(FlexConfigError) as excinfo:
        _combustor(
            3,
            inlet_names=("fuel",),
            heating_values={"fuel": _DIGESTER_HV},
            energy_intensity=1.0 * pyunits.kWh / pyunits.m**3,
        )
    assert excinfo.value.field == "energy_intensity"


@pytest.mark.unit
@pytest.mark.parametrize("inlet_names", [(), ("a", "a")])
def test_combustor_rejects_empty_or_duplicate_inlet_names(inlet_names):
    """Empty or duplicated ``inlet_names`` raise, naming the field."""
    with pytest.raises(FlexConfigError) as excinfo:
        _combustor(3, inlet_names=inlet_names)
    assert excinfo.value.field == "inlet_names"


@pytest.mark.unit
def test_combustor_requires_a_single_phase_property_package():
    """A property package with more than one phase is rejected."""
    m = dummy_gas_time_block(3)
    m.two_phase_properties = _TwoPhaseGasFlow()
    with pytest.raises(FlexConfigError) as excinfo:
        m.unit = Combustor(
            property_package=m.two_phase_properties, inlet_names=("fuel",)
        )
    assert excinfo.value.field == "property_package"


# -- power relation -----------------------------------------------------------


@pytest.mark.unit
def test_combustor_power_relation_body_heating_value():
    """The heating-value relation body is 0 at the hand-computed magnitude."""
    _, unit = _combustor(
        3,
        inlet_names=("digester_gas", "natural_gas"),
        heating_values={"digester_gas": _DIGESTER_HV, "natural_gas": _NATURAL_GAS_HV},
        efficiency=_EFFICIENCY,
    )
    for t in range(3):
        _fix(unit, "flow_in_digester_gas", t, 2.0)
        _fix(unit, "flow_in_natural_gas", t, 3.0)
        expected = _EFFICIENCY * (6.0 * 2.0 + 10.5 * 3.0)
        unit.power_generated[t].fix(expected)
        assert pyo.value(unit.power_electrical_relation[t].body) == pytest.approx(
            0.0, rel=1e-6
        )


@pytest.mark.unit
def test_combustor_power_relation_body_constant_intensity():
    """The constant-intensity relation body is 0 over the total inlet flow."""
    _, unit = _combustor(
        3, inlet_names=("a", "b"), energy_intensity=1.5 * pyunits.kWh / pyunits.m**3
    )
    for t in range(3):
        _fix(unit, "flow_in_a", t, 2.0)
        _fix(unit, "flow_in_b", t, 3.0)
        expected = 1.5 * (2.0 + 3.0)
        unit.power_generated[t].fix(expected)
        assert pyo.value(unit.power_electrical_relation[t].body) == pytest.approx(
            0.0, rel=1e-6
        )


@pytest.mark.unit
def test_combustor_power_electrical_is_an_export():
    """``power_electrical[t]`` is upper-bounded at 0."""
    _, unit = _combustor(3, inlet_names=("fuel",))
    for t in range(3):
        assert unit.power_electrical[t].ub == pytest.approx(0.0)


@pytest.mark.unit
def test_combustor_power_generated_is_non_negative():
    """``power_generated[t]`` -- the relation's target -- is bounded below at 0."""
    _, unit = _combustor(3, inlet_names=("fuel",))
    for t in range(3):
        assert unit.power_generated[t].lb == pytest.approx(0.0)
        assert unit.power_generated[t].ub is None


@pytest.mark.unit
def test_combustor_sign_constraint_ties_export_to_generation():
    """``power_electrical_sign`` holds exactly when power == -power_generated."""
    _, unit = _combustor(3, inlet_names=("fuel",))
    for t in range(3):
        unit.power_generated[t].fix(4.0)
        unit.power_electrical[t].fix(-4.0)
        assert pyo.value(unit.power_electrical_sign[t].body) == pytest.approx(
            0.0, abs=1e-9
        )
        unit.power_electrical[t].fix(4.0)
        assert pyo.value(unit.power_electrical_sign[t].body) != pytest.approx(
            0.0, abs=1e-9
        )


@pytest.mark.unit
def test_combustor_sign_constraint_is_not_swappable():
    """Only the magnitude relation is registered, so only it can be swapped."""
    _, unit = _combustor(3, inlet_names=("fuel",))
    registered = {record.name for record in unit._io_registry.relations}
    assert registered == {"power_electrical_relation"}
    assert unit.find_component("power_electrical_sign") is not None
    with pytest.raises(FlexConfigError) as excinfo:
        unit.swap_relation("power_electrical_sign", _multilinear_surrogate())
    assert excinfo.value.field == "relation_name"


@pytest.mark.unit
def test_combustor_relation_targets_the_generation_magnitude():
    """The registered relation determines ``power_generated``, not the draw."""
    _, unit = _combustor(3, inlet_names=("fuel",))
    record = next(
        r for r in unit._io_registry.relations if r.name == "power_electrical_relation"
    )
    assert record.target is unit.power_generated


@pytest.mark.unit
def test_combustor_registers_power_generated_as_its_io_output():
    """The regressed output is the magnitude, so a fit carries the model's sign."""
    _, unit = _combustor(3, inlet_names=("fuel",))
    outputs = {
        record.var.local_name
        for record in unit._io_registry.io_variables
        if record.role == "output"
    }
    assert "power_generated" in outputs
    assert "power_electrical" not in outputs


@pytest.mark.unit
def test_combustor_energy_intensity_is_non_negative():
    """A design mode or regression cannot unfix the intensity into a sign flip."""
    _, unit = _combustor(3, inlet_names=("fuel",))
    assert unit.energy_intensity.lb == pytest.approx(0.0)


@pytest.mark.unit
def test_combustor_swaps_its_relation_from_a_config_surrogate():
    """A config ``SurrogateSpec`` swaps the magnitude relation at construction."""
    _, unit = _combustor(3, inlet_names=("fuel",), flexops_config=_surrogate_config())
    assert not unit.power_electrical_relation[0].active
    assert unit.power_electrical_sign[0].active
    fitted = unit.surrogate_power_electrical.fitted
    assert fitted is not None
    assert fitted[0].active


@pytest.mark.unit
def test_combustor_swapped_relation_feeds_the_magnitude():
    """The fitted body determines ``power_generated``; the sign makes it an export."""
    _, unit = _combustor(3, inlet_names=("fuel",), flexops_config=_surrogate_config())
    fitted = unit.surrogate_power_electrical.fitted
    for t in range(3):
        _fix(unit, "flow_in_fuel", t, 1.0)
        unit.power_generated[t].fix(_INTERCEPT + _SLOPE)
        unit.power_electrical[t].fix(-(_INTERCEPT + _SLOPE))
        assert pyo.value(fitted[t].body) == pytest.approx(0.0, abs=1e-9)
        assert pyo.value(unit.power_electrical_sign[t].body) == pytest.approx(
            0.0, abs=1e-9
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"heating_values": {"fuel": _DIGESTER_HV}},
    ],
)
def test_combustor_power_relation_constraint_is_named(kwargs):
    """Both relations build ``power_electrical_relation`` (the swap contract)."""
    kwargs.setdefault("inlet_names", ("fuel",))
    _, unit = _combustor(3, **kwargs)
    assert unit.find_component("power_electrical_relation") is not None
    assert unit.find_component("power_eq") is None


@pytest.mark.unit
def test_combustor_registers_no_thermal_power():
    """Only an ELECTRICAL power record is registered; no ``power_thermal``."""
    _, unit = _combustor(3, inlet_names=("fuel",))
    kinds = {rec.kind for rec in unit._io_registry.power}
    assert kinds == {nm.PowerKind.ELECTRICAL}
    assert unit.find_component("power_thermal") is None


# -- integration with the rest of the library --------------------------------


@pytest.mark.unit
def test_combustor_registers_no_fuel_usage():
    """With utility_fuel_source=None (the default), no fuel usage is registered."""
    _, unit = _combustor(3, inlet_names=("fuel",))
    assert unit._io_registry.fuel == []


@pytest.mark.unit
def test_combustor_registers_single_utility_fuel_usage():
    """A single utility_fuel_source registers its own flow Var for costing."""
    _, unit = _combustor(
        3,
        inlet_names=("digester_gas",),
        utility_fuel_source="natural_gas",
    )
    fuel_records = unit._io_registry.fuel
    assert len(fuel_records) == 1
    record = fuel_records[0]
    assert record.fuel_name == "natural_gas"
    assert record.var is unit.utility_flow_natural_gas
    assert unit.find_component("utility_flow_natural_gas") is not None
    assert unit.find_component("fuel_usage_natural_gas") is None


@pytest.mark.unit
def test_combustor_registers_multiple_utility_fuel_usage():
    """Multiple utility_fuel_source names each get their own flow Var."""
    _, unit = _combustor(
        3,
        inlet_names=("digester_gas",),
        utility_fuel_source=["natural_gas", "propane"],
    )
    fuel_records = unit._io_registry.fuel
    assert len(fuel_records) == 2
    names = {rec.fuel_name for rec in fuel_records}
    assert names == {"natural_gas", "propane"}
    assert unit.find_component("utility_flow_natural_gas") is not None
    assert unit.find_component("utility_flow_propane") is not None


@pytest.mark.unit
def test_combustor_utility_fuel_source_is_independent_of_inlets():
    """utility_fuel_source names need not be in inlet_names."""
    _, unit = _combustor(
        3,
        inlet_names=("digester_gas",),
        utility_fuel_source="natural_gas",
    )
    assert unit.find_component("inlet_natural_gas") is None
    assert unit.find_component("utility_flow_natural_gas") is not None


@pytest.mark.unit
def test_combustor_rejects_duplicate_utility_fuel_source():
    """Duplicate names in utility_fuel_source raise FlexConfigError."""
    with pytest.raises(FlexConfigError) as excinfo:
        _combustor(3, utility_fuel_source=["natural_gas", "natural_gas"])
    assert excinfo.value.field == "utility_fuel_source"


@pytest.mark.unit
def test_combustor_rejects_utility_fuel_source_overlapping_inlet():
    """utility_fuel_source names must not duplicate inlet_names."""
    with pytest.raises(FlexConfigError) as excinfo:
        _combustor(
            3,
            inlet_names=("digester_gas",),
            utility_fuel_source="digester_gas",
        )
    assert excinfo.value.field == "utility_fuel_source"


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"inlet_names": None},
        {"inlet_names": ()},
    ],
)
def test_combustor_rejects_empty_inlets_without_utility_fuel(kwargs):
    """Empty or None inlet_names without utility_fuel_source raises FlexConfigError."""
    with pytest.raises(FlexConfigError) as excinfo:
        _combustor(3, **kwargs)
    assert excinfo.value.field == "inlet_names"


@pytest.mark.unit
def test_combustor_utility_fuel_only_constant_intensity():
    """A combustor with no inlets and only utility fuel builds and solves."""
    _, unit = _combustor(
        3,
        inlet_names=(),
        utility_fuel_source="natural_gas",
    )
    ports = {p.local_name for p in unit.component_objects(Port, descend_into=False)}
    assert ports == {"outlet"}
    assert unit.find_component("inlet_natural_gas") is None
    assert unit.find_component("utility_flow_natural_gas") is not None

    fuel_records = unit._io_registry.fuel
    assert len(fuel_records) == 1
    assert fuel_records[0].fuel_name == "natural_gas"
    assert unit._power_relation is CombustorPowerRelation.CONSTANT_INTENSITY

    for t in range(3):
        unit.utility_flow_natural_gas[t].fix(2.0)
        unit.flow_out[t].fix((1 + pyo.value(unit.air_to_fuel_ratio)) * 2.0)
        assert pyo.value(unit.mixing_mass_balance[t].body) == pytest.approx(
            0.0, abs=1e-9
        )
        unit.power_generated[t].fix(pyo.value(unit.energy_intensity) * 2.0)
        assert pyo.value(unit.power_electrical_relation[t].body) == pytest.approx(
            0.0, rel=1e-6
        )


@pytest.mark.unit
def test_combustor_utility_fuel_only_heating_value():
    """A combustor with no inlets uses heating values from utility fuels."""
    _, unit = _combustor(
        3,
        inlet_names=(),
        utility_fuel_source="natural_gas",
        heating_values={"natural_gas": _NATURAL_GAS_HV},
        efficiency=_EFFICIENCY,
    )
    assert unit._power_relation is CombustorPowerRelation.HEATING_VALUE
    assert unit.find_component("heating_value_natural_gas") is not None

    for t in range(3):
        unit.utility_flow_natural_gas[t].fix(2.0)
        unit.flow_out[t].fix((1 + pyo.value(unit.air_to_fuel_ratio)) * 2.0)
        assert pyo.value(unit.mixing_mass_balance[t].body) == pytest.approx(
            0.0, abs=1e-9
        )
        expected = _EFFICIENCY * pyo.value(_NATURAL_GAS_HV) * 2.0
        unit.power_generated[t].fix(expected)
        assert pyo.value(unit.power_electrical_relation[t].body) == pytest.approx(
            0.0, rel=1e-6
        )


@pytest.mark.unit
def test_combustor_utility_fuel_only_with_multiple_blend_ratios():
    """Multiple utility fuels with no inlets and blend ratio constraints."""
    _, unit = _combustor(
        3,
        inlet_names=(),
        utility_fuel_source=["natural_gas", "propane"],
        blend_ratio=[
            {"natural_gas": "propane", "ratio": 2.0},
        ],
    )
    ports = {p.local_name for p in unit.component_objects(Port, descend_into=False)}
    assert ports == {"outlet"}
    for name in ("natural_gas", "propane"):
        assert unit.find_component(f"utility_flow_{name}") is not None

    constraint = unit.find_component("blend_ratio_natural_gas_propane_constraint")
    assert constraint is not None
    for t in range(3):
        unit.utility_flow_propane[t].fix(1.0)
        unit.utility_flow_natural_gas[t].fix(2.0)
        assert pyo.value(constraint[t].body) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_combustor_utility_fuel_only_rejects_unknown_fuel_in_blend():
    """blend_ratio with an unknown fuel name raises when no inlets exist."""
    with pytest.raises(FlexConfigError, match="blend_ratio"):
        _combustor(
            3,
            inlet_names=(),
            utility_fuel_source="natural_gas",
            blend_ratio={"unknown": "natural_gas", "ratio": 1.0},
        )


@pytest.mark.unit
def test_combustor_blend_ratio_builds_parameter_and_constraint():
    """blend_ratio creates a fixed ratio parameter and a flow constraint."""
    _, unit = _combustor(
        3,
        inlet_names=("digester_gas", "natural_gas"),
        blend_ratio={"digester_gas": "natural_gas", "ratio": 1.6},
    )
    ratio_var = unit.find_component("blend_ratio_digester_gas_natural_gas")
    assert ratio_var is not None
    assert ratio_var.fixed
    assert pyo.value(ratio_var) == pytest.approx(1.6)

    constraint = unit.find_component("blend_ratio_digester_gas_natural_gas_constraint")
    assert constraint is not None

    for t in range(3):
        unit.flow_in_natural_gas[t].fix(2.0)
        unit.flow_in_digester_gas[t].fix(3.2)
        assert pyo.value(constraint[t].body) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_combustor_blend_ratio_between_inlet_and_utility():
    """blend_ratio works when one fuel is an inlet and the other is utility."""
    _, unit = _combustor(
        3,
        inlet_names=("digester_gas",),
        utility_fuel_source="natural_gas",
        blend_ratio={"digester_gas": "natural_gas", "ratio": 2.0},
    )
    constraint = unit.find_component("blend_ratio_digester_gas_natural_gas_constraint")
    assert constraint is not None
    for t in range(3):
        unit.utility_flow_natural_gas[t].fix(1.0)
        unit.flow_in_digester_gas[t].fix(2.0)
        assert pyo.value(constraint[t].body) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_combustor_blend_ratio_between_two_utility_fuels():
    """blend_ratio works when both fuels are utility fuel sources."""
    _, unit = _combustor(
        3,
        inlet_names=("digester_gas",),
        utility_fuel_source=["natural_gas", "propane"],
        blend_ratio={"natural_gas": "propane", "ratio": 3.0},
    )
    constraint = unit.find_component("blend_ratio_natural_gas_propane_constraint")
    assert constraint is not None
    for t in range(3):
        unit.utility_flow_propane[t].fix(1.0)
        unit.utility_flow_natural_gas[t].fix(3.0)
        assert pyo.value(constraint[t].body) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_combustor_multiple_blend_ratios_builds_multiple_constraints():
    """A list of blend_ratio dicts builds multiple parameters and constraints."""
    _, unit = _combustor(
        3,
        inlet_names=("digester_gas", "natural_gas", "propane"),
        blend_ratio=[
            {"digester_gas": "natural_gas", "ratio": 1.6},
            {"digester_gas": "propane", "ratio": 2.0},
        ],
    )
    for a, b, expected_ratio in (
        ("digester_gas", "natural_gas", 1.6),
        ("digester_gas", "propane", 2.0),
    ):
        ratio_var = unit.find_component(f"blend_ratio_{a}_{b}")
        assert ratio_var is not None
        assert ratio_var.fixed
        assert pyo.value(ratio_var) == pytest.approx(expected_ratio)

        constraint = unit.find_component(f"blend_ratio_{a}_{b}_constraint")
        assert constraint is not None

    for t in range(3):
        unit.flow_in_natural_gas[t].fix(1.0)
        unit.flow_in_propane[t].fix(1.0)
        unit.flow_in_digester_gas[t].fix(1.6)
        assert pyo.value(
            unit.blend_ratio_digester_gas_natural_gas_constraint[t].body
        ) == pytest.approx(0.0, abs=1e-9)
        assert pyo.value(
            unit.blend_ratio_digester_gas_propane_constraint[t].body
        ) == pytest.approx(-0.4, abs=1e-9)


@pytest.mark.unit
def test_combustor_multiple_blend_ratios_with_utility_fuel():
    """Multiple blend ratios work across inlet and utility fuel sources."""
    _, unit = _combustor(
        3,
        inlet_names=("digester_gas",),
        utility_fuel_source=["natural_gas", "propane"],
        blend_ratio=[
            {"digester_gas": "natural_gas", "ratio": 2.0},
            {"digester_gas": "propane", "ratio": 4.0},
        ],
    )
    for a, b in (("digester_gas", "natural_gas"), ("digester_gas", "propane")):
        constraint = unit.find_component(f"blend_ratio_{a}_{b}_constraint")
        assert constraint is not None

    for t in range(3):
        unit.utility_flow_natural_gas[t].fix(1.0)
        unit.utility_flow_propane[t].fix(1.0)
        unit.flow_in_digester_gas[t].fix(2.0)
        assert pyo.value(
            unit.blend_ratio_digester_gas_natural_gas_constraint[t].body
        ) == pytest.approx(0.0, abs=1e-9)
        assert pyo.value(
            unit.blend_ratio_digester_gas_propane_constraint[t].body
        ) == pytest.approx(-2.0, abs=1e-9)


@pytest.mark.unit
def test_combustor_blend_ratio_constraint_body_is_zero_when_flows_match():
    """The blend ratio constraint body is zero when flows satisfy the ratio."""
    _, unit = _combustor(
        3,
        inlet_names=("digester_gas", "natural_gas"),
        blend_ratio={"digester_gas": "natural_gas", "ratio": 1.6},
    )
    constraint = unit.find_component("blend_ratio_digester_gas_natural_gas_constraint")
    for t in range(3):
        unit.flow_in_digester_gas[t].fix(3.2)
        unit.flow_in_natural_gas[t].fix(2.0)
        assert pyo.value(constraint[t].body) == pytest.approx(0.0, abs=1e-9)

        unit.flow_in_digester_gas[t].fix(2.0)
        assert pyo.value(constraint[t].body) == pytest.approx(-1.2, abs=1e-9)


@pytest.mark.unit
@pytest.mark.parametrize(
    "blend_ratio",
    [
        {"digester_gas": "natural_gas"},
        {"digester_gas": "natural_gas", "ratio": -1.0},
        {"digester_gas": "natural_gas", "ratio": 0.0},
        {"digester_gas": "natural_gas", "ratio": "bad"},
        {"digester_gas": "natural_gas", "extra": 1.0, "ratio": 1.6},
        [],
        [{"digester_gas": "natural_gas"}, "bad"],
        [{"digester_gas": "natural_gas", "ratio": -1.0}],
        [
            {"digester_gas": "natural_gas", "ratio": 1.0},
            {"digester_gas": "natural_gas", "ratio": 2.0},
        ],
    ],
)
def test_combustor_rejects_bad_blend_ratio(blend_ratio):
    """Invalid blend_ratio mappings raise ValueError from the domain."""
    with pytest.raises(ValueError, match="blend_ratio"):
        _combustor(
            3,
            inlet_names=("digester_gas", "natural_gas"),
            blend_ratio=blend_ratio,
        )


@pytest.mark.unit
def test_combustor_rejects_blend_ratio_with_unknown_fuel():
    """blend_ratio naming an unknown inlet raises FlexConfigError."""
    with pytest.raises(FlexConfigError) as excinfo:
        _combustor(
            3,
            inlet_names=("digester_gas", "natural_gas"),
            blend_ratio={"unknown": "natural_gas", "ratio": 1.0},
        )
    assert excinfo.value.field == "blend_ratio"


@pytest.mark.unit
def test_combustor_is_in_the_unit_model_registry():
    """``Combustor`` is reachable from ``flexops`` and its unit-model registry."""
    import flexops
    from flexops import unit_models

    assert "Combustor" in unit_models.__all__
    assert flexops.Combustor is Combustor


@pytest.mark.component
@pytest.mark.needs_highs
def test_combustor_exports_net_against_plant_load():
    """A combustor's export nets against a plant's load rather than adding to it."""
    from flexcore.exceptions import FlexSolverError
    from flexcore.solvers import get_solver
    from flexops import PlantBlock, SimpleAqueousFlow
    from flexops.unit_models import ConstantEnergyIntensityModel

    m = dummy_gas_time_block(3)
    m.aqueous_properties = SimpleAqueousFlow()
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.combustor = Combustor(property_package=m.properties, inlet_names=("fuel",))
    m.plant.load = ConstantEnergyIntensityModel(property_package=m.aqueous_properties)
    m.plant._build_aggregates()

    for t in m.time_block.time_index:
        m.plant.combustor.inlet_fuel_state.flow_vol_phase[t, "Vap"].fix(1.0)
        m.plant.combustor.inlet_fuel_state.pressure[t].fix(101325.0)
        m.plant.combustor.inlet_fuel_state.temperature[t].fix(300.0)
        m.plant.load.inlet_state.flow_vol_phase[t, "Liq"].fix(1.0)

    try:
        solver = get_solver(model=m)
    except FlexSolverError as exc:
        pytest.skip(f"flexcore.solvers.get_solver not available: {exc}")
    results = solver.solve(m)
    pyo.assert_optimal_termination(results)

    for t in m.time_block.time_index:
        assert pyo.value(m.plant.total_electrical_power[t]) == pytest.approx(
            pyo.value(m.plant.load.power_electrical[t])
            + pyo.value(m.plant.combustor.power_electrical[t]),
            rel=1e-6,
        )
        assert pyo.value(m.plant.combustor.power_electrical[t]) < 0.0
