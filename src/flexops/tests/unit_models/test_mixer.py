"""Mixer(OpsBlockData): N named inlets joined into one outlet (§3.2, §3.4)."""

import warnings

import pyomo.environ as pyo
import pytest
from idaes.core import Component, LiquidPhase, declare_process_block_class
from pyomo.environ import units as pyunits
from pyomo.network import Port

from flexcore.exceptions import FlexConfigError
from flexops.properties.simple_aqueous import SimpleAqueousFlow, SimpleAqueousFlowData
from flexops.properties.simple_gas import SimpleGasFlowData
from flexops.testing import (
    UnitModelTestHarness,
    dummy_gas_time_block,
    dummy_time_block,
)
from flexops.testing.harness import _fix_registered_inputs
from flexops.unit_models import Mixer
from flexops.unit_models.mixer import MixerTemperatureRule


@declare_process_block_class("_TwoPhaseMixerFlow")
class _TwoPhaseMixerFlowData(SimpleGasFlowData):
    """Test-only stub: SimpleGasFlow plus a second phase.

    Exists only to exercise Mixer's single-phase guard -- no two-phase
    property package exists in the repository.
    """

    def build(self) -> None:
        super().build()
        self.Liq = LiquidPhase()


@declare_process_block_class("_MultiComponentMixerFlow")
class _MultiComponentMixerFlowData(SimpleAqueousFlowData):
    """Test-only stub: SimpleAqueousFlow plus a second component.

    Exists only to exercise Mixer's multi-component warning -- no
    multi-component property package exists in the repository. Neither shipped
    state block indexes anything by ``component_list``, so the extra component
    changes nothing but the parameter block's own metadata.
    """

    def build(self) -> None:
        super().build()
        self.TDS = Component()


# declare_process_block_class injects the constructible wrapper into this
# module's namespace at runtime; bind it explicitly (as simple_gas.py does)
# so static tools resolve the forward reference used below.
_TwoPhaseMixerFlow = globals()["_TwoPhaseMixerFlow"]
_MultiComponentMixerFlow = globals()["_MultiComponentMixerFlow"]


def _mixer(n: int = 3, **kwargs):
    """Build a Mixer on an ``n``-point aqueous ``dummy_time_block``."""
    m = dummy_time_block(n)
    m.unit = Mixer(property_package=m.properties, **kwargs)
    return m, m.unit


def _gas_mixer(n: int = 3, **kwargs):
    """Build a Mixer on an ``n``-point ``dummy_gas_time_block``."""
    m = dummy_gas_time_block(n)
    m.unit = Mixer(property_package=m.properties, **kwargs)
    return m, m.unit


def _fix(unit, name: str, t, value: float) -> None:
    """Fix a named time-indexed component on ``unit`` at time ``t``."""
    getattr(unit, name)[t].fix(value)


def _registered(unit, role: str) -> set[str]:
    """Return ``{"<state block>.<var>"}`` for IO variables in ``role``."""
    return {
        f"{rec.var.parent_block().local_name}.{rec.name}"
        for rec in unit._io_registry.io_variables
        if rec.role == role
    }


class TestMixerAqueous(UnitModelTestHarness):
    """Three aqueous inlets summed into one outlet.

    The aqueous package carries only ``flow_vol_phase``, so the mixer's whole
    model is the conservation equality. Every inlet flow sits at its
    construction-time initial value of 1.0 m^3/hr, so
    ``flow_out[t] == 1.0 + 1.0 + 1.0 == 3.0`` m^3/hr.
    """

    expected_dof = 0
    expected_solution = {"flow_out[0]": 3.0}

    def configure(self):
        return _mixer(3, inlet_names=("a", "b", "c"))


class TestMixerGasEqualTemperature(UnitModelTestHarness):
    """Two gas inlets under the default ``EQUAL`` temperature rule.

    The gas package always carries pressure and temperature, so the
    non-reference inlet's are tied to the reference inlet's and both pass
    through to the outlet -- the model stays linear.
    """

    expected_dof = 0
    expected_solution = {"flow_out[0]": 2.0}

    def configure(self):
        return _gas_mixer(3, inlet_names=("a", "b"))


class TestMixerGasFlowWeightedTemperature(UnitModelTestHarness):
    """Two gas inlets whose outlet temperature is the volume-weighted blend.

    Both inlet flows and both inlet temperatures are independent inputs here,
    and ``outlet_temperature_eq`` is bilinear -- so ``test_solve`` needs IPOPT
    rather than HiGHS and skips when it is not installed. The relationship
    itself is checked at ``unit`` tier by
    :func:`test_mixer_flow_weighted_outlet_temperature_body`.
    """

    expected_dof = 0

    def configure(self):
        return _gas_mixer(
            3,
            inlet_names=("a", "b"),
            temperature_mixing=MixerTemperatureRule.FLOW_WEIGHTED,
        )


# -- topology and balance ---------------------------------------------------


@pytest.mark.unit
def test_mixer_builds_one_port_per_inlet_name():
    """Ports are named ``inlet_<name>`` per ``inlet_names``, plus one ``outlet``."""
    _, unit = _mixer(3, inlet_names=("sludge", "recycle", "makeup"))

    ports = {p.local_name for p in unit.component_objects(Port, descend_into=False)}
    assert ports == {"inlet_sludge", "inlet_recycle", "inlet_makeup", "outlet"}
    for port in ports:
        assert len(list(getattr(unit, port).values())) > 0
    assert unit.find_component("inlet") is None


@pytest.mark.unit
def test_mixer_mass_balance_body():
    """The outlet flow is the plain sum of the inlet flows -- no scaling term."""
    _, unit = _mixer(3, inlet_names=("a", "b", "c"))
    inlet_sum = 1.0 + 2.0 + 3.0
    for t in range(3):
        _fix(unit, "flow_in_a", t, 1.0)
        _fix(unit, "flow_in_b", t, 2.0)
        _fix(unit, "flow_in_c", t, 3.0)
        _fix(unit, "flow_out", t, inlet_sum)
        assert pyo.value(unit.mixing_mass_balance[t].body) == pytest.approx(
            0.0, abs=1e-9
        )
        unit.flow_out[t].fix(inlet_sum + 1.0)
        assert pyo.value(unit.mixing_mass_balance[t].body) == pytest.approx(
            1.0, abs=1e-9
        )


@pytest.mark.unit
@pytest.mark.parametrize("n_inlets", [1, 2, 5])
def test_mixer_arbitrary_inlet_count_conserves_volume(n_inlets):
    """Any inlet count builds, is units-consistent, and conserves volume."""
    from pyomo.util.check_units import assert_units_consistent

    names = tuple(f"s{i}" for i in range(n_inlets))
    _, unit = _mixer(3, inlet_names=names)
    assert_units_consistent(unit)
    for t in range(3):
        for i, name in enumerate(names):
            _fix(unit, f"flow_in_{name}", t, float(i + 1))
        _fix(unit, "flow_out", t, float(n_inlets * (n_inlets + 1) // 2))
        assert pyo.value(unit.mixing_mass_balance[t].body) == pytest.approx(
            0.0, abs=1e-9
        )


@pytest.mark.unit
def test_mixer_declares_no_power():
    """A mixer applies conservation only -- it draws and exports no energy."""
    _, unit = _mixer(3, inlet_names=("a", "b"))
    assert unit._io_registry.power == []
    assert unit.find_component("power_electrical") is None
    assert unit.find_component("power_thermal") is None
    assert unit._io_registry.fuel == []


# -- intensive states -------------------------------------------------------


@pytest.mark.unit
def test_mixer_equalizes_inlet_intensive_states():
    """Non-reference inlets' pressure equals the reference inlet's."""
    _, unit = _gas_mixer(3, inlet_names=("a", "b"))
    for t in range(3):
        unit.inlet_a_state.pressure[t].fix(101325.0)
        unit.inlet_b_state.pressure[t].fix(101325.0)
        assert pyo.value(
            unit.inlet_state_equality_pressure[t, "b"].body
        ) == pytest.approx(0.0, abs=1e-9)

        unit.inlet_b_state.pressure[t].fix(150000.0)
        assert pyo.value(
            unit.inlet_state_equality_pressure[t, "b"].body
        ) == pytest.approx(150000.0 - 101325.0, abs=1e-9)


@pytest.mark.unit
def test_mixer_single_inlet_builds_no_state_equalities():
    """With one inlet there is no non-reference stream to tie."""
    _, unit = _gas_mixer(3, inlet_names=("only",))
    for state_var in ("pressure", "temperature"):
        assert unit.find_component(f"inlet_state_equality_{state_var}") is None


@pytest.mark.unit
def test_mixer_equal_rule_ties_and_passes_temperature_through():
    """Under ``EQUAL``, temperature is tied across inlets and passed through."""
    _, unit = _gas_mixer(3, inlet_names=("a", "b"))
    assert unit.find_component("inlet_state_equality_temperature") is not None
    assert unit.find_component("pass_through_temperature_eq") is not None
    assert unit.find_component("outlet_temperature_eq") is None


@pytest.mark.unit
def test_mixer_flow_weighted_rule_replaces_the_temperature_pass_through():
    """Under ``FLOW_WEIGHTED``, temperature is neither tied nor passed through."""
    _, unit = _gas_mixer(
        3,
        inlet_names=("a", "b"),
        temperature_mixing=MixerTemperatureRule.FLOW_WEIGHTED,
    )
    assert unit.find_component("inlet_state_equality_temperature") is None
    assert unit.find_component("pass_through_temperature_eq") is None
    assert unit.find_component("outlet_temperature_eq") is not None
    # Pressure is still tied and passed through under either rule.
    assert unit.find_component("inlet_state_equality_pressure") is not None
    assert unit.find_component("pass_through_pressure_eq") is not None


@pytest.mark.unit
def test_mixer_flow_weighted_outlet_temperature_body():
    """The outlet temperature is the volume-weighted average of the inlets'."""
    _, unit = _gas_mixer(
        3,
        inlet_names=("a", "b"),
        temperature_mixing=MixerTemperatureRule.FLOW_WEIGHTED,
    )
    for t in range(3):
        _fix(unit, "flow_in_a", t, 2.0)
        _fix(unit, "flow_in_b", t, 3.0)
        _fix(unit, "flow_out", t, 5.0)
        unit.inlet_a_state.temperature[t].fix(300.0)
        unit.inlet_b_state.temperature[t].fix(400.0)

        blended = (2.0 * 300.0 + 3.0 * 400.0) / 5.0
        unit.outlet_state.temperature[t].fix(blended)
        assert pyo.value(unit.outlet_temperature_eq[t].body) == pytest.approx(
            0.0, abs=1e-9
        )

        unit.outlet_state.temperature[t].fix(blended + 10.0)
        assert pyo.value(unit.outlet_temperature_eq[t].body) == pytest.approx(
            10.0 * 5.0, abs=1e-9
        )


@pytest.mark.unit
def test_mixer_passes_pressure_through_from_the_reference_inlet():
    """``pass_through_pressure_eq`` ties the outlet to the first-named inlet."""
    _, unit = _gas_mixer(3, inlet_names=("a", "b"))
    for t in range(3):
        unit.inlet_a_state.pressure[t].fix(101325.0)
        unit.outlet_state.pressure[t].fix(101325.0)
        assert pyo.value(unit.pass_through_pressure_eq[t].body) == pytest.approx(
            0.0, abs=1e-9
        )


@pytest.mark.unit
def test_mixer_equal_rule_registers_only_the_reference_inlet_intensive_states():
    """Every inlet's flow is an input; only the reference's other states are."""
    _, unit = _gas_mixer(3, inlet_names=("a", "b"))
    inputs = _registered(unit, "input")
    assert "inlet_a_state.flow_vol_phase" in inputs
    assert "inlet_b_state.flow_vol_phase" in inputs
    for state_var in ("pressure", "temperature"):
        assert f"inlet_a_state.{state_var}" in inputs
        assert f"inlet_b_state.{state_var}" not in inputs

    outputs = _registered(unit, "output")
    for state_var in ("flow_vol_phase", "pressure", "temperature"):
        assert f"outlet_state.{state_var}" in outputs


@pytest.mark.unit
def test_mixer_flow_weighted_rule_registers_every_inlet_temperature():
    """Untied inlet temperatures are independent inputs, or DoF would not close."""
    _, unit = _gas_mixer(
        3,
        inlet_names=("a", "b"),
        temperature_mixing=MixerTemperatureRule.FLOW_WEIGHTED,
    )
    inputs = _registered(unit, "input")
    assert "inlet_a_state.temperature" in inputs
    assert "inlet_b_state.temperature" in inputs
    # Pressure is still tied, so only the reference inlet's is an input.
    assert "inlet_a_state.pressure" in inputs
    assert "inlet_b_state.pressure" not in inputs


# -- config rejection -------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("inlet_names", [(), ("a", "a"), ("a", ""), ("a", 1)])
def test_mixer_rejects_bad_inlet_names(inlet_names):
    """Empty, duplicated, or non-string ``inlet_names`` raise, naming the field."""
    with pytest.raises(FlexConfigError) as excinfo:
        _mixer(3, inlet_names=inlet_names)
    assert excinfo.value.field == "inlet_names"


@pytest.mark.unit
def test_mixer_requires_a_single_phase_property_package():
    """A property package with more than one phase is rejected."""
    m = dummy_gas_time_block(3)
    m.two_phase_properties = _TwoPhaseMixerFlow()
    with pytest.raises(FlexConfigError) as excinfo:
        m.unit = Mixer(property_package=m.two_phase_properties)
    assert excinfo.value.field == "property_package"


@pytest.mark.unit
def test_mixer_rejects_flow_weighted_without_a_temperature_state():
    """``FLOW_WEIGHTED`` has nothing to blend on a package with no temperature."""
    with pytest.raises(FlexConfigError) as excinfo:
        _mixer(
            3,
            inlet_names=("a", "b"),
            temperature_mixing=MixerTemperatureRule.FLOW_WEIGHTED,
        )
    assert excinfo.value.field == "temperature_mixing"


@pytest.mark.unit
def test_mixer_rejects_an_unknown_temperature_rule():
    """An unrecognized ``temperature_mixing`` value is rejected by the domain."""
    with pytest.raises(ValueError, match="temperature_mixing"):
        _mixer(3, temperature_mixing="enthalpy")


@pytest.mark.unit
def test_mixer_accepts_the_temperature_rule_as_a_plain_string():
    """The config vocabulary is coerced, so a persisted string value works."""
    _, unit = _gas_mixer(3, inlet_names=("a", "b"), temperature_mixing="flow_weighted")
    assert unit.config.temperature_mixing is MixerTemperatureRule.FLOW_WEIGHTED


# -- physical-simplification warnings ---------------------------------------


def _warning_messages(build) -> list[str]:
    """Return every warning message ``build()`` emits, as strings.

    Recorded rather than turned into errors so an unrelated Pyomo or IDAES
    warning raised during construction cannot masquerade as one of the mixer's.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build()
    return [str(w.message) for w in caught]


@pytest.mark.unit
def test_mixer_warns_on_a_multi_component_property_package():
    """Volumetric mixing tracks no composition, so extra components are flagged."""
    m = dummy_time_block(3)
    m.multi_component_properties = _MultiComponentMixerFlow()
    with pytest.warns(UserWarning, match="multiple components"):
        m.unit = Mixer(
            property_package=m.multi_component_properties, inlet_names=("a", "b")
        )


@pytest.mark.unit
def test_mixer_multi_component_warning_names_the_components():
    """The message lists what the package carries, so the gap is actionable."""
    m = dummy_time_block(3)
    m.multi_component_properties = _MultiComponentMixerFlow()
    with pytest.warns(UserWarning) as caught:
        m.unit = Mixer(
            property_package=m.multi_component_properties, inlet_names=("a", "b")
        )
    text = " ".join(str(w.message) for w in caught)
    assert "H2O" in text and "TDS" in text


@pytest.mark.unit
def test_mixer_does_not_warn_on_a_single_component_property_package():
    """Both shipped packages carry one component -- the common case stays quiet."""
    messages = _warning_messages(lambda: _mixer(3, inlet_names=("a", "b")))
    assert not [msg for msg in messages if "multiple components" in msg]


@pytest.mark.unit
def test_mixer_warns_when_a_gas_is_blended_at_unequal_temperatures():
    """Vapour inlets at different temperatures need an equation of state."""
    with pytest.warns(UserWarning, match="vapor-phase"):
        _gas_mixer(
            3,
            inlet_names=("a", "b"),
            temperature_mixing=MixerTemperatureRule.FLOW_WEIGHTED,
        )


@pytest.mark.unit
def test_mixer_does_not_warn_for_a_gas_under_the_equal_temperature_rule():
    """``EQUAL`` ties every inlet temperature, so volumes really are additive."""
    messages = _warning_messages(lambda: _gas_mixer(3, inlet_names=("a", "b")))
    assert not [msg for msg in messages if "vapor-phase" in msg]


@pytest.mark.unit
def test_mixer_does_not_warn_for_a_liquid_under_the_flow_weighted_rule():
    """A liquid's volume is not set by an equation of state -- nothing to flag."""

    def build():
        m = dummy_time_block(3)
        m.warm_properties = SimpleAqueousFlow(has_temperature=True)
        m.unit = Mixer(
            property_package=m.warm_properties,
            inlet_names=("a", "b"),
            temperature_mixing=MixerTemperatureRule.FLOW_WEIGHTED,
        )

    assert not [msg for msg in _warning_messages(build) if "vapor-phase" in msg]


# -- integration with the rest of the library -------------------------------


@pytest.mark.unit
def test_mixer_is_in_the_unit_model_registry():
    """``Mixer`` is reachable from ``flexops`` and its unit-model registry."""
    import flexops
    from flexops import unit_models

    assert "Mixer" in unit_models.__all__
    assert flexops.Mixer is Mixer


@pytest.mark.component
@pytest.mark.needs_highs
def test_mixer_joins_two_upstream_streams_through_arcs():
    """Two pumps feeding a mixer over arcs solve to the summed outlet flow."""
    from pyomo.network import Arc

    from flexcore.exceptions import FlexSolverError
    from flexcore.solvers import get_solver
    from flexops import PlantBlock
    from flexops.unit_models import Pump

    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.pump_a = Pump(property_package=m.properties)
    m.plant.pump_b = Pump(property_package=m.properties)
    m.plant.mixer = Mixer(property_package=m.properties, inlet_names=("a", "b"))
    m.plant.arc_a = Arc(source=m.plant.pump_a.outlet, destination=m.plant.mixer.inlet_a)
    m.plant.arc_b = Arc(source=m.plant.pump_b.outlet, destination=m.plant.mixer.inlet_b)
    m.plant._build_aggregates()
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    for t in m.time_block.time_index:
        m.plant.pump_a.inlet_state.flow_vol_phase[t, "Liq"].fix(2.0)
        m.plant.pump_b.inlet_state.flow_vol_phase[t, "Liq"].fix(3.0)

    try:
        solver = get_solver(model=m)
    except FlexSolverError as exc:
        pytest.skip(f"flexcore.solvers.get_solver not available: {exc}")
    pyo.assert_optimal_termination(solver.solve(m))

    for t in m.time_block.time_index:
        assert pyo.value(m.plant.mixer.flow_out[t]) == pytest.approx(5.0, rel=1e-6)


@pytest.mark.unit
def test_mixer_closes_degrees_of_freedom_for_any_inlet_count():
    """A mixer is well-posed at every arity once its inlets are fixed."""
    from idaes.core.util.model_statistics import degrees_of_freedom

    for n_inlets in (1, 2, 4):
        m, unit = _gas_mixer(3, inlet_names=tuple(f"s{i}" for i in range(n_inlets)))
        _fix_registered_inputs(unit)
        assert degrees_of_freedom(m) == 0, f"{n_inlets} inlets"


@pytest.mark.unit
def test_mixer_units_are_consistent_under_both_temperature_rules():
    """Both temperature rules are dimensionally exact."""
    from pyomo.util.check_units import assert_units_consistent

    for rule in MixerTemperatureRule:
        _, unit = _gas_mixer(3, inlet_names=("a", "b"), temperature_mixing=rule)
        assert_units_consistent(unit)


@pytest.mark.unit
def test_mixer_outlet_temperature_carries_kelvin():
    """The blended outlet temperature keeps the package's temperature units."""
    _, unit = _gas_mixer(
        3,
        inlet_names=("a", "b"),
        temperature_mixing=MixerTemperatureRule.FLOW_WEIGHTED,
    )
    # pyunits.get_units(...) returns a Pyomo NumericValue; comparing two with
    # == builds a relational expression rather than a bool, so units equality
    # is checked by string, matching OpsBlockData._units_str.
    assert str(pyunits.get_units(unit.outlet_state.temperature[0])) == str(pyunits.K)
