"""Digestor(OpsBlockData): multi-feed anaerobic digester with biogas and optional
sludge outlets."""

import logging

import pyomo.environ as pyo
import pytest
from idaes.core import declare_process_block_class
from pyomo.network import Port

from flexcore import nomenclature as nm
from flexcore.exceptions import FlexConfigError
from flexcore.logger import CONFIGURATION_SIMPLIFICATIONS, DedupHandler
from flexops.properties.simple_aqueous import SimpleAqueousFlow
from flexops.properties.simple_gas import SimpleGasFlow, SimpleGasFlowData
from flexops.testing import (
    UnitModelTestHarness,
    dummy_gas_time_block,
    dummy_time_block,
)
from flexops.testing.harness import _fix_registered_inputs
from flexops.unit_models.watertreatment import Digestor


@pytest.fixture(autouse=True)
def _reset_digestor_logger_dedup():
    logger = logging.getLogger("flexops.unit_models.watertreatment.digestor")
    for handler in logger.handlers:
        if isinstance(handler, DedupHandler):
            handler._deque.clear()
            handler._map.clear()
            handler.dedup_enabled[logging.WARNING] = False
            handler.dedup_enabled[CONFIGURATION_SIMPLIFICATIONS] = False
    yield
    for handler in logger.handlers:
        if isinstance(handler, DedupHandler):
            handler._deque.clear()
            handler._map.clear()
            handler.dedup_enabled[logging.WARNING] = False
            handler.dedup_enabled[CONFIGURATION_SIMPLIFICATIONS] = False


def _digestor_aqueous(n: int = 3, **kwargs):
    """Build a Digestor on an ``n``-point aqueous ``dummy_time_block``."""
    m = dummy_time_block(n)
    inlet_packages = kwargs.pop("inlet_packages", {"feed": m.properties})
    if "biogas_property_package" not in kwargs:
        m._biogas_pkg = SimpleGasFlow()
        kwargs["biogas_property_package"] = m._biogas_pkg
    if "sludge_property_package" not in kwargs:
        m._sludge_pkg = SimpleAqueousFlow()
        kwargs["sludge_property_package"] = m._sludge_pkg
    m.unit = Digestor(inlet_packages=inlet_packages, **kwargs)
    return m, m.unit


def _digestor_multi_feed(n: int = 3, **kwargs):
    """Build a Digestor with two aqueous inlets on an ``n``-point TimeBlock."""
    m = dummy_time_block(n)
    m.properties2 = SimpleAqueousFlow()
    inlet_packages = kwargs.pop(
        "inlet_packages", {"sludge": m.properties, "recycle": m.properties2}
    )
    if "biogas_property_package" not in kwargs:
        m._biogas_pkg = SimpleGasFlow()
        kwargs["biogas_property_package"] = m._biogas_pkg
    if "sludge_property_package" not in kwargs:
        m._sludge_pkg = SimpleAqueousFlow()
        kwargs["sludge_property_package"] = m._sludge_pkg
    m.unit = Digestor(
        inlet_packages=inlet_packages,
        **kwargs,
    )
    return m, m.unit


def _digestor_gas_biogas(n: int = 3, **kwargs):
    """Build a Digestor with a gas biogas outlet on an ``n``-point TimeBlock."""
    m = dummy_time_block(n)
    inlet_packages = kwargs.pop("inlet_packages", {"feed": m.properties})
    if "biogas_property_package" not in kwargs:
        m._biogas_pkg = SimpleGasFlow()
        kwargs["biogas_property_package"] = m._biogas_pkg
    if "sludge_property_package" not in kwargs:
        m._sludge_pkg = SimpleAqueousFlow()
        kwargs["sludge_property_package"] = m._sludge_pkg
    m.unit = Digestor(
        inlet_packages=inlet_packages,
        **kwargs,
    )
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


# -- UnitModelTestHarness classes --------------------------------------------


class TestDigestorSingleAqueousFeed(UnitModelTestHarness):
    """One aqueous feed, biogas_fraction=0.08.

    Inlet flow at its construction-time initial value of 1.0 m^3/hr, so
    biogas_volume[t] == 0.08 * 1.0 == 0.08 m^3/hr.
    """

    expected_dof = 0
    expected_solution = {"biogas_volume[0]": 0.08}

    def configure(self):
        return _digestor_aqueous(3)


class TestDigestorTwoAqueousFeeds(UnitModelTestHarness):
    """Two aqueous feeds, default biogas_fraction=0.08.

    Each inlet at 1.0 m^3/hr, total 2.0, so biogas_volume[t] == 0.16.
    """

    expected_dof = 0
    expected_solution = {"biogas_volume[0]": 0.16}

    def configure(self):
        return _digestor_multi_feed(3)


class TestDigestorNoSludgeOutlet(UnitModelTestHarness):
    """Single feed, no sludge outlet: biogas_volume == 0.08 * 1.0."""

    expected_dof = 0
    expected_solution = {"biogas_volume[0]": 0.08}

    def configure(self):
        return _digestor_aqueous(3, has_sludge_outlet=False)


# -- topology and balance ----------------------------------------------------


@pytest.mark.unit
def test_digestor_builds_one_port_per_inlet_plus_biogas_and_sludge():
    """Ports are named ``inlet_<name>`` per ``inlet_packages``, plus outlets."""
    m = dummy_time_block(3)
    m._inlet_pkg = SimpleAqueousFlow()
    _, unit = _digestor_aqueous(3, inlet_packages={"sludge": m._inlet_pkg})

    ports = {p.local_name for p in unit.component_objects(Port, descend_into=False)}
    assert "inlet_sludge" in ports
    assert "outlet_biogas" in ports
    assert "outlet_sludge" in ports
    for port in ("inlet_sludge", "outlet_biogas", "outlet_sludge"):
        assert len(list(getattr(unit, port).values())) > 0
    assert unit.find_component("inlet") is None


@pytest.mark.unit
def test_digestor_no_sludge_outlet_builds_only_biogas():
    """When ``has_sludge_outlet`` is False, only the biogas outlet is built."""
    _, unit = _digestor_aqueous(3, has_sludge_outlet=False)

    ports = {p.local_name for p in unit.component_objects(Port, descend_into=False)}
    assert "inlet_feed" in ports
    assert "outlet_biogas" in ports
    assert "outlet_sludge" not in ports
    assert unit.find_component("outlet_sludge_state") is None
    assert unit.find_component("sludge_volume") is None
    assert unit.find_component("sludge_mass_eq") is None


@pytest.mark.unit
def test_digestor_mass_balance_body_sludge():
    """sludge_volume[t] == (total_inlet_mass - biogas_mass) / sludge_density."""
    m = dummy_time_block(3)
    m.properties2 = SimpleAqueousFlow()
    _, unit = _digestor_multi_feed(
        3, inlet_packages={"a": m.properties, "b": m.properties2}
    )
    total_vol = 1.0 + 2.0
    total_mass = total_vol * 1000.0
    biogas_mass = 0.293
    for t in range(3):
        _fix(unit, "flow_in_a", t, 1.0)
        _fix(unit, "flow_in_b", t, 2.0)
        _fix(unit, "biogas_volume", t, 0.24)
        unit.outlet_biogas_state.flow_mass_phase[t, "Vap"].fix(biogas_mass)
        desired_sludge_vol = (total_mass - biogas_mass) / 1000.0
        unit.outlet_sludge_state.flow_vol_phase[t, "Liq"].fix(desired_sludge_vol)
        assert pyo.value(unit.sludge_mass_eq[t].body) == pytest.approx(0.0, abs=1e-9)
        unit.outlet_sludge_state.flow_vol_phase[t, "Liq"].fix(
            desired_sludge_vol + 0.001
        )
        assert pyo.value(unit.sludge_mass_eq[t].body) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.unit
def test_digestor_biogas_relation_body():
    """biogas_volume[t] == biogas_fraction * total_inlet_volume[t]."""
    _, unit = _digestor_aqueous(3)
    total = 1.0 + 2.0 + 3.0
    biogas = 0.08 * total
    for t in range(3):
        _fix(unit, "flow_in_feed", t, total)
        _fix(unit, "biogas_volume", t, biogas)
        assert pyo.value(unit.biogas_relation[t].body) == pytest.approx(0.0, abs=1e-9)
        unit.biogas_volume[t].fix(biogas + 1.0)
        assert pyo.value(unit.biogas_relation[t].body) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.unit
def test_digestor_no_sludge_eq_when_has_sludge_outlet_is_false():
    """No sludge_mass_eq is built when the sludge outlet is absent."""
    _, unit = _digestor_aqueous(3, has_sludge_outlet=False)
    assert unit.find_component("sludge_mass_eq") is None
    assert unit.find_component("sludge_volume") is None


@pytest.mark.unit
@pytest.mark.parametrize("n_inlets", [1, 2, 5])
def test_digestor_arbitrary_inlet_count_is_units_consistent(n_inlets):
    """Any inlet count builds and is dimensionally consistent."""
    from pyomo.util.check_units import assert_units_consistent

    m = dummy_time_block(3)
    names = {f"s{i}": m.properties for i in range(n_inlets)}
    _, unit = _digestor_aqueous(3, inlet_packages=names)
    assert_units_consistent(unit)


# -- intensive states --------------------------------------------------------


@pytest.mark.unit
def test_digestor_passes_reference_inlet_states_through_to_biogas():
    """Reactor T/P are tied to the biogas outlet's T/P, not the inlet's."""
    m = dummy_time_block(3)
    m.inlet_props = SimpleGasFlow()
    _, unit = _digestor_gas_biogas(3, inlet_packages={"feed": m.inlet_props})
    assert unit.find_component("reactor_pressure_eq") is not None
    assert unit.find_component("reactor_temperature_eq") is not None
    for t in range(3):
        unit.reactor_pressure[t].fix(101325.0)
        unit.outlet_biogas_state.pressure[t].fix(101325.0)
        assert pyo.value(unit.reactor_pressure_eq[t].body) == pytest.approx(
            0.0, abs=1e-9
        )


@pytest.mark.unit
def test_digestor_passes_reference_inlet_states_through_to_sludge():
    """Reactor T/P are tied to the sludge outlet's T/P when available."""
    m = dummy_time_block(3)
    m.inlet_props = SimpleGasFlow()
    m.sludge_props = SimpleAqueousFlow(has_pressure=True, has_temperature=True)
    _, unit = _digestor_gas_biogas(
        3,
        inlet_packages={"feed": m.inlet_props},
        sludge_property_package=m.sludge_props,
    )
    assert unit.find_component("sludge_pressure_eq") is not None
    assert unit.find_component("sludge_temperature_eq") is not None


@pytest.mark.unit
def test_digestor_registers_reference_inlet_intensive_states_as_inputs():
    """All inlet state vars registered as inputs;
    reactor T/P and biogas volume as outputs."""
    m = dummy_time_block(3)
    m.props_b = SimpleGasFlow()
    _, unit = _digestor_multi_feed(
        3, inlet_packages={"a": m.properties, "b": m.props_b}
    )
    inputs = _registered(unit, "input")
    assert "inlet_a_state.flow_vol_phase" in inputs
    assert "inlet_b_state.flow_vol_phase" in inputs
    assert "inlet_b_state.pressure" in inputs
    assert "inlet_b_state.temperature" in inputs
    assert "inlet_a_state.pressure" not in inputs
    assert "inlet_a_state.temperature" not in inputs

    outputs = _registered(unit, "output")
    assert "unit.biogas_volume" in outputs
    assert "unit.reactor_temperature" in outputs
    assert "unit.reactor_pressure" in outputs
    assert "outlet_biogas_state.flow_vol_phase" not in outputs
    assert "outlet_biogas_state.pressure" not in outputs


# -- biogas fraction parameter -----------------------------------------------


@pytest.mark.unit
def test_digestor_biogas_fraction_is_a_fixed_regressable_parameter():
    """biogas_fraction is a fixed, registered, bounded scalar Var."""
    _, unit = _digestor_aqueous(3, biogas_fraction=0.1)
    assert unit.biogas_fraction.fixed
    assert pyo.value(unit.biogas_fraction) == pytest.approx(0.1)
    assert unit.biogas_fraction.bounds == (0.0, 1.0)
    registered = {rec.name: rec.regressable for rec in unit._io_registry.parameters}
    assert registered["biogas_fraction"] is True


# -- config rejection --------------------------------------------------------


@pytest.mark.unit
def test_digestor_rejects_empty_inlet_packages():
    """An empty ``inlet_packages`` dict raises, naming the field."""
    with pytest.raises(FlexConfigError) as excinfo:
        _digestor_aqueous(3, inlet_packages={})
    assert excinfo.value.field == "inlet_packages"


@pytest.mark.unit
def test_digestor_rejects_non_string_inlet_name():
    """Non-string keys in ``inlet_packages`` raise."""
    m = dummy_time_block(3)
    m.props_1 = SimpleAqueousFlow()
    with pytest.raises(FlexConfigError) as excinfo:
        _digestor_aqueous(3, inlet_packages={1: m.props_1})
    assert excinfo.value.field == "inlet_packages"


@pytest.mark.unit
def test_digestor_rejects_multiphase_inlet_package():
    """An inlet property package with more than one phase is rejected."""
    m = dummy_gas_time_block(3)

    @declare_process_block_class("_TwoPhaseDigestorFlow")
    class _TwoPhaseDigestorFlowData(SimpleGasFlowData):
        def build(self) -> None:
            super().build()
            from idaes.core import LiquidPhase

            self.Liq = LiquidPhase()

    _TwoPhaseDigestorFlow = globals()["_TwoPhaseDigestorFlow"]
    m.two_phase = _TwoPhaseDigestorFlow()
    with pytest.raises(FlexConfigError) as excinfo:
        m.unit = Digestor(inlet_packages={"feed": m.two_phase})
    assert excinfo.value.field == "inlet_packages"


@pytest.mark.unit
def test_digestor_rejects_biogas_fraction_out_of_range():
    """biogas_fraction outside (0, 1) raises."""
    with pytest.raises(ValueError, match="biogas_fraction"):
        _digestor_aqueous(3, biogas_fraction=1.5)


@pytest.mark.unit
def test_digestor_rejects_biogas_fraction_zero():
    """biogas_fraction=0 is rejected (no biogas produced)."""
    with pytest.raises(ValueError, match="biogas_fraction"):
        _digestor_aqueous(3, biogas_fraction=0.0)


@pytest.mark.unit
def test_digestor_accepts_biogas_fraction_as_string_coercion():
    """biogas_fraction is coerced from a plain string value."""
    _, unit = _digestor_aqueous(3, biogas_fraction="0.05")
    assert pyo.value(unit.biogas_fraction) == pytest.approx(0.05)


# -- registration and naming ------------------------------------------------


@pytest.mark.unit
def test_digestor_declares_no_power():
    """A digestor applies conservation only -- it draws and exports no energy."""
    _, unit = _digestor_aqueous(3)
    assert unit._io_registry.power == []
    assert unit.find_component("power_electrical") is None
    assert unit.find_component("power_thermal") is None


@pytest.mark.unit
def test_digestor_biogas_relation_constraint_is_named():
    """The biogas constraint is named biogas_relation (the swap contract)."""
    _, unit = _digestor_aqueous(3)
    assert unit.find_component("biogas_relation") is not None
    assert unit.find_component("biogas_eq") is None


@pytest.mark.unit
def test_digestor_closes_degrees_of_freedom():
    """Fixing every registered input drives DoF to 0."""
    from idaes.core.util.model_statistics import degrees_of_freedom

    for n_inlets in (1, 2):
        m = dummy_time_block(3)
        names = {f"s{i}": m.properties for i in range(n_inlets)}
        m._biogas_pkg = SimpleGasFlow()
        m._sludge_pkg = SimpleAqueousFlow()
        m.unit = Digestor(
            inlet_packages=names,
            biogas_property_package=m._biogas_pkg,
            sludge_property_package=m._sludge_pkg,
        )
        _fix_registered_inputs(m.unit)
        assert degrees_of_freedom(m) == 0, f"{n_inlets} inlets"


@pytest.mark.unit
def test_digestor_is_in_the_unit_model_registry():
    """Digestor is reachable from flexops and its unit-model registry."""
    import flexops
    from flexops import unit_models

    assert "Digestor" in unit_models.__all__
    assert flexops.Digestor is Digestor


# -- surrogate swap ----------------------------------------------------------


@pytest.mark.unit
def test_digestor_swaps_biogas_relation_with_linear_surrogate():
    """A linear surrogate deactivates biogas_relation and adds
    biogas_relation_fitted."""
    from flexcore.config.schema import SurrogateSpec
    from flexops.surrogates import surrogate_from_spec

    _, unit = _digestor_aqueous(3)
    spec = SurrogateSpec(
        surrogate_type="multilinear",
        data={
            "input_variables": {"flow_in_feed": "m^3/hr"},
            "output_variables": {"biogas_volume": "m^3/hr"},
            "coefficients": {"flow_in_feed": 0.08, "intercept": 0.0},
        },
    )
    unit.swap_relation("biogas_relation", surrogate_from_spec(spec))

    assert unit.find_component("biogas_relation") is not None
    assert not unit.biogas_relation.active
    assert unit.surrogate_biogas.fitted is not None
    assert unit.surrogate_biogas.fitted.active


@pytest.mark.unit
def test_digestor_rejects_surrogate_naming_unknown_variable():
    """A surrogate that names a non-existent input variable is rejected."""
    from flexcore.config.schema import SurrogateSpec
    from flexops.surrogates import surrogate_from_spec

    _, unit = _digestor_aqueous(3)
    spec = SurrogateSpec(
        surrogate_type="multilinear",
        data={
            "input_variables": {"nonexistent_var": "m^3/hr"},
            "output_variables": {"biogas_volume": "m^3/hr"},
            "coefficients": {"nonexistent_var": 1.0, "intercept": 0.0},
        },
    )
    with pytest.raises(FlexConfigError) as excinfo:
        unit.swap_relation("biogas_relation", surrogate_from_spec(spec))
    assert excinfo.value.field == "input_variables"


@pytest.mark.unit
def test_digestor_constant_intensity_does_not_swap():
    """Without a surrogate config, biogas_relation is untouched."""
    _, unit = _digestor_aqueous(3)
    assert unit.biogas_relation.active
    assert unit.find_component("biogas_relation_fitted") is None


# -- integration -------------------------------------------------------------


@pytest.mark.unit
def test_digestor_energy_naming():
    """No power_electrical/power_thermal on a digestor."""
    m, unit = _digestor_aqueous(3)
    assert not hasattr(unit, nm.POWER_ELECTRICAL)
    assert not hasattr(unit, nm.POWER_THERMAL)
    for bad_name in ("power", "energy", "work"):
        assert not hasattr(unit, bad_name)


@pytest.mark.unit
def test_digestor_units_consistent():
    """The digestor's variables and constraints are dimensionally consistent."""
    from pyomo.util.check_units import assert_units_consistent

    _, unit = _digestor_aqueous(3)
    assert_units_consistent(unit)


@pytest.mark.unit
def test_digestor_io_registration():
    """Every registered IO variable exists, carries units, and has a doc=."""
    _, unit = _digestor_aqueous(3)
    for record in unit._io_registry.io_variables:
        var = record.var
        assert var.parent_block() is not None
        assert record.units
        assert record.time_indexed
        assert var.doc, f"registered IO variable {record.name!r} has no doc="
