"""Splitter(OpsBlockData): one inlet fanned out to N named outlets (§3.2, §3.4)."""

import pyomo.environ as pyo
import pytest
from idaes.core import LiquidPhase, declare_process_block_class
from pyomo.network import Port

from flexcore.exceptions import FlexConfigError
from flexops.properties.simple_gas import SimpleGasFlowData
from flexops.testing import (
    UnitModelTestHarness,
    dummy_gas_time_block,
    dummy_time_block,
)
from flexops.testing.harness import _fix_registered_inputs
from flexops.unit_models import Splitter


@declare_process_block_class("_TwoPhaseSplitterFlow")
class _TwoPhaseSplitterFlowData(SimpleGasFlowData):
    """Test-only stub: SimpleGasFlow plus a second phase.

    Exists only to exercise Splitter's single-phase guard -- no two-phase
    property package exists in the repository.
    """

    def build(self) -> None:
        super().build()
        self.Liq = LiquidPhase()


# declare_process_block_class injects the constructible wrapper into this
# module's namespace at runtime; bind it explicitly (as simple_gas.py does)
# so static tools resolve the forward reference used below.
_TwoPhaseSplitterFlow = globals()["_TwoPhaseSplitterFlow"]


def _splitter(n: int = 3, **kwargs):
    """Build a Splitter on an ``n``-point aqueous ``dummy_time_block``."""
    m = dummy_time_block(n)
    m.unit = Splitter(property_package=m.properties, **kwargs)
    return m, m.unit


def _gas_splitter(n: int = 3, **kwargs):
    """Build a Splitter on an ``n``-point ``dummy_gas_time_block``."""
    m = dummy_gas_time_block(n)
    m.unit = Splitter(property_package=m.properties, **kwargs)
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


class TestSplitterAqueous(UnitModelTestHarness):
    """One aqueous inlet fanned out to two outlets.

    Conservation is the only flow constraint, so fixing the inlet leaves one
    free outlet flow per time point -- ``(2 - 1) * 3 == 3`` degrees of freedom.
    ``expected_solution`` is deliberately empty: a free split has no unique
    solution to regress against.
    """

    expected_dof = 3

    def configure(self):
        return _splitter(3, outlet_names=("a", "b"))


class TestSplitterGas(UnitModelTestHarness):
    """One gas inlet fanned out to three outlets.

    Pressure and temperature pass through to every outlet, so only the flow
    split stays free: ``(3 - 1) * 3 == 6`` degrees of freedom.
    """

    expected_dof = 6

    def configure(self):
        return _gas_splitter(3, outlet_names=("a", "b", "c"))


# -- topology and balance ---------------------------------------------------


@pytest.mark.unit
def test_splitter_builds_one_port_per_outlet_name():
    """Ports are one ``inlet`` plus ``outlet_<name>`` per ``outlet_names``."""
    _, unit = _splitter(3, outlet_names=("permeate", "brine", "recycle"))

    ports = {p.local_name for p in unit.component_objects(Port, descend_into=False)}
    assert ports == {"inlet", "outlet_permeate", "outlet_brine", "outlet_recycle"}
    for port in ports:
        assert len(list(getattr(unit, port).values())) > 0
    assert unit.find_component("outlet") is None


@pytest.mark.unit
def test_splitter_mass_balance_body():
    """The inlet flow equals the sum of the outlet flows -- nothing else."""
    _, unit = _splitter(3, outlet_names=("a", "b", "c"))
    for t in range(3):
        _fix(unit, "flow_in", t, 6.0)
        _fix(unit, "flow_out_a", t, 1.0)
        _fix(unit, "flow_out_b", t, 2.0)
        _fix(unit, "flow_out_c", t, 3.0)
        assert pyo.value(unit.split_mass_balance[t].body) == pytest.approx(
            0.0, abs=1e-9
        )
        unit.flow_out_c[t].fix(4.0)
        assert pyo.value(unit.split_mass_balance[t].body) == pytest.approx(
            -1.0, abs=1e-9
        )


@pytest.mark.unit
def test_splitter_builds_no_split_fraction_parameter():
    """The split is a decision, not a parameter -- no split fractions are built."""
    _, unit = _splitter(3, outlet_names=("a", "b"))
    assert unit.find_component("split_fraction") is None
    assert unit.find_component("split_definition") is None
    assert unit._io_registry.parameters == []


@pytest.mark.unit
@pytest.mark.parametrize("n_outlets", [2, 3, 4])
def test_splitter_leaves_the_routing_free(n_outlets):
    """Fixing the inlet leaves exactly ``(N - 1) * n_points`` free flows.

    This is the unit's defining contract: conservation alone, so the enclosing
    model's objective (or fixed outlet flows) picks the routing.
    """
    from idaes.core.util.model_statistics import degrees_of_freedom

    names = tuple(f"s{i}" for i in range(n_outlets))
    m, unit = _gas_splitter(3, outlet_names=names)
    _fix_registered_inputs(unit)
    assert degrees_of_freedom(m) == (n_outlets - 1) * m.time_block.n_points


@pytest.mark.unit
@pytest.mark.parametrize("n_outlets", [1, 2, 5])
def test_splitter_arbitrary_outlet_count_conserves_volume(n_outlets):
    """Any outlet count builds, is units-consistent, and conserves volume."""
    from pyomo.util.check_units import assert_units_consistent

    names = tuple(f"s{i}" for i in range(n_outlets))
    _, unit = _splitter(3, outlet_names=names)
    assert_units_consistent(unit)
    for t in range(3):
        for i, name in enumerate(names):
            _fix(unit, f"flow_out_{name}", t, float(i + 1))
        _fix(unit, "flow_in", t, float(n_outlets * (n_outlets + 1) // 2))
        assert pyo.value(unit.split_mass_balance[t].body) == pytest.approx(
            0.0, abs=1e-9
        )


@pytest.mark.unit
def test_splitter_declares_no_power():
    """A splitter applies conservation only -- it draws and exports no energy."""
    _, unit = _splitter(3, outlet_names=("a", "b"))
    assert unit._io_registry.power == []
    assert unit.find_component("power_electrical") is None
    assert unit.find_component("power_thermal") is None
    assert unit._io_registry.fuel == []


# -- intensive states -------------------------------------------------------


@pytest.mark.unit
def test_splitter_passes_intensive_states_to_every_outlet():
    """Each outlet gets its own uniquely named pass-through constraints."""
    _, unit = _gas_splitter(3, outlet_names=("a", "b", "c"))
    for name in ("a", "b", "c"):
        for state_var in ("pressure", "temperature"):
            constraint = unit.find_component(f"pass_through_{name}_{state_var}_eq")
            assert constraint is not None, f"outlet {name}: {state_var}"
            assert len(constraint) == 3
    # The flow basis is governed by the mass balance, never passed through.
    for name in ("a", "b", "c"):
        assert unit.find_component(f"pass_through_{name}_flow_vol_phase_eq") is None


@pytest.mark.unit
def test_splitter_pass_through_body():
    """An outlet's pressure equals the inlet's."""
    _, unit = _gas_splitter(3, outlet_names=("a", "b"))
    for t in range(3):
        unit.inlet_state.pressure[t].fix(101325.0)
        unit.outlet_a_state.pressure[t].fix(101325.0)
        assert pyo.value(unit.pass_through_a_pressure_eq[t].body) == pytest.approx(
            0.0, abs=1e-9
        )
        unit.outlet_a_state.pressure[t].fix(150000.0)
        assert pyo.value(unit.pass_through_a_pressure_eq[t].body) == pytest.approx(
            150000.0 - 101325.0, abs=1e-9
        )


@pytest.mark.unit
def test_splitter_registers_inlet_states_as_inputs_and_outlets_as_outputs():
    """The inlet's states are the boundary conditions; the outlets' are results."""
    _, unit = _gas_splitter(3, outlet_names=("a", "b"))
    inputs = _registered(unit, "input")
    for state_var in ("flow_vol_phase", "pressure", "temperature"):
        assert f"inlet_state.{state_var}" in inputs

    outputs = _registered(unit, "output")
    for name in ("a", "b"):
        for state_var in ("flow_vol_phase", "pressure", "temperature"):
            assert f"outlet_{name}_state.{state_var}" in outputs


# -- config rejection -------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("outlet_names", [(), ("a", "a"), ("a", ""), ("a", 1)])
def test_splitter_rejects_bad_outlet_names(outlet_names):
    """Empty, duplicated, or non-string ``outlet_names`` raise, naming the field."""
    with pytest.raises(FlexConfigError) as excinfo:
        _splitter(3, outlet_names=outlet_names)
    assert excinfo.value.field == "outlet_names"


@pytest.mark.unit
def test_splitter_requires_a_single_phase_property_package():
    """A property package with more than one phase is rejected."""
    m = dummy_gas_time_block(3)
    m.two_phase_properties = _TwoPhaseSplitterFlow()
    with pytest.raises(FlexConfigError) as excinfo:
        m.unit = Splitter(property_package=m.two_phase_properties)
    assert excinfo.value.field == "property_package"


# -- integration with the rest of the library -------------------------------


@pytest.mark.unit
def test_splitter_is_in_the_unit_model_registry():
    """``Splitter`` is reachable from ``flexops`` and its unit-model registry."""
    import flexops
    from flexops import unit_models

    assert "Splitter" in unit_models.__all__
    assert flexops.Splitter is Splitter


@pytest.mark.component
@pytest.mark.needs_highs
def test_splitter_routing_is_chosen_by_the_objective():
    """Minimizing power sends the whole feed down the cheaper of two branches."""
    from pyomo.environ import units as pyunits
    from pyomo.network import Arc

    from flexcore.exceptions import FlexSolverError
    from flexcore.solvers import get_solver
    from flexops import PlantBlock
    from flexops.unit_models import Pump

    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.splitter = Splitter(
        property_package=m.properties, outlet_names=("cheap", "dear")
    )
    m.plant.pump_cheap = Pump(
        property_package=m.properties,
        energy_intensity=0.1 * pyunits.kWh / pyunits.m**3,
    )
    m.plant.pump_dear = Pump(
        property_package=m.properties,
        energy_intensity=0.9 * pyunits.kWh / pyunits.m**3,
    )
    m.plant.arc_cheap = Arc(
        source=m.plant.splitter.outlet_cheap, destination=m.plant.pump_cheap.inlet
    )
    m.plant.arc_dear = Arc(
        source=m.plant.splitter.outlet_dear, destination=m.plant.pump_dear.inlet
    )
    m.plant._build_aggregates()
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    for t in m.time_block.time_index:
        m.plant.splitter.inlet_state.flow_vol_phase[t, "Liq"].fix(5.0)
    m.objective = pyo.Objective(
        expr=sum(m.plant.total_electrical_power[t] for t in m.time_block.time_index),
        sense=pyo.minimize,
    )

    try:
        solver = get_solver(model=m)
    except FlexSolverError as exc:
        pytest.skip(f"flexcore.solvers.get_solver not available: {exc}")
    pyo.assert_optimal_termination(solver.solve(m))

    for t in m.time_block.time_index:
        cheap = pyo.value(m.plant.splitter.flow_out_cheap[t])
        dear = pyo.value(m.plant.splitter.flow_out_dear[t])
        assert cheap + dear == pytest.approx(5.0, rel=1e-6)
        assert cheap == pytest.approx(5.0, rel=1e-6)
        assert dear == pytest.approx(0.0, abs=1e-6)
