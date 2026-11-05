"""Tasks: units grouped behind their resource connections, and the pure RTN check."""

import pyomo.environ as pyo
import pytest
from pyomo.core.expr.visitor import identify_variables
from pyomo.environ import units as pyunits
from pyomo.network import Arc

from flexcore import nomenclature as nm
from flexcore.config.schema import BalanceType
from flexcore.exceptions import FlexConfigError
from flexcore.solvers import facade, get_solver
from flexops import (
    NetworkBlock,
    PlantBlock,
    Resource,
    SimpleUtility,
    Task,
    emit_model,
    resource_graph,
    validate_rtn,
)
from flexops.core.registration import iter_resource_terms
from flexops.testing import dummy_time_block
from flexops.unit_models import Pump

INTENSITY = 0.5 * pyunits.kWh / pyunits.m**3


def _pump(m, block, name):
    """Add a Pump to a block and return it."""
    block.add_component(
        name, Pump(property_package=m.properties, energy_intensity=INTENSITY)
    )
    return block.find_component(name)


def _plant(n: int = 2):
    """Return a plant with a raw water source and a treated water sink."""
    m = dummy_time_block(n)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.raw = Resource(property_package=m.properties, balance=BalanceType.IMPORT)
    m.plant.treated = Resource(
        property_package=m.properties, balance=BalanceType.EXPORT
    )
    return m


def _two_unit_task(m, name):
    """Add a task of two pumps in series between ``raw`` and ``treated``."""
    p = m.plant
    p.add_component(name, Task())
    task = p.find_component(name)
    first, second = _pump(m, task, "first"), _pump(m, task, "second")
    task.link = Arc(source=first.outlet, destination=second.inlet)
    p.add_component(f"{name}_in", Arc(source=p.raw.outlet, destination=first.inlet))
    p.add_component(
        f"{name}_out", Arc(source=second.outlet, destination=p.treated.inlet)
    )
    return task


def _one_unit_task(m, name):
    """Add one bare pump between ``raw`` and ``treated``, serving as a task."""
    p = m.plant
    pump = _pump(m, p, name)
    p.add_component(f"{name}_in", Arc(source=p.raw.outlet, destination=pump.inlet))
    p.add_component(f"{name}_out", Arc(source=pump.outlet, destination=p.treated.inlet))
    return pump


# --- the task block ---------------------------------------------------------


@pytest.mark.unit
def test_a_task_holds_units_built_over_the_plants_time():
    m = _plant()
    m.plant.task = Task()

    pump = _pump(m, m.plant.task, "pump")

    assert pump.flowsheet() is m.plant
    assert {t for t, _ in pump.inlet_state.flow_vol_phase} == set(
        m.time_block.time_index
    )


@pytest.mark.unit
@pytest.mark.parametrize("parent", ["root", "network", "task"])
def test_a_task_must_sit_directly_on_a_plant(parent):
    m = _plant()
    m.plant.task = Task()
    m.net = NetworkBlock(time_block=m.time_block)
    block = {"root": m, "network": m.net, "task": m.plant.task}[parent]

    with pytest.raises(FlexConfigError, match="PlantBlock") as excinfo:
        block.inner = Task()

    assert excinfo.value.field == "parent"


@pytest.mark.unit
def test_a_plant_totals_the_power_of_units_inside_a_task():
    m = _plant()
    task = _two_unit_task(m, "task")

    m.plant._build_aggregates()

    total = m.plant.component(nm.TOTAL_POWER_VARS[nm.PowerKind.ELECTRICAL][0])
    found = {id(var) for var in identify_variables(total[0].expr)}
    assert id(task.first.power_electrical[0]) in found
    assert id(task.second.power_electrical[0]) in found


@pytest.mark.unit
def test_a_plant_utility_collects_terms_from_units_inside_a_task():
    m = _plant()
    task = _two_unit_task(m, "task")
    m.power = SimpleUtility(resource="electricity", units="kW")

    m.plant.power_node = Resource(property_package=m.power)

    assert len(list(m.plant.power_node.edges)) == 2
    terms = iter_resource_terms(m.plant)["electricity"]
    assert {term.unit for term in terms} == {task.first, task.second}


@pytest.mark.unit
def test_a_spec_cannot_hold_a_task_yet():
    m = _plant()
    _two_unit_task(m, "task")

    with pytest.raises(FlexConfigError, match="plant.task"):
        emit_model(m)


# --- the graph ---------------------------------------------------------------


@pytest.mark.unit
def test_graph_counts_a_multi_unit_task_as_one_task():
    m = _plant()
    task = _two_unit_task(m, "task")

    graph = resource_graph(m.plant)

    assert task in graph.tasks
    assert task.first not in graph.tasks
    assert task.second not in graph.tasks


@pytest.mark.unit
def test_graph_counts_a_lone_unit_as_its_own_task():
    m = _plant()
    pump = _one_unit_task(m, "pump")

    assert pump in resource_graph(m.plant).tasks


@pytest.mark.unit
def test_graph_leaves_out_arcs_inside_a_task():
    m = _plant()
    _two_unit_task(m, "task")

    assert "plant.task.link" not in resource_graph(m.plant).resources


@pytest.mark.unit
def test_an_alternative_task_presents_the_same_connections_with_different_units():
    m = _plant()
    single = _one_unit_task(m, "single")
    double = _two_unit_task(m, "double")

    graph = resource_graph(m.plant)

    assert graph.connections(single) == graph.connections(double)
    assert ("plant.raw", False) in graph.connections(double)
    assert ("plant.treated", True) in graph.connections(double)


@pytest.mark.unit
def test_a_task_on_other_resources_presents_different_connections():
    m = _plant()
    m.plant.other = Resource(property_package=m.properties)
    single = _one_unit_task(m, "single")
    moved = _pump(m, m.plant, "moved")
    m.plant.moved_in = Arc(source=m.plant.other.outlet, destination=moved.inlet)
    m.plant.moved_out = Arc(source=moved.outlet, destination=m.plant.treated.inlet)

    graph = resource_graph(m.plant)

    assert graph.connections(single) != graph.connections(moved)


# --- the opt-in validator ----------------------------------------------------


@pytest.mark.unit
def test_a_pure_rtn_validates_and_records_it_on_the_model():
    m = _plant()
    _one_unit_task(m, "single")
    _two_unit_task(m, "double")

    validate_rtn(m)

    assert m.rtn_validated is True


@pytest.mark.unit
def test_a_mixed_flowsheet_builds_without_being_validated():
    m = _plant()
    first, second = _pump(m, m.plant, "first"), _pump(m, m.plant, "second")
    m.plant.link = Arc(source=first.outlet, destination=second.inlet)
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    assert getattr(m, "rtn_validated", False) is False


@pytest.mark.unit
def test_an_arc_between_resources_fails_and_names_a_task_as_the_fix():
    m = _plant()
    m.plant.bypass = Arc(source=m.plant.raw.outlet, destination=m.plant.treated.inlet)

    with pytest.raises(FlexConfigError, match=r"plant\.bypass.*task") as excinfo:
        validate_rtn(m)

    assert excinfo.value.field == "rtn"
    assert m.rtn_validated is False


@pytest.mark.unit
def test_an_arc_between_ungrouped_units_fails():
    m = _plant()
    first, second = _pump(m, m.plant, "first"), _pump(m, m.plant, "second")
    m.plant.link = Arc(source=first.outlet, destination=second.inlet)

    with pytest.raises(FlexConfigError, match=r"plant\.link.*group"):
        validate_rtn(m)


@pytest.mark.unit
def test_an_arc_between_tasks_fails():
    m = _plant()
    a, b = _two_unit_task(m, "a"), _two_unit_task(m, "b")
    m.plant.a_to_b = Arc(source=a.second.outlet, destination=b.first.inlet)

    with pytest.raises(FlexConfigError, match=r"plant\.a_to_b.*resource"):
        validate_rtn(m)


@pytest.mark.unit
def test_an_arc_from_a_task_to_a_bare_unit_fails():
    m = _plant()
    task = _two_unit_task(m, "task")
    bare = _pump(m, m.plant, "bare")
    m.plant.out = Arc(source=task.second.outlet, destination=bare.inlet)

    with pytest.raises(FlexConfigError, match=r"plant\.out"):
        validate_rtn(m)


@pytest.mark.unit
def test_a_resource_inside_a_task_fails():
    m = _plant()
    m.plant.task = Task()
    m.plant.task.hidden = Resource(property_package=m.properties)

    with pytest.raises(FlexConfigError, match=r"plant\.task\.hidden"):
        validate_rtn(m)


@pytest.mark.unit
def test_every_violation_is_listed_at_once():
    m = _plant()
    first, second = _pump(m, m.plant, "first"), _pump(m, m.plant, "second")
    m.plant.link = Arc(source=first.outlet, destination=second.inlet)
    m.plant.bypass = Arc(source=m.plant.raw.outlet, destination=m.plant.treated.inlet)

    with pytest.raises(FlexConfigError) as excinfo:
        validate_rtn(m)

    assert "plant.link" in str(excinfo.value)
    assert "plant.bypass" in str(excinfo.value)


@pytest.mark.unit
def test_validation_still_sees_arcs_after_expansion():
    m = _plant()
    first, second = _pump(m, m.plant, "first"), _pump(m, m.plant, "second")
    m.plant.link = Arc(source=first.outlet, destination=second.inlet)
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    with pytest.raises(FlexConfigError, match=r"plant\.link"):
        validate_rtn(m)


@pytest.mark.unit
def test_a_model_changed_after_validation_is_refused_at_solve():
    m = _plant()
    _one_unit_task(m, "single")
    validate_rtn(m)
    first, second = _pump(m, m.plant, "first"), _pump(m, m.plant, "second")
    m.plant.link = Arc(source=first.outlet, destination=second.inlet)

    with pytest.raises(FlexConfigError, match=r"plant\.link"):
        for check in facade.PRE_SOLVE_CHECKS:
            check(m)


# --- solved ------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.needs_highs
def test_a_task_can_be_swapped_for_its_alternative_with_no_other_change():
    """Both alternatives connect to the same resources; the cheaper one runs."""
    m = _plant(3)
    single = _one_unit_task(m, "single")
    double = _two_unit_task(m, "double")
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)
    steps = list(m.time_block.time_index)
    for t in steps:
        single.inlet_state.flow_vol_phase[t, "Liq"].fix(0.0)
        double.first.inlet_state.flow_vol_phase[t, "Liq"].fix(2.0)
    m.cost = pyo.Objective(expr=sum(double.second.power_electrical[t] for t in steps))

    results = get_solver(model=m, prefer="highs").solve(m)

    pyo.assert_optimal_termination(results)
    assert pyo.value(double.second.outlet_state.flow_vol_phase[0, "Liq"]) == (
        pytest.approx(2.0)
    )
