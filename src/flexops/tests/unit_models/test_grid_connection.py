"""GridConnection(OpsBlockData): a plant's grid import/export with per-period limits."""

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits

from flexcore.exceptions import FlexConfigError
from flexops import PlantBlock
from flexops.testing import dummy_time_block
from flexops.unit_models import ConstantEnergyIntensityModel, GenericRenewables
from flexops.unit_models.watertreatment.grid_connection import GridConnection


def _plant_with_grid(n: int = 3, **kwargs):
    """Build an ``n``-point model with a plant holding only a GridConnection."""
    m = dummy_time_block(n)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.grid = GridConnection(**kwargs)
    return m, m.plant.grid


# -- structure --------------------------------------------------------------


@pytest.mark.unit
def test_grid_connection_builds_nonnegative_import_and_export_in_kw():
    m, grid = _plant_with_grid()
    for var in (grid.grid_import, grid.grid_export):
        assert set(var.index_set()) == set(m.time_block.time_index)
        assert pyunits.get_units(var) == pyunits.kW
        for t in m.time_block.time_index:
            assert var[t].lb == 0.0
            assert var[t].ub is None


@pytest.mark.unit
def test_grid_connection_registers_no_power_draw():
    _m, grid = _plant_with_grid()
    assert grid._io_registry.power == []
    assert grid.find_component("power_electrical") is None


@pytest.mark.unit
def test_grid_connection_outside_a_plant_raises():
    m = dummy_time_block(3)
    with pytest.raises(FlexConfigError) as excinfo:
        m.grid = GridConnection()
    assert excinfo.value.field == "parent"


@pytest.mark.unit
def test_grid_connection_nested_below_the_plant_raises():
    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.sub = pyo.Block()
    with pytest.raises(FlexConfigError) as excinfo:
        m.plant.sub.grid = GridConnection()
    assert excinfo.value.field == "parent"


@pytest.mark.unit
def test_grid_connection_net_power_ties_to_plant_total():
    m, grid = _plant_with_grid()
    for t in m.time_block.time_index:
        # Rewriting the plant total in place must move the grid balance with it.
        m.plant.total_electrical_power[t].set_value(5.0 * pyunits.kW)
        grid.grid_import[t].set_value(5.0)
        grid.grid_export[t].set_value(0.0)
        assert pyo.value(grid.eq_net_power[t].body) == pytest.approx(0.0)
        grid.grid_import[t].set_value(0.0)
        assert pyo.value(grid.eq_net_power[t].body) != pytest.approx(0.0)


@pytest.mark.unit
def test_grid_connection_sees_units_added_after_it():
    m, grid = _plant_with_grid()
    m.plant.solar = GenericRenewables(
        capacity=10 * pyunits.kW, capacity_factor=[0.2, 0.5, 0.8]
    )
    m.plant._build_aggregates()

    for t in m.time_block.time_index:
        m.plant.solar.power_electrical[t].set_value(-4.0)
        grid.grid_import[t].set_value(0.0)
        grid.grid_export[t].set_value(4.0)
        assert pyo.value(grid.eq_net_power[t].body) == pytest.approx(0.0)


# -- limits -----------------------------------------------------------------


@pytest.mark.unit
def test_grid_connection_limits_built_only_when_configured():
    _m, grid = _plant_with_grid(max_import=5 * pyunits.kW)
    assert grid.find_component("grid_import_max") is not None
    assert grid.find_component("grid_import_max_limit") is not None
    for name in ("grid_import_min", "grid_export_min", "grid_export_max"):
        assert grid.find_component(name) is None
        assert grid.find_component(f"{name}_limit") is None


@pytest.mark.unit
def test_grid_connection_builds_all_four_limits():
    _m, grid = _plant_with_grid(
        min_import=1 * pyunits.kW,
        max_import=5 * pyunits.kW,
        min_export=0 * pyunits.kW,
        max_export=3 * pyunits.kW,
    )
    for name in ("grid_import_min", "grid_import_max"):
        assert grid.find_component(name).mutable
    for name in ("grid_export_min", "grid_export_max"):
        assert grid.find_component(name).mutable


@pytest.mark.unit
def test_grid_connection_scalar_limit_converted_to_kw():
    m, grid = _plant_with_grid(max_export=2 * pyunits.MW)
    for t in m.time_block.time_index:
        assert pyo.value(grid.grid_export_max[t]) == pytest.approx(2000.0)


@pytest.mark.unit
def test_grid_connection_series_limit_sets_each_period():
    m, grid = _plant_with_grid(
        max_import=[1 * pyunits.kW, 2 * pyunits.kW, 3 * pyunits.MW]
    )
    values = [pyo.value(grid.grid_import_max[t]) for t in m.time_block.time_index]
    assert values == pytest.approx([1.0, 2.0, 3000.0])


@pytest.mark.unit
def test_grid_connection_series_limit_from_indexed_param():
    m = dummy_time_block(3)
    m.cap = pyo.Param(
        m.time_block.time_index,
        initialize={0: 4.0, 1: 5.0, 2: 6.0},
        units=pyunits.kW,
    )
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.grid = GridConnection(min_import=m.cap)
    values = [
        pyo.value(m.plant.grid.grid_import_min[t]) for t in m.time_block.time_index
    ]
    assert values == pytest.approx([4.0, 5.0, 6.0])


@pytest.mark.unit
def test_grid_connection_series_wrong_length_raises():
    with pytest.raises(FlexConfigError) as excinfo:
        _plant_with_grid(max_import=[1 * pyunits.kW, 2 * pyunits.kW])
    assert excinfo.value.field == "max_import"


@pytest.mark.unit
def test_grid_connection_limit_with_wrong_units_raises():
    with pytest.raises(FlexConfigError) as excinfo:
        _plant_with_grid(max_export=5 * pyunits.m**3 / pyunits.hr)
    assert excinfo.value.field == "max_export"


@pytest.mark.unit
def test_grid_connection_is_in_unit_model_registry():
    import flexops
    from flexops import unit_models

    assert "GridConnection" in unit_models.__all__
    assert flexops.GridConnection is GridConnection


# -- solves -----------------------------------------------------------------


def _solve(m):
    """Solve ``m`` with the best available LP solver, skipping if none."""
    from flexcore.exceptions import FlexSolverError
    from flexcore.solvers import get_solver

    try:
        solver = get_solver(model=m)
    except FlexSolverError as exc:
        pytest.skip(f"flexcore.solvers.get_solver not available: {exc}")
    pyo.assert_optimal_termination(solver.solve(m))


def _solar_and_load_plant(**grid_kwargs):
    """Build a plant with 20 kW of solar, a 10 kWh/m^3 load, and a GridConnection."""
    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.grid = GridConnection(**grid_kwargs)
    m.plant.solar = GenericRenewables(
        capacity=20 * pyunits.kW, capacity_factor=[0.2, 0.5, 0.8]
    )
    m.plant.load = ConstantEnergyIntensityModel(
        property_package=m.properties,
        energy_intensity=10 * pyunits.kWh / pyunits.m**3,
    )
    m.plant._build_aggregates()
    return m


@pytest.mark.component
@pytest.mark.needs_highs
def test_grid_connection_splits_net_draw_into_import_and_export():
    m = _solar_and_load_plant()
    grid = m.plant.grid
    for t in m.time_block.time_index:
        m.plant.load.inlet_state.flow_vol_phase[t, "Liq"].fix(1.0)
    m.obj = pyo.Objective(
        expr=sum(
            grid.grid_import[t] + grid.grid_export[t] for t in m.time_block.time_index
        )
    )
    _solve(m)

    # Load is 10 kW; solar makes 4, 10, 16 kW.
    imports = [pyo.value(grid.grid_import[t]) for t in m.time_block.time_index]
    exports = [pyo.value(grid.grid_export[t]) for t in m.time_block.time_index]
    assert imports == pytest.approx([6.0, 0.0, 0.0], abs=1e-6)
    assert exports == pytest.approx([0.0, 0.0, 6.0], abs=1e-6)


@pytest.mark.component
@pytest.mark.needs_highs
def test_grid_connection_series_max_import_binds_each_period():
    m = _solar_and_load_plant(
        max_import=[3 * pyunits.kW, 5 * pyunits.kW, 1 * pyunits.kW]
    )
    flow = m.plant.load.inlet_state.flow_vol_phase
    m.obj = pyo.Objective(
        expr=sum(flow[t, "Liq"] for t in m.time_block.time_index),
        sense=pyo.maximize,
    )
    _solve(m)

    # The largest load is solar (4, 10, 16 kW) plus the import cap (3, 5, 1 kW).
    loads = [
        pyo.value(m.plant.load.power_electrical[t]) for t in m.time_block.time_index
    ]
    assert loads == pytest.approx([7.0, 15.0, 17.0], rel=1e-6)
