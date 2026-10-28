"""Tests for LumpedReactor: volume holdup with constant composition relations."""

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits
from pyomo.opt import assert_optimal_termination

from flexcore.exceptions import FlexConfigError
from flexcore.solvers import get_solver
from flexops.testing import UnitModelTestHarness, dummy_time_block
from flexops.unit_models import LumpedReactor


class TestLumpedReactor(UnitModelTestHarness):
    """Fixing the inlet and reference outlet flows determines volume and splits."""

    expected_dof = 0

    def configure(self):
        m = dummy_time_block(4)
        m.unit = LumpedReactor(
            property_package=m.properties,
            outlet_names=("gas", "digestate"),
            split_fractions={"digestate": 0.2},
            compositions={"ch4": 0.6, "co2": 0.35},
        )
        return m, m.unit


def _reactor(n: int = 4, **kwargs):
    """Build a fresh LumpedReactor on an ``n``-point dummy_time_block."""
    m = dummy_time_block(n)
    m.unit = LumpedReactor(
        property_package=m.properties,
        max_volume=1000 * pyunits.m**3,
        initial_volume=200 * pyunits.m**3,
        **kwargs,
    )
    return m, m.unit


@pytest.mark.unit
def test_each_composition_gets_its_own_registered_relation():
    _, unit = _reactor(compositions={"ch4": 0.6, "h2s": 0.001})
    relations = {r.name: r for r in unit._io_registry.relations}
    for name in ("ch4", "h2s"):
        record = relations[f"outlet_composition_{name}_relation"]
        assert record.target is unit.find_component(f"outlet_composition_{name}")
        assert record.target.index_set().dimen == 1


@pytest.mark.unit
def test_split_and_composition_parameters_are_regressable():
    _, unit = _reactor(
        outlet_names=("gas", "digestate"),
        split_fractions={"digestate": 0.2},
        compositions={"ch4": 0.6},
    )
    regressable = {r.name for r in unit._io_registry.parameters if r.regressable}
    assert {"split_fraction_digestate", "composition_ch4"} <= regressable
    assert "split_fraction_gas" not in regressable


@pytest.mark.unit
def test_flows_are_inputs_and_volume_level_composition_are_outputs():
    _, unit = _reactor(compositions={"ch4": 0.6})
    outputs = {r.var for r in unit._io_registry.io_variables if r.role == "output"}
    for name in ("volume", "level", "outlet_composition_ch4"):
        assert unit.find_component(name) in outputs
    inputs = {r.var for r in unit._io_registry.io_variables if r.role == "input"}
    assert unit.outlet_product_state.flow_vol_phase in inputs


@pytest.mark.unit
@pytest.mark.parametrize(
    "splits",
    [{"product": 0.2}, {"nope": 0.2}, {"b": 0.7, "c": 0.6}],
)
def test_invalid_split_fractions_rejected(splits):
    m = dummy_time_block(3)
    with pytest.raises(FlexConfigError, match="split_fractions"):
        m.unit = LumpedReactor(
            property_package=m.properties,
            outlet_names=("product", "b", "c"),
            split_fractions=splits,
        )


@pytest.mark.unit
def test_volume_balance_by_hand():
    m, unit = _reactor(n=4)
    flow_in = [100.0, 100.0, 0.0, 0.0]
    flow_out = [50.0, 50.0, 50.0, 50.0]
    volumes = [200.0, 212.5, 200.0, 187.5]
    for t in m.time_block.time_index:
        unit.flow_in_feed[t].set_value(flow_in[t])
        unit.flow_out_product[t].set_value(flow_out[t])
        unit.volume[t].set_value(volumes[t])
    for t in unit.holdup:
        assert pyo.value(unit.holdup[t].body) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.component
@pytest.mark.needs_highs
def test_split_and_constant_composition_hold_in_solution():
    m, unit = _reactor(
        n=3,
        outlet_names=("gas", "digestate"),
        split_fractions={"digestate": 0.2},
        compositions={"ch4": 0.6},
    )
    for t in m.time_block.time_index:
        unit.flow_in_feed[t].fix(100.0)
        unit.flow_out_gas[t].fix(80.0)
    m.obj = pyo.Objective(expr=0)
    assert_optimal_termination(get_solver(model=m, prefer="highs").solve(m))
    for t in m.time_block.time_index:
        assert pyo.value(unit.flow_out_digestate[t]) == pytest.approx(20.0)
        assert pyo.value(unit.volume[t]) == pytest.approx(200.0)
        assert pyo.value(unit.outlet_composition_ch4[t]) == pytest.approx(0.6)


@pytest.mark.unit
def test_level_bounded_and_tied_to_volume_over_capacity():
    m, unit = _reactor(n=3, level_min=0.1, level_max=0.9)
    for t in m.time_block.time_index:
        assert unit.level[t].bounds == (0.1, 0.9)
    assert unit.capacity.fixed
    assert pyo.value(unit.capacity) == pytest.approx(1000.0)
    unit.volume[1].set_value(500.0)
    unit.level[1].set_value(0.5)
    assert pyo.value(unit.level_definition[1].body) == pytest.approx(0.0, abs=1e-9)
