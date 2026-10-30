"""Constraint-body + relaxation tests for the UC status base (M08, §3.5)."""

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits

from flexcore.exceptions import FlexConfigError
from flexops.logic import add_status, relax, unrelax
from flexops.testing import dummy_time_block
from flexops.unit_models import Pump


def _con_satisfied(condata, tol: float = 1e-9) -> bool:
    """Whether a constraint body lies within its (lower, upper) bounds."""
    body = pyo.value(condata.body)
    lower = condata.lower
    upper = condata.upper
    ok = True
    if lower is not None:
        ok = ok and pyo.value(lower) <= body + tol
    if upper is not None:
        ok = ok and body <= pyo.value(upper) + tol
    return ok


def _pump_with_status(min_kw: float = 0.0, max_kw: float = 100.0):
    """Build a 4-step Pump and attach a status binary to its power draw."""
    m = dummy_time_block(4)
    m.unit = Pump(property_package=m.properties)
    status = add_status(
        m.unit, m.unit.power_electrical, min_kw * pyunits.kW, max_kw * pyunits.kW
    )
    return m, m.unit, status


@pytest.mark.unit
def test_add_status_returns_time_indexed_binary():
    """add_status attaches status[t] Binary over the unit's time points."""
    m, unit, status = _pump_with_status()
    assert status is unit.status
    assert status.is_indexed()
    assert set(status.index_set()) == set(m.time_block.time_index)
    for t in m.time_block.time_index:
        assert status[t].domain is pyo.Binary


@pytest.mark.unit
def test_add_status_constraint_bodies():
    """On/at-max is feasible; off-with-output and on-below-min are infeasible."""
    m, unit, status = _pump_with_status(min_kw=10.0, max_kw=100.0)
    power = unit.power_electrical

    # status=1, output=max -> both semicontinuous links satisfied.
    status[0].set_value(1)
    power[0].set_value(100.0)
    assert _con_satisfied(unit.status_min_link[0])
    assert _con_satisfied(unit.status_max_link[0])

    # status=0, output=5 (>0) -> upper link violated (off cannot produce).
    status[1].set_value(0)
    power[1].set_value(5.0)
    assert not _con_satisfied(unit.status_max_link[1])

    # status=1, output=5 (< min=10) -> lower link violated (>= min when on).
    status[2].set_value(1)
    power[2].set_value(5.0)
    assert not _con_satisfied(unit.status_min_link[2])


@pytest.mark.unit
def test_relax_unrelax_round_trip():
    """relax() flips only the tracked binaries' domain; unrelax() restores it."""
    m, unit, status = _pump_with_status()
    n_components = len(list(unit.component_objects()))
    n_constraints = len(list(unit.component_data_objects(pyo.Constraint)))

    assert all(status[t].domain is pyo.Binary for t in m.time_block.time_index)

    relax(unit)
    for t in m.time_block.time_index:
        assert status[t].domain is pyo.UnitInterval
    # Nothing else changed: no components or constraints added/removed.
    assert len(list(unit.component_objects())) == n_components
    assert len(list(unit.component_data_objects(pyo.Constraint))) == n_constraints

    unrelax(unit)
    for t in m.time_block.time_index:
        assert status[t].domain is pyo.Binary


def _counted_pump(max_count: int = 3, min_count_on: int | None = None):
    """Build a 4-step Pump whose status counts up to ``max_count`` trains."""
    m = dummy_time_block(4)
    m.unit = Pump(property_package=m.properties)
    status = add_status(
        m.unit,
        m.unit.power_electrical,
        10.0 * pyunits.kW,
        100.0 * pyunits.kW,
        max_count=max_count,
        min_count_on=min_count_on,
    )
    return m, m.unit, status


@pytest.mark.unit
def test_counted_status_is_a_bounded_integer():
    """max_count = 3 builds status[t] as an integer in 0..3, not a Binary."""
    m, _unit, status = _counted_pump(max_count=3)
    for t in m.time_block.time_index:
        assert status[t].domain is pyo.NonNegativeIntegers
        assert status[t].bounds == (0, 3)


@pytest.mark.unit
def test_default_max_count_builds_the_binary_structure():
    """With the defaults the unit has no status_on and a Binary status."""
    _m, unit, status = _pump_with_status()
    assert all(status[t].domain is pyo.Binary for t in status)
    assert not hasattr(unit, "status_on")
    assert not hasattr(unit, "status_on_min")


@pytest.mark.unit
def test_counted_links_scale_the_output_limits_by_the_count():
    """Two of three trains running may deliver between 2 * min and 2 * max."""
    _m, unit, status = _counted_pump(max_count=3)
    power = unit.power_electrical

    status[0].set_value(2)
    power[0].set_value(200.0)
    assert _con_satisfied(unit.status_min_link[0])
    assert _con_satisfied(unit.status_max_link[0])

    power[0].set_value(201.0)
    assert not _con_satisfied(unit.status_max_link[0])

    power[0].set_value(19.0)
    assert not _con_satisfied(unit.status_min_link[0])


@pytest.mark.unit
def test_status_on_is_built_only_when_min_count_on_is_given():
    """min_count_on adds status_on and the two constraints that tie it to status."""
    _m, unit, _status = _counted_pump(max_count=3)
    assert not hasattr(unit, "status_on")

    _m, unit, status = _counted_pump(max_count=3, min_count_on=2)
    assert unit.status_on[0].domain is pyo.Binary

    # One train alone is not allowed; two or three are.
    for count, status_on, allowed in [(1, 1, False), (2, 1, True), (3, 1, True)]:
        status[0].set_value(count)
        unit.status_on[0].set_value(status_on)
        assert (
            _con_satisfied(unit.status_on_min[0])
            and _con_satisfied(unit.status_on_max[0])
        ) is allowed
    # Off needs status_on = 0; on needs status_on = 1.
    status[0].set_value(0)
    unit.status_on[0].set_value(0)
    assert _con_satisfied(unit.status_on_min[0])
    assert _con_satisfied(unit.status_on_max[0])
    status[0].set_value(2)
    unit.status_on[0].set_value(0)
    assert not _con_satisfied(unit.status_on_max[0])


@pytest.mark.unit
def test_relax_and_unrelax_round_trip_a_count_to_reals_up_to_n():
    """A relaxed count is a real in [0, n]; unrelax restores the integers."""
    m, unit, status = _counted_pump(max_count=3, min_count_on=2)
    time_index = m.time_block.time_index

    relax(unit)
    for t in time_index:
        assert status[t].domain is pyo.NonNegativeReals
        assert status[t].bounds == (0, 3)
        assert unit.status_on[t].domain is pyo.UnitInterval

    unrelax(unit)
    for t in time_index:
        assert status[t].domain is pyo.NonNegativeIntegers
        assert status[t].bounds == (0, 3)
        assert unit.status_on[t].domain is pyo.Binary


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"max_count": 0}, "max_count"),
        ({"max_count": 3, "min_count_on": 1}, "min_count_on"),
        ({"max_count": 3, "min_count_on": 4}, "min_count_on"),
        ({"max_count": 1, "min_count_on": 2}, "min_count_on"),
    ],
)
def test_bad_counts_raise_naming_the_field(kwargs, field):
    """Out-of-range counts raise FlexConfigError and attach no components."""
    m = dummy_time_block(4)
    m.unit = Pump(property_package=m.properties)

    with pytest.raises(FlexConfigError) as excinfo:
        add_status(
            m.unit,
            m.unit.power_electrical,
            0.0 * pyunits.kW,
            1.0 * pyunits.kW,
            **kwargs,
        )

    assert excinfo.value.field == field
    assert not hasattr(m.unit, "status")


@pytest.mark.component
@pytest.mark.needs_highs
def test_min_count_on_holds_in_a_solve():
    """Demand one train could meet runs two, since one alone is below the minimum."""
    from flexcore.solvers import get_solver

    m, unit, status = _counted_pump(max_count=3, min_count_on=2)
    demand = [0.0, 50.0, 50.0, 50.0]

    @m.Constraint(m.time_block.time_index)
    def meet_demand(b, t):
        return unit.power_electrical[t] >= demand[t] * pyunits.kW

    m.cost = pyo.Objective(expr=sum(status[t] for t in status))
    pyo.assert_optimal_termination(get_solver(model=m, prefer="highs").solve(m))

    assert [round(pyo.value(status[t])) for t in status] == [0, 2, 2, 2]


def _multilinear_power(intercept: float):
    """Return a multilinear Pump power relation with the given constant term."""
    from flexops.surrogates.multilinear import MultilinearSurrogate

    return MultilinearSurrogate(
        {
            "input_variables": {"flow_out": "m^3/hr"},
            "output_variables": {"power_electrical": "kW"},
            "coefficients": {"flow_out": 0.4, "intercept": intercept},
        }
    )


@pytest.mark.unit
def test_counting_a_unit_with_a_relation_intercept_raises():
    """A constant term would be counted once, not once per running train."""
    m = dummy_time_block(4)
    m.unit = Pump(property_package=m.properties)
    m.unit.swap_relation("power_electrical_relation", _multilinear_power(5.0))

    with pytest.raises(FlexConfigError, match="constant term") as excinfo:
        add_status(m.unit, m.unit.power_electrical, 0.0, 100.0, max_count=3)

    assert excinfo.value.field == "max_count"
    assert not hasattr(m.unit, "status")


@pytest.mark.unit
def test_swapping_an_intercept_onto_a_counted_unit_raises():
    """The check also holds when the relation is swapped after counting."""
    m, unit, _status = _counted_pump(max_count=3)
    relation = unit.power_electrical_relation

    with pytest.raises(FlexConfigError, match="constant term"):
        unit.swap_relation("power_electrical_relation", _multilinear_power(5.0))

    assert relation.active


@pytest.mark.unit
@pytest.mark.parametrize(("intercept", "max_count"), [(0.0, 3), (5.0, 1)])
def test_zero_intercept_or_single_train_is_accepted(intercept, max_count):
    m = dummy_time_block(4)
    m.unit = Pump(property_package=m.properties)
    m.unit.swap_relation("power_electrical_relation", _multilinear_power(intercept))

    add_status(m.unit, m.unit.power_electrical, 0.0, 100.0, max_count=max_count)
