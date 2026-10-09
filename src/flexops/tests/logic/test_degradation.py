"""Degradation-penalty tests: hinge constraints, pricing, horizon limits, solves."""

import logging

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits
from pyomo.opt import assert_optimal_termination
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.check_units import assert_units_equivalent

from flexcore.config.schema import DegradationSpec, DegradationTerm, DegradationTermSpec
from flexcore.exceptions import FlexConfigError
from flexops.core.ops_block import OpsBlock
from flexops.costing import FlexCosting, currency_units
from flexops.logic import add_degradation
from flexops.logic.status import RollingStateKind
from flexops.surrogates.multilinear import MultilinearSurrogate
from flexops.testing import dummy_time_block

_N = 6
_DT_HR = 0.25  # dummy_time_block is 15-minute resolution
_HORIZON_HR = _N * _DT_HR


def _unit():
    """A bare unit carrying one free, time-indexed kW Var named ``x``."""
    m = dummy_time_block(_N)
    m.unit = OpsBlock()
    m.unit.x = pyo.Var(m.time_block.time_index, units=pyunits.kW, bounds=(0, 100))
    return m, m.unit


def _term(kind=DegradationTerm.VARIATION, **fields) -> DegradationTermSpec:
    """A term on ``x`` with price 2, any field overridden."""
    return DegradationTermSpec(
        **{"kind": kind, "variable": "x", "price": 2.0, **fields}
    )


def _spec(*terms, **fields) -> DegradationSpec:
    """A spec named ``wear`` over ``terms`` (a one-step variation term by default)."""
    return DegradationSpec(name="wear", terms=list(terms) or [_term()], **fields)


def _set(var, values):
    for t, value in enumerate(values):
        var[t].set_value(value)


def _satisfied(condata, tol: float = 1e-9) -> bool:
    """Whether a constraint body lies within its bounds."""
    body = pyo.value(condata.body)
    ok = True
    if condata.lower is not None:
        ok = ok and pyo.value(condata.lower) <= body + tol
    if condata.upper is not None:
        ok = ok and body <= pyo.value(condata.upper) + tol
    return ok


def _hinge_is_tight(unit, k: int, required: dict[int, float]) -> None:
    """Assert the term's hinge admits exactly ``required[t]`` and nothing smaller."""
    excess = unit.find_component(f"wear_{k}_excess")
    cons = [
        c
        for c in (
            unit.find_component(f"wear_{k}_up"),
            unit.find_component(f"wear_{k}_down"),
        )
        if c is not None
    ]
    for t, need in required.items():
        excess[t].set_value(need)
        assert all(_satisfied(c[t]) for c in cons if t in c), t
        if need > 0:
            excess[t].set_value(need - 0.1)
            assert not all(_satisfied(c[t]) for c in cons if t in c), t


@pytest.mark.unit
def test_quantity_relation_ties_the_tracked_quantity_to_the_variable():
    """Each term costs its own tracked quantity, defined equal to the variable."""
    m, unit = _unit()
    add_degradation(unit, _spec())
    quantity = unit.wear_0_quantity
    relation = unit.wear_0_quantity_relation

    _set(unit.x, range(_N))
    _set(quantity, range(_N))
    assert all(_satisfied(c) for c in relation.values())
    quantity[2].set_value(99.0)
    assert not _satisfied(relation[2])


@pytest.mark.unit
def test_variation_charges_change_beyond_the_deadband():
    """|q[t] - q[t-1]| - deadband, floored at zero, is the charged excess."""
    m, unit = _unit()
    add_degradation(unit, _spec(_term(deadband=1.0)))
    _set(unit.wear_0_quantity, [0.0, 0.5, 4.0, 4.0, 1.0, 1.0])

    assert set(unit.wear_0_excess.index_set()) == set(range(1, _N))
    _hinge_is_tight(unit, 0, {1: 0.0, 2: 2.5, 3: 0.0, 4: 2.0, 5: 0.0})


@pytest.mark.unit
def test_variation_window_compares_endpoints_and_skips_missing_history():
    """window=w compares q[t] with q[t-w]; steps without that history carry no term."""
    m, unit = _unit()
    add_degradation(unit, _spec(_term(window=3)))
    _set(unit.wear_0_quantity, [0.0, 9.0, 0.0, 2.0, 2.0, 0.0])

    assert set(unit.wear_0_excess.index_set()) == {3, 4, 5}
    _hinge_is_tight(unit, 0, {3: 2.0, 4: 7.0, 5: 0.0})


@pytest.mark.unit
def test_deviation_charges_distance_from_the_reference_beyond_the_deadband():
    """|q - reference| - deadband, floored at zero, on every step."""
    m, unit = _unit()
    add_degradation(
        unit, _spec(_term(DegradationTerm.DEVIATION, reference=10.0, deadband=1.0))
    )
    _set(unit.wear_0_quantity, [10.0, 12.0, 7.0, 10.5, 0.0, 11.0])

    assert set(unit.wear_0_excess.index_set()) == set(range(_N))
    _hinge_is_tight(unit, 0, {0: 0.0, 1: 1.0, 2: 2.0, 3: 0.0, 4: 9.0, 5: 0.0})


@pytest.mark.unit
def test_exceedance_charges_only_outside_the_band():
    """max(0, q - upper) + max(0, lower - q); a missing bound builds no constraint."""
    m, unit = _unit()
    add_degradation(
        unit, _spec(_term(DegradationTerm.EXCEEDANCE, lower=2.0, upper=8.0))
    )
    _set(unit.wear_0_quantity, [5.0, 9.0, 1.0, 8.0, 2.0, 20.0])
    _hinge_is_tight(unit, 0, {0: 0.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 12.0})

    m2, unit2 = _unit()
    add_degradation(unit2, _spec(_term(DegradationTerm.EXCEEDANCE, upper=8.0)))
    assert unit2.find_component("wear_0_down") is None


@pytest.mark.unit
def test_throughput_charges_the_quantity_beyond_the_deadband():
    """max(0, q - deadband) on every step."""
    m, unit = _unit()
    add_degradation(unit, _spec(_term(DegradationTerm.THROUGHPUT, deadband=3.0)))
    _set(unit.wear_0_quantity, [0.0, 3.0, 5.0, 10.0, 1.0, 4.0])
    assert unit.find_component("wear_0_down") is None
    _hinge_is_tight(unit, 0, {0: 0.0, 1: 0.0, 2: 2.0, 3: 7.0, 4: 0.0, 5: 1.0})


@pytest.mark.unit
def test_rate_charges_variation_per_step_and_other_kinds_per_hour():
    """A variation excess is a per-step amount (divided by dt into a rate); a
    deviation/exceedance/throughput excess is already a level held for the step."""
    m, unit = _unit()
    add_degradation(
        unit,
        _spec(
            _term(price=2.0),
            _term(DegradationTerm.THROUGHPUT, price=3.0),
        ),
    )
    unit.wear_0_excess[2].set_value(1.0)
    unit.wear_1_excess[2].set_value(4.0)

    calculate_variable_from_constraint(unit.wear_rate[2], unit.wear_rate_relation[2])
    assert pyo.value(unit.wear_rate[2]) == pytest.approx(2.0 * 1.0 / _DT_HR + 3.0 * 4.0)
    assert_units_equivalent(pyunits.get_units(unit.wear_rate[2]), 1 / pyunits.hr)


@pytest.mark.unit
def test_total_integrates_the_rate_over_the_horizon():
    """total == sum_t rate[t] * dt, in the model currency's units (a bare number)."""
    m, unit = _unit()
    rate, total = add_degradation(unit, _spec())
    _set(rate, [4.0] * _N)
    calculate_variable_from_constraint(total, unit.wear_total_relation)
    assert pyo.value(total) == pytest.approx(4.0 * _HORIZON_HR)


@pytest.mark.unit
def test_covered_cost_and_budget_are_prorated_from_their_period():
    """Values stated per period_hours are scaled by horizon_hours / period_hours."""
    m, unit = _unit()
    _, total = add_degradation(
        unit,
        _spec(covered_cost=10.0, horizon_budget=40.0, period_hours=3.0),
    )
    scale = _HORIZON_HR / 3.0
    total.set_value(12.0)

    unit.wear_billable.set_value(12.0 - 10.0 * scale)
    assert _satisfied(unit.wear_billable_floor)
    unit.wear_billable.set_value(12.0 - 10.0 * scale - 0.1)
    assert not _satisfied(unit.wear_billable_floor)

    total.set_value(40.0 * scale)
    assert _satisfied(unit.wear_budget)
    total.set_value(40.0 * scale + 0.1)
    assert not _satisfied(unit.wear_budget)


@pytest.mark.unit
def test_no_budget_constraint_unless_one_is_given():
    """horizon_budget=None builds no budget constraint."""
    m, unit = _unit()
    add_degradation(unit, _spec(), costing=None)
    assert unit.find_component("wear_budget") is None


@pytest.mark.unit
def test_billed_rate_spreads_the_billable_amount_evenly():
    """The billed series integrates back to the billable amount over the horizon."""
    m, unit = _unit()
    add_degradation(unit, _spec())
    unit.wear_billable.set_value(6.0)
    billed = sum(pyo.value(unit.wear_billed_rate[t]) * _DT_HR for t in range(_N))
    assert billed == pytest.approx(6.0)


@pytest.mark.unit
def test_prices_and_limits_update_in_place():
    """Prices, deadbands and horizon limits are registered mutable parameters."""
    m, unit = _unit()
    add_degradation(unit, _spec(horizon_budget=5.0))
    unit.update_parameters(
        {
            "wear_0_price": 7.0,
            "wear_0_deadband": 0.5,
            "wear_covered_cost": 1.0,
            "wear_horizon_budget": 9.0,
        }
    )
    assert pyo.value(unit.wear_0_price) == 7.0
    assert pyo.value(unit.wear_0_deadband) == 0.5
    assert pyo.value(unit.wear_horizon_budget) == 9.0
    regressable = {rec.name: rec.regressable for rec in unit._io_registry.parameters}
    assert regressable["wear_0_price"] is False


@pytest.mark.unit
def test_quantity_and_rate_relations_are_swappable():
    """The tracked quantity and the aggregate rate are registered relations, so a
    surrogate can replace either one in place."""
    m, unit = _unit()
    add_degradation(unit, _spec())
    names = {rec.name for rec in unit._io_registry.relations}
    assert {"wear_0_quantity_relation", "wear_rate_relation"} <= names

    surrogate = MultilinearSurrogate(
        {
            "input_variables": {"x": "kW"},
            "output_variables": {"wear_0_quantity": "kW"},
            "coefficients": {"intercept": 0.0, "x": 2.0},
        }
    )
    unit.swap_relation("wear_0_quantity_relation", surrogate)
    assert not unit.wear_0_quantity_relation.active


@pytest.mark.unit
def test_variation_registers_its_history_as_rolling_state():
    """A variation term's trailing window must cross rolling-horizon windows."""
    m, unit = _unit()
    add_degradation(unit, _spec(_term(window=2), _term(DegradationTerm.THROUGHPUT)))
    entries = unit._flexops_rolling_state
    assert [(e["var"], e["k"], e["kind"]) for e in entries] == [
        (unit.wear_0_quantity, 2, RollingStateKind.DEGRADATION)
    ]


@pytest.mark.unit
def test_costing_registers_the_billed_series_as_a_scalar_cost():
    """With a costing package the billable amount enters opex, per unit."""
    m, unit = _unit()
    usd = currency_units("USD")
    m.costing = FlexCosting(
        time_block=m.time_block, energy_prices={"electrical": 0.1 * usd / pyunits.kWh}
    )
    add_degradation(unit, _spec(), costing=m.costing)

    spec = m.costing._registered_scalar_costs["unit_wear"]
    assert spec.unit is unit
    assert spec.quantity is unit.wear_billed_rate

    m.costing.cost_process()
    unit.wear_billable.set_value(6.0)
    cost = m.costing.opex.scalar_cost_unit_wear
    calculate_variable_from_constraint(cost, m.costing.opex.eq_scalar_cost_unit_wear)
    assert pyo.value(cost) == pytest.approx(6.0)


@pytest.mark.unit
def test_unpriced_unbudgeted_penalty_warns(caplog):
    """Without a price or budget nothing drives the hinges tight; say so."""
    m, unit = _unit()
    with caplog.at_level(logging.WARNING):
        add_degradation(unit, _spec())
    assert "neither priced nor budgeted" in caplog.text


@pytest.mark.unit
def test_duplicate_name_raises():
    """A second penalty with the same name on one unit is rejected."""
    m, unit = _unit()
    add_degradation(unit, _spec())
    with pytest.raises(FlexConfigError):
        add_degradation(unit, _spec())


@pytest.mark.unit
def test_unknown_variable_raises():
    """A term naming a variable that is not on the unit is rejected."""
    m, unit = _unit()
    with pytest.raises(FlexConfigError):
        add_degradation(unit, _spec(_term(variable="nope")))


def _solve_against_demand(spec: DegradationSpec) -> tuple[list[float], float]:
    """Minimize sum(x) + degradation total with x[t] >= an alternating demand."""
    from flexcore.solvers import get_solver

    m, unit = _unit()
    demand = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0]
    m.meet = pyo.Constraint(
        m.time_block.time_index, rule=lambda _m, t: unit.x[t] >= demand[t] * pyunits.kW
    )
    _, total = add_degradation(unit, spec)
    m.obj = pyo.Objective(
        expr=sum(unit.x[t] for t in m.time_block.time_index) / pyunits.kW
        + unit.wear_billable
    )
    results = get_solver(model=m, prefer="highs").solve(m)
    assert_optimal_termination(results)
    return [pyo.value(unit.x[t]) for t in range(_N)], pyo.value(m.obj)


@pytest.mark.component
@pytest.mark.needs_highs
def test_variation_price_smooths_an_alternating_schedule():
    """At $2/kW of change, holding 10 kW (cost 60) beats following demand (130)."""
    x, obj = _solve_against_demand(_spec(horizon_budget=1e6))
    assert x == pytest.approx([10.0] * _N)
    assert obj == pytest.approx(60.0)


@pytest.mark.component
@pytest.mark.needs_highs
def test_per_step_deadband_makes_small_moves_free():
    """A 10 kW deadband makes following demand free again."""
    x, obj = _solve_against_demand(_spec(_term(deadband=10.0), horizon_budget=1e6))
    assert x == pytest.approx([0.0, 10.0, 0.0, 10.0, 0.0, 10.0])
    assert obj == pytest.approx(30.0)


@pytest.mark.component
@pytest.mark.needs_highs
def test_covered_cost_covers_the_first_dollars_of_wear():
    """A $100 covered cost absorbs the $100 of cycling wear entirely."""
    x, obj = _solve_against_demand(_spec(covered_cost=100.0, horizon_budget=1e6))
    assert x == pytest.approx([0.0, 10.0, 0.0, 10.0, 0.0, 10.0])
    assert obj == pytest.approx(30.0)


@pytest.mark.component
@pytest.mark.needs_highs
def test_hard_budget_caps_wear_even_when_it_is_free_in_the_objective():
    """A $0 budget forbids any charged change, though the deadband makes it free."""
    x, obj = _solve_against_demand(_spec(covered_cost=1e6, horizon_budget=0.0))
    assert x == pytest.approx([10.0] * _N)
    assert obj == pytest.approx(60.0)
