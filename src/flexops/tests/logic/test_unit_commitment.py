"""Unit-commitment transition + min-uptime/downtime truth table (M08, §5)."""

import itertools

import pyomo.environ as pyo
import pytest

from flexcore.exceptions import FlexConfigError
from flexops.logic import (
    add_conditional,
    add_startup_delay,
    add_startup_shutdown,
    add_status,
    register_parallel_group,
)
from flexops.logic.status import RollingStateKind
from flexops.testing import dummy_time_block
from flexops.unit_models import Pump

_N = 6


def _con_satisfied(condata, tol: float = 1e-9) -> bool:
    """Whether a constraint body lies within its (lower, upper) bounds."""
    body = pyo.value(condata.body)
    lower, upper = condata.lower, condata.upper
    ok = True
    if lower is not None:
        ok = ok and pyo.value(lower) <= body + tol
    if upper is not None:
        ok = ok and body <= pyo.value(upper) + tol
    return ok


def _unit_with_transitions(**kwargs):
    """Build a 6-step Pump with status + startup/shutdown attached."""
    m = dummy_time_block(_N)
    m.unit = Pump(property_package=m.properties)
    status = add_status(m.unit, m.unit.power_electrical, 0.0, 100.0)
    startup, shutdown = add_startup_shutdown(m.unit, status, **kwargs)
    return m, m.unit, status, startup, shutdown


def _fix_transition_schedule(status, startup, shutdown, schedule):
    """Fix status to schedule; derive startup/shutdown via an independent reference."""
    for t in range(len(schedule)):
        status[t].set_value(schedule[t])
    for t in range(1, len(schedule)):
        ref_startup = 1 if (schedule[t - 1] == 0 and schedule[t] == 1) else 0
        ref_shutdown = 1 if (schedule[t - 1] == 1 and schedule[t] == 0) else 0
        startup[t].set_value(ref_startup)
        shutdown[t].set_value(ref_shutdown)


def _min_uptime_feasible(schedule: tuple, k: int) -> bool:
    """Reference: every run of 1s that follows a 0->1 switch is >= k long."""
    n = len(schedule)
    for t in range(1, n):
        if schedule[t - 1] == 0 and schedule[t] == 1:
            run_len = 0
            tau = t
            while tau < n and schedule[tau] == 1:
                run_len += 1
                tau += 1
            if run_len < min(k, n - t):
                return False
    return True


def _min_downtime_feasible(schedule: tuple, k: int) -> bool:
    """Reference: every run of 0s that follows a 1->0 switch is >= k long."""
    n = len(schedule)
    for t in range(1, n):
        if schedule[t - 1] == 1 and schedule[t] == 0:
            run_len = 0
            tau = t
            while tau < n and schedule[tau] == 0:
                run_len += 1
                tau += 1
            if run_len < min(k, n - t):
                return False
    return True


@pytest.mark.unit
def test_add_startup_shutdown_returns_time_indexed_binaries():
    """add_startup_shutdown attaches startup[t]/shutdown[t] over t >= 1."""
    m, unit, status, startup, shutdown = _unit_with_transitions()
    expected = set(m.time_block.time_index) - {0}
    assert set(startup.index_set()) == expected
    assert set(shutdown.index_set()) == expected
    for t in expected:
        assert startup[t].domain is pyo.Binary
        assert shutdown[t].domain is pyo.Binary


@pytest.mark.unit
def test_transition_and_default_bound_truth_table():
    """Default (min_uptime=1, min_downtime=1): every schedule satisfies all bodies."""
    m, unit, status, startup, shutdown = _unit_with_transitions()

    for schedule in itertools.product((0, 1), repeat=_N):
        _fix_transition_schedule(status, startup, shutdown, schedule)
        for t in range(1, _N):
            assert _con_satisfied(unit.transition[t])
            assert _con_satisfied(unit.min_uptime[t])
            assert _con_satisfied(unit.min_downtime[t])


@pytest.mark.unit
@pytest.mark.parametrize("k", [2, 3])
def test_min_uptime_truth_table(k):
    """All 64 schedules: min_uptime body feasibility == pure-Python reference."""
    m, unit, status, startup, shutdown = _unit_with_transitions(min_uptime=k)

    for schedule in itertools.product((0, 1), repeat=_N):
        _fix_transition_schedule(status, startup, shutdown, schedule)
        bodies_ok = all(_con_satisfied(c) for c in unit.min_uptime.values())
        assert bodies_ok == _min_uptime_feasible(schedule, k), schedule


@pytest.mark.unit
@pytest.mark.parametrize("k", [2, 3])
def test_min_downtime_truth_table(k):
    """All 64 schedules: min_downtime body feasibility == pure-Python reference."""
    m, unit, status, startup, shutdown = _unit_with_transitions(min_downtime=k)

    for schedule in itertools.product((0, 1), repeat=_N):
        _fix_transition_schedule(status, startup, shutdown, schedule)
        bodies_ok = all(_con_satisfied(c) for c in unit.min_downtime.values())
        assert bodies_ok == _min_downtime_feasible(schedule, k), schedule


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [{"min_uptime": 0}, {"min_downtime": 0}, {"min_uptime": -1}, {"min_downtime": -2}],
)
def test_bad_min_uptime_downtime_raise(kwargs):
    """min_uptime/min_downtime < 1 raises FlexConfigError."""
    m = dummy_time_block(_N)
    m.unit = Pump(property_package=m.properties)
    status = add_status(m.unit, m.unit.power_electrical, 0.0, 100.0)
    with pytest.raises(FlexConfigError):
        add_startup_shutdown(m.unit, status, **kwargs)


@pytest.mark.unit
def test_rolling_state_registered_only_when_dwell_requested():
    """min_uptime/min_downtime > 1 registers rolling state; the k=1 default does not."""
    _, default_unit, _, _, _ = _unit_with_transitions()
    assert getattr(default_unit, "_flexops_rolling_state", []) == []

    _, dwell_unit, _, startup, shutdown = _unit_with_transitions(
        min_uptime=3, min_downtime=2
    )
    entries = dwell_unit._flexops_rolling_state
    kinds = {e["kind"] for e in entries}
    assert RollingStateKind.MIN_UPTIME in kinds
    assert RollingStateKind.MIN_DOWNTIME in kinds

    uptime_entry = next(e for e in entries if e["kind"] == RollingStateKind.MIN_UPTIME)
    assert uptime_entry["k"] == 3
    assert uptime_entry["var"] is startup

    downtime_entry = next(
        e for e in entries if e["kind"] == RollingStateKind.MIN_DOWNTIME
    )
    assert downtime_entry["k"] == 2
    assert downtime_entry["var"] is shutdown


def _counted_unit(max_count: int):
    """Build a 6-step Pump whose status counts up to ``max_count`` trains."""
    m = dummy_time_block(_N)
    m.unit = Pump(property_package=m.properties)
    status = add_status(
        m.unit, m.unit.power_electrical, 0.0, 100.0, max_count=max_count
    )
    startup, shutdown = add_startup_shutdown(m.unit, status, min_downtime=2)
    return m, m.unit, status, startup, shutdown


@pytest.mark.unit
def test_counted_transitions_are_bounded_integers():
    """startup/shutdown count trains: integers in 0..n."""
    _m, _unit, _status, startup, shutdown = _counted_unit(3)
    for var in (startup, shutdown):
        for t in var:
            assert var[t].domain is pyo.NonNegativeIntegers
            assert var[t].bounds == (0, 3)


@pytest.mark.unit
def test_min_downtime_uses_the_train_count():
    """At n = 3 the right side is 3 - status, so two trains running is feasible."""
    _m, unit, status, _startup, shutdown = _counted_unit(3)

    # Two trains running, one stopped at t = 1: needs 1 <= 3 - 2.
    status[1].set_value(2)
    shutdown[1].set_value(1)
    assert _con_satisfied(unit.min_downtime[1])

    # All three running cannot also have just stopped one: 1 <= 3 - 3 fails.
    status[1].set_value(3)
    assert not _con_satisfied(unit.min_downtime[1])


@pytest.mark.unit
def test_binary_unit_commitment_expressions_are_unchanged():
    """At the default max_count the rolling-sum constraints read as before."""
    _m, unit, _status, _startup, _shutdown = _unit_with_transitions(min_downtime=2)

    assert str(unit.min_downtime[2].expr) == (
        "unit.shutdown[1] + unit.shutdown[2]  <=  1 - unit.status[2]"
    )
    assert unit.startup[1].domain is pyo.Binary


@pytest.mark.unit
def test_counted_transition_identity():
    """The count change between steps equals trains started minus stopped."""
    _m, unit, status, startup, shutdown = _counted_unit(3)
    status[0].set_value(1)
    status[1].set_value(3)
    startup[1].set_value(2)
    shutdown[1].set_value(0)
    assert _con_satisfied(unit.transition[1])


def _two_units():
    """Two pumps with a status each, the second counting three trains."""
    m = dummy_time_block(_N)
    m.a = Pump(property_package=m.properties)
    m.b = Pump(property_package=m.properties)
    add_status(m.a, m.a.power_electrical, 0.0, 100.0)
    add_status(m.b, m.b.power_electrical, 0.0, 100.0, max_count=3)
    return m


@pytest.mark.unit
@pytest.mark.parametrize(
    "attach",
    [
        lambda m: add_conditional(m.a, m.b),
        lambda m: add_conditional(m.b, m.a, then="off"),
        lambda m: add_startup_delay(m.a, m.b, k=1),
        lambda m: add_startup_delay(m.b, m.a, k=1),
        lambda m: register_parallel_group([m.a, m.b]),
    ],
    ids=["conditional", "conditional_off", "delay_upstream", "delay_unit", "group"],
)
def test_cross_unit_logic_refuses_counted_units(attach):
    """Helpers that read status as on/off raise instead of building a wrong model."""
    m = _two_units()

    with pytest.raises(FlexConfigError, match="resource balance") as excinfo:
        attach(m)

    assert excinfo.value.field == "max_count"
    assert excinfo.value.value == 3


@pytest.mark.unit
def test_parallel_group_without_status_ordering_accepts_counted_units():
    """order_status=False never compares statuses, so a count is harmless."""
    m = _two_units()
    m.a.recovery = pyo.Var(m.time_block.time_index)
    m.b.recovery = pyo.Var(m.time_block.time_index)

    register_parallel_group([m.a, m.b], variables=["recovery"], order_status=False)


@pytest.mark.component
@pytest.mark.needs_highs
def test_counted_unit_matches_three_ordered_trains():
    """One counted unit reaches the objective of three identical, ordered units."""
    from flexcore.solvers import get_solver

    demand = [30.0, 150.0, 250.0, 250.0, 120.0, 40.0, 0.0, 260.0]
    startup_cost, running_cost = 20.0, 5.0

    def solve(build_units) -> float:
        m = dummy_time_block(len(demand))
        units = build_units(m)
        steps = list(m.time_block.time_index)

        @m.Constraint(steps)
        def meet_demand(b, t):
            return sum(u.power_electrical[t] for u in units) >= demand[t]

        @m.Objective(sense=pyo.minimize)
        def cost(b):
            return sum(
                running_cost * u.status[t] + startup_cost * u.startup[t]
                for u in units
                for t in steps
                if t >= 1
            )

        results = get_solver(model=m, prefer="highs").solve(m)
        pyo.assert_optimal_termination(results)
        return pyo.value(m.cost)

    def three_trains(m):
        units = []
        for name in ("t0", "t1", "t2"):
            unit = Pump(property_package=m.properties)
            m.add_component(name, unit)
            status = add_status(unit, unit.power_electrical, 10.0, 100.0)
            add_startup_shutdown(unit, status)
            units.append(unit)
        register_parallel_group(units)
        return units

    def counted(m):
        unit = Pump(property_package=m.properties)
        m.add_component("trains", unit)
        status = add_status(unit, unit.power_electrical, 10.0, 100.0, max_count=3)
        add_startup_shutdown(unit, status)
        return [unit]

    assert solve(counted) == pytest.approx(solve(three_trains))


@pytest.mark.component
@pytest.mark.needs_highs
def test_counted_min_uptime_holds_in_a_solve():
    """Two trains started for one step keep running for min_uptime steps."""
    from flexcore.solvers import get_solver

    m = dummy_time_block(_N)
    m.unit = Pump(property_package=m.properties)
    status = add_status(m.unit, m.unit.power_electrical, 0.0, 100.0, max_count=2)
    add_startup_shutdown(m.unit, status, min_uptime=3)
    status[0].fix(0)
    demand = [0.0, 150.0, 0.0, 0.0, 0.0, 0.0]

    @m.Constraint(m.time_block.time_index)
    def meet_demand(b, t):
        return m.unit.power_electrical[t] >= demand[t]

    m.cost = pyo.Objective(expr=sum(status[t] for t in status))
    pyo.assert_optimal_termination(get_solver(model=m, prefer="highs").solve(m))

    assert [round(pyo.value(status[t])) for t in status] == [0, 2, 2, 2, 0, 0]
