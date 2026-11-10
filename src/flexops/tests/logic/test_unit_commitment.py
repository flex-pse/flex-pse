"""Unit-commitment transition + min-uptime/downtime truth table (M08, §5)."""

import itertools

import pyomo.environ as pyo
import pytest

from flexcore.exceptions import FlexConfigError
from flexops.logic import add_startup_shutdown, add_status
from flexops.logic.status import RollingStateKind, relax, unrelax
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
def test_add_startup_shutdown_returns_time_indexed_unit_interval_vars():
    """startup[t]/shutdown[t] over t >= 1 are continuous on [0, 1], not Binary."""
    m, unit, status, startup, shutdown = _unit_with_transitions()
    expected = set(m.time_block.time_index) - {0}
    assert set(startup.index_set()) == expected
    assert set(shutdown.index_set()) == expected
    for t in expected:
        assert startup[t].domain is pyo.UnitInterval
        assert shutdown[t].domain is pyo.UnitInterval


@pytest.mark.unit
def test_relax_unrelax_leave_startup_shutdown_continuous():
    """Only status is a tracked binary: unrelax must not make startup Binary."""
    m, unit, status, startup, shutdown = _unit_with_transitions(min_uptime=2)
    relax(unit)
    unrelax(unit)
    for t in startup:
        assert status[t].domain is pyo.Binary
        assert startup[t].domain is pyo.UnitInterval
        assert shutdown[t].domain is pyo.UnitInterval


@pytest.mark.component
@pytest.mark.needs_highs
@pytest.mark.parametrize("kwargs", [{}, {"min_uptime": 3, "min_downtime": 2}])
def test_integral_status_forces_integral_startup_shutdown(kwargs):
    """Every feasible status schedule pins startup/shutdown to the 0/1 edges.

    With status fixed, transition gives startup - shutdown == delta and
    startup >= 0 gives shutdown >= max(0, -delta) at every t, so if the LP
    maximum of sum(startup + shutdown) equals the edge count, every
    shutdown[t] (hence startup[t]) is at its integral reference value.
    """
    from flexcore.solvers import get_solver

    m, unit, status, startup, shutdown = _unit_with_transitions(**kwargs)
    m.obj = pyo.Objective(
        expr=sum(startup[t] + shutdown[t] for t in startup), sense=pyo.maximize
    )
    up, down = kwargs.get("min_uptime", 1), kwargs.get("min_downtime", 1)
    checked = 0
    for schedule in itertools.product((0, 1), repeat=_N):
        if not (
            _min_uptime_feasible(schedule, up)
            and _min_downtime_feasible(schedule, down)
        ):
            continue
        for t, value in enumerate(schedule):
            status[t].fix(value)
        results = get_solver(model=m, prefer="highs").solve(m)
        assert results.solver.termination_condition == pyo.TerminationCondition.optimal
        edges = sum(a != b for a, b in itertools.pairwise(schedule))
        assert pyo.value(m.obj) == pytest.approx(edges), schedule
        for t in startup:
            ref_up = int(schedule[t - 1] == 0 and schedule[t] == 1)
            assert startup[t].value == pytest.approx(ref_up), (schedule, t)
        checked += 1
    assert checked > 0


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
