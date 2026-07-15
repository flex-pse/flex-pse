"""Tests for TimeBlock: the discrete-time substrate."""

import time

import pandas as pd
import pyomo.environ as pyo
import pytest
from dateutil.relativedelta import relativedelta
from pyomo.environ import units as pyunits

from flexcore.exceptions import FlexConfigError
from flexops.core.time_block import TimeBlock


@pytest.fixture
def tb():
    """A 1-day, 15-minute-resolution TimeBlock (96 points)."""
    m = pyo.ConcreteModel()
    m.time_block = TimeBlock(
        start_date="2025-01-01", end_date="2025-01-02", time_step=15 * pyunits.min
    )
    return m.time_block


@pytest.mark.unit
def test_n_points_and_horizon(tb):
    """96 points at 15 min resolution span exactly 24 hours."""
    assert tb.n_points == 96
    assert pyo.value(pyunits.convert(tb.horizon, pyunits.hr)) == pytest.approx(
        24.0, rel=1e-9
    )


@pytest.mark.unit
def test_time_index_is_integer_set(tb):
    """time_index holds the plain integers 0..N-1, never timestamps."""
    assert list(tb.time_index) == list(range(96))


@pytest.mark.unit
def test_time_param_elapsed(tb):
    """time[i] is the elapsed time i*dt, carrying the user's units."""
    assert set(tb.time.keys()) == set(tb.time_index)
    assert pyo.value(tb.time[0]) == 0.0
    # 15-minute step: point 4 is at 1 hour of elapsed time.
    assert pyo.value(pyunits.convert(tb.time[4], pyunits.hr)) == pytest.approx(
        1.0, rel=1e-9
    )
    assert pyo.value(pyunits.convert(tb.time[95], pyunits.min)) == pytest.approx(
        95 * 15, rel=1e-9
    )
    assert pyunits.get_units(tb.time[1]) == pyunits.min


@pytest.mark.unit
def test_index_roundtrip(tb):
    """timestamp_of/index_of round-trip for interior and boundary indices."""
    for i in (0, 1, 47, 95):
        assert tb.index_of(tb.timestamp_of(i)) == i


@pytest.mark.unit
@pytest.mark.parametrize(
    ("time_step", "end_date", "n_points", "horizon_hr"),
    [
        (5 * pyunits.min, "2025-01-02", 288, 24.0),
        (15 * pyunits.min, "2025-01-02", 96, 24.0),
        (30 * pyunits.min, "2025-01-03", 96, 48.0),
        (1 * pyunits.hr, "2025-01-08", 168, 168.0),
        (24 * pyunits.hr, "2025-01-29", 28, 672.0),
    ],
)
def test_resolutions_and_horizons(time_step, end_date, n_points, horizon_hr):
    """TimeBlock builds correctly across time resolutions and horizon lengths."""
    m = pyo.ConcreteModel()
    m.time_block = TimeBlock(
        start_date="2025-01-01", end_date=end_date, time_step=time_step
    )
    tb = m.time_block
    assert tb.n_points == n_points
    assert pyo.value(pyunits.convert(tb.horizon, pyunits.hr)) == pytest.approx(
        horizon_hr, rel=1e-9
    )
    for i in (0, n_points - 1):
        assert tb.index_of(tb.timestamp_of(i)) == i


@pytest.mark.unit
def test_datetime_index_matches(tb):
    """The datetime index matches a plain pandas date_range."""
    expected = pd.date_range("2025-01-01", periods=96, freq="15min")
    assert tb.datetime_index.equals(expected)


@pytest.mark.unit
def test_dt_units_min_vs_hr_consistent():
    """Equivalent time_steps in different units produce identical blocks."""
    m = pyo.ConcreteModel()
    m.tb_min = TimeBlock(
        start_date="2025-01-01", end_date="2025-01-02", time_step=15 * pyunits.min
    )
    m.tb_hr = TimeBlock(
        start_date="2025-01-01", end_date="2025-01-02", time_step=0.25 * pyunits.hr
    )
    assert m.tb_min.n_points == m.tb_hr.n_points
    assert m.tb_min.datetime_index.equals(m.tb_hr.datetime_index)
    assert pyo.value(pyunits.convert(m.tb_min.dt, pyunits.s)) == pytest.approx(
        pyo.value(pyunits.convert(m.tb_hr.dt, pyunits.s)), rel=1e-9
    )


@pytest.mark.unit
def test_bare_number_time_step_raises():
    """A bare-number time_step raises FlexConfigError mentioning pyunits."""
    m = pyo.ConcreteModel()
    with pytest.raises(FlexConfigError, match="pyunits"):
        m.time_block = TimeBlock(
            start_date="2025-01-01", end_date="2025-01-02", time_step=15
        )


@pytest.mark.unit
def test_off_grid_timestamp_raises(tb):
    """A timestamp that doesn't fall on the grid raises FlexConfigError."""
    with pytest.raises(FlexConfigError, match="2025-01-01 00:07:00"):
        tb.index_of("2025-01-01 00:07:00")


@pytest.mark.unit
def test_out_of_range_raises(tb):
    """Out-of-horizon timestamps/indices raise FlexConfigError."""
    with pytest.raises(FlexConfigError):
        tb.index_of("2026-01-01")
    with pytest.raises(FlexConfigError):
        tb.timestamp_of(9999)


@pytest.mark.unit
def test_non_divisible_dt_raises():
    """A span that isn't an integer multiple of dt raises FlexConfigError."""
    m = pyo.ConcreteModel()
    with pytest.raises(FlexConfigError, match="time_step"):
        m.time_block = TimeBlock(
            start_date="2025-01-01",
            end_date="2025-01-01T00:25:00",
            time_step=15 * pyunits.min,
        )


@pytest.mark.unit
def test_over_one_month_rejected():
    """The one-calendar-month bound is calendar-based, not a fixed 30 days."""
    m = pyo.ConcreteModel()
    with pytest.raises(FlexConfigError, match="one-calendar-month"):
        m.tb_over = TimeBlock(
            start_date="2025-01-01", end_date="2025-02-02", time_step=15 * pyunits.min
        )
    m.tb_exact = TimeBlock(
        start_date="2025-01-01", end_date="2025-02-01", time_step=15 * pyunits.min
    )
    assert m.tb_exact.n_points > 0

    m.tb_feb_exact = TimeBlock(
        start_date="2025-02-01", end_date="2025-03-01", time_step=15 * pyunits.min
    )
    assert m.tb_feb_exact.n_points > 0
    with pytest.raises(FlexConfigError, match="one-calendar-month"):
        m.tb_feb_over = TimeBlock(
            start_date="2025-02-01", end_date="2025-03-02", time_step=15 * pyunits.min
        )


@pytest.mark.unit
def test_max_length_configurable():
    """max_length overrides the default one-calendar-month horizon cap."""
    # A two-month horizon is rejected under the default one-month cap...
    m = pyo.ConcreteModel()
    with pytest.raises(FlexConfigError, match="one-calendar-month"):
        m.tb_default = TimeBlock(
            start_date="2025-01-01", end_date="2025-03-01", time_step=1 * pyunits.hr
        )
    # ...but builds when max_length is widened to two months.
    m.tb_wide = TimeBlock(
        start_date="2025-01-01",
        end_date="2025-03-01",
        time_step=1 * pyunits.hr,
        max_length=relativedelta(months=2),
    )
    assert m.tb_wide.n_points == (31 + 28) * 24

    # A tighter max_length rejects a horizon the default would allow.
    m2 = pyo.ConcreteModel()
    with pytest.raises(FlexConfigError, match="maximum-length"):
        m2.tb_tight = TimeBlock(
            start_date="2025-01-01",
            end_date="2025-01-15",
            time_step=1 * pyunits.hr,
            max_length=relativedelta(weeks=1),
        )


@pytest.mark.unit
def test_coarse_resolution_horizon_builds():
    """A 1-hour resolution horizon builds with the expected point count."""
    m = pyo.ConcreteModel()
    m.time_block = TimeBlock(
        start_date="2025-01-01", end_date="2025-01-15", time_step=1 * pyunits.hr
    )
    assert m.time_block.n_points == 336
    expected = pd.date_range("2025-01-01", periods=336, freq="1h")
    assert m.time_block.datetime_index.equals(expected)


@pytest.mark.unit
def test_fine_resolution_horizon_builds():
    """A 1-minute resolution horizon builds with the expected point count."""
    m = pyo.ConcreteModel()
    m.time_block = TimeBlock(
        start_date="2025-01-01", end_date="2025-01-02", time_step=1 * pyunits.min
    )
    assert m.time_block.n_points == 1440


@pytest.mark.unit
def test_build_speed_worst_case():
    """A full month at 15-min resolution builds in well under 1 second."""
    m = pyo.ConcreteModel()
    start = time.perf_counter()
    m.time_block = TimeBlock(
        start_date="2025-01-01", end_date="2025-02-01", time_step=15 * pyunits.min
    )
    elapsed = time.perf_counter() - start
    assert m.time_block.n_points == 2976
    assert elapsed < 1.0


@pytest.mark.unit
def test_register_initial_state_roundtrip(tb):
    """Registered mutable Params show up in the registry and stay mutable."""
    tb.model().p = pyo.Param(initialize=0.5, mutable=True)
    tb.register_initial_state(tb.model().p)
    assert tb.initial_state_params == (tb.model().p,)

    tb.model().p.set_value(0.75)
    assert tb.initial_state_params[0].value == 0.75

    tb.model().q = pyo.Param(initialize=0.5, mutable=False)
    with pytest.raises(FlexConfigError):
        tb.register_initial_state(tb.model().q)


@pytest.mark.unit
def test_window_metadata(tb):
    """window() returns correct metadata and clips at the horizon."""
    w = tb.window(4, 8)
    assert w.indices == range(4, 12)
    assert w.start_time == tb.timestamp_of(4)
    assert w.end_time == tb.timestamp_of(12)

    w_clip = tb.window(90, 20)
    assert w_clip.indices == range(90, 96)

    with pytest.raises(FlexConfigError):
        tb.window(96, 1)
