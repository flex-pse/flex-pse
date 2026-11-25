"""A time point's timestamp, its rates, and EECO's tariff windows line up.

flex-pse labels each time point with the timestamp at the **start** of the
step it covers: point ``i`` is ``start_date + i * dt`` and covers
``[start_date + i * dt, start_date + (i + 1) * dt)``. A rate at point ``i``
(power, flow) is that step's average, and a state at point ``i`` (tank volume,
battery charge) is the level at the step's end.

EECO uses the same convention. ``create_charge_array`` puts a timestamp in a
charge window when ``hour_start <= hour < hour_end``. So with a 16-21 peak, the
step stamped 16:00 (16:00-17:00) is peak, the step stamped 20:00 (20:00-21:00)
is peak, and the step stamped 21:00 (21:00-22:00) is not.

These tests pin that agreement. If either side switched to stamping a step by
its end, the step stamped 21:00 would cover 20:00-21:00, which is still peak,
but would be billed off-peak, and the 16:00 step would be billed peak for an
hour (15:00-16:00) that is off-peak.
"""

import pandas as pd
import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits
from pyomo.network import Arc
from pyomo.opt import assert_optimal_termination

from flexops.core.time_block import TimeBlock
from flexops.costing import FlexCosting, evaluate_cost, load_tariff, price_series
from flexops.costing import opex as opex_mod
from flexops.properties.simple_aqueous import SimpleAqueousFlow
from flexops.unit_models import Pump, Tank

_PEAK = 0.18
_OFF_PEAK = 0.09
_DAY = "2025-07-08"  # a Tuesday in July, inside the tariff's weekday summer peak


def _energy_only_tariff():
    """A tariff with only energy charges: peak 16:00-21:00, off-peak otherwise.

    No demand or customer charge, so a load's bill is exactly its energy times
    the price of the window its step falls in.
    """

    def charge(name, hour_start, hour_end, price):
        return {
            "utility": "electric",
            "type": "energy",
            "name": name,
            "month_start": 1,
            "month_end": 12,
            "weekday_start": 0,
            "weekday_end": 6,
            "hour_start": hour_start,
            "hour_end": hour_end,
            "basic_charge_limit (metric)": 0,
            "charge (metric)": price,
            "units": "$/kWh",
        }

    return load_tariff(
        [
            charge("offpeak-morning", 0, 16, _OFF_PEAK),
            charge("peak", 16, 21, _PEAK),
            charge("offpeak-evening", 21, 24, _OFF_PEAK),
        ]
    )


def _time_block(minutes: int) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel()
    m.time_block = TimeBlock(
        start_date=_DAY, end_date="2025-07-09", time_step=minutes * pyunits.min
    )
    return m


@pytest.mark.unit
@pytest.mark.parametrize("minutes", [60, 15])
def test_each_time_point_is_stamped_with_the_start_of_its_step(minutes):
    """Point i is start_date + i*dt; the last point starts one step before end."""
    tb = _time_block(minutes).time_block
    step = pd.Timedelta(minutes=minutes)

    assert tb.datetime_index[0] == pd.Timestamp(_DAY)
    assert tb.datetime_index[-1] == pd.Timestamp("2025-07-09") - step
    assert tb.index_of(f"{_DAY} 16:00") == 16 * 60 // minutes


@pytest.mark.unit
@pytest.mark.parametrize("minutes", [60, 15])
def test_flexcosting_hands_eeco_the_time_blocks_own_stamps(minutes, monkeypatch):
    """The horizon EECO builds its charge arrays on is the TimeBlock's index.

    FlexCosting passes EECO a start, an exclusive end, and a resolution, and
    EECO rebuilds the timestamps from them. Those must be exactly
    ``datetime_index``, so position ``t`` of every charge array prices
    ``power[t]``.
    """
    eeco_costs = opex_mod._eeco_costs
    calls = []
    original = eeco_costs.get_charge_dict

    def capture(start, end, rate_data, resolution="15m"):
        calls.append((start, end, resolution))
        return original(start, end, rate_data, resolution=resolution)

    monkeypatch.setattr(eeco_costs, "get_charge_dict", capture)

    m = _time_block(minutes)
    m.properties = SimpleAqueousFlow()
    m.costing = FlexCosting(time_block=m.time_block, tariff=_energy_only_tariff())
    m.pump = Pump(
        property_package=m.properties,
        energy_intensity=0.5 * pyunits.kWh / pyunits.m**3,
        costing_package=m.costing,
    )
    m.costing.cost_process()

    assert calls, "FlexCosting never asked EECO for its charge arrays"
    for start, end, resolution in calls:
        _, eeco_stamps = eeco_costs.get_timesteps(start, end, resolution)
        assert list(eeco_stamps["DateTime"]) == list(m.time_block.datetime_index)


@pytest.mark.unit
@pytest.mark.parametrize("minutes", [60, 15])
def test_the_peak_window_covers_the_steps_that_start_inside_it(minutes):
    """Steps stamped 16:00 to 20:59 are peak; 15:xx and 21:00 are not."""
    index = _time_block(minutes).time_block.datetime_index
    prices = price_series(_energy_only_tariff(), index)

    step = pd.Timedelta(minutes=minutes)
    peak_start = pd.Timestamp(f"{_DAY} 16:00")
    peak_end = pd.Timestamp(f"{_DAY} 21:00")
    assert prices[peak_start - step] == pytest.approx(_OFF_PEAK)
    assert prices[peak_start] == pytest.approx(_PEAK)
    assert prices[peak_end - step] == pytest.approx(_PEAK)
    assert prices[peak_end] == pytest.approx(_OFF_PEAK)
    in_peak = (index >= peak_start) & (index < peak_end)
    assert int(in_peak.sum()) == 5 * 60 // minutes


@pytest.mark.unit
@pytest.mark.parametrize(
    ("stamp", "price"),
    [
        ("15:00", _OFF_PEAK),  # 15:00-16:00
        ("16:00", _PEAK),  # 16:00-17:00
        ("20:00", _PEAK),  # 20:00-21:00
        ("21:00", _OFF_PEAK),  # 21:00-22:00
    ],
)
def test_a_load_is_billed_at_the_price_of_the_step_it_starts(stamp, price):
    """The post-hoc bill of a 10 kW, one-hour load at one step."""
    tb = _time_block(60).time_block
    power = [0.0] * tb.n_points
    power[tb.index_of(f"{_DAY} {stamp}")] = 10.0

    bill = evaluate_cost(
        power, _energy_only_tariff(), 1.0, time_index=tb.datetime_index
    )

    assert bill == pytest.approx(10.0 * price)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("stamp", "billed"),
    [("15:00", False), ("16:00", True), ("20:00", True), ("21:00", False)],
)
def test_a_peak_demand_charge_counts_the_steps_that_start_in_its_window(stamp, billed):
    """Demand charges use the same window rule as energy charges.

    A $10/kW peak demand charge over 16:00-21:00 and no other charge: a 10 kW,
    one-hour load sets the peak demand only if its step starts in the window.
    """
    tariff = load_tariff(
        [
            {
                "utility": "electric",
                "type": "demand",
                "name": "peak-demand",
                "month_start": 1,
                "month_end": 12,
                "weekday_start": 0,
                "weekday_end": 6,
                "hour_start": 16,
                "hour_end": 21,
                "basic_charge_limit (metric)": 0,
                "charge (metric)": 10.0,
                "units": "$/kW",
            }
        ]
    )
    tb = _time_block(60).time_block
    power = [0.0] * tb.n_points
    power[tb.index_of(f"{_DAY} {stamp}")] = 10.0

    bill = evaluate_cost(
        power, tariff, 1.0, time_index=tb.datetime_index, prorate=False
    )

    assert bill == pytest.approx(100.0 if billed else 0.0)


def _fixed_pumping_model(step: int) -> pyo.ConcreteModel:
    """Pump 100 m^3/hr into a tank during one hour only, all flows fixed."""
    m = _time_block(60)
    m.properties = SimpleAqueousFlow()
    m.costing = FlexCosting(time_block=m.time_block, tariff=_energy_only_tariff())
    m.pump = Pump(
        property_package=m.properties,
        energy_intensity=0.5 * pyunits.kWh / pyunits.m**3,
        costing_package=m.costing,
    )
    m.tank = Tank(
        property_package=m.properties,
        max_volume=1000 * pyunits.m**3,
        initial_volume=200 * pyunits.m**3,
    )
    m.arc = Arc(source=m.pump.outlet, destination=m.tank.inlet)
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)
    for t in m.time_block.time_index:
        m.pump.inlet_state.flow_vol_phase[t, "Liq"].fix(100.0 if t == step else 0.0)
        m.tank.outlet_state.flow_vol_phase[t, "Liq"].fix(0.0)
    m.costing.cost_process()
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
    return m


@pytest.mark.component
@pytest.mark.needs_highs
@pytest.mark.parametrize(
    ("stamp", "price"),
    [("15:00", _OFF_PEAK), ("16:00", _PEAK), ("20:00", _PEAK), ("21:00", _OFF_PEAK)],
)
def test_the_optimized_cost_prices_a_step_by_the_window_it_starts_in(stamp, price):
    """The objective EECO builds in the model bills the same step the same way.

    100 m^3/hr at 0.5 kWh/m^3 is 50 kW for one hour, 50 kWh. The tank's
    volume at that step is the level after the hour's pumping.
    """
    from flexcore.solvers import get_solver

    step = _time_block(60).time_block.index_of(f"{_DAY} {stamp}")
    m = _fixed_pumping_model(step)
    results = get_solver(model=m, prefer="highs").solve(m)
    assert_optimal_termination(results)

    assert pyo.value(m.objective) == pytest.approx(50.0 * price, rel=1e-6)
    assert pyo.value(m.pump.power_electrical[step]) == pytest.approx(50.0)
    assert pyo.value(m.tank.volume[step]) == pytest.approx(300.0)
    if step > 0:
        assert pyo.value(m.tank.volume[step - 1]) == pytest.approx(200.0)
