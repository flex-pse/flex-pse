"""Tests for demand-response revenue: the eeco DR bridge and FlexCosting's DR leg."""

import math

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest
from eeco import demandresponse as eeco_dr
from pyomo.environ import units as pyunits
from pyomo.network import Arc
from pyomo.opt import assert_optimal_termination
from pyomo.util.calc_var_value import calculate_variable_from_constraint

from flexcore.config.schema import DRBaseline, DRConfig, DRPayment, HistoricalPower
from flexops.core.time_block import TimeBlock
from flexops.costing import (
    FlexCosting,
    currency_units,
    dr_baseline_method,
    dr_payment_structure,
    evaluate_dr_revenue,
    historical_power_series,
)
from flexops.properties.simple_aqueous import SimpleAqueousFlow
from flexops.unit_models import Pump, Tank

# The event runs 16:00-19:00 on the one-day horizon 2025-07-08.
_EVENT_HOURS = (16, 17, 18)
# Pays linearly up to full delivery, then flat.
_REGIONS = [
    {"x1": 0.0, "x2": 1.0, "y1": 0.0, "y2": 1.0},
    {"x1": 1.0, "x2": None, "y1": 1.0, "y2": 1.0},
]


def _dr_config(baseline: dict, *, bid_kw=100.0, history_kw=100.0, **event) -> DRConfig:
    """Return a one-event DR config over hourly history from 2025-06-16 to 07-07."""
    history_days = pd.bdate_range("2025-06-16", "2025-07-07")
    return DRConfig.model_validate(
        {
            "events": [
                {
                    "event_date": "2025-07-08",
                    "start_hour": 16,
                    "duration_hours": 3,
                    "baseline_days": [d.date().isoformat() for d in history_days],
                    "bid_capacity_kW": bid_kw,
                    "capacity_price": 10.0,
                    **event,
                }
            ],
            "historical_power": {
                "start": "2025-06-16T00:00",
                "time_step_minutes": 60,
                "values_kW": [history_kw] * (22 * 24),
            },
            "baseline": baseline,
            "payment": {"kind": "capacity", "regions": _REGIONS},
        }
    )


_FIXED_LEVEL = {"method": "fixed_level", "firm_level_kW": 150.0}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("spec", "cls"),
    [
        ({"method": "similar_days"}, eeco_dr.SimilarDaysBaseline),
        ({"method": "top_usage_days"}, eeco_dr.TopUsageDaysBaseline),
        (_FIXED_LEVEL, eeco_dr.FixedLevelBaseline),
        ({"method": "unilateral_interruption"}, eeco_dr.UnilateralInterruptionBaseline),
    ],
)
def test_each_baseline_method_maps_to_its_eeco_class(spec, cls):
    """Every baseline method in the spec builds the matching eeco baseline."""
    assert type(dr_baseline_method(DRBaseline.model_validate(spec))) is cls


@pytest.mark.unit
def test_similar_days_options_pass_through_to_eeco():
    """Similar-days options reach eeco unchanged."""
    baseline = dr_baseline_method(
        DRBaseline(
            method="similar_days",
            n_baseline_days=5,
            adjustment_offset_hours=None,
            holiday_country=None,
            resolution="1h",
        )
    )
    assert baseline.n_baseline_days == 5
    assert baseline.adjustment_offset_hours is None
    assert baseline.holiday_country is None
    assert baseline.resolution == "1h"


@pytest.mark.unit
def test_payment_regions_map_open_bounds_to_infinity():
    """A capacity payment builds eeco's PaymentStructure with open bounds infinite."""
    payment = dr_payment_structure(DRPayment(kind="capacity", regions=_REGIONS))
    assert type(payment) is eeco_dr.PaymentStructure
    assert payment.regions[1]["x2"] == math.inf
    assert payment.settlement == "interval"


@pytest.mark.unit
def test_capacity_energy_payment_carries_its_energy_price():
    """A capacity-plus-energy payment builds eeco's CapacityEnergyPayment."""
    payment = dr_payment_structure(
        DRPayment(kind="capacity_energy", regions=_REGIONS, energy_price=0.5)
    )
    assert type(payment) is eeco_dr.CapacityEnergyPayment
    assert payment.energy_price == 0.5


@pytest.mark.unit
def test_historical_power_series_is_indexed_from_its_start():
    """Meter history becomes a series on a naive, evenly spaced datetime index."""
    series = historical_power_series(
        HistoricalPower(
            start="2025-07-07T00:00", time_step_minutes=15, values_kW=[1.0] * 8
        )
    )
    assert series.index[0] == pd.Timestamp("2025-07-07T00:00")
    assert series.index[-1] == pd.Timestamp("2025-07-07T01:45")
    assert series.index.tz is None


@pytest.mark.unit
def test_evaluate_dr_revenue_matches_a_direct_eeco_settlement():
    """Ex-post settlement equals eeco's itemized revenue on history plus realized."""
    config = _dr_config(_FIXED_LEVEL)
    realized = pd.Series(
        [40.0] * 24, index=pd.date_range("2025-07-08", periods=24, freq="h")
    )

    per_event, total = evaluate_dr_revenue(realized, config)

    power = realized.combine_first(historical_power_series(config.historical_power))
    _, expected = eeco_dr.calculate_itemized_dr_revenue(
        power,
        [config.events[0].model_dump(exclude={"region_x1"})],
        eeco_dr.FixedLevelBaseline(150.0),
        eeco_dr.PaymentStructure(
            [
                {"x1": 0.0, "x2": 1.0, "y1": 0.0, "y2": 1.0},
                {"x1": 1.0, "x2": math.inf, "y1": 1.0, "y2": 1.0},
            ]
        ),
    )
    assert total == pytest.approx(expected)
    assert total == pytest.approx(1000.0)  # full delivery: 10 $/kW x 100 kW
    assert len(per_event) == 1


def _pump_tank(demand_response=None) -> pyo.ConcreteModel:
    """Pump -> Tank at a flat 0.10 USD/kWh, objective = operating cost."""
    m = pyo.ConcreteModel()
    m.time_block = TimeBlock(
        start_date="2025-07-08", end_date="2025-07-09", time_step=1 * pyunits.hr
    )
    m.properties = SimpleAqueousFlow()
    m.costing = FlexCosting(
        time_block=m.time_block,
        energy_prices={"electrical": 0.10 * currency_units("USD") / pyunits.kWh},
        demand_response=demand_response,
    )
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
        m.tank.outlet_state.flow_vol_phase[t, "Liq"].fix(100.0)
        m.pump.inlet_state.flow_vol_phase[t, "Liq"].setlb(0.0)
        m.pump.inlet_state.flow_vol_phase[t, "Liq"].setub(300.0)
    m.costing.cost_process()
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
    m.terminal = pyo.Constraint(expr=m.tank.volume[23] >= 200.0)
    return m


def _solve(m) -> None:
    """Solve ``m`` with HiGHS and require optimality."""
    from flexcore.solvers import get_solver

    assert_optimal_termination(get_solver(model=m, prefer="highs").solve(m))


def _event_power(m) -> list[float]:
    """Return the solved site power in each event hour (kW)."""
    return [pyo.value(m.costing.aggregate_electrical_power[t]) for t in _EVENT_HOURS]


@pytest.mark.unit
def test_no_demand_response_builds_a_zero_dr_line_item():
    """Without DR the opex block carries dr_revenue == 0 and no DR sub-block."""
    m = _pump_tank()
    opex = m.costing.opex
    assert opex.find_component("dr") is None
    opex.dr_revenue.set_value(5.0)
    calculate_variable_from_constraint(opex.dr_revenue, opex.eq_dr_revenue)
    assert pyo.value(opex.dr_revenue) == pytest.approx(0.0)


@pytest.mark.unit
def test_demand_response_accepts_a_plain_dict():
    """demand_response takes the config as plain data and validates it."""
    m = _pump_tank(_dr_config(_FIXED_LEVEL).model_dump(mode="json"))
    assert isinstance(m.costing.config.demand_response, DRConfig)
    assert m.costing.opex.find_component("dr") is not None


@pytest.mark.component
@pytest.mark.needs_highs
def test_fixed_level_dr_moves_load_out_of_the_event_and_is_netted():
    """DR revenue pays to curtail in the event and is subtracted from opex."""
    m = _pump_tank(_dr_config(_FIXED_LEVEL))
    _solve(m)

    opex = m.costing.opex
    assert max(_event_power(m)) <= 50.0 + 1e-6  # 150 kW firm level - 100 kW bid
    assert pyo.value(opex.dr_revenue) == pytest.approx(1000.0, rel=1e-6)
    assert pyo.value(opex.total_operating_cost) == pytest.approx(
        pyo.value(
            opex.electricity_cost
            + opex.fuel_cost
            + opex.fixed_operating_cost
            + opex.scalar_cost
            - opex.dr_revenue
        )
    )
    report = m.costing.report_cost(m)
    assert report.operating.dr_revenue == pytest.approx(1000.0, rel=1e-6)
    assert report.operating.total == pytest.approx(
        report.operating.electricity - report.operating.dr_revenue
    )


@pytest.mark.component
@pytest.mark.needs_highs
def test_fixed_region_keeps_dr_free_of_free_binaries():
    """An event with region_x1 set fixes every region binary and still pays."""
    m = _pump_tank(_dr_config(_FIXED_LEVEL, region_x1=1.0))
    binaries = [
        v
        for v in m.costing.opex.dr.component_data_objects(pyo.Var)
        if v.domain is pyo.Binary
    ]
    assert binaries and all(v.fixed for v in binaries)

    _solve(m)
    assert pyo.value(m.costing.opex.dr_revenue) == pytest.approx(1000.0, rel=1e-6)


@pytest.mark.component
@pytest.mark.needs_highs
def test_similar_days_dr_settles_against_meter_history():
    """A similar-days baseline from history pays and reconciles ex post."""
    baseline = {
        "method": "similar_days",
        "adjustment_offset_hours": None,
        "holiday_country": None,
    }
    m = _pump_tank(_dr_config(baseline, bid_kw=70.0, history_kw=120.0))
    _solve(m)

    assert max(_event_power(m)) <= 50.0 + 1e-6  # 120 kW baseline - 70 kW bid
    in_model = pyo.value(m.costing.opex.dr_revenue)
    assert in_model == pytest.approx(700.0, rel=1e-6)
    assert m.costing.report_cost(m).operating.dr_revenue == pytest.approx(
        in_model, rel=1e-6
    )
    assert np.isfinite(pyo.value(m.objective))
