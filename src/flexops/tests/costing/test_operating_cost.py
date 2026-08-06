"""Golden-bill and in-objective tests for the EECO cost bridge.

``test_golden_monthly_bill`` is the ``unit``-tier truth check: it evaluates a
hand-computed PG&E-B-20-flavored bill on a fixed realized load, no solve. The
remaining tests are ``component`` tier: they build the convex-relaxed
in-objective cost on a toy model, solve the trivial LP with HiGHS, and check the
relaxed proxy against the post-hoc bill, the DR no-op, and LP classification.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits

from flexcore.exceptions import FlexConfigError
from flexcore.solvers import ProblemClass, classify
from flexops.core.time_block import TimeBlock
from flexops.costing import (
    DRConfig,
    add_electricity_cost,
    add_operating_cost,
    evaluate_cost,
    evaluate_fuel_cost,
    load_dr_program,
    load_tariff,
    opex,
)
from flexops.costing.opex import _itemized_electricity_cost

_FIXTURES = Path(__file__).parent.parent / "fixtures"
_TARIFF_JSON = _FIXTURES / "tariff_tou_demo.json"
_DR_JSON = _FIXTURES / "dr_events_demo.json"

_N = 744  # hours in July 2025


def _july_index() -> pd.DatetimeIndex:
    """Hourly July-2025 datetime index (744 stamps)."""
    return pd.date_range("2025-07-01", periods=_N, freq="h")


def _reference_load() -> np.ndarray:
    """Realized aggregate power: 100 kW everywhere, 200 kW at 2025-07-10 03:00."""
    index = _july_index()
    load = np.full(_N, 100.0)
    load[index.get_loc(pd.Timestamp("2025-07-10 03:00"))] = 200.0
    return load


def _line_item(itemized_total: dict, substr: str) -> float:
    """Sum the by-charge-key costs whose key contains ``substr``."""
    return float(sum(v for k, v in itemized_total.items() if substr in k))


def _build_toy_model(load: np.ndarray) -> pyo.ConcreteModel:
    """A minimal ConcreteModel carrying the fixed kW load as a Var (no FlexOps)."""
    m = pyo.ConcreteModel()
    m.step = pyo.RangeSet(0, len(load) - 1)
    m.agg = pyo.Var(m.step, initialize=100.0, bounds=(0, None))
    for i in m.step:
        m.agg[i].fix(float(load[i]))
    return m


@pytest.mark.unit
def test_rate_data_columns_track_eeco_constants():
    """opex's column/utility names are sourced from ``eeco.costs`` constants.

    Guards against an upstream EECO column rename silently diverging from the
    literals flex-pse validates against: the required-column and charge-column
    tuples must equal the corresponding ``eeco.costs`` string constants, and the
    utility sentinels must equal EECO's.
    """
    from eeco import costs as eeco_costs

    assert opex._REQUIRED_COLUMNS == (
        eeco_costs.UTILITY,
        eeco_costs.TYPE,
        eeco_costs.NAME,
        eeco_costs.MONTH_START,
        eeco_costs.MONTH_END,
        eeco_costs.WEEKDAY_START,
        eeco_costs.WEEKDAY_END,
        eeco_costs.HOUR_START,
        eeco_costs.HOUR_END,
    )
    assert opex._CHARGE_COLUMNS == (
        eeco_costs.CHARGE_METRIC,
        eeco_costs.CHARGE_IMPERIAL,
        eeco_costs.CHARGE,
    )
    assert opex._ELECTRIC == eeco_costs.ELECTRIC
    assert opex._GAS == eeco_costs.GAS


@pytest.mark.unit
def test_golden_monthly_bill():
    """`evaluate_cost` reproduces every line item and the 14,085.00 total."""
    tariff = load_tariff(_TARIFF_JSON)
    load = _reference_load()

    index = _july_index()
    total = evaluate_cost(load, tariff, dt_hours=1.0, time_index=index)
    assert total == pytest.approx(14085.00, abs=0.005)

    itemized = _itemized_electricity_cost(load, tariff, dt_hours=1.0, time_index=index)[
        "total"
    ]
    assert _line_item(itemized, "energy_peak_") == pytest.approx(2070.00, abs=0.005)
    assert _line_item(itemized, "energy_offpeak_") == pytest.approx(5670.00, abs=0.005)
    assert _line_item(itemized, "energy_tier2_") == pytest.approx(245.00, abs=0.005)
    assert _line_item(itemized, "demand_peak-demand_") == pytest.approx(
        2150.00, abs=0.005
    )
    assert _line_item(itemized, "demand_anytime-demand_") == pytest.approx(
        3800.00, abs=0.005
    )
    assert _line_item(itemized, "customer_fixed_") == pytest.approx(150.00, abs=0.005)


@pytest.mark.component
@pytest.mark.needs_highs
def test_relaxed_leq_or_approx_true():
    """The relaxed in-objective total is <= or ~= the post-hoc true bill."""
    from flexcore.solvers import get_solver

    tariff = load_tariff(_TARIFF_JSON)
    load = _reference_load()
    index = _july_index()

    m = _build_toy_model(load)
    handles = add_operating_cost(
        block=m,
        electrical_power=m.agg,
        time_index=index,
        dt_hours=1.0,
        tariff=tariff,
    )
    m.objective = pyo.Objective(expr=handles.total_operating_cost, sense=pyo.minimize)
    get_solver(model=m, prefer="highs").solve(m)

    relaxed = pyo.value(handles.total_operating_cost)
    true_cost = evaluate_cost(load, tariff, dt_hours=1.0, time_index=index)
    assert relaxed <= true_cost + 1e-3


@pytest.mark.component
@pytest.mark.needs_highs
def test_dr_container_is_noop_on_objective():
    """Supplying a loaded DRConfig leaves the in-objective cost unchanged."""
    from flexcore.solvers import get_solver

    tariff = load_tariff(_TARIFF_JSON)
    load = _reference_load()
    index = _july_index()

    def _solve(dr_config):
        m = _build_toy_model(load)
        handles = add_operating_cost(
            block=m,
            electrical_power=m.agg,
            time_index=index,
            dt_hours=1.0,
            tariff=tariff,
            dr_config=dr_config,
        )
        m.objective = pyo.Objective(
            expr=handles.total_operating_cost, sense=pyo.minimize
        )
        get_solver(model=m, prefer="highs").solve(m)
        return pyo.value(handles.total_operating_cost)

    without_dr = _solve(None)
    with_dr = _solve(DRConfig(program=load_dr_program(_DR_JSON)))
    assert with_dr == pytest.approx(without_dr, abs=1e-6)


@pytest.mark.component
@pytest.mark.needs_highs
def test_demand_charge_is_linear():
    """The built in-objective model classifies LP (no max()/nonlinearity)."""
    tariff = load_tariff(_TARIFF_JSON)
    load = _reference_load()
    index = _july_index()

    m = _build_toy_model(load)
    handles = add_operating_cost(
        block=m,
        electrical_power=m.agg,
        time_index=index,
        dt_hours=1.0,
        tariff=tariff,
    )
    m.objective = pyo.Objective(expr=handles.total_operating_cost, sense=pyo.minimize)
    assert classify(m) is ProblemClass.LP


# --------------------------------------------------------------------------- #
# Combined opex block: one block carries both electric and gas utility costs.
# --------------------------------------------------------------------------- #
_N24 = 24


def _flat_two_utility_tariff():
    """A flat (no-tier, no-demand) tariff with one electric and one gas charge."""
    records = [
        {
            "utility": "electric",
            "type": "energy",
            "name": "allday",
            "month_start": 1,
            "month_end": 12,
            "weekday_start": 0,
            "weekday_end": 6,
            "hour_start": 0,
            "hour_end": 24,
            "basic_charge_limit (metric)": 0,
            "charge (metric)": 0.10,
            "units": "$/kWh",
        },
        {
            "utility": "gas",
            "type": "energy",
            "name": "allday",
            "month_start": 1,
            "month_end": 12,
            "weekday_start": 0,
            "weekday_end": 6,
            "hour_start": 0,
            "hour_end": 24,
            "basic_charge_limit (metric)": 0,
            "charge (metric)": 0.50,
            "units": "$/m3",
        },
    ]
    return load_tariff(records)


def _two_utility_model(elec_kw: np.ndarray, gas_flow: np.ndarray) -> pyo.ConcreteModel:
    """A toy block carrying the standard `power_electrical`/`fuel_usage` Vars, fixed."""
    m = pyo.ConcreteModel()
    m.t = pyo.RangeSet(0, _N24 - 1)
    m.power_electrical = pyo.Var(m.t, bounds=(0, None))
    m.fuel_usage = pyo.Var(m.t, bounds=(0, None))
    for i in m.t:
        m.power_electrical[i].fix(float(elec_kw[i]))
        m.fuel_usage[i].fix(float(gas_flow[i]))
    return m


@pytest.mark.unit
def test_add_electricity_cost_builds_electric_components():
    """`add_electricity_cost` (the renamed electric builder) builds electric terms."""
    tariff = load_tariff(_TARIFF_JSON)
    m = _build_toy_model(_reference_load())
    handles = add_electricity_cost(
        block=m,
        electrical_power=m.agg,
        time_index=_july_index(),
        dt_hours=1.0,
        tariff=tariff,
    )
    assert "electric" in handles.eeco_block
    assert any(v.name.startswith("electric_") for v in m.component_objects(pyo.Var))


@pytest.mark.component
@pytest.mark.needs_highs
def test_add_operating_cost_combines_electric_and_gas():
    """One opex block carries both utilities; its total is electric + gas."""
    from flexcore.solvers import get_solver

    tariff = _flat_two_utility_tariff()
    elec = np.full(_N24, 100.0)
    gas = np.full(_N24, 10.0)
    index = pd.date_range("2025-07-01", periods=_N24, freq="h")

    m = _two_utility_model(elec, gas)
    handles = add_operating_cost(
        block=m,
        electrical_power=m.power_electrical,
        fuel_power=m.fuel_usage,
        time_index=index,
        dt_hours=1.0,
        tariff=tariff,
    )
    m.objective = pyo.Objective(expr=handles.total_operating_cost, sense=pyo.minimize)
    get_solver(model=m, prefer="highs").solve(m)

    combined = pyo.value(handles.total_operating_cost)
    expected = evaluate_cost(
        elec, tariff, dt_hours=1.0, time_index=index
    ) + evaluate_fuel_cost(gas, tariff, dt_hours=1.0, time_index=index)
    assert set(handles.eeco_block) == {"electric", "gas"}
    assert combined == pytest.approx(expected, abs=1e-3)


@pytest.mark.component
@pytest.mark.needs_highs
def test_add_operating_cost_reads_block_defaults():
    """With no power args, reads block.power_electrical / block.fuel_usage."""
    from flexcore.solvers import get_solver

    tariff = _flat_two_utility_tariff()
    elec = np.full(_N24, 100.0)
    gas = np.full(_N24, 10.0)
    index = pd.date_range("2025-07-01", periods=_N24, freq="h")

    m = _two_utility_model(elec, gas)
    handles = add_operating_cost(block=m, time_index=index, dt_hours=1.0, tariff=tariff)
    m.objective = pyo.Objective(expr=handles.total_operating_cost, sense=pyo.minimize)
    get_solver(model=m, prefer="highs").solve(m)

    combined = pyo.value(handles.total_operating_cost)
    expected = evaluate_cost(
        elec, tariff, dt_hours=1.0, time_index=index
    ) + evaluate_fuel_cost(gas, tariff, dt_hours=1.0, time_index=index)
    assert set(handles.eeco_block) == {"electric", "gas"}
    assert combined == pytest.approx(expected, abs=1e-3)


@pytest.mark.unit
def test_add_operating_cost_requires_a_utility():
    """No power passed and no standard components on the block → FlexConfigError."""
    tariff = _flat_two_utility_tariff()
    index = pd.date_range("2025-07-01", periods=_N24, freq="h")
    m = pyo.ConcreteModel()
    with pytest.raises(FlexConfigError, match="power_electrical"):
        add_operating_cost(block=m, time_index=index, dt_hours=1.0, tariff=tariff)


@pytest.mark.unit
def test_wrapper_block_t_matches_time_block_time_index():
    """The injected `block.t` EECO iterates aligns with the TimeBlock's time_index.

    Makes sure the power Var indexed on`time_block.time_index` is consumed correctly
    by EECO under `block.t`.
    """
    tariff = load_tariff(_TARIFF_JSON)
    m = pyo.ConcreteModel()
    m.time_block = TimeBlock(
        start_date="2025-01-01", end_date="2025-01-02", time_step=15 * pyunits.min
    )
    tb = m.time_block
    m.power_electrical = pyo.Var(tb.time_index, initialize=0.0, units=pyunits.kW)
    for i in tb.time_index:
        m.power_electrical[i].fix(100.0)
    dt_hours = pyo.value(pyunits.convert(tb.dt, pyunits.hr))

    assert not hasattr(m, "t")  # TimeBlock exposes `time_index`, not `t`
    add_operating_cost(
        block=m,
        electrical_power=m.power_electrical,
        time_index=tb.datetime_index,
        dt_hours=dt_hours,
        tariff=tariff,
    )
    # Three-way alignment: injected step set == TimeBlock time set == Var domain,
    # and all N match the datetime stamp count.
    assert list(m.t) == list(tb.time_index)
    assert set(m.t) == set(m.power_electrical.index_set())
    assert len(m.t) == len(tb.datetime_index)


@pytest.mark.unit
def test_wrapper_preserves_existing_block_t():
    """When the block already defines `t`, the wrapper reuses it, not re-creates it."""
    tariff = load_tariff(_TARIFF_JSON)
    index = _july_index()
    m = _build_toy_model(_reference_load())
    m.t = pyo.RangeSet(0, len(index) - 1)
    existing = m.t
    add_operating_cost(
        block=m,
        electrical_power=m.agg,
        time_index=index,
        dt_hours=1.0,
        tariff=tariff,
    )
    assert m.t is existing  # guard did not overwrite it


# --------------------------------------------------------------------------- #
# Sub-monthly prorating of monthly-assessed charges
# --------------------------------------------------------------------------- #
def _one_day_index():
    """Hourly index over a single summer weekday (2025-07-08, a Tuesday)."""
    return pd.date_range("2025-07-08", periods=24, freq="h")


# The demo tariff's monthly-assessed charges, at a flat 100 kW draw:
# $150/month customer + ($21.50 + $19.00)/kW demand.
_MONTHLY_ASSESSED_AT_100KW = 150.0 + (21.5 + 19.0) * 100.0
# One summer weekday's energy at 100 kW: 5 peak hours @ $0.18, 19 off-peak @ $0.09.
_ONE_DAY_ENERGY_AT_100KW = 100.0 * (5 * 0.18 + 19 * 0.09)


@pytest.mark.unit
def test_prorate_false_bills_full_monthly_charges():
    """prorate=False bills the whole monthly demand + fixed charge on one day."""
    tariff = load_tariff(_TARIFF_JSON)
    load = np.full(24, 100.0)
    total = evaluate_cost(
        load, tariff, dt_hours=1.0, time_index=_one_day_index(), prorate=False
    )
    assert total == pytest.approx(
        _ONE_DAY_ENERGY_AT_100KW + _MONTHLY_ASSESSED_AT_100KW, abs=0.005
    )


@pytest.mark.unit
def test_prorate_scales_only_the_monthly_assessed_charges():
    """Prorating scales demand + fixed by days/month and leaves energy alone."""
    tariff = load_tariff(_TARIFF_JSON)
    load = np.full(24, 100.0)
    scale = 24.0 / 744.0  # one day of a 31-day month
    total = evaluate_cost(
        load, tariff, dt_hours=1.0, time_index=_one_day_index(), prorate=True
    )
    assert total == pytest.approx(
        _ONE_DAY_ENERGY_AT_100KW + _MONTHLY_ASSESSED_AT_100KW * scale, abs=0.005
    )


@pytest.mark.unit
def test_prorate_is_a_noop_on_a_full_month():
    """A full-month horizon covers the whole month, so nothing is scaled."""
    tariff = load_tariff(_TARIFF_JSON)
    load = _reference_load()
    index = _july_index()
    prorated = evaluate_cost(load, tariff, dt_hours=1.0, time_index=index, prorate=True)
    assert prorated == pytest.approx(14085.00, abs=0.005)


@pytest.mark.unit
def test_daily_assessed_demand_charge_is_not_prorated():
    """A demand charge assessed per day is already horizon-scaled; leave it be."""
    tariff = load_tariff(_TARIFF_JSON)
    # Reassess the anytime demand charge daily; the peak one stays monthly.
    tariff = tariff.copy()
    tariff["assessed"] = "monthly"
    tariff.loc[tariff["name"] == "anytime-demand", "assessed"] = "daily"
    load = np.full(24, 100.0)
    scale = 24.0 / 744.0

    total = evaluate_cost(
        load, tariff, dt_hours=1.0, time_index=_one_day_index(), prorate=True
    )
    # Daily anytime demand bills in full ($19/kW x 100 kW); the monthly peak
    # demand and the customer charge are prorated.
    expected = _ONE_DAY_ENERGY_AT_100KW + 19.0 * 100.0 + (150.0 + 21.5 * 100.0) * scale
    assert total == pytest.approx(expected, abs=0.005)
