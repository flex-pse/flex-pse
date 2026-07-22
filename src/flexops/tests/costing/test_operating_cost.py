"""Golden-bill and in-objective tests for the EECO cost bridge (M06).

``test_golden_monthly_bill`` is the ``unit``-tier truth check: it evaluates a
hand-computed PG&E-B-20-flavored bill on a fixed realized load, no solve. The
remaining tests are ``component`` tier: they build the convex-relaxed
in-objective cost on a toy model, solve the trivial LP with HiGHS, and check the
relaxed proxy against the post-hoc bill, the DR no-op, and LP classification.
"""

from pathlib import Path

import numpy as np
import pyomo.environ as pyo
import pytest

from flexcore.solvers import ProblemClass, classify
from flexops.costing import (
    DRConfig,
    add_operating_cost,
    evaluate_cost,
    load_dr_program,
    load_tariff,
)
from flexops.costing.opex import _itemized_electricity_cost

_FIXTURES = Path(__file__).parent.parent / "fixtures"
_TARIFF_JSON = _FIXTURES / "tariff_tou_demo.json"
_DR_JSON = _FIXTURES / "dr_events_demo.json"

_N = 744  # hours in July 2025


def _july_index():
    """Hourly July-2025 datetime index (744 stamps)."""
    import pandas as pd

    return pd.date_range("2025-07-01", periods=_N, freq="h")


def _reference_load() -> np.ndarray:
    """Realized aggregate power: 100 kW everywhere, 200 kW at 2025-07-10 03:00."""
    index = _july_index()
    load = np.full(_N, 100.0)
    load[index.get_loc(__import__("pandas").Timestamp("2025-07-10 03:00"))] = 200.0
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
