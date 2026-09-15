"""Cross-validation tests for ArimaSurrogate against ArimaRegressor.

These tests require ``flexparameterize`` and validate that the Pyomo
surrogate reproduces the predictions of the fitted regressor, and that
it can be regressed via ipopt standalone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits

from flexops.core.ops_block import OpsBlock
from flexops.core.time_block import TimeBlock
from flexops.properties.simple_aqueous import SimpleAqueousFlow
from flexops.surrogates import ArimaSurrogate
from flexparameterize.regression.arima import ArimaRegressor


def _assert_tail_matches_target(pyomo_opt, target_biogas, *, lookback=10):
    """Assert the optimized tail reaches the in-sample mean target."""
    pyomo_opt = np.asarray(pyomo_opt, dtype=float)
    target = float(target_biogas)
    assert len(pyomo_opt) >= lookback
    tail = pyomo_opt[-lookback:]
    target_tol = 1e-3 * abs(target)
    assert np.all(
        np.abs(tail - target) < target_tol
    ), f"Optimized tail {tail} not within {target_tol:.3e} of target {target:.6f}"


# -- cross-validation: Pyomo vs direct fit -----------------------------------


@pytest.mark.unit
def test_pyomo_matches_direct_fit_for_multiple_arima_orders():
    """Pyomo surrogate matches ArimaRegressor predictions for multiple orders."""
    np.random.seed(42)
    n_train = 100
    n_insample = 20
    n_fcst = 10
    idx = pd.date_range("2024-01-01", periods=n_train, freq="1h")
    feed = pd.Series(np.random.uniform(0.1, 1.0, size=n_train), index=idx, name="feed")

    orders_to_test = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 1),
    ]

    for order in orders_to_test:
        p, d, q = order

        y_values = np.zeros(n_train)
        params = {
            (1, 0, 0): {"phi": 0.5, "const": 0.1},
            (0, 1, 0): {"phi": 0.3, "const": 0.05},
            (0, 0, 1): {"theta": 0.4, "const": 0.2},
            (1, 1, 1): {"phi": 0.4, "theta": 0.3, "const": 0.08},
        }
        param = params.get(order, {})

        for t in range(1, n_train):
            exog = 0.8 * float(feed.iloc[t])
            if d == 0:
                if p == 1 and q == 0:
                    y_values[t] = (
                        param["const"]
                        + param["phi"] * y_values[t - 1]
                        + exog
                        + np.random.normal(0, 0.05)
                    )
                elif p == 0 and q == 1:
                    y_values[t] = param["const"] + exog + np.random.normal(0, 0.05)
                elif p == 1 and q == 1:
                    y_values[t] = (
                        param["const"]
                        + param["phi"] * y_values[t - 1]
                        + exog
                        + param["theta"]
                        * (np.random.normal(0, 0.05) if t == 1 else 0.0)
                        + np.random.normal(0, 0.05)
                    )
            else:
                if p == 1 and q == 1:
                    y_values[t] = (
                        y_values[t - 1]
                        + param["const"]
                        + exog
                        + param["phi"] * (y_values[t - 1] - y_values[t - 2])
                        + np.random.normal(0, 0.05)
                    )

        y = pd.DataFrame({"biogas": y_values}, index=idx)

        regressor = ArimaRegressor(order=order, max_ar_persistence=None).fit(
            pd.DataFrame({"feed": feed}),
            y,
            input_units={"feed": "dimensionless"},
            output_units="m^3/hr",
        )
        assert regressor.fitted is True

        all_exog = np.zeros((n_insample + n_fcst, 1))
        all_exog[:n_insample] = feed.iloc[-n_insample:].values.reshape(-1, 1)
        direct_all = np.asarray(
            regressor.model.predict(
                steps=n_insample + n_fcst,
                exog=all_exog,
                start=n_train - n_insample,
                dynamic=True,
            )
        )
        direct_fcst = direct_all[n_insample:]

        m = pyo.ConcreteModel()
        start_idx = idx[-n_insample]
        m.time_block = TimeBlock(
            start_date=start_idx.strftime("%Y-%m-%dT%H:%M"),
            end_date=(start_idx + pd.Timedelta(hours=n_insample + n_fcst)).strftime(
                "%Y-%m-%dT%H:%M"
            ),
            time_step=1 * pyunits.hr,
        )
        m.props = SimpleAqueousFlow(has_pressure=False)
        m.unit = OpsBlock(property_package=m.props)
        m.unit.add_stream_ports()
        m.unit.add_component(
            "biogas_m3_hour",
            pyo.Var(
                m.time_block.time_index, initialize=0.0, units=pyunits.m**3 / pyunits.hr
            ),
        )
        m.unit.register_io_variable(m.unit.biogas_m3_hour, role="output")
        m.unit.add_component(
            "feed",
            pyo.Var(
                m.time_block.time_index, initialize=0.0, units=pyunits.dimensionless
            ),
        )
        m.unit.register_io_variable(m.unit.feed, role="input")

        m.unit.add_component(
            "biogas_m3_hour_relation",
            pyo.Constraint(
                m.time_block.time_index,
                rule=lambda b, t: pyo.Constraint.Skip,
            ),
        )
        m.unit.register_relation(
            m.unit.biogas_m3_hour_relation, target=m.unit.biogas_m3_hour
        )

        spec = regressor.to_surrogate_spec()
        surrogate = ArimaSurrogate(spec.data)
        m.unit.swap_relation("biogas_m3_hour_relation", surrogate)

        for t in range(n_insample):
            m.unit.feed[t].set_value(float(feed.iloc[-n_insample + t]))
            m.unit.feed[t].fix()
        for t in range(n_fcst):
            m.unit.feed[n_insample + t].set_value(0.0)
            m.unit.feed[n_insample + t].fix()

        for t in range(n_insample + n_fcst):
            m.unit.biogas_m3_hour[t].set_value(float(direct_all[t]))

        m.obj = pyo.Objective(expr=0.0)

        solver = pyo.SolverFactory("ipopt")
        result = solver.solve(m, tee=False)
        assert result.solver.termination_condition == pyo.TerminationCondition.optimal

        pyomo_fcst = np.array(
            [
                float(m.unit.biogas_m3_hour[t].value)
                for t in range(n_insample, n_insample + n_fcst)
            ]
        )

        fcst_rmse = np.sqrt(np.mean((pyomo_fcst - direct_fcst) ** 2))

        print(f"ARIMA{order}: forecast RMSE={fcst_rmse:.6e}")
        assert fcst_rmse < 1e-4, f"ARIMA{order} forecast mismatch: {fcst_rmse}"


# -- reswap ----------------------------------------------------------------


@pytest.mark.unit
def test_reswapping_arima_relation_succeeds_and_uses_latest_coefficients():
    """A second re-fit-and-reswap must not raise and must use latest coefficients."""
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")

    def _fit(phi, const, seed):
        rng = np.random.default_rng(seed)
        y_values = np.zeros(n)
        for t in range(1, n):
            y_values[t] = const + phi * y_values[t - 1] + rng.normal(0, 0.01)
        y = pd.DataFrame({"y": y_values}, index=idx)
        return ArimaRegressor(order=(1, 0, 0), max_ar_persistence=None).fit(
            pd.DataFrame(index=idx),
            y,
            input_units={},
            output_units="m^3/hr",
        )

    regressor1 = _fit(phi=0.3, const=0.1, seed=1)
    regressor2 = _fit(phi=0.6, const=0.4, seed=2)

    m = pyo.ConcreteModel()
    m.time_block = TimeBlock(
        start_date=idx[10].strftime("%Y-%m-%dT%H:%M"),
        end_date=(idx[10] + pd.Timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M"),
        time_step=1 * pyunits.hr,
    )
    m.props = SimpleAqueousFlow(has_pressure=False)
    m.unit = OpsBlock(property_package=m.props)
    m.unit.add_stream_ports()
    m.unit.add_component(
        "y",
        pyo.Var(
            m.time_block.time_index,
            initialize=0.0,
            units=pyunits.m**3 / pyunits.hr,
        ),
    )
    m.unit.register_io_variable(m.unit.y, role="output")
    m.unit.add_component(
        "y_relation",
        pyo.Constraint(m.time_block.time_index, rule=lambda b, t: pyo.Constraint.Skip),
    )
    m.unit.register_relation(m.unit.y_relation, target=m.unit.y)

    spec1 = regressor1.to_surrogate_spec()
    block1 = m.unit.swap_relation("y_relation", ArimaSurrogate(spec1.data))

    fitted_1 = block1.find_component("fitted")
    assert fitted_1 is not None
    assert fitted_1[0].active

    spec2 = regressor2.to_surrogate_spec()
    block2 = m.unit.swap_relation("y_relation", ArimaSurrogate(spec2.data))

    assert not fitted_1[0].active
    fitted_2 = block2.find_component("fitted_2")
    assert fitted_2 is not None
    assert fitted_2[0].active

    m.obj = pyo.Objective(expr=0.0)
    solver = pyo.SolverFactory("ipopt")
    result = solver.solve(m, tee=False)
    assert result.solver.termination_condition == pyo.TerminationCondition.optimal

    coef2 = regressor2.coefficients
    expected_y1 = coef2.get("const", 0.0) + coef2["ar1"] * float(m.unit.y[0].value)
    assert float(m.unit.y[1].value) == pytest.approx(expected_y1, rel=1e-4)


# -- ipopt roundtrip tests --------------------------------------------------


@pytest.mark.component
@pytest.mark.needs_ipopt
@pytest.mark.parametrize(
    "order,auto,auto_kwargs",
    [
        ((1, 0, 0), False, {}),
        ((2, 0, 0), False, {}),
        ((3, 0, 0), False, {}),
        ((0, 0, 1), False, {}),
        ((0, 0, 2), False, {}),
        ((0, 0, 3), False, {}),
        ((1, 0, 1), False, {}),
        ((2, 0, 2), False, {}),
        ((3, 0, 3), False, {}),
        ((0, 1, 0), False, {}),
        ((1, 1, 0), False, {}),
        ((0, 1, 1), False, {}),
        ((1, 1, 1), False, {}),
        ((2, 1, 2), False, {}),
        (None, True, {"max_p": 3, "max_q": 3}),
    ],
    ids=[
        "explicit-ar1",
        "explicit-ar2",
        "explicit-ar3",
        "explicit-ma1",
        "explicit-ma2",
        "explicit-ma3",
        "explicit-ar1-ma1",
        "explicit-ar2-ma2",
        "explicit-ar3-ma3",
        "explicit-i1",
        "explicit-ar1-i1",
        "explicit-i1-ma1",
        "explicit-ar1-i1-ma1",
        "explicit-ar2-i1-ma2",
        "auto",
    ],
)
def test_arima_roundtrip(order, auto, auto_kwargs):
    """Fit, build Pyomo surrogate, optimize exog controls, verify against direct fit."""
    np.random.seed(0)
    n_train = 100
    n_insample = 20
    n_fcst = 10
    n_opt = 100
    n_total = n_insample + n_fcst + n_opt
    idx = pd.date_range("2024-01-01", periods=n_train, freq="1h")

    feed = pd.Series(np.random.uniform(0.1, 1.0, size=n_train), index=idx, name="feed")
    y_values = np.zeros(n_train)
    for t in range(1, n_train):
        y_values[t] = (
            0.1
            + 0.5 * y_values[t - 1]
            + 0.8 * float(feed.iloc[t])
            + np.random.normal(0, 0.05)
        )
    y = pd.DataFrame({"biogas": y_values}, index=idx)
    X = pd.DataFrame({"feed": feed})

    regressor = ArimaRegressor(
        order=order, auto=auto, max_ar_persistence=None, **auto_kwargs
    ).fit(
        X,
        y,
        input_units={"feed": "dimensionless"},
        output_units="m^3/hr",
    )
    assert regressor.fitted is True

    spec = regressor.to_surrogate_spec()

    insample_exog = X.iloc[-n_insample:].values
    forecast_exog = np.zeros((n_fcst, 1))
    all_exog = np.concatenate([insample_exog, forecast_exog])
    sm_all = np.asarray(
        regressor.model.predict(
            steps=n_insample + n_fcst,
            exog=all_exog,
            start=n_train - n_insample,
            dynamic=True,
        )
    )
    direct_insample = sm_all[:n_insample]
    direct_fcst = sm_all[n_insample:]

    mean_feed = float(X["feed"].mean())
    target_biogas = float(y.iloc[-n_insample:]["biogas"].mean())

    m = pyo.ConcreteModel()
    start_idx = idx[n_train - n_insample]
    m.time_block = TimeBlock(
        start_date=start_idx.strftime("%Y-%m-%dT%H:%M"),
        end_date=(start_idx + pd.Timedelta(hours=n_total)).strftime("%Y-%m-%dT%H:%M"),
        time_step=1 * pyunits.hr,
    )
    m.props = SimpleAqueousFlow(has_pressure=False)
    m.unit = OpsBlock(property_package=m.props)
    m.unit.add_stream_ports()

    m.unit.add_component(
        "biogas_m3_hour",
        pyo.Var(
            m.time_block.time_index, initialize=0.0, units=pyunits.m**3 / pyunits.hr
        ),
    )
    m.unit.register_io_variable(m.unit.biogas_m3_hour, role="output")
    m.unit.add_component(
        "feed",
        pyo.Var(m.time_block.time_index, initialize=0.0, units=pyunits.dimensionless),
    )
    m.unit.register_io_variable(m.unit.feed, role="input")

    m.unit.add_component(
        "biogas_m3_hour_relation",
        pyo.Constraint(m.time_block.time_index, rule=lambda b, t: pyo.Constraint.Skip),
    )
    m.unit.register_relation(
        m.unit.biogas_m3_hour_relation, target=m.unit.biogas_m3_hour
    )

    surrogate = ArimaSurrogate(spec.data)
    m.unit.swap_relation("biogas_m3_hour_relation", surrogate)

    for t in range(n_insample + n_fcst):
        m.unit.feed[t].set_value(float(all_exog[t].item()))
        m.unit.feed[t].fix()
    feed_min = float(X["feed"].min())
    feed_max = float(X["feed"].max())
    for t in range(n_insample + n_fcst, n_total):
        m.unit.feed[t].set_value(mean_feed)
        m.unit.feed[t].setlb(feed_min)
        m.unit.feed[t].setub(feed_max)

    m.obj = pyo.Objective(
        expr=sum(
            (m.unit.biogas_m3_hour[t] - target_biogas) ** 2
            for t in range(n_insample + n_fcst, n_total)
        ),
        sense=pyo.minimize,
    )

    solver = pyo.SolverFactory("ipopt")
    result = solver.solve(m, tee=False)
    assert result.solver.termination_condition == pyo.TerminationCondition.optimal

    pyomo_insample = np.array(
        [float(m.unit.biogas_m3_hour[t].value) for t in range(n_insample)]
    )
    pyomo_fcst = np.array(
        [
            float(m.unit.biogas_m3_hour[t].value)
            for t in range(n_insample, n_insample + n_fcst)
        ]
    )
    pyomo_opt = np.array(
        [
            float(m.unit.biogas_m3_hour[t].value)
            for t in range(n_insample + n_fcst, n_total)
        ]
    )
    inital_point = (
        abs(float(pyomo_insample[0] - direct_insample[0])) / direct_insample[0] * 100
    )
    instample_delta = np.max((pyomo_insample - direct_insample) / direct_insample) * 100
    print(f"In-sample initial point difference: {inital_point}")
    print(f"pyomo_insample: {pyomo_insample}")
    print(f"direct_insample: {direct_insample}")
    forcast_delta = np.max((pyomo_fcst - direct_fcst) / direct_fcst) * 100
    assert inital_point < 1e-4, f"In-sample point too high: {inital_point}"
    assert instample_delta < 1e-4, f"In-sample delta too high: {instample_delta}"
    assert forcast_delta < 1e-4, f"Forecast delta too high: {forcast_delta}"
    _assert_tail_matches_target(pyomo_opt, target_biogas, lookback=10)


@pytest.mark.component
@pytest.mark.needs_ipopt
def test_arima_roundtrip_d1_at_offset_zero():
    """Fit ARIMA(1,1,0), build surrogate at offset==0, fix burn-in, optimize."""
    np.random.seed(7)
    n_train = 100
    n_insample = 20
    n_fcst = 10
    n_opt = 100
    p = 1
    seed_count = p + 1
    n_model = (n_insample - seed_count) + n_fcst + n_opt
    idx = pd.date_range("2024-01-01", periods=n_train, freq="1h")

    feed = pd.Series(np.random.uniform(0.1, 1.0, size=n_train), index=idx, name="feed")
    y_values = np.zeros(n_train)
    for t in range(1, n_train):
        if t == 1:
            y_values[t] = y_values[t - 1] - 0.4 + 0.8 * float(feed.iloc[t])
        else:
            y_values[t] = (
                y_values[t - 1]
                - 0.4
                + 0.8 * float(feed.iloc[t])
                + 0.5 * (y_values[t - 1] - y_values[t - 2])
            )
        y_values[t] += np.random.normal(0, 0.05)
    y = pd.DataFrame({"biogas": y_values}, index=idx)
    X = pd.DataFrame({"feed": feed})

    regressor = ArimaRegressor(order=(1, 1, 0), max_ar_persistence=None).fit(
        X,
        y,
        input_units={"feed": "dimensionless"},
        output_units="m^3/hr",
    )
    assert regressor.fitted is True

    spec = regressor.to_surrogate_spec()

    insample_exog = X.iloc[seed_count:n_insample].values
    forecast_exog = np.zeros((n_fcst, 1))
    all_exog = np.concatenate([insample_exog, forecast_exog])
    assert len(all_exog) == (n_insample - seed_count) + n_fcst
    direct_insample = np.asarray(
        regressor.model.predict(
            steps=n_insample - seed_count,
            exog=insample_exog,
            start=seed_count,
            dynamic=True,
        )
    )
    sm_fcst = np.asarray(
        regressor.model.predict(
            steps=(n_insample - seed_count) + n_fcst,
            exog=all_exog,
            start=seed_count,
            dynamic=True,
        )
    )
    direct_fcst = sm_fcst[-n_fcst:]

    mean_feed = float(X["feed"].mean())
    target_biogas = float(y.iloc[-n_insample:]["biogas"].mean())

    m = pyo.ConcreteModel()
    start_idx = idx[seed_count]
    m.time_block = TimeBlock(
        start_date=start_idx.strftime("%Y-%m-%dT%H:%M"),
        end_date=(start_idx + pd.Timedelta(hours=n_model)).strftime("%Y-%m-%dT%H:%M"),
        time_step=1 * pyunits.hr,
    )
    m.props = SimpleAqueousFlow(has_pressure=False)
    m.unit = OpsBlock(property_package=m.props)
    m.unit.add_stream_ports()

    m.unit.add_component(
        "biogas_m3_hour",
        pyo.Var(
            m.time_block.time_index, initialize=0.0, units=pyunits.m**3 / pyunits.hr
        ),
    )
    m.unit.register_io_variable(m.unit.biogas_m3_hour, role="output")
    m.unit.add_component(
        "feed",
        pyo.Var(m.time_block.time_index, initialize=0.0, units=pyunits.dimensionless),
    )
    m.unit.register_io_variable(m.unit.feed, role="input")

    m.unit.add_component(
        "biogas_m3_hour_relation",
        pyo.Constraint(m.time_block.time_index, rule=lambda b, t: pyo.Constraint.Skip),
    )
    m.unit.register_relation(
        m.unit.biogas_m3_hour_relation, target=m.unit.biogas_m3_hour
    )

    surrogate = ArimaSurrogate(spec.data)
    m.unit.swap_relation("biogas_m3_hour_relation", surrogate)

    # The surrogate's historical state supplies the seed window for this
    # offset-zero case. Exogenous inputs are fixed only for the modeled horizon,
    # with the downstream optimization window left free.
    for t in range(len(all_exog)):
        m.unit.feed[t].set_value(float(all_exog[t].item()))
        m.unit.feed[t].fix()
    feed_min = float(X["feed"].min())
    feed_max = float(X["feed"].max())
    for t in range(len(all_exog), n_model):
        m.unit.feed[t].set_value(mean_feed)
        m.unit.feed[t].setlb(feed_min)
        m.unit.feed[t].setub(feed_max)

    m.obj = pyo.Objective(
        expr=sum(
            (m.unit.biogas_m3_hour[t] - target_biogas) ** 2
            for t in range(len(all_exog), n_model)
        ),
        sense=pyo.minimize,
    )

    solver = pyo.SolverFactory("ipopt")
    result = solver.solve(m, tee=False)
    assert result.solver.termination_condition == pyo.TerminationCondition.optimal

    pyomo_insample = np.array(
        [float(m.unit.biogas_m3_hour[t].value) for t in range(len(insample_exog))]
    )
    pyomo_fcst = np.array(
        [
            float(m.unit.biogas_m3_hour[t].value)
            for t in range(len(insample_exog), len(all_exog))
        ]
    )
    pyomo_opt = np.array(
        [float(m.unit.biogas_m3_hour[t].value) for t in range(len(all_exog), n_model)]
    )

    insample_rmse = float(np.sqrt(np.mean((pyomo_insample - direct_insample) ** 2)))
    forecast_rmse = float(np.sqrt(np.mean((pyomo_fcst - direct_fcst) ** 2)))

    assert insample_rmse < 1e-4, f"In-sample RMSE too high: {insample_rmse}"
    assert forecast_rmse < 1e-4, f"Forecast RMSE too high: {forecast_rmse}"
    _assert_tail_matches_target(pyomo_opt, target_biogas, lookback=10)


@pytest.mark.component
@pytest.mark.needs_ipopt
def test_arima_roundtrip_d1_with_drift():
    """Fit ARIMA(1,1,0) with drift, build surrogate, verify fidelity and optimize."""
    np.random.seed(11)
    n_train = 100
    n_insample = 20
    n_fcst = 10
    n_opt = 100
    n_total = n_insample + n_fcst + n_opt
    idx = pd.date_range("2024-01-01", periods=n_train, freq="1h")

    feed = pd.Series(np.random.uniform(0.1, 1.0, size=n_train), index=idx, name="feed")
    y_values = np.zeros(n_train)
    for t in range(1, n_train):
        if t == 1:
            y_values[t] = y_values[t - 1] - 0.4 + 0.8 * float(feed.iloc[t])
        else:
            y_values[t] = (
                y_values[t - 1]
                - 0.4
                + 0.8 * float(feed.iloc[t])
                + 0.3 * (y_values[t - 1] - y_values[t - 2])
            )
        y_values[t] += np.random.normal(0, 0.05)
    y = pd.DataFrame({"biogas": y_values}, index=idx)
    X = pd.DataFrame({"feed": feed})

    regressor = ArimaRegressor(
        order=(1, 1, 0), include_drift=True, max_ar_persistence=None
    ).fit(
        X,
        y,
        input_units={"feed": "dimensionless"},
        output_units="m^3/hr",
    )
    assert regressor.fitted is True
    assert "drift" in regressor.model_["coef"]

    spec = regressor.to_surrogate_spec()
    assert "drift" in spec.data["coefficients"]

    insample_exog = X.iloc[-n_insample:].values
    forecast_exog = np.zeros((n_fcst, 1))
    all_exog = np.concatenate([insample_exog, forecast_exog])
    sm_all = np.asarray(
        regressor.model.predict(
            steps=n_insample + n_fcst,
            exog=all_exog,
            start=n_train - n_insample,
            dynamic=True,
        )
    )
    direct_insample = sm_all[:n_insample]
    direct_fcst = sm_all[n_insample:]

    mean_feed = float(X["feed"].mean())
    target_biogas = float(y.iloc[-n_insample:]["biogas"].mean())

    m = pyo.ConcreteModel()
    start_idx = idx[-n_insample]
    m.time_block = TimeBlock(
        start_date=start_idx.strftime("%Y-%m-%dT%H:%M"),
        end_date=(start_idx + pd.Timedelta(hours=n_total)).strftime("%Y-%m-%dT%H:%M"),
        time_step=1 * pyunits.hr,
    )
    m.props = SimpleAqueousFlow(has_pressure=False)
    m.unit = OpsBlock(property_package=m.props)
    m.unit.add_stream_ports()

    m.unit.add_component(
        "biogas_m3_hour",
        pyo.Var(
            m.time_block.time_index, initialize=0.0, units=pyunits.m**3 / pyunits.hr
        ),
    )
    m.unit.register_io_variable(m.unit.biogas_m3_hour, role="output")
    m.unit.add_component(
        "feed",
        pyo.Var(m.time_block.time_index, initialize=0.0, units=pyunits.dimensionless),
    )
    m.unit.register_io_variable(m.unit.feed, role="input")

    m.unit.add_component(
        "biogas_m3_hour_relation",
        pyo.Constraint(m.time_block.time_index, rule=lambda b, t: pyo.Constraint.Skip),
    )
    m.unit.register_relation(
        m.unit.biogas_m3_hour_relation, target=m.unit.biogas_m3_hour
    )

    surrogate = ArimaSurrogate(spec.data)
    m.unit.swap_relation("biogas_m3_hour_relation", surrogate)

    for t in range(n_insample + n_fcst):
        m.unit.biogas_m3_hour[t].set_value(float(sm_all[t]))

    for t in range(n_insample + n_fcst):
        m.unit.feed[t].set_value(float(all_exog[t].item()))
        m.unit.feed[t].fix()
    feed_min = float(X["feed"].min())
    feed_max = float(X["feed"].max())
    for t in range(n_insample + n_fcst, n_total):
        m.unit.feed[t].set_value(mean_feed)
        m.unit.feed[t].setlb(feed_min)
        m.unit.feed[t].setub(feed_max)

    m.obj = pyo.Objective(
        expr=sum(
            (m.unit.biogas_m3_hour[t] - target_biogas) ** 2
            for t in range(n_insample + n_fcst, n_total)
        ),
        sense=pyo.minimize,
    )

    solver = pyo.SolverFactory("ipopt")
    result = solver.solve(m, tee=False)
    assert result.solver.termination_condition == pyo.TerminationCondition.optimal
    pyomo_insample = np.array(
        [float(m.unit.biogas_m3_hour[t].value) for t in range(n_insample)]
    )
    pyomo_fcst = np.array(
        [
            float(m.unit.biogas_m3_hour[t].value)
            for t in range(n_insample, n_insample + n_fcst)
        ]
    )
    pyomo_opt = np.array(
        [
            float(m.unit.biogas_m3_hour[t].value)
            for t in range(n_insample + n_fcst, n_total)
        ]
    )
    print("Pyomo optimization results:")
    print("In-sample:", pyomo_insample)
    print("Forecast:", pyomo_fcst)
    print("Optimized:", pyomo_opt)
    print("Target:", target_biogas)
    insample_rmse = float(np.sqrt(np.mean((pyomo_insample - direct_insample) ** 2)))
    forecast_rmse = float(np.sqrt(np.mean((pyomo_fcst - direct_fcst) ** 2)))
    assert insample_rmse < 1e-4, f"In-sample RMSE too high: {insample_rmse}"
    assert forecast_rmse < 1e-4, f"Forecast RMSE too high: {forecast_rmse}"
    _assert_tail_matches_target(pyomo_opt, target_biogas, lookback=10)
