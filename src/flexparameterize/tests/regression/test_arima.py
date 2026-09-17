"""Tests for ArimaRegressor: ARIMA time-series regression with exogenous inputs.

Importing this module (and ``ArimaRegressor`` itself) never requires
scipy or statsforecast -- only :meth:`ArimaRegressor.fit` does, lazily.
Tests that exercise a real fit call ``pytest.importorskip("scipy")``
themselves so the absence test below still runs (and passes) without the
extra installed.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

from flexcore.config.schema import SurrogateType
from flexcore.exceptions import FlexConfigError, FlexDataError
from flexops.surrogates import ArimaSurrogate
from flexparameterize.regression import Regressor
from flexparameterize.regression.arima import ArimaRegressor

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "test_time_series_data")
_BIO_GAS_PATH = os.path.join(_TEST_DATA_DIR, "imputed_bio_gas_generation.csv")


def _bio_gas_dataframe() -> pd.DataFrame:
    """Load the biogas test data set, parsed with a DatetimeIndex."""
    return pd.read_csv(_BIO_GAS_PATH, parse_dates=["timestamp"]).set_index("timestamp")


# -- absence tests -----------------------------------------------------------


@pytest.mark.unit
def test_scipy_absent_raises(monkeypatch):
    """With scipy unimportable, fitting raises FlexConfigError."""
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.optimize", None)

    y = pd.DataFrame({"biogas_m3_hour": [1.0, 2.0, 3.0]})
    with pytest.raises(FlexConfigError, match=r"flex-pse\[parameterize\]"):
        ArimaRegressor(order=(1, 0, 0)).fit(pd.DataFrame(index=y.index), y)


@pytest.mark.unit
def test_no_order_no_auto_raises():
    """Passing neither order nor auto=True raises FlexConfigError."""
    y = pd.DataFrame({"biogas_m3_hour": [1.0, 2.0, 3.0]})
    with pytest.raises(FlexConfigError, match="order"):
        ArimaRegressor().fit(
            pd.DataFrame(index=y.index),
            y,
            input_units={},
            output_units="m^3/hr",
        )


# -- synthetic-data fitting tests --------------------------------------------


@pytest.mark.unit
def test_fits_ar1_no_exog():
    """AR(1) with no exogenous regressors recovers a known coefficient."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(0)
    n = 300
    phi = 0.7
    y_values = np.zeros(n)
    for t in range(1, n):
        y_values[t] = phi * y_values[t - 1] + rng.normal(0, 0.1)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    y = pd.DataFrame({"biogas": y_values}, index=idx)

    regressor = ArimaRegressor(order=(1, 0, 0), max_ar_persistence=None).fit(
        pd.DataFrame(index=idx), y
    )
    assert regressor.coefficients is not None
    assert "ar1" in regressor.coefficients
    assert regressor.coefficients["ar1"] == pytest.approx(phi, rel=0.1)


@pytest.mark.unit
def test_fits_arima_with_exog():
    """ARIMA(1,1,1) with one exogenous regressor."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(7)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    feed = pd.Series(rng.uniform(0.1, 1.0, size=n), index=idx, name="feed")
    biogas = np.zeros(n)
    for t in range(1, n):
        biogas[t] = 0.5 * biogas[t - 1] + 2.0 * feed.iloc[t] + rng.normal(0, 0.05)
    y = pd.DataFrame({"biogas": biogas}, index=idx)

    regressor = ArimaRegressor(order=(1, 0, 1), max_ar_persistence=None).fit(
        pd.DataFrame({"feed": feed}),
        y,
        input_units={"feed": "kg"},
        output_units="m^3/hr",
    )
    assert regressor.exogenous_variables == ["feed"]
    assert "feed" in regressor.coefficients
    assert regressor.coefficients["feed"] > 0.5


@pytest.mark.unit
def test_fits_multiple_exog_columns():
    """Two exogenous columns are both recovered."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(99)
    n = 150
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    x1 = pd.Series(rng.uniform(1.0, 10.0, size=n), index=idx, name="x1")
    x2 = pd.Series(rng.uniform(1e5, 5e5, size=n), index=idx, name="x2")
    y_vals = 0.4 * x1 + 1e-5 * x2 + rng.normal(0, 0.01, size=n)
    y = pd.DataFrame({"output": y_vals}, index=idx)

    regressor = ArimaRegressor(order=(0, 0, 0)).fit(
        pd.DataFrame({"x1": x1, "x2": x2}),
        y,
        input_units={"x1": "unit", "x2": "unit"},
        output_units="unit",
    )
    assert sorted(regressor.exogenous_variables) == ["x1", "x2"]
    assert "x1" in regressor.coefficients
    assert "x2" in regressor.coefficients
    assert regressor.coefficients["x1"] == pytest.approx(0.4, rel=0.2)
    assert regressor.coefficients["x2"] == pytest.approx(1e-5, rel=0.2)


# -- protocol conformance ----------------------------------------------------


@pytest.mark.unit
def test_isinstance_regressor():
    """ArimaRegressor structurally conforms to Regressor and behaves as one."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(1)
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    vals = np.cumsum(rng.normal(0, 0.1, size=n))
    y = pd.DataFrame({"y": vals}, index=idx)

    regressor = ArimaRegressor(order=(1, 0, 0), max_ar_persistence=None).fit(
        pd.DataFrame(index=idx), y
    )
    assert isinstance(regressor, Regressor)
    result = regressor.to_fit_result()
    assert isinstance(result.coefficients, dict)
    assert result.n_samples == n
    assert np.isfinite(result.metrics["aic"])
    assert np.isfinite(result.metrics["rmse"])


@pytest.mark.unit
def test_provenance_populated():
    """Fit metrics are finite and the emitted spec's provenance is JSON-safe."""
    import json

    pytest.importorskip("scipy")

    rng = np.random.default_rng(2)
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    vals = np.cumsum(rng.normal(0, 0.1, size=n))
    y = pd.DataFrame({"y": vals}, index=idx)

    regressor = ArimaRegressor(order=(1, 0, 0), max_ar_persistence=None).fit(
        pd.DataFrame(index=idx), y
    )
    result = regressor.to_fit_result()
    assert np.isfinite(result.metrics["aic"])
    assert np.isfinite(result.metrics["rmse"])
    assert result.n_samples == n
    assert len(result.data_window) == 2

    spec = regressor.to_surrogate_spec()
    assert spec.surrogate_type == SurrogateType.ARIMA
    provenance = {"n_samples": result.n_samples, **result.metrics}
    json.dumps(provenance)


# -- SurrogateSpec data contract ----------------------------------------------


@pytest.mark.unit
def test_surrogate_spec_data_contract():
    """SurrogateSpec.data contains all keys the ArimaSurrogate build needs."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(3)
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    feed = pd.Series(rng.uniform(0.1, 1.0, size=n), index=idx, name="feed")
    vals = 2.0 * feed + rng.normal(0, 0.05, size=n)
    y = pd.DataFrame({"biogas": vals}, index=idx)

    regressor = ArimaRegressor(order=(0, 0, 0)).fit(
        pd.DataFrame({"feed": feed}),
        y,
        input_units={"feed": "kg"},
        output_units="m^3/hr",
    )
    spec = regressor.to_surrogate_spec()

    data = spec.data
    assert set(data) == {
        "input_variables",
        "output_variables",
        "coefficients",
        "history",
    }
    assert data["coefficients"]["order"] == [0, 0, 0]
    assert "intercept" in data["coefficients"]
    assert "exog_coefs" in data["coefficients"]
    assert len(data["coefficients"]["exog_coefs"]) == 1
    assert isinstance(data["coefficients"]["intercept"], float)
    assert isinstance(data["coefficients"]["exog_coefs"], list)
    assert len(data["coefficients"]["exog_coefs"]) == 1
    assert len(data["history"]["y_values"]) == n
    assert len(data["history"]["eps_values"]) == n


# -- to_fit_result / to_surrogate_spec guards ---------------------------------


@pytest.mark.unit
def test_surrogate_spec_omits_intercept_without_a_deterministic_term():
    """include_mean=False emits no intercept key, not an intercept of 0.0.

    The surrogate reads a missing key as "no deterministic term" and builds
    no Var for it, so emitting 0.0 would fix a parameter the fit never had.
    """
    pytest.importorskip("scipy")

    rng = np.random.default_rng(31)
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    vals = np.zeros(n)
    for t in range(1, n):
        vals[t] = 0.4 * vals[t - 1] + rng.normal(0, 0.1)
    y = pd.DataFrame({"y": vals}, index=idx)

    regressor = ArimaRegressor(
        order=(1, 0, 0), include_mean=False, max_ar_persistence=None
    ).fit(pd.DataFrame(index=idx), y, output_units="m^3/hr")

    spec = regressor.to_surrogate_spec()
    assert "intercept" not in spec.data["coefficients"]
    assert "drift" not in spec.data["coefficients"]
    assert "const" not in regressor.to_fit_result().coefficients
    ArimaSurrogate(spec.data)


@pytest.mark.unit
def test_surrogate_spec_keeps_a_zero_valued_fitted_intercept():
    """A deterministic term that fits to ~0.0 is still reported.

    Its presence is what tells the surrogate to build the Var; only
    include_mean=False should omit it.
    """
    pytest.importorskip("scipy")

    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    y = pd.DataFrame({"y": np.zeros(n)}, index=idx)

    regressor = ArimaRegressor(order=(1, 0, 0), max_ar_persistence=None).fit(
        pd.DataFrame(index=idx), y, output_units="m^3/hr"
    )

    coefficients = regressor.to_surrogate_spec().data["coefficients"]
    assert coefficients["intercept"] == pytest.approx(0.0)
    assert regressor.to_fit_result().coefficients["const"] == pytest.approx(0.0)


@pytest.mark.unit
def test_aicc_is_none_when_undefined_instead_of_raising(caplog):
    """Too few effective rows makes the AICc correction term undefined.

    This used to raise ZeroDivisionError from inside fit() itself, via
    model_ -> _aicc_from_direct, on a fit the row check accepts.
    """
    pytest.importorskip("scipy")

    idx = pd.date_range("2024-01-01", periods=3, freq="1h")
    y = pd.DataFrame({"y": [1.0, 2.0, 3.5]}, index=idx)

    regressor = ArimaRegressor(order=(0, 0, 0)).fit(pd.DataFrame(index=idx), y)

    assert regressor.fitted is True
    assert regressor.model_["aicc"] is None
    assert regressor.fit_diagnostics()["aicc"] is None


@pytest.mark.unit
def test_information_criteria_exclude_zero_padded_residuals():
    """AIC/BIC/sigma2 must not be diluted by the zero-padded leading lags.

    `resid` keeps its full length for alignment, but the first max(p, q)
    entries carry no residual; dividing by them understates sigma2.
    """
    pytest.importorskip("scipy")

    rng = np.random.default_rng(5)
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    vals = np.zeros(n)
    for t in range(3, n):
        vals[t] = 0.5 * vals[t - 1] + rng.normal(0, 0.1)
    y = pd.DataFrame({"y": vals}, index=idx)

    regressor = ArimaRegressor(order=(3, 0, 0), max_ar_persistence=None).fit(
        pd.DataFrame(index=idx), y
    )
    model = regressor.model

    assert len(model.resid) == n
    assert model.nobs_effective == n - 3
    assert len(model.effective_resid) == n - 3
    expected_sigma2 = float(np.sum(model.effective_resid**2)) / (n - 3)
    assert model.sigma2 == pytest.approx(expected_sigma2)
    # Padding-inclusive variance is strictly smaller, so this pins direction.
    assert model.sigma2 > float(np.sum(model.resid**2)) / n


@pytest.mark.unit
def test_to_fit_result_before_fit_raises():
    """to_fit_result before fit raises FlexDataError."""
    regressor = ArimaRegressor(order=(1, 0, 0))
    with pytest.raises(FlexDataError, match="no fit yet"):
        regressor.to_fit_result()


@pytest.mark.unit
def test_to_surrogate_spec_before_fit_raises():
    """to_surrogate_spec before fit raises FlexDataError."""
    regressor = ArimaRegressor(order=(1, 0, 0))
    with pytest.raises(FlexDataError, match="no fit yet"):
        regressor.to_surrogate_spec()


@pytest.mark.unit
def test_missing_input_units_raises():
    """Missing input_units for an exogenous column raises FlexConfigError."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(5)
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    feed = pd.Series(rng.uniform(0.1, 1.0, size=n), index=idx, name="feed")
    vals = 2.0 * feed.values + rng.normal(0, 0.05, size=n)
    y = pd.DataFrame({"biogas": vals}, index=idx)

    with pytest.raises(FlexConfigError, match="feed"):
        ArimaRegressor(order=(0, 0, 0)).fit(
            pd.DataFrame({"feed": feed}),
            y,
            input_units={},
            output_units="m^3/hr",
        )


# -- diagnostics -------------------------------------------------------------


@pytest.mark.unit
def test_fit_diagnostics_keys():
    """fit_diagnostics returns AIC, BIC, AICc, RMSE, log-likelihood."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(6)
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    vals = np.cumsum(rng.normal(0, 0.1, size=n))
    y = pd.DataFrame({"y": vals}, index=idx)

    regressor = ArimaRegressor(order=(1, 0, 0), max_ar_persistence=None).fit(
        pd.DataFrame(index=idx), y
    )
    diag = regressor.fit_diagnostics()
    assert np.isfinite(diag["aic"])
    assert np.isfinite(diag["bic"])
    assert np.isfinite(diag["aicc"])
    assert np.isfinite(diag["rmse"])
    assert np.isfinite(diag["log_likelihood"])


@pytest.mark.unit
def test_fit_diagnostics_before_fit_raises():
    """fit_diagnostics before fit raises FlexDataError."""
    regressor = ArimaRegressor(order=(1, 0, 0))
    with pytest.raises(FlexDataError, match="no fit yet"):
        regressor.fit_diagnostics()


# -- auto_arima --------------------------------------------------------------


@pytest.mark.unit
def test_auto_arima_selects_order():
    """auto=True runs AutoARIMA and stores a valid order."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(8)
    n = 120
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    vals = np.cumsum(rng.normal(0, 0.1, size=n))
    y = pd.DataFrame({"y": vals}, index=idx)

    regressor = ArimaRegressor(
        auto=True, max_p=2, max_q=2, max_ar_persistence=None
    ).fit(pd.DataFrame(index=idx), y, input_units={}, output_units="m^3/hr")
    assert regressor.order is not None
    assert len(regressor.order) == 3
    assert all(isinstance(v, int) for v in regressor.order)
    result = regressor.to_fit_result()
    assert np.isfinite(result.metrics["aic"])


# -- biogas test data --------------------------------------------------------


@pytest.mark.unit
def test_fits_biogas_with_feed_and_ts_exog():
    """Fits biogas_m3_hour from feed_volume_kg and TS_pct as exogenous."""
    pytest.importorskip("scipy")

    if not os.path.exists(_BIO_GAS_PATH):
        pytest.skip(f"Test data not found at {_BIO_GAS_PATH}")

    df = _bio_gas_dataframe()
    X = df[["feed_volume_kg", "TS_pct"]]
    y = df[["biogas_m3_hour"]]

    regressor = ArimaRegressor(order=(1, 0, 1), max_ar_persistence=None).fit(
        X,
        y,
        input_units={"feed_volume_kg": "kg", "TS_pct": "%"},
        output_units="m^3/hr",
    )

    assert regressor.exogenous_variables == ["feed_volume_kg", "TS_pct"]
    assert regressor.output_variable == "biogas_m3_hour"
    assert regressor.n_samples == len(
        df.dropna(subset=["feed_volume_kg", "TS_pct", "biogas_m3_hour"])
    )
    assert np.isfinite(regressor.metrics["aic"])
    assert np.isfinite(regressor.metrics["rmse"])

    spec = regressor.to_surrogate_spec()
    assert spec.surrogate_type == SurrogateType.ARIMA
    assert spec.data["coefficients"]["order"] == [1, 0, 1]
    assert len(spec.data["coefficients"]["exog_coefs"]) == 2
    assert spec.data["output_variables"] == {"biogas_m3_hour": "m^3/hr"}


@pytest.mark.unit
def test_biogas_surrogate_spec_has_all_keys():
    """Spec produced from biogas data carries the full ArimaSurrogate contract."""
    pytest.importorskip("scipy")

    if not os.path.exists(_BIO_GAS_PATH):
        pytest.skip(f"Test data not found at {_BIO_GAS_PATH}")

    df = _bio_gas_dataframe()
    X = df[["feed_volume_kg", "TS_pct"]]
    y = df[["biogas_m3_hour"]]

    regressor = ArimaRegressor(order=(2, 0, 1), max_ar_persistence=None).fit(
        X,
        y,
        input_units={"feed_volume_kg": "kg", "TS_pct": "%"},
        output_units="m^3/hr",
    )
    spec = regressor.to_surrogate_spec()

    data = spec.data
    assert "input_variables" in data
    assert "output_variables" in data
    assert "coefficients" in data
    assert "order" in data["coefficients"]
    assert "intercept" in data["coefficients"]
    assert "ar_coefs" in data["coefficients"]
    assert "ma_coefs" in data["coefficients"]
    assert "exog_coefs" in data["coefficients"]

    assert len(data["coefficients"]["ar_coefs"]) == 2  # AR(2)
    assert len(data["coefficients"]["ma_coefs"]) == 1  # MA(1)
    assert len(data["coefficients"]["exog_coefs"]) == 2  # two exogenous columns


# -- cross-validation against statsmodels on real biogas data -----------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "order",
    [
        (1, 0, 0),
        (2, 0, 0),
        (3, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (0, 0, 3),
        (1, 0, 1),
        (2, 0, 2),
        (0, 1, 0),
        (1, 1, 0),
        (0, 1, 1),
        (1, 1, 1),
        (2, 1, 2),
    ],
    ids=[
        "ar1",
        "ar2",
        "ar3",
        "ma1",
        "ma2",
        "ma3",
        "ar1-ma1",
        "ar2-ma2",
        "i1",
        "ar1-i1",
        "i1-ma1",
        "ar1-i1-ma1",
        "ar2-i1-ma2",
    ],
)
def test_biogas_matches_statsmodels_insample_and_forecast(order):
    """Direct ARIMA fits and forecasts on real biogas data match statsmodels."""
    pytest.importorskip("scipy")
    pytest.importorskip("statsmodels")
    import warnings

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    if not os.path.exists(_BIO_GAS_PATH):
        pytest.skip(f"Test data not found at {_BIO_GAS_PATH}")

    df = _bio_gas_dataframe().dropna(
        subset=["biogas_m3_hour", "feed_volume_kg", "TS_pct"]
    )
    n_train = 120
    n_fcst = 24
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train : n_train + n_fcst]

    X_train = train_df[["feed_volume_kg", "TS_pct"]]
    y_train = train_df[["biogas_m3_hour"]]
    X_test = test_df[["feed_volume_kg", "TS_pct"]]

    p, d, q = order
    regressor = ArimaRegressor(order=order, max_ar_persistence=None).fit(
        X_train,
        y_train,
        input_units={"feed_volume_kg": "kg", "TS_pct": "%"},
        output_units="m^3/hr",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trend = "c"
        sm_model = SARIMAX(y_train, exog=X_train, order=order, trend=trend)
        sm_result = sm_model.fit(disp=False)

    # 1. In-sample level fitted values (skip t=0 Kalman diffuse prior for d=1)
    offset_in = 1 if d == 1 else 0
    reg_fitted = regressor.model.fittedvalues_level[offset_in:]
    sm_fitted = sm_result.fittedvalues.values[offset_in:]
    valid = ~np.isnan(reg_fitted)
    in_sample_mape = float(
        np.mean(
            np.abs(reg_fitted[valid] - sm_fitted[valid])
            / np.abs(reg_fitted[valid])
            * 100
        )
    )
    assert (
        in_sample_mape < 10.0
    ), f"ARIMA{order} in-sample MAPE too large: {in_sample_mape:.2f}%"

    # 2. Out-of-sample multi-step forecast
    reg_fcst = regressor.model.predict(steps=n_fcst, exog=X_test.values)
    sm_fcst = sm_result.forecast(steps=n_fcst, exog=X_test).values
    fcst_mape = float(np.mean(np.abs(reg_fcst - sm_fcst) / np.abs(reg_fcst) * 100))
    assert fcst_mape < 10.0, f"ARIMA{order} forecast MAPE too large: {fcst_mape:.2f}%"

    # 3. In-sample dynamic recursive prediction
    offset = max(p + d, q) + (1 if d == 1 else 0)
    steps = n_train - offset
    reg_dyn = regressor.model.predict(
        steps=steps,
        exog=X_train.values[offset:],
        start=offset,
        dynamic=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm_dyn = sm_result.predict(
            start=offset,
            end=n_train - 1,
            exog=X_train.values[offset:],
            dynamic=True,
        ).values
    dyn_mape = float(np.mean(np.abs(reg_dyn - sm_dyn) / np.abs(reg_dyn) * 100))
    assert (
        dyn_mape < 10.0
    ), f"ARIMA{order} dynamic in-sample MAPE too large: {dyn_mape:.2f}%"


@pytest.mark.unit
def test_biogas_auto_arima_matches_statsmodels():
    """auto=True order selection on biogas data matches statsmodels predictions."""
    pytest.importorskip("scipy")
    pytest.importorskip("statsmodels")
    pytest.importorskip("statsforecast")
    import warnings

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    if not os.path.exists(_BIO_GAS_PATH):
        pytest.skip(f"Test data not found at {_BIO_GAS_PATH}")

    df = _bio_gas_dataframe().dropna(
        subset=["biogas_m3_hour", "feed_volume_kg", "TS_pct"]
    )
    n_train = 120
    n_fcst = 24
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train : n_train + n_fcst]

    X_train = train_df[["feed_volume_kg", "TS_pct"]]
    y_train = train_df[["biogas_m3_hour"]]
    X_test = test_df[["feed_volume_kg", "TS_pct"]]

    regressor = ArimaRegressor(
        auto=True, max_ar_persistence=None, max_p=3, max_q=3, max_d=1
    ).fit(
        X_train,
        y_train,
        input_units={"feed_volume_kg": "kg", "TS_pct": "%"},
        output_units="m^3/hr",
    )
    assert regressor.fitted is True
    order = regressor.order
    assert order is not None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trend = "c"
        sm_model = SARIMAX(y_train, exog=X_train, order=order, trend=trend)
        sm_result = sm_model.fit(disp=False)

    reg_fcst = regressor.model.predict(steps=n_fcst, exog=X_test.values)
    sm_fcst = sm_result.forecast(steps=n_fcst, exog=X_test).values
    fcst_mape = float(np.mean(np.abs(reg_fcst - sm_fcst) / np.abs(reg_fcst) * 100))
    assert (
        fcst_mape < 10.0
    ), f"Auto ARIMA{order} forecast MAPE too large: {fcst_mape:.2f}%"


# -- validation: non-differenced only -----------------------------------------


@pytest.mark.unit
def test_d_greater_than_zero_raises():
    """ARIMA order with d>1 raises FlexConfigError; d=1 is allowed."""
    with pytest.raises(FlexConfigError, match="d=0 or d=1"):
        ArimaRegressor(order=(1, 2, 0))
    # d=1 should NOT raise during construction
    regressor = ArimaRegressor(order=(1, 1, 0))
    assert regressor.order == (1, 1, 0)


@pytest.mark.unit
def test_D_greater_than_zero_raises():
    """Seasonal order with D>0 raises FlexConfigError."""
    with pytest.raises(FlexConfigError, match="D=0"):
        ArimaRegressor(order=(1, 0, 0), seasonal_order=(0, 1, 0, 24))


@pytest.mark.unit
def test_seasonal_P_or_Q_greater_than_zero_raises():
    """Seasonal AR/MA terms (P>0 or Q>0) raise FlexConfigError at construction.

    The direct-fit backend does not support seasonal AR/MA terms at all;
    this must fail fast (at __init__), not deep inside fit() with a
    NotImplementedError.
    """
    with pytest.raises(FlexConfigError, match="seasonal"):
        ArimaRegressor(order=(1, 0, 0), seasonal_order=(1, 0, 0, 24))
    with pytest.raises(FlexConfigError, match="seasonal"):
        ArimaRegressor(order=(1, 0, 0), seasonal_order=(0, 0, 1, 24))
    # The trivial (0, 0, 0, m) seasonal order is accepted (a no-op).
    regressor = ArimaRegressor(order=(1, 0, 0), seasonal_order=(0, 0, 0, 24))
    assert regressor.seasonal_order == (0, 0, 0, 24)


@pytest.mark.unit
def test_auto_arima_seasonal_search_with_nonzero_PQ_raises(monkeypatch):
    """auto=True with a seasonal search that selects P>0/Q>0 raises
    FlexConfigError (not a bare NotImplementedError) rather than crashing
    deep inside the direct-fit backend.

    ``_auto_select_order`` is monkeypatched to deterministically return a
    seasonal order with P>0, since AutoARIMA's own seasonal search is a
    heuristic that cannot be relied on to pick one on demand.
    """
    pytest.importorskip("scipy")

    import flexparameterize.regression.arima as arima_module

    monkeypatch.setattr(
        arima_module,
        "_auto_select_order",
        lambda *a, **k: ((1, 0, 0), (1, 0, 0, 24)),
    )

    rng = np.random.default_rng(3)
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    y = pd.DataFrame({"y": rng.normal(size=n)}, index=idx)

    regressor = ArimaRegressor(
        auto=True, seasonal_order=(0, 0, 0, 24), max_ar_persistence=None
    )
    with pytest.raises(FlexConfigError, match="seasonal"):
        regressor.fit(pd.DataFrame(index=idx), y)


@pytest.mark.unit
def test_include_drift_raises_with_d0():
    """include_drift=True with d=0 raises FlexConfigError."""
    with pytest.raises(FlexConfigError, match="include_drift"):
        ArimaRegressor(order=(1, 0, 0), include_drift=True)


@pytest.mark.unit
def test_include_drift_with_d1_fits_and_spec_includes_drift():
    """include_drift=True with d=1 fits and spec emits drift."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(99)
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    # Generate data with a linear trend so drift is meaningful
    vals = 0.1 * np.arange(n) + rng.normal(0, 0.1, size=n)
    y = pd.DataFrame({"y": vals}, index=idx)

    regressor = ArimaRegressor(
        order=(1, 1, 0), include_drift=True, max_ar_persistence=None
    ).fit(pd.DataFrame(index=idx), y)
    assert regressor.fitted is True
    assert regressor.order[1] == 1

    spec = regressor.to_surrogate_spec()
    assert "drift" in spec.data["coefficients"]
    assert isinstance(spec.data["coefficients"]["drift"], float)


@pytest.mark.unit
def test_d1_default_mean_uses_constant_drift_for_pyomo_contract():
    """The d=1 default deterministic term matches ArimaSurrogate's drift."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(12)
    n = 120
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    differences = np.empty(n - 1)
    differences[0] = 0.2
    for position in range(1, n - 1):
        differences[position] = (
            0.35 + 0.4 * differences[position - 1] + rng.normal(0.0, 0.01)
        )
    y = pd.DataFrame(
        {"y": np.concatenate(([1.0], 1.0 + np.cumsum(differences)))}, index=idx
    )

    regressor = ArimaRegressor(order=(1, 1, 0), max_ar_persistence=None).fit(
        pd.DataFrame(index=idx), y, output_units="m^3/hr"
    )

    assert "const" not in regressor.coefficients
    assert regressor.coefficients["drift"] == pytest.approx(0.35, abs=0.03)
    spec = regressor.to_surrogate_spec()
    assert "const" not in spec.data["coefficients"]
    assert "drift" in spec.data["coefficients"]
    ArimaSurrogate(spec.data)


@pytest.mark.unit
def test_auto_arima_d_in_zero_or_one():
    """auto=True allows d=0 or d=1, but rejects d>1."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(42)
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    vals = np.cumsum(rng.normal(0, 0.1, size=n))
    y = pd.DataFrame({"y": vals}, index=idx)

    regressor = ArimaRegressor(
        auto=True, max_p=2, max_q=2, max_d=1, max_ar_persistence=None
    ).fit(pd.DataFrame(index=idx), y, input_units={}, output_units="m^3/hr")
    assert regressor.order is not None
    assert regressor.order[1] in (0, 1)


@pytest.mark.unit
def test_auto_arima_d_greater_than_one_raises(monkeypatch):
    """auto=True with a search that selects d>1 raises FlexConfigError."""
    pytest.importorskip("scipy")

    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    y = pd.DataFrame({"y": np.random.default_rng(0).normal(size=n)}, index=idx)

    # _auto_select_order is monkeypatched to return d=2
    from flexparameterize.regression import arima as arima_module

    original = arima_module._auto_select_order

    def _fake_auto(*_args, **_kwargs):
        return (1, 2, 1), None

    monkeypatch.setattr(arima_module, "_auto_select_order", _fake_auto)
    try:
        regressor = ArimaRegressor(auto=True, max_ar_persistence=None)
        with pytest.raises(FlexConfigError, match="d=2"):
            regressor.fit(pd.DataFrame(index=idx), y)
    finally:
        monkeypatch.setattr(arima_module, "_auto_select_order", original)


@pytest.mark.unit
def test_high_ar_persistence_is_bounded_not_rejected():
    """A near-unit-root fit is bounded to max_ar_persistence, not rejected."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(99)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    vals = np.cumsum(rng.normal(0, 0.1, size=n))
    y = pd.DataFrame({"y": vals}, index=idx)

    # AR(1) on a random walk wants |phi| ~= 1; the default
    # max_ar_persistence=0.85 constrains the fit itself instead of raising.
    regressor = ArimaRegressor(order=(1, 0, 0)).fit(pd.DataFrame(index=idx), y)
    assert regressor.fitted is True
    ar_coef = regressor.coefficients["ar1"]
    assert abs(ar_coef) <= 0.85 + 1e-6


@pytest.mark.unit
def test_low_ar_persistence_passes():
    """AR coefficient below max_ar_persistence fits successfully."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(1)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    phi = 0.5
    vals = np.zeros(n)
    for t in range(1, n):
        vals[t] = phi * vals[t - 1] + rng.normal(0, 0.1)
    y = pd.DataFrame({"y": vals}, index=idx)

    # AR(1) with relaxed persistence threshold should fit
    regressor = ArimaRegressor(order=(1, 0, 0), max_ar_persistence=1.0).fit(
        pd.DataFrame(index=idx), y
    )
    assert regressor.fitted is True


# -- registry -----------------------------------------------------------------


@pytest.mark.unit
def test_arima_in_registry():
    """get_regressor('arima') now returns ArimaRegressor (not raises)."""
    from flexparameterize.regression import get_regressor

    assert get_regressor(SurrogateType.ARIMA) is ArimaRegressor
    assert get_regressor("arima") is ArimaRegressor


@pytest.mark.unit
def test_fits_arima_d1_no_exog():
    """ARIMA(1,1,0) with no exogenous regressors fits and predicts."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(0)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    phi = 0.6
    y_values = np.zeros(n)
    for t in range(1, n):
        y_values[t] = (
            y_values[t - 1]
            + phi * (y_values[t - 1] - y_values[t - 2])
            + rng.normal(0, 0.05)
        )

    y = pd.DataFrame({"y": y_values}, index=idx)

    regressor = ArimaRegressor(order=(1, 1, 0), max_ar_persistence=None).fit(
        pd.DataFrame(index=idx), y
    )
    assert regressor.fitted is True
    assert regressor.order == (1, 1, 0)
    assert regressor.model.model.k_diff == 1

    # predict should return original-scale values
    fcst = regressor.model.predict(steps=10)
    assert len(fcst) == 10
    assert not np.any(np.isnan(fcst))

    # In-sample dynamic prediction should match observed scale
    sm_insample = regressor.model.predict(
        steps=20,
        start=n - 20,
        dynamic=True,
    )
    assert len(sm_insample) == 20
    assert not np.any(np.isnan(sm_insample))


# -- include_mean=False predict correctness (H3) ------------------------------


@pytest.mark.unit
def test_predict_with_include_mean_false_no_exog():
    """`.model.predict()` does not crash and matches a hand-computed
    no-constant forecast when the model was fit with include_mean=False.

    Previously `predict()` hardcoded `has_const=True` regardless of how the
    model was actually fit, which crashed with a negative-array-size
    ValueError whenever include_mean=False and there were no MA/exog terms
    to absorb the resulting off-by-one parameter misread.
    """
    pytest.importorskip("scipy")

    rng = np.random.default_rng(0)
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    phi = 0.5
    y_values = np.zeros(n)
    for t in range(1, n):
        y_values[t] = phi * y_values[t - 1] + rng.normal(0, 0.01)
    y = pd.DataFrame({"y": y_values}, index=idx)

    regressor = ArimaRegressor(
        order=(1, 0, 0), include_mean=False, max_ar_persistence=None
    ).fit(pd.DataFrame(index=idx), y)
    assert "const" not in regressor.coefficients

    fcst = regressor.model.predict(steps=3)
    assert len(fcst) == 3
    assert not np.any(np.isnan(fcst))

    ar1 = regressor.coefficients["ar1"]
    manual = []
    prev = float(y_values[-1])
    for _ in range(3):
        prev = ar1 * prev
        manual.append(prev)
    assert list(fcst) == pytest.approx(manual, rel=1e-6)


@pytest.mark.unit
def test_predict_with_include_mean_false_and_exog():
    """include_mean=False with exogenous regressors also predicts correctly."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(1)
    n = 150
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    feed = pd.Series(rng.uniform(0.1, 1.0, size=n), index=idx, name="feed")
    phi = 0.4
    y_values = np.zeros(n)
    for t in range(1, n):
        y_values[t] = phi * y_values[t - 1] + 2.0 * feed.iloc[t] + rng.normal(0, 0.01)
    y = pd.DataFrame({"y": y_values}, index=idx)

    regressor = ArimaRegressor(
        order=(1, 0, 0), include_mean=False, max_ar_persistence=None
    ).fit(
        pd.DataFrame({"feed": feed}),
        y,
        input_units={"feed": "dimensionless"},
        output_units="unit",
    )
    assert "const" not in regressor.coefficients

    exog_future = np.array([[0.5], [0.6], [0.7]])
    fcst = regressor.model.predict(steps=3, exog=exog_future)
    assert len(fcst) == 3
    assert not np.any(np.isnan(fcst))

    ar1 = regressor.coefficients["ar1"]
    beta = regressor.coefficients["feed"]
    manual = []
    eta_prev = float(y_values[-1]) - beta * float(feed.iloc[-1])
    for i in range(3):
        eta_prev = ar1 * eta_prev
        manual.append(beta * exog_future[i, 0] + eta_prev)
    assert list(fcst) == pytest.approx(manual, rel=1e-6)


# -- row-sufficiency validation (H4) ------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "order,n_rows",
    [
        ((4, 0, 0), 5),  # AR(4)+const needs far more than 1 usable row
        ((0, 0, 3), 3),  # MA(3)+const needs far more than 0 usable rows
        ((2, 1, 2), 4),  # mixed AR/MA with differencing
    ],
)
def test_underdetermined_order_raises_flex_data_error(order, n_rows):
    """Fitting an order with too few rows raises FlexDataError instead of
    silently returning a rank-deficient, meaningless `lstsq` solution.
    """
    pytest.importorskip("scipy")

    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n_rows, freq="1h")
    y = pd.DataFrame({"y": rng.normal(size=n_rows)}, index=idx)

    with pytest.raises(FlexDataError, match="needs at least"):
        ArimaRegressor(order=order, max_ar_persistence=None).fit(
            pd.DataFrame(index=idx),
            y,
            input_units={},
            output_units="m^3/hr",
        )


@pytest.mark.unit
def test_sufficiently_sized_order_still_fits():
    """The tightened row-sufficiency check does not false-positive on a
    comfortably-sized fit."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(0)
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    y_values = np.zeros(n)
    for t in range(1, n):
        y_values[t] = 0.5 * y_values[t - 1] + rng.normal(0, 0.05)
    y = pd.DataFrame({"y": y_values}, index=idx)

    regressor = ArimaRegressor(order=(1, 0, 0), max_ar_persistence=None).fit(
        pd.DataFrame(index=idx), y
    )
    assert regressor.fitted is True


# -- predict(dynamic=False) (M4) -----------------------------------------------


@pytest.mark.unit
def test_predict_dynamic_false_raises():
    """dynamic=False is documented as unsupported and must raise loudly,
    not silently fall back to dynamic=True's behaviour (the parameter was
    previously read from the signature but never actually used).
    """
    pytest.importorskip("scipy")

    rng = np.random.default_rng(0)
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    y_values = np.zeros(n)
    for t in range(1, n):
        y_values[t] = 0.5 * y_values[t - 1] + rng.normal(0, 0.05)
    y = pd.DataFrame({"y": y_values}, index=idx)

    regressor = ArimaRegressor(order=(1, 0, 0), max_ar_persistence=None).fit(
        pd.DataFrame(index=idx), y
    )
    with pytest.raises(FlexConfigError, match="dynamic"):
        regressor.model.predict(steps=3, dynamic=False)


# -- FlexConfigError paths that were not previously exercised ------------------


@pytest.mark.unit
def test_statsforecast_absent_raises(monkeypatch):
    """With statsforecast unimportable, auto=True raises FlexConfigError."""
    monkeypatch.setitem(sys.modules, "statsforecast", None)
    monkeypatch.setitem(sys.modules, "statsforecast.models", None)

    y = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0]}, index=pd.date_range("2024-01-01", periods=3, freq="1h")
    )
    with pytest.raises(FlexConfigError, match=r"flex-pse\[parameterize\]"):
        ArimaRegressor(auto=True).fit(
            pd.DataFrame(index=y.index),
            y,
            input_units={},
            output_units="m^3/hr",
        )


@pytest.mark.unit
def test_to_surrogate_spec_missing_input_units_raises():
    """fit() raises when input_units is missing an exogenous column."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(0)
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    feed = pd.Series(rng.uniform(0.1, 1.0, size=n), index=idx, name="feed")
    vals = 2.0 * feed + rng.normal(0, 0.05, size=n)
    y = pd.DataFrame({"biogas": vals}, index=idx)

    with pytest.raises(FlexConfigError, match="input_units"):
        ArimaRegressor(order=(0, 0, 0)).fit(
            pd.DataFrame({"feed": feed}),
            y,
            input_units={},
            output_units="m^3/hr",
        )


@pytest.mark.unit
def test_max_ar_persistence_invalid_type_raises():
    """max_ar_persistence given a non-numeric value raises FlexConfigError."""
    with pytest.raises(FlexConfigError, match="max_ar_persistence"):
        ArimaRegressor(order=(1, 0, 0), max_ar_persistence="high")
