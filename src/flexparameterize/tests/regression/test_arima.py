"""Tests for ArimaRegressor: ARIMA time-series regression with exogenous inputs.

Importing this module (and ``ArimaRegressor`` itself) never requires
scipy -- only :meth:`ArimaRegressor.fit` does, lazily.
Tests that exercise a real fit call ``pytest.importorskip("scipy")``
themselves so the absence test below still runs (and passes) without the
extra installed.
"""

from __future__ import annotations

import os

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


def _capture_arima_warnings(monkeypatch) -> list[str]:
    """Record the ARIMA module's warnings.

    The flexcore logger does not propagate, so ``caplog`` never sees them.
    """
    from flexparameterize.regression import arima

    messages: list[str] = []
    monkeypatch.setattr(
        arima._log, "warning", lambda msg, *args: messages.append(msg % args)
    )
    return messages


def _free_run_rmse(regressor, X, y, horizon: int) -> float:
    """Score a fit by windowed free-run error at ``horizon``, whatever horizon
    it was fitted at, so fits at different horizons compare on one criterion."""
    from flexparameterize.regression.utils.arima_utils import output_error_residuals

    x = X.to_numpy(dtype=float) if X.shape[1] else None
    residuals = output_error_residuals(
        regressor.model.params,
        regressor.model.terms,
        y.to_numpy(dtype=float).ravel(),
        x,
        horizon,
    )
    return float(np.sqrt(np.mean(residuals**2)))


# -- absence tests -----------------------------------------------------------
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
    assert "ar.L1" in regressor.coefficients
    assert regressor.coefficients["ar.L1"] == pytest.approx(phi, rel=0.1)


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
    assert "exog.feed" in regressor.coefficients
    assert regressor.coefficients["exog.feed"] > 0.5


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
    assert "exog.x1" in regressor.coefficients
    assert "exog.x2" in regressor.coefficients
    assert regressor.coefficients["exog.x1"] == pytest.approx(0.4, rel=0.2)
    assert regressor.coefficients["exog.x2"] == pytest.approx(1e-5, rel=0.2)


# -- protocol conformance ----------------------------------------------------


def _clean_feed_data(n: int = 48):
    """Return evenly spaced ``(X, y)`` frames for the data-cleanliness tests."""
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(12)
    feed = rng.uniform(0.1, 1.0, size=n)
    y = pd.DataFrame({"y": 2.0 * feed + rng.normal(0, 0.01, size=n)}, index=idx)
    return pd.DataFrame({"feed": feed}, index=idx), y


def _fit_feed(X, y):
    return ArimaRegressor(order=(1, 0, 0)).fit(
        X, y, input_units={"feed": "dimensionless"}
    )


@pytest.mark.unit
@pytest.mark.parametrize("frame", ["X", "y"])
def test_fit_rejects_null_values(frame):
    """A null in X or y is refused rather than silently dropped."""
    X, y = _clean_feed_data()
    (X if frame == "X" else y).iloc[10] = np.nan

    column = "feed" if frame == "X" else "y"
    with pytest.raises(FlexDataError, match=rf"1 row\(s\) hold a null.*\['{column}'\]"):
        _fit_feed(X, y)


@pytest.mark.unit
def test_fit_rejects_misaligned_x_and_y():
    """Rows present in only one of X and y surface as nulls, so fit refuses."""
    X, y = _clean_feed_data()
    with pytest.raises(FlexDataError, match="share the same index"):
        _fit_feed(X.iloc[:-1], y)


@pytest.mark.unit
@pytest.mark.parametrize(
    "mangle",
    [
        pytest.param(lambda df: df.drop(df.index[20]), id="gap"),
        pytest.param(lambda df: df.iloc[::-1], id="decreasing"),
        pytest.param(lambda df: pd.concat([df.iloc[:5], df.iloc[4:]]), id="duplicate"),
    ],
)
def test_fit_rejects_unevenly_spaced_timestamps(mangle):
    """The lags and the surrogate history assume one constant time step."""
    X, y = _clean_feed_data()
    with pytest.raises(FlexDataError, match="evenly spaced, increasing"):
        _fit_feed(mangle(X), mangle(y))


@pytest.mark.unit
@pytest.mark.parametrize("role", ["X", "y"])
def test_fit_rejects_a_series(role):
    """X and y must be DataFrames; a Series gets a clear error, not AttributeError."""
    X, y = _clean_feed_data()
    args = {"X": X, "y": y}
    args[role] = args[role].iloc[:, 0]

    with pytest.raises(FlexDataError, match=rf"takes {role} as a DataFrame.*Series"):
        _fit_feed(args["X"], args["y"])


@pytest.mark.unit
def test_fit_rejects_multi_column_y():
    X, y = _clean_feed_data()
    y["other"] = y["y"]

    with pytest.raises(FlexDataError, match=r"one output column; y has 2"):
        _fit_feed(X, y)


@pytest.mark.unit
def test_fit_accepts_clean_evenly_spaced_data():
    pytest.importorskip("scipy")
    X, y = _clean_feed_data()

    regressor = _fit_feed(X, y)

    assert regressor.n_samples == len(y)
    assert regressor.coefficients["exog.feed"] == pytest.approx(2.0, abs=0.05)


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
    assert len(data["history"]["eta_values"]) == n
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
    assert "intercept" not in regressor.to_fit_result().coefficients
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
    assert regressor.to_fit_result().coefficients["intercept"] == pytest.approx(0.0)


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
    """fit_diagnostics returns finite fit statistics under fixed keys."""
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
    assert set(diag) == {
        "aic",
        "bic",
        "log_likelihood",
        "n_parameters",
        "n_samples",
        "sigma2",
        "rmse",
        "free_run_rmse",
        "forecast_horizon",
        "ma_max_root",
    }
    assert all(np.isfinite(value) for value in diag.values())


@pytest.mark.unit
def test_fit_diagnostics_before_fit_raises():
    """fit_diagnostics before fit raises FlexDataError."""
    regressor = ArimaRegressor(order=(1, 0, 0))
    with pytest.raises(FlexDataError, match="no fit yet"):
        regressor.fit_diagnostics()


# -- auto_arima --------------------------------------------------------------


@pytest.mark.unit
def test_auto_arima_selects_order():
    """auto=True runs the order search and stores a valid order."""
    pytest.importorskip("statsmodels")  # the d test runs ADF
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
    """Direct ARIMA fits and forecasts on real biogas data match statsmodels.

    Uses the default ``forecast_horizon=1`` on purpose: statsmodels SARIMAX
    maximizes the exact Gaussian likelihood, which is a one-step
    (equation-error) criterion. Comparing a multi-step fit against it would
    be comparing two different estimators, not validating ours.
    """
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
        auto=True,
        max_ar_persistence=None,
        max_p=3,
        max_q=3,
        max_d=1,
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

    assert "intercept" not in regressor.coefficients
    assert regressor.coefficients["drift"] == pytest.approx(0.35, abs=0.03)
    spec = regressor.to_surrogate_spec()
    assert "const" not in spec.data["coefficients"]
    assert "drift" in spec.data["coefficients"]
    ArimaSurrogate(spec.data)


@pytest.mark.unit
def test_auto_arima_d_in_zero_or_one():
    """auto=True allows d=0 or d=1, but rejects d>1."""
    pytest.importorskip("statsmodels")  # the d test runs ADF
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
@pytest.mark.parametrize("key", ["seasonal", "ic", "stepwise"])
def test_auto_kwargs_reject_unknown_options(key):
    """The search is always non-seasonal and AIC-ranked; other options are refused."""
    with pytest.raises(FlexConfigError, match=rf"does not accept \['{key}'\]"):
        ArimaRegressor(auto=True, **{key: True})


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [{"include_drift": True}, {"seasonal_order": (0, 0, 0, 24)}, {"max_p": 3}],
)
def test_extra_keywords_require_auto(kwargs):
    """Without auto=True, extra keywords (or removed options) are refused."""
    with pytest.raises(FlexConfigError, match="only apply with auto=True"):
        ArimaRegressor(order=(1, 1, 0), **kwargs)


@pytest.mark.unit
@pytest.mark.parametrize("kwargs", [{"max_d": 2}, {"d": 2}])
def test_auto_arima_d_greater_than_one_raises(kwargs):
    """auto=True refuses a search that could reach d>1, before any fit."""
    with pytest.raises(FlexConfigError, match="at most d=1"):
        ArimaRegressor(auto=True, **kwargs)


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
    ar_coef = regressor.coefficients["ar.L1"]
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


@pytest.mark.unit
def test_nonstationary_ar_block_is_rejected_even_within_the_coefficient_bound():
    """Bounding each AR coefficient does not make the AR block stationary.

    ``ar = [0.85, 0.85]`` sits inside the default per-coefficient bound, yet
    its characteristic root is ~1.44, so a forecast from it explodes.
    """
    pytest.importorskip("scipy")

    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(3)
    vals = np.zeros(n)
    vals[:2] = 1.0
    for t in range(2, n):
        vals[t] = 0.85 * vals[t - 1] + 0.85 * vals[t - 2] + rng.normal(0, 0.01)
    y = pd.DataFrame({"y": vals}, index=idx)

    with pytest.raises(FlexConfigError, match="not stationary"):
        ArimaRegressor(order=(2, 0, 0), include_mean=False).fit(
            pd.DataFrame(index=idx), y
        )


@pytest.mark.unit
def test_output_error_rejects_a_noninvertible_ma_block(monkeypatch):
    """A multi-step fit with MA roots outside the unit circle is refused.

    The forecast seeds its MA lags from the one-step residuals, which a
    non-invertible MA makes explode. Each coefficient of ``[-0.99, -0.99]``
    is inside the MA bound, but the block's largest root is ~1.6.
    """
    pytest.importorskip("scipy")
    from flexparameterize.regression import arima

    real_fit_direct = arima.fit_direct

    def noninvertible(*args, **kwargs):
        results = real_fit_direct(*args, **kwargs)
        results.params[2:4] = -0.99  # layout: const, ar1, ma1, ma2
        return results

    monkeypatch.setattr(arima, "fit_direct", noninvertible)

    n = 120
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    y = pd.DataFrame({"y": np.random.default_rng(4).normal(0, 0.1, size=n)}, index=idx)
    with pytest.raises(FlexConfigError, match="non-invertible"):
        ArimaRegressor(order=(1, 0, 2), forecast_horizon=24).fit(
            pd.DataFrame(index=idx), y
        )


@pytest.mark.unit
def test_equation_error_keeps_a_noninvertible_ma_block_with_a_warning(monkeypatch):
    """The default one-step fit only warns about a non-invertible MA block.

    Its forecast seeds from the very residuals it minimized, so they stay
    small; on the biogas data most MA fits land here and forecast fine.
    """
    pytest.importorskip("scipy")

    if not os.path.exists(_BIO_GAS_PATH):
        pytest.skip(f"Test data not found at {_BIO_GAS_PATH}")

    warnings = _capture_arima_warnings(monkeypatch)
    df = _bio_gas_dataframe().asfreq("15min").dropna().iloc[:384]
    regressor = ArimaRegressor(order=(0, 0, 3)).fit(
        df[["feed_volume_kg", "TS_pct"]],
        df[["biogas_m3_hour"]],
        input_units={"feed_volume_kg": "kg", "TS_pct": "dimensionless"},
        output_units="m^3/hr",
    )

    assert regressor.fit_diagnostics()["ma_max_root"] >= 1.0
    assert any("non-invertible" in message for message in warnings)


@pytest.mark.component
def test_output_error_ma_block_stays_bounded_on_biogas():
    """The held-out failure: biogas MA(1), output error, origin row 4224.

    Unbounded, this fit landed at ``ma1 = 1.116`` and its first forecast step
    was ~1e6. Bounded, it forecasts at the equation-error level.

    Marked component purely on runtime (several seconds); no solver.
    """
    pytest.importorskip("scipy")

    if not os.path.exists(_BIO_GAS_PATH):
        pytest.skip(f"Test data not found at {_BIO_GAS_PATH}")

    df = _bio_gas_dataframe().asfreq("15min").dropna()
    X = df[["feed_volume_kg", "TS_pct"]]
    y = df[["biogas_m3_hour"]]
    train, test = slice(4224 - 384, 4224), slice(4224, 4224 + 192)

    regressor = ArimaRegressor(order=(0, 0, 1), forecast_horizon=192).fit(
        X.iloc[train],
        y.iloc[train],
        input_units={"feed_volume_kg": "kg", "TS_pct": "dimensionless"},
        output_units="m^3/hr",
    )
    forecast = regressor.model.predict(steps=192, exog=X.iloc[test].to_numpy())
    rmse = float(np.sqrt(np.mean((forecast - y.iloc[test].to_numpy().ravel()) ** 2)))

    assert abs(regressor.coefficients["ma.L1"]) <= 0.99 + 1e-9
    assert rmse < 0.02


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
    assert regressor.model.terms.d == 1

    # predict should return original-scale values
    fcst = regressor.model.predict(steps=10)
    assert len(fcst) == 10
    assert not np.any(np.isnan(fcst))

    # In-sample dynamic prediction should match observed scale
    sm_insample = regressor.model.predict(
        steps=20,
        start=n - 20,
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
    assert "intercept" not in regressor.coefficients

    fcst = regressor.model.predict(steps=3)
    assert len(fcst) == 3
    assert not np.any(np.isnan(fcst))

    ar1 = regressor.coefficients["ar.L1"]
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
    assert "intercept" not in regressor.coefficients

    exog_future = np.array([[0.5], [0.6], [0.7]])
    fcst = regressor.model.predict(steps=3, exog=exog_future)
    assert len(fcst) == 3
    assert not np.any(np.isnan(fcst))

    ar1 = regressor.coefficients["ar.L1"]
    beta = regressor.coefficients["exog.feed"]
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


# -- FlexConfigError paths that were not previously exercised ------------------


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


# -- forecast_horizon: the fitting objective ----------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "horizon", [0, -5, 2.5, "auto", True, "0D", "-1D", "NaT", None]
)
def test_forecast_horizon_rejects_invalid_values(horizon):
    """forecast_horizon must be a positive int or a positive duration."""
    with pytest.raises(FlexConfigError, match="forecast_horizon"):
        ArimaRegressor(order=(1, 0, 0), forecast_horizon=horizon)


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs", [{"fit_objective": "output_error"}, {"fit_solver": "ipopt"}]
)
def test_removed_fitting_options_are_rejected(kwargs):
    """fit_objective was merged into forecast_horizon and the ipopt fitting
    backend was removed; passing either is an error rather than a silently
    ignored keyword."""
    with pytest.raises(FlexConfigError, match=next(iter(kwargs))):
        ArimaRegressor(order=(1, 0, 0), forecast_horizon=24, **kwargs)


@pytest.mark.unit
def test_default_horizon():
    """Defaults stay on the classical one-step fit.

    A longer horizon only changes much when ``d >= 1`` -- it exists to stop
    a drift error integrating over a free run -- and it costs an order of
    magnitude more time, so it is opt-in.
    """
    regressor = ArimaRegressor(order=(1, 0, 0))
    assert regressor.forecast_horizon == 1


@pytest.mark.unit
def test_explicit_forecast_horizon_is_used_verbatim():
    """An explicit horizon is visible before the fit and survives it."""
    pytest.importorskip("scipy")

    n = 150
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(4)
    y = pd.DataFrame({"y": np.cumsum(rng.normal(0, 0.1, size=n))}, index=idx)

    regressor = ArimaRegressor(
        order=(1, 1, 0),
        forecast_horizon=20,
        max_ar_persistence=None,
    )
    assert regressor.forecast_horizon == 20
    regressor.fit(pd.DataFrame(index=idx), y)
    assert regressor.forecast_horizon == 20
    assert regressor.fit_diagnostics()["forecast_horizon"] == 20.0


@pytest.mark.unit
def test_duration_horizon_fits_the_same_as_its_step_count():
    """A duration is the same fit as the step count it covers on the
    training grid, and stays a duration on the regressor across refits."""
    pytest.importorskip("scipy")

    n = 200
    rng = np.random.default_rng(4)
    values = np.cumsum(rng.normal(0, 0.1, size=n))

    def frames(freq):
        idx = pd.date_range("2024-01-01", periods=n, freq=freq)
        return pd.DataFrame(index=idx), pd.DataFrame({"y": values}, index=idx)

    by_steps = ArimaRegressor(order=(1, 1, 0), forecast_horizon=8).fit(*frames("15min"))
    by_duration = ArimaRegressor(order=(1, 1, 0), forecast_horizon="2h")
    assert by_duration.forecast_horizon == pd.Timedelta(hours=2)

    by_duration.fit(*frames("15min"))
    assert by_duration.fit_diagnostics()["forecast_horizon"] == 8.0
    assert by_duration.coefficients == pytest.approx(by_steps.coefficients)

    # Refit on an hourly grid: the same duration is now 2 steps.
    by_duration.fit(*frames("1h"))
    assert by_duration.forecast_horizon == pd.Timedelta(hours=2)
    assert by_duration.fit_diagnostics()["forecast_horizon"] == 2.0


@pytest.mark.unit
def test_free_run_rmse_matches_hand_computation_at_the_fitted_horizon():
    """metrics["free_run_rmse"] is the windowed free-run error at the fit's
    own horizon; at ``forecast_horizon=1`` that is the one-step rmse.

    This is the diagnostic whose absence let a drifting d=1 fit look good:
    one-step rmse improves with order while the free run degrades.
    """
    pytest.importorskip("scipy")
    from flexparameterize.regression.utils.arima_utils import (
        output_error_residuals,
    )

    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(5)
    values = np.zeros(n)
    for t in range(2, n):
        values[t] = (
            values[t - 1] + 0.4 * (values[t - 1] - values[t - 2]) + rng.normal(0, 0.05)
        )
    y = pd.DataFrame({"y": values}, index=idx)
    X = pd.DataFrame(index=idx)

    for horizon in (1, 24):
        regressor = ArimaRegressor(order=(1, 1, 0), forecast_horizon=horizon).fit(X, y)
        reported = regressor.metrics["free_run_rmse"]
        if horizon == 1:
            assert reported == pytest.approx(regressor.metrics["rmse"], rel=1e-12)
        assert np.isfinite(reported)

        coefficients = regressor.coefficients
        theta = np.array([coefficients["drift"], coefficients["ar.L1"]], dtype=float)
        terms = regressor.model.terms
        residuals = output_error_residuals(
            theta, terms, values, None, regressor.forecast_horizon
        )
        expected = float(np.sqrt(np.mean(residuals**2)))
        assert reported == pytest.approx(expected, rel=1e-9)


@pytest.mark.unit
def test_output_error_trades_parameter_recovery_for_forecast_accuracy():
    """Document the estimator trade, so neither half surprises a caller.

    On a well-specified series the one-step criterion is consistent and
    recovers the true parameters; the free-run criterion is not consistent
    and returns whatever forecasts best over its horizon. Each horizon wins
    on its own criterion, and that is the whole choice between them.
    """
    pytest.importorskip("scipy")

    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(7)
    true_drift, true_phi = 0.05, 0.4
    values = np.zeros(n)
    for t in range(2, n):
        values[t] = (
            values[t - 1]
            + true_drift
            + true_phi * (values[t - 1] - values[t - 2])
            + rng.normal(0, 0.01)
        )
    y = pd.DataFrame({"y": values}, index=idx)
    X = pd.DataFrame(index=idx)

    equation = ArimaRegressor(order=(1, 1, 0), max_ar_persistence=None).fit(X, y)
    output = ArimaRegressor(
        order=(1, 1, 0), forecast_horizon=192, max_ar_persistence=None
    ).fit(X, y)

    # The one-step fit recovers the generating parameters.
    assert equation.coefficients["drift"] == pytest.approx(true_drift, rel=0.15)
    assert equation.coefficients["ar.L1"] == pytest.approx(true_phi, rel=0.2)
    # The 192-step fit wins on the criterion it minimizes.
    assert output.metrics["free_run_rmse"] <= _free_run_rmse(equation, X, y, 192)


@pytest.mark.unit
def test_output_error_curbs_the_spurious_d1_drift_ramp():
    """A driftless d=1 series must not acquire a ramping drift term.

    Equation error leaves drift nearly unidentified, so it floats to a value
    that costs almost nothing per step but accumulates over the horizon. The
    output-error objective sees the accumulation.
    """
    pytest.importorskip("scipy")

    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(8)
    values = np.zeros(n)
    for t in range(2, n):
        values[t] = (
            values[t - 1] + 0.45 * (values[t - 1] - values[t - 2]) + rng.normal(0, 0.05)
        )
    y = pd.DataFrame({"y": values}, index=idx)
    X = pd.DataFrame(index=idx)

    equation = ArimaRegressor(order=(1, 1, 1), max_ar_persistence=None).fit(X, y)
    output = ArimaRegressor(
        order=(1, 1, 1), forecast_horizon=192, max_ar_persistence=None
    ).fit(X, y)

    assert output.metrics["free_run_rmse"] <= _free_run_rmse(
        equation, X, y, 192
    ), "a 192-step fit must not be worse on the criterion it optimizes"


@pytest.mark.unit
def test_output_error_spec_satisfies_the_surrogate_contract():
    """An output-error fit still emits a spec ArimaSurrogate accepts."""
    pytest.importorskip("scipy")

    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(9)
    feed = pd.Series(rng.uniform(0.1, 1.0, size=n), index=idx, name="feed")
    values = np.zeros(n)
    for t in range(1, n):
        values[t] = 0.4 * values[t - 1] + 2.0 * feed.iloc[t] + rng.normal(0, 0.02)
    y = pd.DataFrame({"y": values}, index=idx)

    regressor = ArimaRegressor(order=(1, 0, 1), max_ar_persistence=None).fit(
        pd.DataFrame({"feed": feed}),
        y,
        input_units={"feed": "dimensionless"},
        output_units="m^3/hr",
    )
    spec = regressor.to_surrogate_spec()

    assert spec.surrogate_type == SurrogateType.ARIMA
    assert spec.data["coefficients"]["order"] == [1, 0, 1]
    ArimaSurrogate(spec.data)
    assert np.isfinite(regressor.to_fit_result().metrics["free_run_rmse"])


@pytest.mark.component
def test_output_error_beats_equation_error_on_biogas_high_order():
    """The reported bug: real biogas data, ARIMA(3,1,3), d=1 drift ramp.

    The equation-error fit drifts upward over the horizon because drift is
    unidentified by one-step residuals (|t| ~ 1.4) while it accumulates as
    drift*steps/(1-sum(ar)) in the free run the surrogate performs.

    Marked component rather than unit purely on runtime: two fits of a
    seven-parameter model over 384 rows exceeds the sub-second unit budget.
    No solver is involved.
    """
    pytest.importorskip("scipy")

    if not os.path.exists(_BIO_GAS_PATH):
        pytest.skip(f"Test data not found at {_BIO_GAS_PATH}")

    df = _bio_gas_dataframe().dropna(
        subset=["biogas_m3_hour", "feed_volume_kg", "TS_pct"]
    )
    X = df[["feed_volume_kg", "TS_pct"]].iloc[:384]
    y = df[["biogas_m3_hour"]].iloc[:384]
    units = {
        "input_units": {"feed_volume_kg": "kg", "TS_pct": "dimensionless"},
        "output_units": "m^3/hr",
    }

    equation = ArimaRegressor(order=(3, 1, 3), max_ar_persistence=None).fit(
        X, y, **units
    )
    output = ArimaRegressor(
        order=(3, 1, 3), forecast_horizon=192, max_ar_persistence=None
    ).fit(X, y, **units)

    baseline = _free_run_rmse(equation, X, y, 192)
    assert output.metrics["free_run_rmse"] < 0.6 * baseline, (
        f"192-step free-run rmse {output.metrics['free_run_rmse']:.5f} vs "
        f"one-step fit {baseline:.5f}"
    )


@pytest.mark.unit
def test_whole_series_horizon_leaves_the_ma_block_at_its_warm_start(monkeypatch):
    """A window spanning the series must not disturb the MA coefficients.

    At a full-series horizon the MA block reaches only the first q of
    several hundred residuals, so its gradient is negligible and the
    coefficients keep the one-step warm-start values, where they were
    identified. An intermediate horizon identifies them weakly instead,
    which is worse: it drags them out of the unit circle -- which is why the
    MA bound and invertibility guard exist for every other case.
    """
    pytest.importorskip("scipy")

    n = 150
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(11)
    values = np.zeros(n)
    noise = rng.normal(0, 0.05, size=n)
    for t in range(2, n):
        values[t] = (
            values[t - 1]
            + 0.4 * (values[t - 1] - values[t - 2])
            + noise[t]
            + 0.3 * noise[t - 1]
        )
    y = pd.DataFrame({"y": values}, index=idx)
    X = pd.DataFrame(index=idx)

    equation = ArimaRegressor(order=(1, 1, 1)).fit(X, y)
    warnings = _capture_arima_warnings(monkeypatch)
    # Windows start at row max(p, q) + d = 2, so one window spans the rest.
    output = ArimaRegressor(order=(1, 1, 1), forecast_horizon=n - 2).fit(X, y)

    assert output.coefficients["ma.L1"] == pytest.approx(
        equation.coefficients["ma.L1"], rel=1e-3
    )
    assert output.fit_diagnostics()["ma_max_root"] < 1.0
    assert not any("non-invertible" in message for message in warnings)


@pytest.mark.unit
def test_fit_does_not_hide_warnings_from_the_refit(monkeypatch):
    """fit() does not filter the warnings its least-squares fit raises."""
    pytest.importorskip("scipy")
    import warnings

    from flexparameterize.regression import arima

    real_fit_direct = arima.fit_direct

    def noisy_fit_direct(*args, **kwargs):
        warnings.warn("refit warning", RuntimeWarning, stacklevel=2)
        return real_fit_direct(*args, **kwargs)

    monkeypatch.setattr(arima, "fit_direct", noisy_fit_direct)
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=60, freq="1h")
    y = pd.DataFrame({"y": rng.normal(size=60)}, index=idx)

    with pytest.warns(RuntimeWarning, match="refit warning"):
        ArimaRegressor(order=(1, 0, 0)).fit(pd.DataFrame(index=idx), y)


@pytest.mark.unit
def test_fit_requires_datetime_index():
    """Without timestamps the surrogate history cannot be dated, so fit refuses."""
    y = pd.DataFrame({"y": np.random.default_rng(0).normal(size=40)})
    with pytest.raises(FlexDataError, match="DatetimeIndex.*got RangeIndex"):
        ArimaRegressor(order=(1, 0, 0)).fit(pd.DataFrame(index=y.index), y)


# -- history_length -----------------------------------------------------------


def _fitted_arimax(n: int = 400):
    """Return an ARIMAX(1, 1, 1) fit on ``n`` hourly rows."""
    pytest.importorskip("scipy")
    rng = np.random.default_rng(17)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    feed = pd.Series(rng.uniform(0.5, 1.5, size=n), index=idx, name="feed")
    eta = np.cumsum(rng.normal(0, 0.05, size=n))
    y = pd.DataFrame({"biogas": 2.0 * feed.to_numpy() + eta}, index=idx)
    regressor = ArimaRegressor(order=(1, 1, 1), max_ar_persistence=None).fit(
        pd.DataFrame({"feed": feed}),
        y,
        input_units={"feed": "kg/hr"},
        output_units="m^3/hr",
    )
    return regressor, idx


def _initial_state_at(spec, start: pd.Timestamp):
    """Return the surrogate's pre-horizon ``(eta, eps)`` lags at ``start``."""
    import pyomo.environ as pyo
    from pyomo.environ import units as pyunits

    from flexops.core.time_block import TimeBlock

    model = pyo.ConcreteModel()
    model.time_block = TimeBlock(
        start_date=start.isoformat(),
        end_date=(start + pd.Timedelta(hours=3)).isoformat(),
        time_step=1 * pyunits.hr,
    )
    return ArimaSurrogate(spec.data)._initial_state(model.time_block)


@pytest.mark.unit
def test_surrogate_spec_default_history_keeps_one_week():
    regressor, idx = _fitted_arimax()

    history = regressor.to_surrogate_spec().data["history"]

    # 168 hourly points plus the max(p + d, q) = 2 lags ahead of them.
    assert len(history["eta_values"]) == 168 + 2
    assert len(history["eps_values"]) == 168 + 2
    assert pd.Timestamp(history["start_date"]) == idx[-168]


@pytest.mark.unit
@pytest.mark.parametrize("offset", [-72, -1])
def test_trimmed_history_matches_full_history_inside_the_window(offset):
    """Trimming loses nothing for a horizon that starts in the kept window."""
    regressor, idx = _fitted_arimax()
    full = regressor.to_surrogate_spec(history_length=None)
    trimmed = regressor.to_surrogate_spec(history_length="3D")

    full_state = _initial_state_at(full, idx[offset])
    trimmed_state = _initial_state_at(trimmed, idx[offset])

    for full_lags, trimmed_lags in zip(full_state, trimmed_state, strict=True):
        assert trimmed_lags == pytest.approx(full_lags)


@pytest.mark.unit
def test_zero_history_holds_the_terminal_state_for_forecasting(monkeypatch):
    """history_length=0 starts a horizon right after training with no forecast."""
    from flexops.surrogates import arima as arima_surrogate

    messages: list[str] = []
    monkeypatch.setattr(
        arima_surrogate._log, "warning", lambda msg, *args: messages.append(msg)
    )
    regressor, idx = _fitted_arimax()
    full = regressor.to_surrogate_spec(history_length=None)
    terminal = regressor.to_surrogate_spec(history_length=0)
    after_training = idx[-1] + pd.Timedelta(hours=1)

    assert len(terminal.data["history"]["eta_values"]) == 2
    assert pd.Timestamp(terminal.data["history"]["start_date"]) == after_training
    for full_lags, terminal_lags in zip(
        _initial_state_at(full, after_training),
        _initial_state_at(terminal, after_training),
        strict=True,
    ):
        assert terminal_lags == pytest.approx(full_lags)
    assert messages == []


@pytest.mark.unit
def test_horizon_before_trimmed_history_names_the_fix():
    regressor, idx = _fitted_arimax()
    trimmed = regressor.to_surrogate_spec(history_length="1D")

    with pytest.raises(FlexConfigError, match="longer history_length"):
        _initial_state_at(trimmed, idx[-48])


@pytest.mark.unit
def test_surrogate_spec_rejects_invalid_history_length():
    regressor, _idx = _fitted_arimax()

    with pytest.raises(FlexConfigError, match="history_length"):
        regressor.to_surrogate_spec(history_length=-1)
