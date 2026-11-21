"""Tests for LinearRegressor: OLS-fitted multilinear regression.

Importing this module (and ``LinearRegressor`` itself) never requires
scikit-learn -- only ``fit`` does, lazily -- so each test that exercises a
real fit calls ``pytest.importorskip("sklearn")`` itself; the absence test
below is deliberately exempt so it still runs (and passes) with the extra
uninstalled.
"""

import sys

import numpy as np
import pandas as pd
import pytest

from flexcore.config.schema import SurrogateType
from flexcore.exceptions import FlexConfigError
from flexparameterize.regression import Regressor
from flexparameterize.regression.linear import LinearRegressor


@pytest.mark.unit
def test_recovers_coefficients_noisy_pump():
    """OLS recovers a known slope and intercept from noisy synthetic data."""
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(42)
    flow = rng.uniform(1.0, 20.0, size=200)
    noise = rng.normal(0.0, 0.01, size=200)
    power = 0.4 * flow + 2.0 + noise
    X = pd.DataFrame({"flow": flow})
    y = pd.DataFrame({"power": power})

    regressor = LinearRegressor().fit(
        X, y, input_units={"flow": "m^3/hr"}, output_units="kW"
    )

    assert regressor.coefficients["flow"] == pytest.approx(0.4, rel=1e-2)
    assert regressor.coefficients["intercept"] == pytest.approx(2.0, abs=0.05)


@pytest.mark.unit
def test_two_input_columns():
    """Two linear input columns (no cross term) are both recovered."""
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(7)
    flow = rng.uniform(1.0, 20.0, size=200)
    pressure = rng.uniform(1e5, 5e5, size=200)
    power = 0.4 * flow + 1e-5 * pressure + 2.0
    X = pd.DataFrame({"flow": flow, "pressure": pressure})
    y = pd.DataFrame({"power": power})

    regressor = LinearRegressor().fit(
        X, y, input_units={"flow": "m^3/hr", "pressure": "Pa"}, output_units="kW"
    )

    assert regressor.coefficients["flow"] == pytest.approx(0.4, rel=1e-2)
    assert regressor.coefficients["pressure"] == pytest.approx(1e-5, rel=1e-2)

    with pytest.raises(FlexConfigError, match="pressure"):
        LinearRegressor().fit(X, y, input_units={"flow": "m^3/hr"}, output_units="kW")


@pytest.mark.unit
def test_isinstance_regressor():
    """LinearRegressor structurally conforms to Regressor and behaves as one."""
    pytest.importorskip("sklearn")
    assert isinstance(LinearRegressor(), Regressor)
    X = pd.DataFrame({"flow": [1.0, 2.0, 3.0, 4.0]})
    y = pd.DataFrame({"power": [2.4, 2.8, 3.2, 3.6]})
    regressor = LinearRegressor().fit(
        X, y, input_units={"flow": "m^3/hr"}, output_units="kW"
    )
    result = regressor.to_fit_result()
    assert result.coefficients["flow"] == pytest.approx(0.4, rel=1e-6)


@pytest.mark.unit
def test_provenance_populated():
    """Fit metrics are finite and the emitted spec's provenance is JSON-safe."""
    import json

    pytest.importorskip("sklearn")
    X = pd.DataFrame({"flow": [1.0, 2.0, 3.0, 4.0]})
    y = pd.DataFrame({"power": [2.4, 2.8, 3.2, 3.6]})
    regressor = LinearRegressor().fit(
        X, y, input_units={"flow": "m^3/hr"}, output_units="kW"
    )

    result = regressor.to_fit_result()
    assert np.isfinite(result.metrics["r2"])
    assert np.isfinite(result.metrics["rmse"])

    spec = regressor.to_surrogate_spec()
    assert spec.surrogate_type == SurrogateType.MULTILINEAR
    provenance = {"n_samples": result.n_samples, **result.metrics}
    json.dumps(provenance)


@pytest.mark.unit
def test_sklearn_absent_raises(monkeypatch):
    """With sklearn unimportable, fitting raises FlexConfigError naming the extra."""
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.linear_model", None)

    X = pd.DataFrame({"flow": [1.0, 2.0, 3.0]})
    y = pd.DataFrame({"power": [1.0, 2.0, 3.0]})
    with pytest.raises(FlexConfigError, match=r"flex-pse\[parameterize\]"):
        LinearRegressor().fit(X, y, input_units={"flow": "m^3/hr"}, output_units="kW")


def _cross_term_frames():
    """Noise-free ``y = 1 + 2a + 3b + 4ab`` with an engineered ``a*b`` column."""
    rng = np.random.default_rng(11)
    a = rng.uniform(0.0, 1.0, size=50)
    b = rng.uniform(0.0, 1.0, size=50)
    X = pd.DataFrame({"a": a, "b": b, "a*b": a * b})
    y = pd.DataFrame({"y": 1.0 + 2.0 * a + 3.0 * b + 4.0 * a * b})
    return X, y


@pytest.mark.unit
def test_cross_term_fit_recovers_known_coefficients():
    """A product column is fitted as a cross-term coefficient."""
    pytest.importorskip("sklearn")
    X, y = _cross_term_frames()

    regressor = LinearRegressor().fit(
        X, y, input_units={"a": "m^3/s", "b": "dimensionless"}, output_units="W"
    )

    expected = {"intercept": 1.0, "a": 2.0, "b": 3.0, "a*b": 4.0}
    for key, value in expected.items():
        assert regressor.coefficients[key] == pytest.approx(value, abs=1e-10)


@pytest.mark.unit
def test_cross_term_spec_lists_factors_as_input_variables():
    """The spec's inputs are the factors; the product is only a coefficient key."""
    pytest.importorskip("sklearn")
    X, y = _cross_term_frames()

    spec = (
        LinearRegressor()
        .fit(X, y, input_units={"a": "m^3/s", "b": "dimensionless"}, output_units="W")
        .to_surrogate_spec()
    )

    assert spec.data["input_variables"] == {"a": "m^3/s", "b": "dimensionless"}
    assert "a*b" in spec.data["coefficients"]


@pytest.mark.unit
def test_cross_term_needs_units_only_for_factors():
    """A product column needs no units of its own, but every factor does."""
    pytest.importorskip("sklearn")
    X, y = _cross_term_frames()
    product_only = X[["a*b"]]

    regressor = LinearRegressor().fit(
        product_only,
        y,
        input_units={"a": "m^3/s", "b": "dimensionless"},
        output_units="W",
    )
    assert list(regressor.to_surrogate_spec().data["input_variables"]) == ["a", "b"]

    with pytest.raises(FlexConfigError, match="'b'"):
        LinearRegressor().fit(X, y, input_units={"a": "m^3/s"}, output_units="W")
