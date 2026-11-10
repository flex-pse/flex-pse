"""Tests for the Regressor Protocol and FitResult dataclass."""

import pandas as pd
import pytest

from flexcore.exceptions import FlexDataError
from flexparameterize.regression import ConstantIntensityRegressor
from flexparameterize.regression.base import (
    FitResult,
    Regressor,
    require_frame,
    single_column,
)
from flexparameterize.regression.linear import LinearRegressor


class _NotARegressor:
    """Missing to_surrogate_spec -- should fail the structural check."""

    def fit(self, X, y):
        return self

    def to_fit_result(self):
        return None


@pytest.mark.unit
def test_protocol_conformance():
    """A conforming class isinstance-checks true; calling it returns real types."""
    assert isinstance(ConstantIntensityRegressor(), Regressor)
    assert not isinstance(_NotARegressor(), Regressor)

    X = pd.DataFrame({"flow": [1.0, 2.0, 4.0]})
    y = pd.DataFrame({"power": [0.5, 1.0, 2.0]})
    regressor = ConstantIntensityRegressor().fit(X, y)

    result = regressor.to_fit_result()
    assert isinstance(result, FitResult)
    assert result.coefficients == {"energy_intensity": pytest.approx(0.5)}
    assert result.metrics["r2"] == pytest.approx(1.0)
    assert result.n_samples == 3
    assert result.data_window == (0, 2)


@pytest.mark.unit
def test_to_fit_result_before_fit_raises():
    """Calling to_fit_result before fit raises FlexDataError."""
    with pytest.raises(FlexDataError):
        ConstantIntensityRegressor().to_fit_result()


@pytest.mark.unit
def test_require_frame_rejects_a_series_naming_the_regressor():
    series = pd.Series([1.0, 2.0], name="flow")
    with pytest.raises(FlexDataError, match=r"^SomeRegressor\.fit takes X as a Data"):
        require_frame(series, regressor="SomeRegressor", role="X")


@pytest.mark.unit
def test_single_column_returns_the_named_column():
    frame = pd.DataFrame({"power": [1.0, 2.0]})
    column = single_column(frame, regressor="R", role="y", what="power")
    assert column.name == "power"
    assert list(column) == [1.0, 2.0]


@pytest.mark.unit
def test_single_column_rejects_extra_columns_with_hint():
    frame = pd.DataFrame({"a": [1.0], "b": [2.0]})
    with pytest.raises(
        FlexDataError,
        match=r"^R fits one flow column; X has 2 \(\['a', 'b'\]\)\. "
        r"Select the single flow column, or split it\.$",
    ):
        single_column(frame, regressor="R", role="X", what="flow", hint=", or split it")


@pytest.mark.unit
def test_constant_regressor_errors_name_itself():
    two = pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0]})
    one = pd.DataFrame({"power": [1.0, 2.0]})
    with pytest.raises(FlexDataError, match="^ConstantIntensityRegressor fits one"):
        ConstantIntensityRegressor().fit(two, one)
    with pytest.raises(FlexDataError, match="takes y as a DataFrame, got Series"):
        ConstantIntensityRegressor().fit(one, one["power"])


@pytest.mark.unit
def test_linear_regressor_errors_name_itself():
    """LinearRegressor used to borrow the constant regressor's error message."""
    pytest.importorskip("sklearn")
    X = pd.DataFrame({"flow": [1.0, 2.0, 3.0]})
    y = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
    units = {"flow": "m^3/hr"}

    with pytest.raises(FlexDataError, match="^LinearRegressor fits one output column"):
        LinearRegressor().fit(X, y, input_units=units, output_units="kW")
    with pytest.raises(FlexDataError, match="^LinearRegressor.fit takes X as a Data"):
        LinearRegressor().fit(X["flow"], y[["a"]], input_units=units, output_units="kW")
