"""Tests for get_regressor: the SurrogateType-keyed regressor registry."""

import importlib
import sys

import pytest

from flexcore.config.schema import SurrogateType
from flexcore.exceptions import FlexConfigError
from flexparameterize.regression import (
    ArimaRegressor,
    ConstantIntensityRegressor,
    LinearRegressor,
    get_regressor,
)

RESERVED = (
    SurrogateType.QUADRATIC,
    SurrogateType.EXPONENTIAL,
    SurrogateType.NEURAL_NETWORK,
)


@pytest.mark.unit
def test_get_regressor_known_names():
    """Both members and their string values resolve to the right classes."""
    assert get_regressor(SurrogateType.CONSTANT_INTENSITY) is ConstantIntensityRegressor
    assert get_regressor("constant_intensity") is ConstantIntensityRegressor
    assert get_regressor(SurrogateType.MULTILINEAR) is LinearRegressor
    assert get_regressor("multilinear") is LinearRegressor
    assert get_regressor(SurrogateType.ARIMA) is ArimaRegressor
    assert get_regressor("arima") is ArimaRegressor


@pytest.mark.unit
@pytest.mark.parametrize("surrogate_type", RESERVED)
def test_reserved_names_raise_notimplemented(surrogate_type):
    """A reserved-but-unbuilt SurrogateType raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        get_regressor(surrogate_type)


@pytest.mark.unit
def test_unknown_name_raises():
    """An unknown name raises FlexConfigError listing valid and reserved names."""
    with pytest.raises(FlexConfigError, match="constant_intensity"):
        get_regressor("not_a_real_type")


@pytest.mark.unit
def test_package_import_without_sklearn(monkeypatch):
    """flexparameterize and its regression package import fine without sklearn."""
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.linear_model", None)

    import flexparameterize
    import flexparameterize.regression

    importlib.reload(flexparameterize.regression)
    importlib.reload(flexparameterize)
