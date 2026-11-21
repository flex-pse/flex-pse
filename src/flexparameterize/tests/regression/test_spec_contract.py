"""Regressors produce specs only through their surrogate's contract.

Each regressor's ``to_surrogate_spec()`` must build a spec its flexops
surrogate class accepts, with parameters named exactly as the fit names its
coefficients and every parameter fixed. No flexparameterize module may build a
``SurrogateSpec`` by hand: specs come from ``<Surrogate>.make_spec``.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import flexparameterize
from flexcore.config.schema import SurrogateType
from flexops.surrogates import surrogate_from_spec
from flexparameterize.regression import (
    COEFFICIENT_NAME,
    ArimaRegressor,
    ConstantIntensityRegressor,
    LinearRegressor,
)

_INDEX = pd.date_range("2024-01-01", periods=60, freq="1h")


def _flow() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({"flow": rng.uniform(1.0, 5.0, size=len(_INDEX))}, index=_INDEX)


def _linear():
    flow = _flow()
    power = pd.DataFrame({"power": 0.5 * flow["flow"] + 2.0}, index=_INDEX)
    regressor = LinearRegressor().fit(
        flow, power, input_units={"flow": "m^3/hr"}, output_units="kW"
    )
    return regressor, regressor.coefficients


def _constant():
    flow = _flow()
    power = pd.DataFrame({"power": 0.5 * flow["flow"]}, index=_INDEX)
    regressor = ConstantIntensityRegressor().fit(flow, power)
    return regressor, regressor.to_fit_result().coefficients


def _arima():
    pytest.importorskip("scipy")
    flow = _flow()
    rng = np.random.default_rng(11)
    values = np.zeros(len(_INDEX))
    for t in range(1, len(_INDEX)):
        values[t] = (
            0.4 * values[t - 1] + 0.8 * flow["flow"].iloc[t] + rng.normal(0, 0.05)
        )
    regressor = ArimaRegressor(order=(1, 0, 1), max_ar_persistence=None).fit(
        flow,
        pd.DataFrame({"power": values}, index=_INDEX),
        input_units={"flow": "m^3/hr"},
        output_units="kW",
    )
    return regressor, regressor.coefficients


@pytest.mark.unit
@pytest.mark.parametrize(
    ("fit", "surrogate_type"),
    [
        (_linear, SurrogateType.MULTILINEAR),
        (_constant, SurrogateType.CONSTANT_INTENSITY),
        (_arima, SurrogateType.ARIMA),
    ],
    ids=["linear", "constant_intensity", "arima"],
)
def test_regressor_spec_satisfies_its_surrogate_contract(fit, surrogate_type):
    """The spec validates, names the fitted coefficients, and fixes them all."""
    regressor, coefficients = fit()
    spec = regressor.to_surrogate_spec()

    assert spec.surrogate_type is surrogate_type
    surrogate = surrogate_from_spec(spec)
    assert surrogate.spec == spec
    assert set(spec.parameters) == set(coefficients)
    for name, value in coefficients.items():
        assert spec.parameters[name].value == pytest.approx(value)
    assert spec.free_parameters() == []
    assert spec.config_version == type(surrogate).config_version


@pytest.mark.unit
def test_constant_intensity_spec_carries_its_one_parameter():
    """The constant-intensity fit's one parameter is COEFFICIENT_NAME."""
    regressor, _coefficients = _constant()
    assert list(regressor.to_surrogate_spec().parameters) == [COEFFICIENT_NAME]


def _hand_built_specs(path: Path) -> list[int]:
    """Return the line numbers where ``path`` calls ``SurrogateSpec(...)``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name == "SurrogateSpec":
            lines.append(node.lineno)
    return lines


@pytest.mark.unit
def test_no_flexparameterize_module_builds_a_surrogate_spec_by_hand():
    """Specs come from the surrogate classes' make_spec, never a literal."""
    package = Path(flexparameterize.__file__).parent
    offenders = {
        str(path.relative_to(package)): lines
        for path in sorted(package.rglob("*.py"))
        if "tests" not in path.relative_to(package).parts
        and (lines := _hand_built_specs(path))
    }
    assert offenders == {}, (
        "Build surrogate specs with <Surrogate>.make_spec, not SurrogateSpec(...): "
        f"{offenders}"
    )
