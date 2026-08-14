"""Regressors that turn plant data into a ``SurrogateSpec``.

One regressor so far; the pluggable regressor protocol and its registry arrive
alongside the linear regressor.
"""

from flexparameterize.regression.constant import (
    COEFFICIENT_NAME,
    ConstantIntensityRegressor,
    constant_intensity_coefficient,
)

__all__ = [
    "COEFFICIENT_NAME",
    "ConstantIntensityRegressor",
    "constant_intensity_coefficient",
]
