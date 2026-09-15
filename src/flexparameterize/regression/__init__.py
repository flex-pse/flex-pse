"""Regressors that turn plant data into a ``SurrogateSpec``.

:func:`get_regressor` resolves a :class:`~flexcore.config.schema.SurrogateType`
(member or string value) to the regressor class that fits it, so a config
naming which regressor produced a spec is a lookup, not a hand-maintained
if/else. Importing this package never imports scikit-learn -- only
:class:`~flexparameterize.regression.linear.LinearRegressor.fit` does, lazily.
"""

from flexcore.config.schema import SurrogateType
from flexcore.exceptions import FlexConfigError
from flexparameterize.regression.arima import ArimaRegressor
from flexparameterize.regression.base import FitResult, Regressor
from flexparameterize.regression.constant import (
    COEFFICIENT_NAME,
    ConstantIntensityRegressor,
    constant_intensity_coefficient,
)
from flexparameterize.regression.linear import LinearRegressor

_REGRESSORS: dict[SurrogateType, type] = {
    SurrogateType.CONSTANT_INTENSITY: ConstantIntensityRegressor,
    SurrogateType.MULTILINEAR: LinearRegressor,
    SurrogateType.ARIMA: ArimaRegressor,
}

_RESERVED = {
    SurrogateType.QUADRATIC: "QuadraticSurrogate",
    SurrogateType.EXPONENTIAL: "ExponentialSurrogate",
    SurrogateType.NEURAL_NETWORK: "NeuralNetworkSurrogate",
}
"""dict: SurrogateType -> the reserved (not yet implemented) surrogate class
name in ``flexops.surrogates``, named in the NotImplementedError message."""


def get_regressor(name: SurrogateType | str) -> type:
    """Resolve a ``SurrogateType`` to the regressor class that fits it.

    Args:
        name: A :class:`~flexcore.config.schema.SurrogateType` member or its
            string value.

    Returns:
        The regressor class.

    Raises:
        NotImplementedError: If ``name`` is a reserved ``SurrogateType`` with
            no regressor yet (mirrors the corresponding
            ``flexops.surrogates`` stub class).
        FlexConfigError: If ``name`` is not a known ``SurrogateType`` value at
            all, listing the valid and reserved names.
    """
    try:
        surrogate_type = SurrogateType(name)
    except ValueError as exc:
        known = ", ".join(repr(member.value) for member in SurrogateType)
        raise FlexConfigError(
            f"{name!r} is not a known SurrogateType. Known: {known}.",
            field="surrogate_type",
            value=name,
        ) from exc
    if surrogate_type in _RESERVED:
        raise NotImplementedError(
            f"No regressor is implemented for {surrogate_type.value!r} yet; "
            f"see flexops.surrogates.{_RESERVED[surrogate_type]}, not yet "
            "implemented."
        )
    return _REGRESSORS[surrogate_type]


__all__ = [
    "ArimaRegressor",
    "COEFFICIENT_NAME",
    "ConstantIntensityRegressor",
    "FitResult",
    "LinearRegressor",
    "Regressor",
    "constant_intensity_coefficient",
    "get_regressor",
]
