"""The surrogate registry: which class implements which ``SurrogateType``.

:data:`SURROGATES` maps every
:class:`~flexcore.config.schema.SurrogateType` member to the class that
implements its contract, and :func:`surrogate_from_spec` is the one way to
turn a :class:`~flexcore.config.schema.SurrogateSpec` into a validated
:class:`~flexops.surrogates.base.Surrogate`. Both are re-exported from the
package (:mod:`flexops.surrogates`).
"""

from flexcore.config.schema import SurrogateSpec, SurrogateType
from flexcore.exceptions import FlexConfigError
from flexops.surrogates.arima import ArimaSurrogate
from flexops.surrogates.base import Surrogate
from flexops.surrogates.constant_intensity import ConstantIntensitySurrogate
from flexops.surrogates.exponential import ExponentialSurrogate
from flexops.surrogates.multilinear import MultilinearSurrogate
from flexops.surrogates.neural_network import NeuralNetworkSurrogate
from flexops.surrogates.quadratic import QuadraticSurrogate

SURROGATES: dict[SurrogateType, type[Surrogate]] = {
    SurrogateType.CONSTANT_INTENSITY: ConstantIntensitySurrogate,
    SurrogateType.MULTILINEAR: MultilinearSurrogate,
    SurrogateType.QUADRATIC: QuadraticSurrogate,
    SurrogateType.EXPONENTIAL: ExponentialSurrogate,
    SurrogateType.ARIMA: ArimaSurrogate,
    SurrogateType.NEURAL_NETWORK: NeuralNetworkSurrogate,
}
"""dict: SurrogateType -> the class that implements it. The extension point
for a new relationship shape: add the member to
:class:`~flexcore.config.schema.SurrogateType`, a class in this subpackage
defining its contract, and an entry here."""


def surrogate_class(surrogate_type: SurrogateType | str) -> type[Surrogate]:
    """Return the class registered for ``surrogate_type``.

    Args:
        surrogate_type: A :class:`~flexcore.config.schema.SurrogateType` or
            its value.

    Returns:
        The registered class.

    Raises:
        FlexConfigError: If no class is registered for it.
    """
    surrogate_class = SURROGATES.get(SurrogateType(surrogate_type))
    if surrogate_class is None:
        known = ", ".join(repr(name.value) for name in SURROGATES)
        raise FlexConfigError(
            f"No surrogate class is registered for surrogate_type "
            f"{surrogate_type!r}. Known: {known}.",
            field="surrogate_type",
            value=surrogate_type,
        )
    return surrogate_class


def surrogate_from_spec(spec: SurrogateSpec) -> Surrogate:
    """Construct and validate the surrogate ``spec`` names.

    The surrogate migrates an older ``config_version`` of its own config.

    Args:
        spec: The :class:`~flexcore.config.schema.SurrogateSpec` to realize.

    Returns:
        The constructed, validated
        :class:`~flexops.surrogates.base.Surrogate`.

    Raises:
        FlexConfigError: If no class is registered for ``spec.surrogate_type``
            or ``spec`` does not match its contract.
        NotImplementedError: If the named class is not yet implemented.
    """
    return surrogate_class(spec.surrogate_type)(spec)


def is_registered(surrogate: Surrogate) -> bool:
    """Return whether ``surrogate``'s class is the one registered for its type.

    Only a registered class can be rebuilt from a spec, so only such a
    surrogate is written out by ``emit_model``; a custom subclass is not.

    Args:
        surrogate: A surrogate instance.

    Returns:
        ``True`` when ``type(surrogate)`` is ``SURROGATES[surrogate_type]``.
    """
    return SURROGATES.get(getattr(surrogate, "surrogate_type", None)) is type(surrogate)
