"""The surrogate registry: which class implements which ``SurrogateType``.

:data:`SURROGATES` maps every
:class:`~flexcore.config.schema.SurrogateType` member that has a class to that
class, and :func:`surrogate_from_spec` is the one way to turn a
:class:`~flexcore.config.schema.SurrogateSpec` into a built, validated
:class:`~flexops.surrogates.base.Surrogate`. Both are re-exported from the
package (:mod:`flexops.surrogates`).
"""

from flexcore.config.schema import SurrogateSpec, SurrogateType
from flexcore.exceptions import FlexConfigError
from flexops.surrogates.arima import ArimaSurrogate
from flexops.surrogates.base import Surrogate
from flexops.surrogates.exponential import ExponentialSurrogate
from flexops.surrogates.grey_box import ExternalModelSurrogate
from flexops.surrogates.multilinear import MultilinearSurrogate
from flexops.surrogates.neural_network import NeuralNetworkSurrogate
from flexops.surrogates.quadratic import QuadraticSurrogate

SURROGATES: dict[SurrogateType, type[Surrogate]] = {
    SurrogateType.MULTILINEAR: MultilinearSurrogate,
    SurrogateType.QUADRATIC: QuadraticSurrogate,
    SurrogateType.EXPONENTIAL: ExponentialSurrogate,
    SurrogateType.ARIMA: ArimaSurrogate,
    SurrogateType.NEURAL_NETWORK: NeuralNetworkSurrogate,
    SurrogateType.EXTERNAL_MODEL: ExternalModelSurrogate,
}
"""dict: SurrogateType -> the class that implements it. The extension point
for a new relationship shape: add the member to
:class:`~flexcore.config.schema.SurrogateType`, a class in this subpackage,
and an entry here. Deliberately excludes
``SurrogateType.CONSTANT_INTENSITY`` (see the package docstring)."""


def surrogate_from_spec(spec: SurrogateSpec) -> Surrogate:
    """Construct and validate the surrogate ``spec`` names.

    Args:
        spec: The :class:`~flexcore.config.schema.SurrogateSpec` to realize.

    Returns:
        The constructed, validated
        :class:`~flexops.surrogates.base.Surrogate`.

    Raises:
        FlexConfigError: If ``spec.surrogate_type`` is
            ``SurrogateType.CONSTANT_INTENSITY`` (which has no class; see the
            package docstring) or is otherwise not in
            :data:`SURROGATES` (not reachable through the enum today, but
            guarded for a future member added without a registry entry).
        NotImplementedError: If the named class is not yet implemented.
    """
    if spec.surrogate_type is SurrogateType.CONSTANT_INTENSITY:
        raise FlexConfigError(
            "'constant_intensity' has no surrogate class: it fixes a "
            "process parameter rather than swapping a Constraint. Fix the "
            "parameter directly instead of calling surrogate_from_spec.",
            field="surrogate_type",
            value=spec.surrogate_type,
        )
    surrogate_class = SURROGATES.get(spec.surrogate_type)
    if surrogate_class is None:
        known = ", ".join(repr(name.value) for name in SURROGATES)
        raise FlexConfigError(
            f"No surrogate class is registered for surrogate_type "
            f"{spec.surrogate_type!r}. Known: {known}.",
            field="surrogate_type",
            value=spec.surrogate_type,
        )
    return surrogate_class(spec.data)
