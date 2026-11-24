"""ExponentialSurrogate: not yet implemented."""

from typing import ClassVar

from flexcore.config.schema import SurrogateType
from flexops.surrogates._stub import StubSurrogate


class ExponentialSurrogate(StubSurrogate):
    """An exponential relationship in its inputs.

    Not yet implemented; use
    :class:`~flexops.surrogates.multilinear.MultilinearSurrogate`
    for a relationship linear in each input.
    """

    surrogate_type: ClassVar[SurrogateType] = SurrogateType.EXPONENTIAL
    not_implemented: ClassVar[str] = (
        "ExponentialSurrogate is not yet implemented. Use "
        "MultilinearSurrogate (surrogate_type='multilinear') for a "
        "relationship linear in each input."
    )
