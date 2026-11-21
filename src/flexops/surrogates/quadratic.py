"""QuadraticSurrogate: not yet implemented."""

from typing import ClassVar

from flexcore.config.schema import SurrogateType
from flexops.surrogates._stub import StubSurrogate


class QuadraticSurrogate(StubSurrogate):
    """A quadratic (up to degree-2, including squared terms) relationship.

    Not yet implemented; use
    :class:`~flexops.surrogates.multilinear.MultilinearSurrogate`
    for a relationship with no squared term.
    """

    surrogate_type: ClassVar[SurrogateType] = SurrogateType.QUADRATIC
    not_implemented: ClassVar[str] = (
        "QuadraticSurrogate is not yet implemented. Use "
        "MultilinearSurrogate (surrogate_type='multilinear') for a "
        "relationship with no squared term."
    )
