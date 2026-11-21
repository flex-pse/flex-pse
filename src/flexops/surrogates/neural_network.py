"""NeuralNetworkSurrogate: not yet implemented."""

from typing import ClassVar

from flexcore.config.schema import SurrogateType
from flexops.surrogates._stub import StubSurrogate


class NeuralNetworkSurrogate(StubSurrogate):
    """A neural-network (e.g. ReLU big-M or ICNN) forward-pass relationship.

    Not yet implemented; use
    :class:`~flexops.surrogates.multilinear.MultilinearSurrogate`
    for a relationship expressible without a network.
    """

    surrogate_type: ClassVar[SurrogateType] = SurrogateType.NEURAL_NETWORK
    not_implemented: ClassVar[str] = (
        "NeuralNetworkSurrogate is not yet implemented. Use "
        "MultilinearSurrogate (surrogate_type='multilinear') for a "
        "relationship expressible without a network."
    )
