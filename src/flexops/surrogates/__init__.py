"""Predefined surrogate-structure classes.

Every relationship a unit's energy draw (or any other registered relation,
see :meth:`~flexops.core.ops_block.OpsBlockData.register_relation`) can be
swapped to is one of these classes: it validates its
:class:`~flexcore.config.schema.SurrogateSpec`'s ``data`` mapping in
``__init__`` and builds the Pyomo objects in
:meth:`~flexops.surrogates.base.Surrogate.build`. A leaf subpackage of
``flexops`` (imports only ``flexcore`` and Pyomo), so
``flexops.core.ops_block`` can import
:data:`~flexops.surrogates.surrogates.SURROGATES` with no cross-package
import -- this is what lets a config's surrogate be realized at
unit construction time (``build_model``), with no ``flexparameterize`` import
anywhere in ``flexops``.

:class:`~multilinear.MultilinearSurrogate` and
:class:`~grey_box.ExternalModelSurrogate` are implemented; the rest raise
``NotImplementedError`` at construction, naming
``MultilinearSurrogate`` as the implemented alternative.
``SurrogateType.CONSTANT_INTENSITY`` has no class here at all -- it fixes a
process parameter rather than swapping a Constraint, so
``flexparameterize.apply.apply_to_model`` handles it directly.
"""

from flexops.surrogates.arima import ArimaSurrogate
from flexops.surrogates.base import Surrogate
from flexops.surrogates.exponential import ExponentialSurrogate
from flexops.surrogates.grey_box import (
    ExternalFramework,
    ExternalModelSurrogate,
    get_driver,
)
from flexops.surrogates.multilinear import MultilinearSurrogate
from flexops.surrogates.neural_network import NeuralNetworkSurrogate
from flexops.surrogates.quadratic import QuadraticSurrogate
from flexops.surrogates.surrogates import SURROGATES, surrogate_from_spec

__all__ = [
    "SURROGATES",
    "ArimaSurrogate",
    "ExponentialSurrogate",
    "ExternalFramework",
    "ExternalModelSurrogate",
    "MultilinearSurrogate",
    "NeuralNetworkSurrogate",
    "QuadraticSurrogate",
    "Surrogate",
    "get_driver",
    "surrogate_from_spec",
]
