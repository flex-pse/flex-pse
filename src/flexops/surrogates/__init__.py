"""Predefined surrogate-structure classes and their spec contracts.

Every relationship a unit's energy draw (or any other registered relation,
see :meth:`~flexops.core.ops_block.OpsBlockData.register_relation`) can be
swapped to is one of these classes. Each defines its contract -- the
``config`` it accepts and the ``parameters`` it fits -- validates a
:class:`~flexcore.config.schema.SurrogateSpec` against it in ``__init__``,
builds the Pyomo objects in :meth:`~flexops.surrogates.base.Surrogate.build`,
and reads its live spec back with
:meth:`~flexops.surrogates.base.Surrogate.read_spec`. flexparameterize
produces specs only through each class's ``make_spec``.

A leaf subpackage of ``flexops`` (imports only ``flexcore``, pydantic and
Pyomo), so ``flexops.core.ops_block`` can import
:data:`~flexops.surrogates.surrogates.SURROGATES` with no cross-package import.

:class:`~multilinear.MultilinearSurrogate`, :class:`~arima.ArimaSurrogate` and
:class:`~constant_intensity.ConstantIntensitySurrogate` are implemented; the
rest raise ``NotImplementedError`` at construction, naming an implemented
alternative. ``constant_intensity`` swaps no relation: it sets the unit's own
``energy_intensity``.
"""

from flexops.surrogates.arima import ArimaSurrogate
from flexops.surrogates.base import Surrogate
from flexops.surrogates.constant_intensity import ConstantIntensitySurrogate
from flexops.surrogates.exponential import ExponentialSurrogate
from flexops.surrogates.multilinear import MultilinearSurrogate
from flexops.surrogates.neural_network import NeuralNetworkSurrogate
from flexops.surrogates.quadratic import QuadraticSurrogate
from flexops.surrogates.surrogates import (
    SURROGATES,
    is_registered,
    surrogate_class,
    surrogate_from_spec,
)

__all__ = [
    "SURROGATES",
    "ArimaSurrogate",
    "ConstantIntensitySurrogate",
    "ExponentialSurrogate",
    "MultilinearSurrogate",
    "NeuralNetworkSurrogate",
    "QuadraticSurrogate",
    "Surrogate",
    "is_registered",
    "surrogate_class",
    "surrogate_from_spec",
]
