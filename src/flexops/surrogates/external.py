"""Framework enum and driver registry for external-model grey-box surrogates.

Imports **no** framework: resolving a framework to its driver class
(:func:`get_driver`) is the only place a heavy dependency (``torch``, ...) is
imported, and that happens lazily, at call time. This is what keeps
``import flexops.surrogates`` clean on a bare install.
"""

import enum
import importlib
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from flexcore.exceptions import FlexConfigError


class ExternalFramework(enum.StrEnum):
    """Which autograd framework an external model's driver is written for."""

    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    JAX = "jax"


class ExternalModelDriver(ABC):
    """Evaluate one external model and its first two derivatives at a point.

    Attributes:
        framework: The :class:`ExternalFramework` this driver implements.
    """

    framework: ClassVar[ExternalFramework]

    def __init__(self, model, n_inputs: int) -> None:
        """Store the model and its input dimension.

        Args:
            model: The fitted, callable external model.
            n_inputs: Number of scalar inputs the model takes.
        """
        self._model = model
        self._n_inputs = n_inputs

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> float:
        """Return the model's scalar output at ``x``."""

    @abstractmethod
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Dense gradient, shape ``(n_inputs,)``."""

    @abstractmethod
    def hessian(self, x: np.ndarray) -> np.ndarray:
        """Dense *full symmetric* Hessian, shape ``(n_inputs, n_inputs)``."""

    @abstractmethod
    def check_differentiable(self, x: np.ndarray) -> None:
        """Raise FlexConfigError if the model is not differentiable at ``x``."""


_DRIVERS: dict[ExternalFramework, str] = {
    ExternalFramework.PYTORCH: "flexops.surrogates.drivers.torch_driver.TorchDriver",
}
"""dict: implemented framework -> dotted path of its driver class."""

_RESERVED_FRAMEWORKS: dict[ExternalFramework, str] = {
    ExternalFramework.TENSORFLOW: "TensorFlowDriver",
    ExternalFramework.ONNX: "OnnxDriver",
    ExternalFramework.JAX: "JaxDriver",
}
"""dict: reserved (not yet implemented) framework -> its driver class name."""


def get_driver(framework: ExternalFramework | str) -> type[ExternalModelDriver]:
    """Resolve a framework to its driver class, importing it lazily.

    Mirrors :func:`flexparameterize.regression.get_regressor`: a
    ``NotImplementedError`` names the reserved driver class for a reserved
    member; a ``FlexConfigError`` lists known values for an unknown string.
    The driver module (and therefore its framework) is imported only here,
    not at this module's top level.

    Args:
        framework: An :class:`ExternalFramework` member or its string value.

    Returns:
        The driver class.

    Raises:
        FlexConfigError: If ``framework`` is not a known ``ExternalFramework``
            value.
        NotImplementedError: If ``framework`` is a reserved member with no
            driver yet.
    """
    try:
        member = ExternalFramework(framework)
    except ValueError as exc:
        known = ", ".join(repr(m.value) for m in ExternalFramework)
        raise FlexConfigError(
            f"{framework!r} is not a known ExternalFramework. Known: {known}.",
            field="framework",
            value=framework,
        ) from exc
    if member in _RESERVED_FRAMEWORKS:
        raise NotImplementedError(
            f"No driver is implemented for {member.value!r} yet; see "
            f"flexops.surrogates.drivers.{_RESERVED_FRAMEWORKS[member]}, not "
            "yet implemented."
        )
    dotted = _DRIVERS[member]
    module_name, class_name = dotted.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
