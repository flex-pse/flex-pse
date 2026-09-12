"""TorchDriver: an :class:`~flexops.surrogates.external.ExternalModelDriver`
backed by ``torch.autograd``. The only module in this milestone that imports
``torch`` at module scope.
"""

import numpy as np
import torch

from flexcore.exceptions import FlexConfigError
from flexcore.logger import get_logger
from flexops.surrogates.external import ExternalFramework, ExternalModelDriver

_log = get_logger(__name__)


class TorchDriver(ExternalModelDriver):
    """Evaluates a PyTorch model and its exact first/second derivatives.

    Uses ``torch.autograd.functional.jacobian``/``hessian`` (not manual
    ``create_graph=True`` chains), so there is no double-backward footgun to
    get wrong -- the higher-level API sidesteps it.
    """

    framework = ExternalFramework.PYTORCH

    def __init__(self, model, n_inputs: int) -> None:
        """Store ``model``, coercing an ``nn.Module``'s dtype to float64 once.

        Args:
            model: A fitted, callable PyTorch model (an ``nn.Module`` or a
                plain closure/function).
            n_inputs: Number of scalar inputs the model takes.
        """
        super().__init__(model, n_inputs)
        if isinstance(model, torch.nn.Module):
            model.double()

    def _as_tensor(self, x: np.ndarray) -> torch.Tensor:
        """Convert ``x`` to a float64 tensor (PyNumero always hands float64)."""
        return torch.as_tensor(x, dtype=torch.float64)

    def _scalar_fn(self, t: torch.Tensor) -> torch.Tensor:
        """Call the model and normalize its output to a 0-d tensor."""
        return self._model(t).reshape(())

    def evaluate(self, x: np.ndarray) -> float:
        """Return the model's scalar output at ``x``."""
        return float(self._scalar_fn(self._as_tensor(x)))

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Dense gradient at ``x``, shape ``(n_inputs,)``."""
        t = self._as_tensor(x)
        jac = torch.autograd.functional.jacobian(self._scalar_fn, t)
        return jac.reshape(self._n_inputs).detach().numpy()

    def hessian(self, x: np.ndarray) -> np.ndarray:
        """Dense full symmetric Hessian at ``x``, shape ``(n_inputs, n_inputs)``."""
        t = self._as_tensor(x)
        hess = torch.autograd.functional.hessian(self._scalar_fn, t)
        return hess.reshape(self._n_inputs, self._n_inputs).detach().numpy()

    def check_differentiable(self, x: np.ndarray) -> None:
        """Raise if the forward pass breaks autograd; warn on a non-finite grad.

        Uses ``torch.autograd.grad`` rather than ``.backward()``, which would
        accumulate into the caller's own ``model.parameters()[...].grad`` as a
        side effect. A non-finite gradient is not rejected: ``sqrt``/``log``/
        ``1/x`` have legitimately infinite derivatives at ordinary points.

        Args:
            x: The probe point, in the surrogate's declared input units.

        Raises:
            FlexConfigError: If the forward pass disconnects the autograd
                graph (an in-place op, ``.detach()``, ``.item()``/``float()``
                cast, or ``torch.no_grad()`` anywhere inside it).
        """
        t = self._as_tensor(x).requires_grad_(True)
        (grad,) = torch.autograd.grad(self._scalar_fn(t), t, allow_unused=True)
        if grad is None:
            raise FlexConfigError(
                f"model {self._model!r} is not differentiable at the probe "
                "point: its forward pass disconnects the autograd graph (an "
                "in-place op, .detach(), .item()/float() cast, or "
                "torch.no_grad() somewhere inside it). Fix the forward pass "
                "so gradients flow through to every input.",
                field="model_path",
            )
        if not torch.isfinite(grad).all():
            _log.warning(
                "model %r has a non-finite gradient at the probe point %s "
                "(expected for e.g. sqrt/log/1/x evaluated near a "
                "singularity).",
                self._model,
                x,
            )
