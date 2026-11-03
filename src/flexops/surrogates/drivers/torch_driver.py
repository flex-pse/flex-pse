"""An ``ExternalModelDriver`` that uses ``torch.autograd``."""

import copy

import numpy as np
import torch

from flexcore.exceptions import FlexConfigError
from flexcore.logger import get_logger
from flexops.surrogates.grey_box import ExternalFramework, ExternalModelDriver

_log = get_logger(__name__)


def _model_device(model) -> torch.device:
    """Read a model's device: an nn.Module's own parameter device, or CPU
    for a parameter-free module or a plain closure."""
    if isinstance(model, torch.nn.Module):
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")
    return torch.device("cpu")


class TorchDriver(ExternalModelDriver):
    """Evaluates a PyTorch model and its exact first and second derivatives."""

    framework = ExternalFramework.PYTORCH

    def __init__(self, model, n_inputs: int) -> None:
        """Store ``model`` and its device, using a float64 eval-mode copy of a module.

        Args:
            model: A fitted, callable PyTorch model (an ``nn.Module`` or a
                plain closure/function).
            n_inputs: Number of scalar inputs the model takes.
        """
        # A copy, so the caller's module keeps its dtype and training mode.
        if isinstance(model, torch.nn.Module):
            model = copy.deepcopy(model).double().eval()
        super().__init__(model, n_inputs)
        self._device = _model_device(model)

    def _as_tensor(self, x: np.ndarray) -> torch.Tensor:
        """Convert ``x`` to a float64 tensor on the model's device (PyNumero
        always hands float64 numpy on CPU; the model may live on GPU)."""
        return torch.as_tensor(x, dtype=torch.float64, device=self._device)

    def _scalar_fn(self, t: torch.Tensor) -> torch.Tensor:
        """Call the model and normalize its output to a 0-d tensor."""
        return self._model(t).reshape(())

    def evaluate(self, x: np.ndarray) -> float:
        """Return the model's scalar output at ``x``."""
        with torch.no_grad():
            return float(self._scalar_fn(self._as_tensor(x)))

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Dense gradient at ``x``, shape ``(n_inputs,)``."""
        t = self._as_tensor(x)
        jac = torch.autograd.functional.jacobian(self._scalar_fn, t)
        return jac.reshape(self._n_inputs).detach().cpu().numpy()

    def hessian(self, x: np.ndarray) -> np.ndarray:
        """Dense full symmetric Hessian at ``x``, shape ``(n_inputs, n_inputs)``."""
        t = self._as_tensor(x)
        hess = torch.autograd.functional.hessian(self._scalar_fn, t)
        return hess.reshape(self._n_inputs, self._n_inputs).detach().cpu().numpy()

    def check_differentiable(self, x: np.ndarray) -> None:
        """Raise if the forward pass fails or breaks autograd; warn on a bad gradient.

        Args:
            x: The probe point, in the surrogate's declared input units.

        Raises:
            FlexConfigError: If the model can't run on ``n_inputs`` inputs and
                return one scalar, or its forward pass disconnects autograd.
        """

        def _not_differentiable():
            raise FlexConfigError(
                f"model {self._model!r} is not differentiable at the probe "
                "point: its forward pass disconnects the autograd graph (an "
                "in-place op, .detach(), .item()/float() cast, or "
                "torch.no_grad() somewhere inside it). Fix the forward pass "
                "so gradients flow through to every input.",
                field="model_path",
            )

        t = self._as_tensor(x).requires_grad_(True)
        try:
            output = self._scalar_fn(t)
        except (RuntimeError, IndexError, TypeError, AttributeError) as exc:
            raise FlexConfigError(
                f"model {self._model!r} could not run on {self._n_inputs} input(s) "
                f"and return one scalar tensor: {exc}",
                field="input_variables",
            ) from exc
        # Fully detached output: torch.autograd.grad would raise, not return None.
        if not output.requires_grad:
            _not_differentiable()
        (grad,) = torch.autograd.grad(output, t, allow_unused=True)
        if grad is None:
            _not_differentiable()
        if not torch.isfinite(grad).all():
            _log.warning(
                "model %r has a non-finite gradient at the probe point %s "
                "(expected for e.g. sqrt/log/1/x evaluated near a "
                "singularity).",
                self._model,
                x,
            )
