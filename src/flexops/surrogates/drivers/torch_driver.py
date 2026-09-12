"""TorchDriver: an :class:`~flexops.surrogates.grey_box.ExternalModelDriver`
backed by ``torch.autograd``. The only module in this milestone that imports
``torch`` at module scope.
"""

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
    """Evaluates a PyTorch model and its exact first/second derivatives.

    Uses ``torch.autograd.functional.jacobian``/``hessian`` (not manual
    ``create_graph=True`` chains), so there is no double-backward footgun to
    get wrong -- the higher-level API sidesteps it.

    The forward/backward pass runs on whatever device the model's own
    parameters already live on; call ``model.to("cuda")`` yourself before
    passing it in via ``model_path`` to run on GPU. The rest of the
    optimization (CyIpopt, Pyomo) always stays on CPU -- every value handed
    back crosses to CPU first (see :meth:`jacobian`/:meth:`hessian`).
    """

    framework = ExternalFramework.PYTORCH

    def __init__(self, model, n_inputs: int) -> None:
        """Store ``model`` and its device; coerce an ``nn.Module``'s dtype
        to float64 once.

        Args:
            model: A fitted, callable PyTorch model (an ``nn.Module`` or a
                plain closure/function).
            n_inputs: Number of scalar inputs the model takes.
        """
        super().__init__(model, n_inputs)
        if isinstance(model, torch.nn.Module):
            model.double()
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
        output = self._scalar_fn(t)
        if not output.requires_grad:
            # Fully detached: torch.autograd.grad itself raises here rather
            # than returning None, since the whole output has no grad_fn.
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
