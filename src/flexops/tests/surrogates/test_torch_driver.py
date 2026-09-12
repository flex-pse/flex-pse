"""Tests for TorchDriver's device handling: the wrapped model's forward pass
runs on whatever device its own parameters already live on; optimization
(cyipopt/Pyomo) always sees plain CPU floats/numpy back."""

from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.unit
@pytest.mark.needs_torch
def test_model_device_defaults_to_cpu_for_plain_closure():
    """A bare closure has no device of its own -- same as today's default."""
    torch = pytest.importorskip("torch")
    from flexops.surrogates.drivers.torch_driver import _model_device

    assert _model_device(lambda x: x) == torch.device("cpu")


@pytest.mark.unit
@pytest.mark.needs_torch
def test_model_device_defaults_to_cpu_for_parameter_free_module():
    """An nn.Module with no parameters falls back to CPU, not StopIteration."""
    torch = pytest.importorskip("torch")
    from flexops.surrogates.drivers.torch_driver import _model_device

    assert _model_device(torch.nn.Identity()) == torch.device("cpu")


@pytest.mark.unit
@pytest.mark.needs_torch
def test_model_device_reads_from_module_parameters():
    """A real (CPU) module's device is read straight off its parameters."""
    torch = pytest.importorskip("torch")
    from flexops.surrogates.drivers.torch_driver import _model_device

    module = torch.nn.Linear(1, 1)
    assert _model_device(module) == torch.device("cpu")


@pytest.mark.unit
@pytest.mark.needs_torch
def test_model_device_reads_non_cpu_device_from_parameters(monkeypatch):
    """Proves the device-reading logic picks up a non-CPU device, without
    requiring real GPU hardware."""
    torch = pytest.importorskip("torch")
    from flexops.surrogates.drivers.torch_driver import _model_device

    module = torch.nn.Linear(1, 1)
    fake_device = torch.device("cuda", 0)
    monkeypatch.setattr(
        module, "parameters", lambda: iter([SimpleNamespace(device=fake_device)])
    )

    assert _model_device(module) == fake_device


@pytest.mark.unit
@pytest.mark.needs_torch
def test_as_tensor_uses_model_device(monkeypatch):
    """TorchDriver._as_tensor places the input tensor on the model's device."""
    torch = pytest.importorskip("torch")
    from flexops.surrogates.drivers.torch_driver import TorchDriver

    module = torch.nn.Linear(1, 1)
    fake_device = torch.device("cuda", 0)
    monkeypatch.setattr(
        module, "parameters", lambda: iter([SimpleNamespace(device=fake_device)])
    )

    driver = TorchDriver(module, n_inputs=1)
    assert driver._device == fake_device

    # Patch torch.as_tensor itself (rather than really allocating on cuda,
    # which this build of torch may not even support) to prove _as_tensor
    # requests the model's device from it.
    captured = {}
    real_as_tensor = torch.as_tensor

    def _fake_as_tensor(x, **kwargs):
        captured.update(kwargs)
        return real_as_tensor(x, dtype=kwargs.get("dtype"))

    monkeypatch.setattr(torch, "as_tensor", _fake_as_tensor)

    driver._as_tensor(np.array([1.0]))

    assert captured["device"] == fake_device


@pytest.mark.unit
@pytest.mark.needs_torch
def test_gpu_model_evaluates_end_to_end():
    """A GPU-resident model evaluates end to end, returning plain CPU values
    that match a CPU-run reference -- the one test needing real hardware."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no GPU available")
    from flexops.surrogates.drivers.torch_driver import TorchDriver

    def make_module():
        torch.manual_seed(0)
        module = torch.nn.Linear(2, 1)
        return module

    cpu_driver = TorchDriver(make_module(), n_inputs=2)
    gpu_module = make_module().to("cuda")
    gpu_driver = TorchDriver(gpu_module, n_inputs=2)

    x = np.array([1.5, -0.5])
    assert gpu_driver.evaluate(x) == pytest.approx(cpu_driver.evaluate(x), rel=1e-5)
    assert isinstance(gpu_driver.evaluate(x), float)

    jac = gpu_driver.jacobian(x)
    assert isinstance(jac, np.ndarray)
    assert jac.tolist() == pytest.approx(cpu_driver.jacobian(x).tolist(), rel=1e-5)

    hess = gpu_driver.hessian(x)
    assert isinstance(hess, np.ndarray)
    assert hess.tolist() == pytest.approx(cpu_driver.hessian(x).tolist(), rel=1e-5)
