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


@pytest.fixture
def single_thread():
    """Run torch on one thread: BatchNorm's backward pass deadlocks in OpenMP on
    macOS conda, where a second OpenMP runtime is loaded."""
    torch = pytest.importorskip("torch")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _stochastic_module(torch):
    """A float32 module whose output depends on train/eval mode."""

    class _Stochastic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
            self.norm = torch.nn.BatchNorm1d(2)
            self.drop = torch.nn.Dropout(p=0.5)

        def forward(self, x):
            hidden = self.norm(self.linear(x).reshape(1, 2))
            return self.drop(hidden).sum()

    return _Stochastic()


@pytest.mark.unit
@pytest.mark.needs_torch
def test_repeated_evaluation_of_a_stochastic_module_returns_the_same_value(
    single_thread,
):
    torch = pytest.importorskip("torch")
    from flexops.surrogates.drivers.torch_driver import TorchDriver

    module = _stochastic_module(torch)
    driver = TorchDriver(module, n_inputs=2)
    x = np.array([1.0, 2.0])
    assert driver.evaluate(x) == driver.evaluate(x)
    np.testing.assert_array_equal(driver.jacobian(x), driver.jacobian(x))


@pytest.mark.unit
@pytest.mark.needs_torch
def test_driver_leaves_the_callers_module_untouched(single_thread):
    torch = pytest.importorskip("torch")
    from flexops.surrogates.drivers.torch_driver import TorchDriver

    module = _stochastic_module(torch)
    stats_before = module.norm.running_mean.clone()
    batches_before = module.norm.num_batches_tracked.clone()

    driver = TorchDriver(module, n_inputs=2)
    driver.evaluate(np.array([1.0, 2.0]))
    driver.jacobian(np.array([1.0, 2.0]))

    assert module.training
    assert next(module.parameters()).dtype == torch.float32
    assert torch.equal(module.norm.running_mean, stats_before)
    assert torch.equal(module.norm.num_batches_tracked, batches_before)


@pytest.mark.unit
@pytest.mark.needs_torch
def test_evaluate_emits_no_requires_grad_warning(recwarn):
    torch = pytest.importorskip("torch")
    from flexops.surrogates.drivers.torch_driver import TorchDriver

    driver = TorchDriver(torch.nn.Linear(1, 1), n_inputs=1)
    driver.evaluate(np.array([1.0]))
    assert not [w for w in recwarn if "requires_grad" in str(w.message)]


@pytest.mark.unit
@pytest.mark.needs_torch
def test_check_differentiable_names_the_input_count_on_a_shape_error():
    torch = pytest.importorskip("torch")
    from flexcore.exceptions import FlexConfigError
    from flexops.surrogates.drivers.torch_driver import TorchDriver

    driver = TorchDriver(torch.nn.Linear(3, 1), n_inputs=2)
    with pytest.raises(FlexConfigError, match="2 input"):
        driver.check_differentiable(np.array([1.0, 2.0]))


@pytest.mark.unit
@pytest.mark.needs_torch
def test_check_differentiable_rejects_a_non_scalar_output():
    torch = pytest.importorskip("torch")
    from flexcore.exceptions import FlexConfigError
    from flexops.surrogates.drivers.torch_driver import TorchDriver

    driver = TorchDriver(torch.nn.Linear(2, 3), n_inputs=2)
    with pytest.raises(FlexConfigError, match="scalar"):
        driver.check_differentiable(np.array([1.0, 2.0]))
