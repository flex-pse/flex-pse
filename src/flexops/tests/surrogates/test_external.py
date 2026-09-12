"""Tests for the external-model framework/driver registry (no framework, no
solver -- must run and pass on a bare install)."""

import subprocess
import sys

import pytest

from flexcore.exceptions import FlexConfigError
from flexops.surrogates.external import ExternalFramework, get_driver

RESERVED = (
    ExternalFramework.TENSORFLOW,
    ExternalFramework.ONNX,
    ExternalFramework.JAX,
)


@pytest.mark.unit
@pytest.mark.needs_torch
def test_get_driver_resolves_pytorch():
    """The implemented framework resolves to its driver class."""
    from flexops.surrogates.drivers.torch_driver import TorchDriver

    assert get_driver(ExternalFramework.PYTORCH) is TorchDriver
    assert get_driver("pytorch") is TorchDriver


@pytest.mark.unit
@pytest.mark.parametrize("framework", RESERVED)
def test_get_driver_reserved_frameworks_raise_not_implemented(framework):
    """A reserved-but-unbuilt framework raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        get_driver(framework)


@pytest.mark.unit
def test_get_driver_unknown_framework_raises_config_error():
    """An unknown framework name raises FlexConfigError listing known values."""
    with pytest.raises(FlexConfigError, match="pytorch"):
        get_driver("not_a_real_framework")


@pytest.mark.unit
def test_importing_flexops_surrogates_does_not_import_torch():
    """A bare install imports flexops.surrogates with no torch import."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import flexops.surrogates; "
            "assert 'torch' not in sys.modules, sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
