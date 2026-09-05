"""Smoke test: __version__ matches the installed distribution."""

import re
from importlib.metadata import version

import pytest

import flexcore


@pytest.mark.unit
def test_version_matches_distribution():
    """flexcore.__version__ matches the flex-pse distribution version."""
    assert flexcore.__version__ == version("flex-pse")
    assert re.match(r"^\d+\.\d+\.\d+", flexcore.__version__)
