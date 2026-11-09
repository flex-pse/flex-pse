"""Smoke test: __version__ matches the installed distribution."""

import re
from importlib.metadata import version

import pytest

import flexdesign


@pytest.mark.unit
def test_version_matches_distribution():
    """flexdesign.__version__ matches the flex-pse distribution version."""
    assert flexdesign.__version__ == version("flex-pse")
    assert re.match(r"^\d+\.\d+\.\d+", flexdesign.__version__)
