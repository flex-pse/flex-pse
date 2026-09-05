"""Smoke test: __version__ matches the installed distribution."""

import re
from importlib.metadata import version

import pytest

import flexschedule


@pytest.mark.unit
def test_version_matches_distribution():
    """flexschedule.__version__ matches the flex-pse distribution version."""
    assert flexschedule.__version__ == version("flex-pse")
    assert re.match(r"^\d+\.\d+\.\d+", flexschedule.__version__)
