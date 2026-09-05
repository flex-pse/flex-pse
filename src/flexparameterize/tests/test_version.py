"""Smoke test: __version__ matches the installed distribution."""

import re
from importlib.metadata import version

import pytest

import flexparameterize


@pytest.mark.unit
def test_version_matches_distribution():
    """flexparameterize.__version__ matches the flex-pse distribution version."""
    assert flexparameterize.__version__ == version("flex-pse")
    assert re.match(r"^\d+\.\d+\.\d+", flexparameterize.__version__)
