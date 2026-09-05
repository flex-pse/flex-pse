"""Smoke test: __version__ matches the installed distribution."""

import re
from importlib.metadata import version

import pytest

import flexops


@pytest.mark.unit
def test_version_matches_distribution():
    """flexops.__version__ matches the flex-pse distribution version."""
    assert flexops.__version__ == version("flex-pse")
    assert re.match(r"^\d+\.\d+\.\d+", flexops.__version__)
