"""Smoke test: the flexdesign package is importable."""

import pytest


@pytest.mark.unit
def test_import():
    """Importing flexdesign succeeds."""
    import flexdesign  # noqa: F401
