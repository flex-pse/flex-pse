"""Smoke test: the flexops package is importable."""

import pytest


@pytest.mark.unit
def test_import():
    """Importing flexops succeeds."""
    import flexops  # noqa: F401


@pytest.mark.unit
def test_top_level_exports():
    """TimeBlock, the property packages, and PowerKind are importable from flexops."""
    from flexops import PowerKind, SimpleAqueousFlow, SimpleGasFlow, TimeBlock

    assert PowerKind.ELECTRICAL == "electrical"
    assert PowerKind.THERMAL == "thermal"
    assert SimpleAqueousFlow is not None
    assert SimpleGasFlow is not None
    assert TimeBlock is not None
