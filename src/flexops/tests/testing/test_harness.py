"""Self-checks for the public test harness and the ``dummy_time_block`` helper.

The behavioral checks of the harness stages themselves live in the unit-model
test classes that subclass it (``TestPump``, ``TestStorageTank``); this module
verifies the two harness-module contracts: the bare base class is never
collected by pytest, and ``dummy_time_block`` has the exact shape the M14 docs
generator relies on.
"""

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits

from flexops.properties.simple_aqueous import SimpleAqueousFlowData
from flexops.testing import UnitModelTestHarness, dummy_time_block


@pytest.mark.unit
def test_base_class_not_collected(pytester):
    """Pytest collects no items from the bare ``UnitModelTestHarness``.

    The base class's name does not match pytest's ``Test*`` collection pattern
    (pitfall 5): its ``configure`` raises, so collecting it would fail every
    stage on every run.
    """
    assert not UnitModelTestHarness.__name__.startswith("Test")
    pytester.makepyfile("""
        from flexops.testing import UnitModelTestHarness
        """)
    items, _ = pytester.inline_genitems()
    assert items == []


@pytest.mark.unit
def test_dummy_time_block_shape():
    """``dummy_time_block(3)`` has 3 points, SimpleAqueousFlow, 15-min dt."""
    m = dummy_time_block(3)
    assert m.time_block.n_points == 3
    assert isinstance(m.properties, SimpleAqueousFlowData)
    dt_min = pyo.value(pyunits.convert(m.time_block.dt, pyunits.min))
    assert dt_min == pytest.approx(15.0, rel=1e-9)


@pytest.mark.unit
def test_dummy_time_block_default_n():
    """The signature is exactly ``dummy_time_block(n: int = 3)`` (M14 imports it)."""
    m = dummy_time_block()
    assert m.time_block.n_points == 3
