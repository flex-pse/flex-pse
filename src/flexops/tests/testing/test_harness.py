"""Self-checks for the public UnitModelTestHarness (M04, plan §2)."""

import pytest
from pyomo.environ import units as pyunits

from flexops.properties.simple_aqueous import SimpleAqueousFlowData
from flexops.testing import dummy_time_block


@pytest.mark.unit
def test_base_class_not_collected(pytester):
    """Pytest collects zero tests from the bare harness (its configure() raises)."""
    pytester.makepyfile("""
        from flexops.testing import UnitModelTestHarness

        class NotATest(UnitModelTestHarness):
            pass
        """)
    result = pytester.runpytest("-p", "no:cacheprovider")
    result.assert_outcomes(passed=0, failed=0, errors=0)


@pytest.mark.unit
def test_dummy_time_block_shape():
    """dummy_time_block(3) has 3 time points, a SimpleAqueousFlow, 15-min dt."""
    m = dummy_time_block(3)
    assert m.time_block.n_points == 3
    assert m.time_block.dt.value == pytest.approx(15)
    assert pyunits.get_units(m.time_block.dt) == pyunits.min
    assert isinstance(m.properties, SimpleAqueousFlowData)
