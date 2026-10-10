"""Tests for flexops.core.compare.model_differences."""

import pyomo.environ as pyo
import pytest

from flexops.core.compare import model_differences
from flexops.testing import dummy_time_block
from flexops.unit_models import Tank


def build():
    """Return a small model with one Tank."""
    m = dummy_time_block(3)
    m.tank = Tank(property_package=m.properties)
    return m


@pytest.mark.unit
def test_identical_models_have_no_differences():
    assert model_differences(build(), build()) == []


@pytest.mark.unit
def test_extra_constraint_is_listed():
    a, b = build(), build()
    b.extra = pyo.Constraint(expr=b.tank.flow_in[0] >= 1)

    assert model_differences(a, b) == ["Only in b: extra (Constraint)"]


@pytest.mark.unit
def test_compare_module_does_not_import_pytest():
    import flexops.core.compare as compare

    assert "pytest" not in vars(compare)
