"""SpaceCooling and Chiller: cooling load to electrical draw through a COP."""

import pyomo.environ as pyo
import pytest

from flexcore.solvers import ProblemClass, classify
from flexops.testing import UnitModelTestHarness, dummy_time_block
from flexops.unit_models import Chiller, SpaceCooling


def with_load(unit, load_kw: float = 100.0):
    """Set every step of ``unit.cooling_load`` to ``load_kw`` and return the unit."""
    for t in unit.cooling_load:
        unit.cooling_load[t].set_value(load_kw)
    return unit


class TestSpaceCooling(UnitModelTestHarness):
    """100 kW of load at COP 4 with 0.05 kW/kW ancillary draws 30 kW."""

    expected_dof = 0
    expected_solution = {f"power_electrical[{t}]": 30.0 for t in range(3)}

    def configure(self):
        m = dummy_time_block(3)
        m.unit = SpaceCooling(cop=4.0, ancillary_intensity=0.05)
        return m, with_load(m.unit)


class TestChiller(UnitModelTestHarness):
    """100 kW of load at the default constant COP 5 and 0.05 ancillary draws 25 kW."""

    expected_dof = 0
    expected_solution = {f"power_electrical[{t}]": 25.0 for t in range(3)}

    def configure(self):
        m = dummy_time_block(3)
        m.unit = Chiller()
        return m, with_load(m.unit)


@pytest.mark.unit
def test_space_cooling_cooling_load_is_a_registered_input():
    m = dummy_time_block(3)
    m.unit = SpaceCooling()
    inputs = {r.name for r in m.unit._io_registry.io_variables if r.role == "input"}
    assert "cooling_load" in inputs


@pytest.mark.unit
def test_space_cooling_is_lp():
    m = dummy_time_block(3)
    m.unit = SpaceCooling()
    assert classify(m) is ProblemClass.LP


@pytest.mark.unit
def test_chiller_defaults_to_cop_of_five():
    m = dummy_time_block(3)
    m.unit = Chiller()
    assert pyo.value(m.unit.cop) == pytest.approx(5.0)
    assert classify(m) is ProblemClass.LP
