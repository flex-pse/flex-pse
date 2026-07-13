"""Harness-driven tests for ``Pump`` (constant energy intensity, LP)."""

from flexops.testing import UnitModelTestHarness, dummy_time_block
from flexops.unit_models import Pump


class TestPump(UnitModelTestHarness):
    """A Pump on a 3-point dummy model.

    With ``flow_vol`` at 100 m³/hr and the default energy intensity of
    0.5 kWh/m³, the electrical draw is 100 x 0.5 = 50 kW at every time point
    (kWh/m³ x m³/hr = kW, no conversion factor).
    """

    expected_dof = 0
    expected_solution = {
        "electrical_power[0]": 50.0,
        "electrical_power[1]": 50.0,
        "electrical_power[2]": 50.0,
    }

    def configure(self):
        """Build a 3-point dummy model with one Pump; fix nothing."""
        m = dummy_time_block(3)
        m.unit = Pump(property_package=m.properties)
        for t in m.time_block.time_index:
            m.unit.flow_vol[t].set_value(100.0)
        return m, m.unit
