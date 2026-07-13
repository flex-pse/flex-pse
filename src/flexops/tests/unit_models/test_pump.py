"""Harness-driven tests for ``Pump`` (both energy relations, LP)."""

import pytest

from flexcore.exceptions import FlexConfigError
from flexops.testing import UnitModelTestHarness, dummy_time_block
from flexops.unit_models import Pump

# Hand-computed electrical draw for the efficiency/head relation:
# density * g * head * flow / efficiency, converted to kW.
_HEAD_MODE_KW = 1000.0 * 9.80665 * 50.0 * (100.0 / 3600.0) / 0.8 / 1000.0


class TestPump(UnitModelTestHarness):
    """A Pump on a 3-point dummy model (default energy-intensity relation).

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


class TestPumpEfficiencyHead(UnitModelTestHarness):
    """A Pump on a 3-point dummy model with the efficiency/head relation.

    With ``flow_vol`` at 100 m³/hr, 50 m head, 0.8 efficiency, and the
    property package's fixed 1000 kg/m³ density, the electrical draw is
    density * g * head * flow / efficiency ≈ 17.0254 kW at every time point.
    """

    expected_dof = 0
    expected_solution = {
        "electrical_power[0]": _HEAD_MODE_KW,
        "electrical_power[1]": _HEAD_MODE_KW,
        "electrical_power[2]": _HEAD_MODE_KW,
    }

    def configure(self):
        """Build a 3-point dummy model with one efficiency/head Pump."""
        m = dummy_time_block(3)
        m.unit = Pump(
            property_package=m.properties,
            energy_relation="efficiency_head",
            efficiency=0.8,
            head=50.0,
        )
        for t in m.time_block.time_index:
            m.unit.flow_vol[t].set_value(100.0)
        return m, m.unit


@pytest.mark.unit
def test_energy_relation_selects_params():
    """Each relation builds only its own parameters, both regressable.

    FlexParameterize may later swap the efficiency Param for a fitted
    function; in v0 both relations expose plain mutable Params.
    """
    m = dummy_time_block(3)
    m.unit = Pump(property_package=m.properties)
    assert m.unit.component("energy_intensity") is not None
    assert m.unit.component("efficiency") is None
    assert m.unit.component("head") is None

    m2 = dummy_time_block(3)
    m2.unit = Pump(
        property_package=m2.properties,
        energy_relation="efficiency_head",
        efficiency=0.8,
        head=50.0,
    )
    assert m2.unit.component("energy_intensity") is None
    assert m2.unit.component("efficiency") is not None
    assert m2.unit.component("head") is not None
    registered = {r.name: r for r in m2.unit._io_registry.parameters}
    assert registered["efficiency"].regressable is True
    assert registered["head"].regressable is True


@pytest.mark.unit
def test_efficiency_head_requires_head():
    """The efficiency/head relation without a head is a config error."""
    m = dummy_time_block(3)
    with pytest.raises(FlexConfigError, match="head"):
        m.unit = Pump(property_package=m.properties, energy_relation="efficiency_head")


@pytest.mark.unit
def test_invalid_energy_relation():
    """An unknown energy_relation value is a config error naming the options."""
    m = dummy_time_block(3)
    with pytest.raises(FlexConfigError, match="energy_relation"):
        m.unit = Pump(property_package=m.properties, energy_relation="affinity")
