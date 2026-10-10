"""OpsBlockData.build_from_config: real config-driven unit construction (§3.2, R3)."""

import pytest
from pydantic import ValidationError

from flexcore.config.schema import UnitConfig
from flexops.core.ops_block import OpsBlockData
from flexops.testing import dummy_time_block

_VALID = {
    "unit_model_class": "ConstantEnergyIntensityModel",
    "construction_options": {"energy_intensity": {"value": 0.42, "units": "kWh/m^3"}},
    "io_variables": [
        {"name": "flow_vol_phase", "role": "input", "units": "m^3/hr"},
        {"name": "power_electrical", "role": "output", "units": "kW"},
    ],
}


@pytest.mark.unit
def test_build_from_config_builds_unit():
    """A valid unit config builds the named unit with the declared IO registered."""
    m = dummy_time_block(3)
    cfg = UnitConfig.model_validate(_VALID)
    m.unit = OpsBlockData.build_from_config(cfg, property_package=m.properties)

    registered = {record.name for record in m.unit._io_registry.io_variables}
    assert {spec.name for spec in cfg.io_variables} <= registered
    assert m.unit.energy_intensity.value == pytest.approx(0.42, rel=1e-6)


@pytest.mark.unit
def test_build_from_config_accepts_a_raw_mapping_by_round_tripping_it():
    """A dict is validated through the pydantic schema, never passed on raw."""
    m = dummy_time_block(3)
    m.unit = OpsBlockData.build_from_config(_VALID, property_package=m.properties)
    assert m.unit.find_component("power_electrical_relation") is not None


@pytest.mark.unit
def test_build_from_config_bad_config_raises():
    """A wrong-typed field raises ValidationError naming the field path."""
    bad = {
        "unit_model_class": "ConstantEnergyIntensityModel",
        "io_variables": [{"name": "flow_vol_phase", "role": "sideways", "units": "m"}],
    }
    with pytest.raises(ValidationError) as excinfo:
        OpsBlockData.build_from_config(bad)
    assert "io_variables.0.role" in str(excinfo.value)


@pytest.mark.unit
def test_unset_unit_commitment_keeps_unit_default():
    """A config that omits unit_commitment gets the unit's own default."""
    m = dummy_time_block(3)
    m.unit = OpsBlockData.build_from_config(
        UnitConfig(
            unit_model_class="BatteryModel",
            construction_options={"capacity": {"value": 1.0, "units": "kWh"}},
        )
    )

    assert m.unit.config.unit_commitment.status is False
    assert m.unit.find_component("status") is None
