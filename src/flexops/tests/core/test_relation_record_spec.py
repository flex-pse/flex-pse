"""Tests that a swap records the SurrogateSpec its surrogate was built from."""

import pytest

from flexcore.config.schema import SurrogateSpec, SurrogateType
from flexops.surrogates import MultilinearSurrogate, surrogate_from_spec
from flexops.testing import dummy_time_block
from flexops.unit_models import ConstantEnergyIntensityModel

RELATION = "power_electrical_relation"
DATA = {
    "input_variables": {"flow_out": "m^3/hr"},
    "output_variables": {"power_electrical": "kW"},
    "coefficients": {"flow_out": 0.8, "intercept": 0.0},
}


def build_unit():
    """Return a model with one constant-intensity unit ``m.unit``."""
    m = dummy_time_block(3)
    m.unit = ConstantEnergyIntensityModel(property_package=m.properties)
    return m


def record(unit):
    """Return the unit's energy relation record."""
    return next(r for r in unit._io_registry.relations if r.name == RELATION)


@pytest.mark.unit
def test_swap_with_spec_built_surrogate_stores_the_spec():
    m = build_unit()
    spec = SurrogateSpec(surrogate_type=SurrogateType.MULTILINEAR, data=DATA)

    m.unit.swap_relation(RELATION, surrogate_from_spec(spec))

    assert record(m.unit).spec == spec


@pytest.mark.unit
def test_surrogate_from_spec_sets_spec_on_the_surrogate():
    spec = SurrogateSpec(surrogate_type=SurrogateType.MULTILINEAR, data=DATA)

    assert surrogate_from_spec(spec).spec is spec


@pytest.mark.unit
def test_swap_with_hand_built_surrogate_stores_no_spec():
    m = build_unit()

    m.unit.swap_relation(RELATION, MultilinearSurrogate(DATA))

    assert record(m.unit).spec is None
