"""Tests for apply_to_model: the mutate-a-live-model FlexParameterize direction."""

import pyomo.environ as pyo
import pytest
from idaes.core.util.model_statistics import degrees_of_freedom
from pyomo.environ import units as pyunits

from flexcore.config.schema import SurrogateSpec
from flexcore.exceptions import FlexDataError
from flexops.unit_models import ConstantEnergyIntensityModel, ReverseOsmosis
from flexparameterize.apply import apply_to_model
from flexparameterize.tags import TagMap, model_alias
from flexparameterize.tests.helpers import INTENSITY, build_plant, evaluate_data

ALIASED = TagMap({})
"""An empty TagMap: the fixture data already carries model aliases."""


def _linear_spec(coefficient: float) -> SurrogateSpec:
    """A hand-built linear relationship on the unit's ``flow_in`` reference."""
    return SurrogateSpec(
        functional_form="linear",
        input_variables=["flow_in"],
        coefficients={"flow_in": coefficient, "intercept": 0.0},
    )


@pytest.mark.component
def test_apply_fixes_params_in_place():
    """The regressed parameter ends up fixed at the truth and the DOF drops."""
    m, unit = build_plant()
    data = evaluate_data(unit)
    unit.energy_intensity.unfix()
    dof_before = degrees_of_freedom(m)

    report = apply_to_model(m, data, ALIASED)

    assert unit.energy_intensity.fixed
    assert pyo.value(unit.energy_intensity) == pytest.approx(INTENSITY, rel=1e-6)
    assert degrees_of_freedom(m) == dof_before - 1
    assert report.dof_before - report.dof_after == 1
    assert report.fixed_parameters[unit.name]["energy_intensity"] == pytest.approx(
        INTENSITY, rel=1e-6
    )


@pytest.mark.component
def test_apply_swaps_energy_relation_in_place():
    """A richer relationship deactivates the default one on the same unit object.

    The constant-intensity regressor is the only one that ships so far and its
    fit never warrants a richer form, so the spec is supplied here;
    ``apply_to_model`` attaches a supplied and a fitted spec identically.
    """
    m, unit = build_plant()
    data = evaluate_data(unit)
    inlet, outlet = unit.inlet, unit.outlet
    components_before = set(unit.component_map())

    report = apply_to_model(
        m, data, ALIASED, surrogates={unit.name: _linear_spec(INTENSITY)}
    )

    relation = unit.power_electrical_relation
    fitted = unit.find_component("power_electrical_relation_fitted")
    assert all(not relation[t].active for t in m.time_block.time_index)
    assert fitted is not None
    assert all(fitted[t].active for t in m.time_block.time_index)
    assert unit.inlet is inlet and unit.outlet is outlet
    assert set(unit.component_map()) - components_before == {
        "power_electrical_relation_fitted"
    }
    assert report.swapped_relations == {unit.name: ["power_electrical_relation"]}


@pytest.mark.component
def test_apply_swaps_energy_relation_to_quadratic():
    """A quadratic power curve replaces the unit's constant-intensity relation."""
    m, unit = build_plant()
    data = evaluate_data(unit)
    spec = SurrogateSpec(
        functional_form="quadratic",
        input_variables=["flow_out"],
        coefficients={"intercept": 1.0, "flow_out": 0.4, "flow_out^2": 0.02},
    )

    report = apply_to_model(m, data, ALIASED, surrogates={unit.name: spec})

    assert report.swapped_relations == {unit.name: ["power_electrical_relation"]}
    assert all(
        not unit.power_electrical_relation[t].active for t in m.time_block.time_index
    )
    unit.flow_out[0].set_value(10.0)
    unit.power_electrical[0].set_value(0.0)
    # power - (1.0 + 0.4*10 + 0.02*100) == -7.0
    assert pyo.value(unit.power_electrical_relation_fitted[0].body) == pytest.approx(
        -7.0
    )


@pytest.mark.component
def test_apply_swaps_energy_relation_to_expanded_bilinear():
    """Net draw as a function of outlet flow, outlet pressure and their product."""
    m, unit = build_plant(has_pressure=True)
    data = evaluate_data(unit)
    spec = SurrogateSpec(
        functional_form="bilinear",
        input_variables=["flow_out", "outlet_state.pressure"],
        coefficients={
            "intercept": 1.0,
            "flow_out": 0.4,
            "outlet_state.pressure": 1e-5,
            "flow_out*outlet_state.pressure": 2e-6,
        },
    )

    report = apply_to_model(m, data, ALIASED, surrogates={unit.name: spec})

    assert report.swapped_relations == {unit.name: ["power_electrical_relation"]}
    unit.flow_out[0].set_value(10.0)
    unit.outlet_state.pressure[0].set_value(3.0e5)
    unit.power_electrical[0].set_value(0.0)
    # power - (1.0 + 0.4*10 + 1e-5*3e5 + 2e-6*10*3e5) == -(1 + 4 + 3 + 6)
    assert pyo.value(unit.power_electrical_relation_fitted[0].body) == pytest.approx(
        -14.0
    )


@pytest.mark.unit
def test_apply_insufficient_data_raises():
    """Insufficient data raises before anything on the model is mutated."""
    m, unit = build_plant()
    data = evaluate_data(unit).drop(columns=[model_alias(unit.power_electrical)])
    unit.energy_intensity.unfix()

    with pytest.raises(FlexDataError, match="power_electrical"):
        apply_to_model(m, data, ALIASED)

    assert not unit.energy_intensity.fixed


@pytest.mark.component
def test_apply_with_supplied_surrogate_skips_fit():
    """A supplied surrogate needs no data; a second unit still fits from data."""
    m, fitted_unit = build_plant()
    m.facility.vendor = ConstantEnergyIntensityModel(
        property_package=m.properties,
        energy_intensity=1.0 * pyunits.kWh / pyunits.m**3,
    )
    vendor = m.facility.vendor
    data = evaluate_data(fitted_unit)
    fitted_unit.energy_intensity.unfix()

    report = apply_to_model(
        m, data, ALIASED, surrogates={vendor.name: _linear_spec(2.0)}
    )

    assert report.swapped_relations == {vendor.name: ["power_electrical_relation"]}
    assert vendor.find_component("power_electrical_relation_fitted") is not None
    assert vendor.name not in report.fixed_parameters
    assert pyo.value(fitted_unit.energy_intensity) == pytest.approx(INTENSITY, rel=1e-6)
    assert fitted_unit.name in report.fixed_parameters


@pytest.mark.component
def test_apply_swaps_a_named_relation():
    """A ``{relation_name: spec}`` mapping swaps a unit's OTHER relation.

    Mixed in one call with a second unit that still fits its own energy
    relation from data normally.
    """
    m, fitted_unit = build_plant()
    data = evaluate_data(fitted_unit)
    fitted_unit.energy_intensity.unfix()
    m.facility.ro = ReverseOsmosis(property_package=m.properties)
    ro = m.facility.ro
    recovery_spec = SurrogateSpec(
        functional_form="linear",
        input_variables=["feed"],
        coefficients={"feed": 0.01, "intercept": 0.4},
    )

    report = apply_to_model(
        m, data, ALIASED, surrogates={ro.name: {"split_definition": recovery_spec}}
    )

    assert report.swapped_relations[ro.name] == ["split_definition"]
    assert ro.split_definition[0].active is False
    assert ro.split_mass_balance[0].active is True
    assert ro.name not in report.fixed_parameters
    assert fitted_unit.name in report.fixed_parameters
    assert pyo.value(fitted_unit.energy_intensity) == pytest.approx(INTENSITY, rel=1e-6)
