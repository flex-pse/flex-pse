"""Tests for apply_to_model: the mutate-a-live-model FlexParameterize direction."""

import pyomo.environ as pyo
import pytest
from idaes.core.util.model_statistics import degrees_of_freedom
from pyomo.environ import units as pyunits

from flexcore.config.schema import SurrogateSpec, SurrogateType
from flexcore.exceptions import FlexDataError
from flexops.surrogates import surrogate_from_spec
from flexops.surrogates.base import Surrogate
from flexops.unit_models import ConstantEnergyIntensityModel, ReverseOsmosis
from flexparameterize.apply import apply_to_model
from flexparameterize.tags import TagMap, model_alias
from flexparameterize.tests.helpers import INTENSITY, build_plant, evaluate_data

ALIASED = TagMap({})
"""An empty TagMap: the fixture data already carries model aliases."""


def _multilinear_spec(coefficient: float) -> SurrogateSpec:
    """A hand-built multilinear relationship on the unit's ``flow_in`` reference."""
    return SurrogateSpec(
        surrogate_type=SurrogateType.MULTILINEAR,
        data={
            "input_variables": {"flow_in": "m^3/hr"},
            "output_variables": {"power_electrical": "kW"},
            "coefficients": {"flow_in": coefficient, "intercept": 0.0},
        },
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
        m, data, ALIASED, surrogates={unit.name: _multilinear_spec(INTENSITY)}
    )

    relation = unit.power_electrical_relation
    fitted = unit.surrogate_power_electrical.fitted
    assert all(not relation[t].active for t in m.time_block.time_index)
    assert fitted is not None
    assert all(fitted[t].active for t in m.time_block.time_index)
    assert unit.inlet is inlet and unit.outlet is outlet
    assert set(unit.component_map()) - components_before == {
        "surrogate_power_electrical"
    }
    assert report.swapped_relations == {unit.name: ["power_electrical_relation"]}


@pytest.mark.component
def test_apply_swaps_energy_relation_to_multilinear():
    """Net draw as a function of outlet flow, outlet pressure and their product."""
    m, unit = build_plant(has_pressure=True)
    data = evaluate_data(unit)
    spec = SurrogateSpec(
        surrogate_type=SurrogateType.MULTILINEAR,
        data={
            "input_variables": {"flow_out": "m^3/hr", "outlet_state.pressure": "Pa"},
            "output_variables": {"power_electrical": "kW"},
            "coefficients": {
                "intercept": 1.0,
                "flow_out": 0.4,
                "outlet_state.pressure": 1e-5,
                "flow_out*outlet_state.pressure": 2e-6,
            },
        },
    )

    report = apply_to_model(m, data, ALIASED, surrogates={unit.name: spec})

    assert report.swapped_relations == {unit.name: ["power_electrical_relation"]}
    unit.flow_out[0].set_value(10.0)
    unit.outlet_state.pressure[0].set_value(3.0e5)
    unit.power_electrical[0].set_value(0.0)
    # power - (1.0 + 0.4*10 + 1e-5*3e5 + 2e-6*10*3e5) == -(1 + 4 + 3 + 6)
    assert pyo.value(unit.surrogate_power_electrical.fitted[0].body) == pytest.approx(
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
        m, data, ALIASED, surrogates={vendor.name: _multilinear_spec(2.0)}
    )

    assert report.swapped_relations == {vendor.name: ["power_electrical_relation"]}
    assert vendor.surrogate_power_electrical.fitted is not None
    assert report.fixed_parameters[vendor.name] == {"flow_in": 2.0, "intercept": 0.0}
    assert vendor.name not in {
        k for k, v in report.fixed_parameters.items() if k != vendor.name
    }
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
        surrogate_type=SurrogateType.MULTILINEAR,
        data={
            "input_variables": {"feed": "m^3/hr"},
            "output_variables": {"permeate": "m^3/hr"},
            "coefficients": {"feed": 0.01, "intercept": 0.4},
        },
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


@pytest.mark.component
def test_apply_switches_to_active_block():
    """``active_surrogates`` reactivates a previously built surrogate block.

    The block must already exist from a prior ``swap_relation`` (here the first
    ``apply_to_model`` call creates it). The second call switches back to it
    via ``switch_surrogate_block`` without refitting.
    """
    m, unit = build_plant()
    data = evaluate_data(unit)
    unit.energy_intensity.unfix()

    block_name = "surrogate_power_electrical"
    first_report = apply_to_model(
        m, data, ALIASED, surrogates={unit.name: _multilinear_spec(INTENSITY)}
    )
    assert first_report.swapped_relations == {unit.name: ["power_electrical_relation"]}
    assert unit.current_surrogate_block("power_electrical_relation") == block_name

    second_report = apply_to_model(
        m, data, ALIASED, active_surrogates={unit.name: block_name}
    )

    assert second_report.swapped_relations == {unit.name: [block_name]}
    assert unit.current_surrogate_block("power_electrical_relation") == block_name
    assert unit.surrogate_power_electrical.fitted is not None


class _NoCoefSurrogate(Surrogate):
    """A surrogate whose block carries no ``coefficients`` registry."""

    def _validate(self):
        pass

    @property
    def input_variables(self):
        return {}

    @property
    def output_variables(self):
        return {"power_electrical": "kW"}

    def build(self, unit, target):
        def body(t):
            return 2.0 * pyunits.get_units(target[t])

        block = pyo.Block(concrete=True)
        block.body = body
        return block, body


@pytest.mark.component
def test_apply_surrogate_without_coefficients_skips_registry():
    """A surrogate block with no ``coefficients`` is attached without error.

    ``apply_to_model`` must not assume every surrogate block carries a
    ``CoefficientRegistry``; surrogates that encode their relationship
    directly in the fitted Constraint have nothing to register or fix.
    """
    m, unit = build_plant()
    data = evaluate_data(unit)
    unit.energy_intensity.unfix()

    no_coef_spec = SurrogateSpec(
        surrogate_type=SurrogateType.MULTILINEAR,
        data={
            "input_variables": {},
            "output_variables": {"power_electrical": "kW"},
        },
    )

    original_surrogate_from_spec = surrogate_from_spec

    def _fake_surrogate_from_spec(spec):
        if spec is no_coef_spec:
            return _NoCoefSurrogate(spec.data)
        return original_surrogate_from_spec(spec)

    import flexops.core.stages as stages_module

    stages_module.surrogate_from_spec = _fake_surrogate_from_spec
    try:
        report = apply_to_model(m, data, ALIASED, surrogates={unit.name: no_coef_spec})
    finally:
        stages_module.surrogate_from_spec = original_surrogate_from_spec

    assert report.swapped_relations == {unit.name: ["power_electrical_relation"]}
    assert unit.surrogate_power_electrical.fitted is not None
    assert not hasattr(unit.surrogate_power_electrical, "coefficients")


@pytest.mark.component
def test_apply_surrogate_idempotent_on_second_call():
    """Calling apply_to_model twice with the same surrogate does not duplicate
    coefficient ParameterRecords in the unit's IO registry.
    """
    m, unit = build_plant()
    data = evaluate_data(unit)
    unit.energy_intensity.unfix()

    spec = _multilinear_spec(INTENSITY)

    first_report = apply_to_model(m, data, ALIASED, surrogates={unit.name: spec})
    assert first_report.swapped_relations == {unit.name: ["power_electrical_relation"]}
    param_count_after_first = len(unit._io_registry.parameters)

    second_report = apply_to_model(m, data, ALIASED, surrogates={unit.name: spec})
    assert second_report.swapped_relations == {unit.name: ["power_electrical_relation"]}
    assert len(unit._io_registry.parameters) == param_count_after_first

    param_names = [p.name for p in unit._io_registry.parameters]
    assert param_names.count("flow_in") == 1
    assert param_names.count("intercept") == 1


@pytest.mark.unit
def test_apply_relation_spec_constants_match():
    """flexops' shared relation constants equal flexparameterize's own."""
    from flexops.core.stages import ENERGY_RELATION, INTENSITY_PARAMETER
    from flexparameterize.apply import POWER_ELECTRICAL_RELATION
    from flexparameterize.regression import COEFFICIENT_NAME

    assert ENERGY_RELATION == POWER_ELECTRICAL_RELATION
    assert INTENSITY_PARAMETER == COEFFICIENT_NAME
