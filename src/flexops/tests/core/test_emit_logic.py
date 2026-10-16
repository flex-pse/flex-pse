"""Tests for emit_model writing logic, ramp and extension elements back."""

import warnings

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits

import flexops as fo
from flexcore.exceptions import FlexEmitWarning
from flexops import emit_model, logic
from flexops.tests import sample_extensions
from flexops.tests.core.test_emit_model import base_model

KW = pyunits.kW


def emit_quietly(model):
    """Process costing, then emit with every FlexEmitWarning an error."""
    if model.costing.find_component("aggregate_operating_cost") is None:
        model.costing.cost_process()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FlexEmitWarning)
        return emit_model(model)


def trains(model):
    """Add an indexed pump ``p.ro`` over ``p.trains = [0, 1, 2]``."""
    model.p.trains = pyo.Set(initialize=[0, 1, 2], ordered=True)
    model.p.ro = fo.Pump(model.p.trains, property_package=model.properties)
    return list(model.p.ro.values())


def dumped(spec, *kinds) -> list[dict]:
    """Return the spec's elements of ``kinds`` as JSON dicts without defaults."""
    return [
        el.model_dump(mode="json", exclude_defaults=True) for el in spec.of_kind(*kinds)
    ]


@pytest.mark.unit
def test_equal_member_logic_is_one_element_on_the_indexed_unit():
    m = base_model()
    for member in trains(m):
        logic.add_status(member, member.power_electrical, 0 * KW, 10 * KW)

    assert dumped(emit_quietly(m), "status") == [
        {
            "kind": "status",
            "unit": "p.ro",
            "variable": "power_electrical",
            "min_output": {"value": 0, "units": "kW"},
            "max_output": {"value": 10, "units": "kW"},
        }
    ]


@pytest.mark.unit
def test_differing_member_logic_is_one_element_per_member():
    m = base_model()
    members = trains(m)
    logic.add_dwell(members[0].flow_in, 2)
    logic.add_dwell(members[2].flow_in, 3)

    assert sorted((el["unit"], el["k"]) for el in dumped(emit_quietly(m), "dwell")) == [
        ("p.ro[0]", 2),
        ("p.ro[2]", 3),
    ]


@pytest.mark.unit
def test_cross_unit_logic_is_written_with_full_unit_names():
    m = base_model()
    m.p.a = fo.Pump(property_package=m.properties)
    m.p.b = fo.Pump(property_package=m.properties)
    for unit in (m.p.a, m.p.b):
        logic.add_status(unit, unit.power_electrical, 0 * KW, 10 * KW)
    logic.add_conditional(m.p.a, m.p.b, then="off")

    assert dumped(emit_quietly(m), "conditional") == [
        {"kind": "conditional", "unit": "p.a", "implies": "p.b", "state": "off"}
    ]


@pytest.mark.unit
def test_ramp_is_written_with_its_units_and_window():
    m = base_model()
    m.p.a = fo.Pump(property_package=m.properties)
    logic.add_ramp_rate(
        m.p.a.power_electrical, ramp_down=2.0, ramp_rate_units=KW / pyunits.hr, window=2
    )

    assert dumped(emit_quietly(m), "ramp") == [
        {
            "kind": "ramp",
            "unit": "p.a",
            "variable": "power_electrical",
            "ramp_down": 2.0,
            "units": "kW/h",
            "window": 2,
        }
    ]


@pytest.mark.unit
def test_extension_and_its_objective_are_emitted_without_a_warning():
    m = base_model()
    m.p.a = fo.Pump(property_package=m.properties)
    m.costing.cost_process()
    sample_extensions.minimize_power(m, units=["p.a"])

    spec = emit_quietly(m)

    assert dumped(spec, "extension") == [
        {
            "kind": "extension",
            "entry_point": "flexops.tests.sample_extensions:minimize_power",
            "options": {"units": ["p.a"]},
            "name": "minimize_power",
        }
    ]
    assert dumped(spec, "objective") == [
        {"kind": "objective", "name": "power_objective", "expression": "extension"}
    ]


@pytest.mark.unit
def test_custom_objective_without_an_extension_still_warns():
    m = base_model()
    m.p.a = fo.Pump(property_package=m.properties)
    m.costing.cost_process()
    m.obj = pyo.Objective(expr=sum(m.p.a.power_electrical.values()))

    with pytest.warns(FlexEmitWarning, match="obj"):
        emit_model(m, check=False)
