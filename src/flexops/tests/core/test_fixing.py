"""Tests for fixing and unfixing variables from fix, unfix and dispatch elements."""

import copy
import gzip
import json
import logging
import random
import warnings

import pyomo.environ as pyo
import pytest
from pydantic import TypeAdapter
from pyomo.environ import units as pyunits

import flexops as fo
from flexcore.config.spec import SCHEMA_VERSION, Element, FlowsheetSpec
from flexcore.exceptions import FlexConfigError, FlexEmitWarning, FlexFixWarning
from flexops import apply_spec, build_model, emit_model, logic
from flexops.core.compare import assert_models_equivalent
from flexops.core.fixing import FixReport, FixViolation, apply_state

M3_HR = pyunits.m**3 / pyunits.hr


def elements(*dicts) -> list:
    """Validate element dicts as spec elements."""
    return TypeAdapter(list[Element]).validate_python(list(dicts))


def fix(variable, value, **fields) -> dict:
    """Return a fix element dict."""
    return {"kind": "fix", "variable": variable, "value": value, **fields}


def unfix(variable, **fields) -> dict:
    """Return an unfix element dict."""
    return {"kind": "unfix", "variable": variable, **fields}


def dispatch(values, **fields) -> dict:
    """Return a dispatch element dict on the pump's power."""
    return {
        "kind": "dispatch",
        "unit": "p.pump",
        "variable": "power_electrical",
        "values": values,
        **fields,
    }


def plant_model() -> pyo.ConcreteModel:
    """Return a 3-step model with a pump, two RO trains with status, and test vars.

    ``p.x`` is time-indexed in m^3/s with bounds [0, 1]; ``p.flows`` is indexed
    by time and phase; ``p.scalar`` is a scalar in kW; ``p.k`` is a Param.
    """
    m = pyo.ConcreteModel(name="site")
    m.time_block = fo.TimeBlock(
        start_date="2025-01-01", end_date="2025-01-01T03:00", time_step=1 * pyunits.hr
    )
    m.properties = fo.SimpleAqueousFlow()
    m.p = fo.PlantBlock(time_block=m.time_block)
    intensity = 0.5 * pyunits.kWh / pyunits.m**3
    m.p.pump = fo.ConstantEnergyIntensityModel(
        property_package=m.properties, energy_intensity=intensity
    )
    m.p.trains = pyo.Set(initialize=[0, 1], ordered=True)
    m.p.ro = fo.ConstantEnergyIntensityModel(
        m.p.trains, property_package=m.properties, energy_intensity=intensity
    )
    for train in m.p.ro.values():
        logic.add_status(train, train.flow_in, 0 * M3_HR, 5 * M3_HR)
    time = m.time_block.time_index
    m.p.x = pyo.Var(time, bounds=(0, 1), units=pyunits.m**3 / pyunits.s)
    m.p.flows = pyo.Var(time, ["Liq", "Vap"], units=M3_HR)
    m.p.scalar = pyo.Var(units=pyunits.kW)
    m.p.k = pyo.Param(initialize=1.0, mutable=True)
    return m


def fixed(var) -> dict:
    """Return ``{index: value}`` for the fixed points of an indexed var."""
    return {i: pyo.value(v) for i, v in var.items() if v.fixed}


def apply(model, *dicts, **kwargs) -> FixReport:
    """Apply element dicts to ``model`` with apply_state."""
    return apply_state(model, elements(*dicts), **kwargs)


# -- value forms --------------------------------------------------------------


@pytest.mark.unit
def test_fix_scalar():
    m = plant_model()

    apply(m, fix("p.scalar", 3.0))

    assert m.p.scalar.fixed
    assert pyo.value(m.p.scalar) == 3.0


@pytest.mark.unit
def test_fix_single_index():
    m = plant_model()

    apply(m, fix("p.x[1]", 0.5))

    assert fixed(m.p.x) == {1: 0.5}


@pytest.mark.unit
def test_fix_whole_indexed_scalar_value():
    m = plant_model()

    apply(m, fix("p.x", 0.5))

    assert fixed(m.p.x) == {0: 0.5, 1: 0.5, 2: 0.5}


@pytest.mark.unit
def test_fix_time_series_list():
    m = plant_model()

    apply(m, fix("p.x", [0.1, 0.2, 0.3]))

    assert fixed(m.p.x) == {0: 0.1, 1: 0.2, 2: 0.3}


@pytest.mark.unit
def test_fix_time_series_dict():
    m = plant_model()

    apply(m, fix("p.x", {"0": 0.1, "1": 0.2, "2": 0.3}))

    assert fixed(m.p.x) == {0: 0.1, 1: 0.2, 2: 0.3}


@pytest.mark.unit
def test_fix_at_subset():
    m = plant_model()

    apply(m, fix("p.x", [0.2, 0.4], at=[0, 2]))

    assert fixed(m.p.x) == {0: 0.2, 2: 0.4}


@pytest.mark.unit
@pytest.mark.parametrize("phase", ["Liq", "'Liq'"])
def test_fix_component_index_wildcard(phase):
    m = plant_model()

    apply(m, fix(f"p.flows[*,{phase}]", 2.0))

    assert fixed(m.p.flows) == {(0, "Liq"): 2.0, (1, "Liq"): 2.0, (2, "Liq"): 2.0}


@pytest.mark.unit
def test_fix_indexed_members_wildcard():
    m = plant_model()

    apply(m, fix("p.ro[*].status", 1))

    for train in m.p.ro.values():
        assert fixed(train.status) == {0: 1, 1: 1, 2: 1}


@pytest.mark.unit
def test_fix_through_a_reference():
    m = plant_model()

    apply(m, fix("p.pump.flow_in[1]", 2.0))

    assert fixed(m.p.pump.inlet_state.flow_vol_phase) == {(1, "Liq"): 2.0}


@pytest.mark.unit
def test_fix_units_conversion():
    m = plant_model()

    apply(m, fix("p.x[0]", 1800.0, units="m^3/hr"))

    assert pyo.value(m.p.x[0]) == pytest.approx(0.5)


@pytest.mark.unit
def test_fix_units_on_a_unitless_var_rejected():
    m = plant_model()

    with pytest.raises(FlexConfigError, match="units"):
        apply(m, fix("p.ro[0].status[0]", 1, units="kW"))


@pytest.mark.unit
def test_fix_source_gz(tmp_path):
    (tmp_path / "x.json.gz").write_bytes(
        gzip.compress(json.dumps([0.1, 0.2, 0.3]).encode())
    )
    m = plant_model()

    apply(m, fix("p.x", {"$source": "x.json.gz"}), base_dir=tmp_path)

    assert fixed(m.p.x) == {0: 0.1, 1: 0.2, 2: 0.3}


@pytest.mark.unit
def test_unfix_whole_var():
    m = plant_model()

    report = apply(m, unfix("p.pump.energy_intensity"))

    assert not m.p.pump.energy_intensity.fixed
    assert report.unfixed == ["p.pump.energy_intensity"]


# -- helpful errors -------------------------------------------------------------


@pytest.mark.unit
def test_unknown_path_suggests_close_match():
    m = plant_model()

    with pytest.raises(FlexConfigError, match="matches no component") as excinfo:
        apply(m, fix("p.pump.power_electricl", 1.0))

    assert "p.pump.power_electrical" in str(excinfo.value)


@pytest.mark.unit
def test_malformed_path_is_a_config_error():
    m = plant_model()

    with pytest.raises(FlexConfigError, match="p.x\\["):
        apply(m, fix("p.x[", 1.0))


@pytest.mark.unit
def test_fix_param_rejected():
    m = plant_model()

    with pytest.raises(FlexConfigError, match="Params are set, not fixed"):
        apply(m, fix("p.k", 2.0))


@pytest.mark.unit
def test_fix_non_variable_rejected():
    m = plant_model()

    with pytest.raises(FlexConfigError, match="not a variable"):
        apply(m, fix("p.trains", 2.0))


@pytest.mark.unit
def test_wrong_length_message():
    m = plant_model()

    with pytest.raises(FlexConfigError, match="expected 3 values.*got 2"):
        apply(m, fix("p.x", [0.1, 0.2]))


@pytest.mark.unit
def test_missing_time_key_message():
    m = plant_model()

    with pytest.raises(FlexConfigError, match="missing time indices \\[2\\]"):
        apply(m, fix("p.x", {"0": 0.1, "1": 0.2}))


@pytest.mark.unit
def test_time_values_on_a_var_without_time_rejected():
    m = plant_model()

    with pytest.raises(FlexConfigError, match="no time index"):
        apply(m, fix("p.scalar", 1.0, at=[0]))


# -- policies -------------------------------------------------------------------


@pytest.mark.unit
def test_default_policy_is_error():
    spec = FlowsheetSpec.model_validate(
        {
            "schema_version": SCHEMA_VERSION,
            "name": "m",
            "elements": [
                {
                    "kind": "time",
                    "start_date": "2025-01-01",
                    "end_date": "2025-01-02",
                    "time_step": "1 hr",
                }
            ],
        }
    )
    m = plant_model()

    assert spec.fix_on_violation == "error"
    with pytest.raises(FlexConfigError, match="above_ub"):
        apply(m, fix("p.x[0]", 2.0))
    assert not m.p.x[0].fixed


@pytest.mark.unit
def test_error_policy_whole_stage_atomic():
    m = plant_model()

    with pytest.raises(FlexConfigError, match="p.x\\[1\\]"):
        apply(
            m,
            fix("p.x[0]", 0.5, name="good"),
            fix("p.x[1]", 5.0, name="bad"),
            unfix("p.pump.energy_intensity"),
        )

    assert fixed(m.p.x) == {}
    assert m.p.pump.energy_intensity.fixed


@pytest.mark.unit
def test_warn_policy():
    m = plant_model()

    with pytest.warns(FlexFixWarning) as caught:
        report = apply(m, fix("p.x", [0.5, 5.0, 0.5], on_violation="warn"))

    assert len(caught) == 1
    assert fixed(m.p.x) == {0: 0.5, 1: 5.0, 2: 0.5}
    assert report.violations == [
        FixViolation(
            variable="p.x[1]",
            value=5.0,
            lb=0,
            ub=1,
            kind="above_ub",
            action="fixed",
            element="fix p.x",
        )
    ]


@pytest.mark.unit
def test_skip_policy_leaves_point_free():
    m = plant_model()
    m.p.x[1].set_value(0.3)

    with pytest.warns(FlexFixWarning, match="p.x\\[1\\]"):
        report = apply(m, fix("p.x", [0.5, 5.0, 0.5], on_violation="skip"))

    assert fixed(m.p.x) == {0: 0.5, 2: 0.5}
    assert pyo.value(m.p.x[1]) == 0.3
    assert report.skipped == ["p.x[1]"]
    assert report.violations[0].action == "skipped"


@pytest.mark.unit
def test_domain_violation_each_policy():
    bad = "p.ro[0].status[0]"
    with pytest.raises(FlexConfigError, match="domain"):
        apply(plant_model(), fix(bad, 0.5))
    with pytest.raises(FlexConfigError, match="domain"):
        apply(plant_model(), fix(bad, 0.5, on_violation="warn"))

    m = plant_model()
    with pytest.warns(FlexFixWarning):
        report = apply(m, fix(bad, 0.5, on_violation="skip"))

    assert not m.p.ro[0].status[0].fixed
    assert report.violations[0].kind == "domain"


@pytest.mark.unit
def test_tolerance():
    apply(plant_model(), fix("p.x[0]", 1 + 1e-12))
    with pytest.raises(FlexConfigError):
        apply(plant_model(), fix("p.x[0]", 1.001))

    apply(plant_model(), fix("p.x[0]", 1.001), tolerance=0.01)
    apply(plant_model(), fix("p.x[0]", 1.001, tolerance=0.01))


@pytest.mark.unit
def test_element_policy_overrides_spec_default():
    with pytest.raises(FlexConfigError):
        apply(
            plant_model(),
            fix("p.x[0]", 2.0, on_violation="error"),
            on_violation="skip",
        )

    m = plant_model()
    with pytest.warns(FlexFixWarning):
        apply(m, fix("p.x[0]", 2.0, on_violation="warn"), on_violation="error")
    assert fixed(m.p.x) == {0: 2.0}


@pytest.mark.unit
def test_fix_warning_escalates():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FlexFixWarning)
        with pytest.raises(FlexFixWarning):
            apply(plant_model(), fix("p.x[0]", 2.0, on_violation="warn"))
        warnings.warn("not escalated", FlexEmitWarning, stacklevel=1)


@pytest.mark.unit
def test_violations_logged(caplog):
    logger = logging.getLogger("flexops.core.fixing")
    logger.addHandler(caplog.handler)
    try:
        with pytest.warns(FlexFixWarning):
            apply(
                plant_model(),
                fix("p.x", [2.0, 0.5, 3.0], on_violation="warn"),
                fix("p.ro[0].status[0]", 0.5, on_violation="skip"),
            )
    finally:
        logger.removeHandler(caplog.handler)

    warned = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warned) == 3


@pytest.mark.unit
def test_report_lists_every_violation_with_the_element_name():
    m = plant_model()

    with pytest.warns(FlexFixWarning):
        report = apply(m, fix("p.x", 2.0, on_violation="skip", name="levels"))

    assert {v.element for v in report.violations} == {"levels"}
    assert sorted(report.skipped) == ["p.x[0]", "p.x[1]", "p.x[2]"]


# -- conflicts ------------------------------------------------------------------


@pytest.mark.unit
def test_conflicting_fixes_error():
    with pytest.raises(FlexConfigError, match="p.x\\[1\\]") as excinfo:
        apply(
            plant_model(),
            fix("p.x", 0.2, name="first"),
            fix("p.x[1]", 0.3, name="second"),
        )
    assert "first" in str(excinfo.value)
    assert "second" in str(excinfo.value)

    m = plant_model()
    apply(m, fix("p.x", 0.2), fix("p.x[1]", 0.2))
    assert fixed(m.p.x) == {0: 0.2, 1: 0.2, 2: 0.2}


@pytest.mark.unit
def test_fix_unfix_conflict_error():
    with pytest.raises(FlexConfigError, match="conflict"):
        apply(plant_model(), fix("p.x[1]", 0.3), unfix("p.x"))


@pytest.mark.unit
def test_fix_dispatch_conflict_error():
    with pytest.raises(FlexConfigError, match="conflict"):
        apply(
            plant_model(),
            fix("p.pump.power_electrical[0]", 9.0),
            dispatch([1.0, 2.0, 3.0]),
        )


@pytest.mark.unit
def test_dispatch_fixes_and_records_the_series():
    m = plant_model()

    report = apply(m, dispatch({"0": 1.0, "1": 2.0, "2": 3.0}))

    assert fixed(m.p.pump.power_electrical) == {0: 1.0, 1: 2.0, 2: 3.0}
    assert m.p.pump._flex_dispatch["power_electrical"]["values"] == [1.0, 2.0, 3.0]
    assert "p.pump.power_electrical[0]" in report.fixed


@pytest.mark.unit
def test_dispatch_unknown_variable_raises():
    m = plant_model()

    with pytest.raises(FlexConfigError, match="not on"):
        apply(m, dispatch([1.0, 2.0, 3.0], variable="nope"))


# -- built models ---------------------------------------------------------------


TIME = {
    "kind": "time",
    "start_date": "2025-01-01",
    "end_date": "2025-01-01T03:00",
    "time_step": "1 hr",
}
INTENSITY = {"energy_intensity": {"value": 0.5, "units": "kWh/m^3"}}
STATE = [
    fix("p.ro[*].status", 1, name="always_on"),
    fix("p.pump.flow_in", [1.0, 2.0, 3.0]),
    unfix("p.ro[*].energy_intensity"),
    dispatch([1.0, 2.0, 3.0], unit="p.other"),
]


def spec_dict(*extra) -> dict:
    """Return a costed spec with a pump, two RO trains with status, and ``extra``."""
    return {
        "schema_version": SCHEMA_VERSION,
        "name": "site",
        "elements": copy.deepcopy(
            [
                TIME,
                {
                    "kind": "property_package",
                    "name": "properties",
                    "property_class": "SimpleAqueousFlow",
                },
                {
                    "kind": "costing",
                    "name": "costing",
                    "energy_prices": {"electrical": {"value": 0.1, "units": "USD/kWh"}},
                },
                {"kind": "plant", "name": "p"},
                {"kind": "set", "name": "p.trains", "values": [0, 1]},
                *(
                    {
                        "kind": "unit",
                        "name": name,
                        "unit_model_class": "ConstantEnergyIntensityModel",
                        "construction_options": INTENSITY,
                        **fields,
                    }
                    for name, fields in (
                        ("p.pump", {}),
                        ("p.other", {}),
                        ("p.ro", {"index": "p.trains"}),
                    )
                ),
                {
                    "kind": "status",
                    "unit": "p.ro",
                    "variable": "flow_in",
                    "min_output": {"value": 0.0, "units": "m^3/hr"},
                    "max_output": {"value": 5.0, "units": "m^3/hr"},
                },
                {"kind": "objective"},
                *extra,
            ]
        ),
    }


@pytest.mark.unit
def test_order_agnostic_state():
    data = spec_dict(*STATE)
    model = build_model(data)

    for seed in range(3):
        shuffled = copy.deepcopy(data)
        random.Random(seed).shuffle(shuffled["elements"])
        rebuilt = build_model(shuffled)

        assert_models_equivalent(model, rebuilt)
        assert set(rebuilt._flex_fix_report.fixed) == set(model._flex_fix_report.fixed)


@pytest.mark.unit
def test_build_rejects_a_conflicting_spec():
    with pytest.raises(FlexConfigError, match="conflict"):
        build_model(
            spec_dict(fix("p.pump.flow_in", 1.0), fix("p.pump.flow_in[0]", 2.0))
        )


@pytest.mark.unit
def test_build_uses_the_spec_policy():
    data = spec_dict(fix("p.ro[0].status[0]", 0.5))
    data["fix_on_violation"] = "skip"

    with pytest.warns(FlexFixWarning):
        model = build_model(data)

    assert not model.p.ro[0].status[0].fixed


@pytest.mark.unit
def test_apply_spec_fix_live_model():
    model = build_model(spec_dict(fix("p.pump.flow_in", 1.0)))
    identity = id(model)

    with pytest.raises(FlexConfigError, match="conflict"):
        apply_spec(model, [fix("p.other.flow_in[0]", 1.0), unfix("p.other.flow_in")])
    assert not model.p.other.inlet_state.flow_vol_phase[0, "Liq"].fixed

    report = apply_spec(
        model, [fix("p.pump.flow_in", 4.0), fix("p.other.flow_in", 2.0)]
    )

    assert id(model) == identity
    assert fixed(model.p.pump.flow_in) == {0: 4.0, 1: 4.0, 2: 4.0}
    assert "p.other.inlet_state.flow_vol_phase[0,Liq]" in report.fixed
    with warnings.catch_warnings():
        warnings.simplefilter("error", FlexEmitWarning)
        spec = emit_model(model)
    assert_models_equivalent(model, build_model(spec))


@pytest.mark.unit
def test_apply_spec_policy_comes_from_the_built_spec():
    data = spec_dict()
    data["fix_on_violation"] = "skip"
    model = build_model(data)

    with pytest.warns(FlexFixWarning):
        apply_spec(model, [fix("p.ro[0].status[0]", 0.5)])
    with pytest.raises(FlexConfigError, match="domain"):
        apply_spec(model, [fix("p.ro[0].status[0]", 0.5)], on_violation="error")


@pytest.mark.unit
def test_apply_spec_replaces_an_earlier_record_for_the_same_variable():
    model = build_model(spec_dict())

    apply_spec(model, [fix("p.pump.flow_in", 1.0)])
    apply_spec(model, [unfix("p.pump.flow_in")])

    recorded = [(e.kind, e.variable) for e in model._flex_applied]
    assert recorded == [("unfix", "p.pump.flow_in")]
