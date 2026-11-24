"""The surrogate spec envelope and the 0.1.1 -> 0.2.0 surrogate migration."""

import json
import math

import pytest

from flexcore.config.io import dump_spec, load_spec
from flexcore.config.schema import ParameterState, SurrogateSpec, _SurrogateFields
from flexcore.config.spec import SCHEMA_VERSION, SurrogateElement
from flexcore.config.surrogate_migration import migrate_surrogate_spec
from flexcore.exceptions import FlexConfigError, FlexMigrationWarning

HISTORY = {
    "start_date": "2025-01-01T00:00:00",
    "time_step_seconds": 3600.0,
    "eta_values": [1.0],
    "eps_values": [0.0],
}


def _flowsheet(*surrogates, version=SCHEMA_VERSION):
    """A minimal valid flowsheet: elements 0-1 are fixed, surrogates follow."""
    return {
        "schema_version": version,
        "name": "p",
        "elements": [
            {"kind": "plant", "name": "p"},
            {
                "kind": "unit",
                "name": "p.ro",
                "unit_model_class": "ReverseOsmosis",
                "property_package": None,
                "costing_package": None,
            },
            *surrogates,
            {
                "kind": "time",
                "start_date": "2025-01-01",
                "end_date": "2025-01-02",
                "time_step": "1 hr",
            },
        ],
    }


# -- envelope ----------------------------------------------------------------


@pytest.mark.unit
def test_parameter_state_defaults_to_fixed_and_always_writes_it():
    """fixed defaults to true, and a dump keeps it even when defaults are dropped."""
    state = ParameterState(value=0.5)
    assert state.fixed is True
    assert json.loads(state.model_dump_json(exclude_defaults=True)) == {
        "value": 0.5,
        "fixed": True,
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "value", [2, 0.5, [1.0, 2.0], [[0.1, 0.2], [0.3, 0.4]], [[[1.0]]]]
)
def test_parameter_value_is_a_number_or_nested_lists(value):
    """Scalars, lists and nested lists are accepted and dumped as floats."""
    dumped = ParameterState(value=value).model_dump()["value"]

    def leaves(node):
        return (
            [x for item in node for x in leaves(item)]
            if isinstance(node, list)
            else [node]
        )

    assert dumped == value
    assert all(type(leaf) is float for leaf in leaves(dumped))


@pytest.mark.unit
@pytest.mark.parametrize("value", [True, "1.0", math.nan, math.inf, {"a": 1.0}])
def test_parameter_value_rejects_non_numbers(value):
    """Booleans, strings, NaN, infinities and mappings are not values."""
    with pytest.raises(ValueError):
        ParameterState(value=value)


@pytest.mark.unit
def test_fixed_parameter_needs_a_value_but_a_free_one_does_not():
    """Only a free parameter may leave its value to the surrogate's guess."""
    assert ParameterState(fixed=False).value is None
    with pytest.raises(ValueError, match="fixed parameter needs a value"):
        ParameterState(fixed=True)
    with pytest.raises(ValueError, match="fixed parameter needs a value"):
        ParameterState()


@pytest.mark.unit
def test_parameter_state_rejects_unknown_fields_and_non_bool_fixed():
    """A parameter carries exactly value and fixed, fixed as a real bool."""
    with pytest.raises(ValueError):
        ParameterState(value=1.0, bounds=[0, 1])
    with pytest.raises(ValueError):
        ParameterState(value=1.0, fixed="yes")
    with pytest.raises(ValueError):
        ParameterState(value=1.0, fixed=1)


@pytest.mark.unit
def test_element_carries_exactly_the_surrogate_spec_fields():
    """A surrogate element is placement plus every SurrogateSpec field."""
    assert set(SurrogateSpec.model_fields) == set(_SurrogateFields.model_fields)
    assert set(SurrogateElement.model_fields) == set(SurrogateSpec.model_fields) | {
        "kind",
        "unit",
        "relation",
        "name",
    }


@pytest.mark.unit
def test_element_and_spec_convert_both_ways():
    """to_surrogate_spec and from_surrogate_spec round-trip every field."""
    spec = SurrogateSpec(
        surrogate_type="multilinear",
        config_version=2,
        config={"terms": ["feed"]},
        parameters={"feed": {"value": 0.4, "fixed": False}},
        provenance={"rmse": 0.1},
    )
    element = SurrogateElement.from_surrogate_spec(
        spec, unit="p.ro", relation="split_definition", name="split"
    )
    assert (element.unit, element.relation, element.name) == (
        "p.ro",
        "split_definition",
        "split",
    )
    assert element.to_surrogate_spec() == spec
    assert element.free_parameters() == ["feed"]


@pytest.mark.unit
def test_element_dump_puts_placement_first_and_keeps_config_version():
    """A dumped element reads kind/unit/relation first and names its version."""
    element = SurrogateElement(
        kind="surrogate",
        unit="p.ro",
        surrogate_type="multilinear",
        parameters={"intercept": {"value": 0.0}},
    )
    dumped = json.loads(element.model_dump_json(exclude_defaults=True))
    assert list(dumped)[:2] == ["kind", "unit"]
    assert dumped["config_version"] == 1
    assert dumped["parameters"] == {"intercept": {"value": 0.0, "fixed": True}}


@pytest.mark.unit
def test_spec_dump_keeps_config_version():
    """A standalone spec dump names its config version even at the default."""
    spec = SurrogateSpec(surrogate_type="multilinear")
    assert json.loads(spec.model_dump_json(exclude_defaults=True)) == {
        "surrogate_type": "multilinear",
        "config_version": 1,
    }


@pytest.mark.unit
def test_surrogate_element_survives_a_file_round_trip(tmp_path):
    """dump_spec then load_spec keeps config, parameters and fixed flags."""
    element = {
        "kind": "surrogate",
        "unit": "p.ro",
        "surrogate_type": "arima",
        "config": {"order": [1, 0, 0], "history": HISTORY},
        "parameters": {
            "intercept": {"value": 2.0},
            "ar.L1": {"value": 0.5, "fixed": False},
            "exog.feed": {"fixed": False},
        },
    }
    path = tmp_path / "spec.json"
    dump_spec(load_spec(_flowsheet(element)), path)
    written = next(
        el
        for el in json.loads(path.read_text())["elements"]
        if el["kind"] == "surrogate"
    )
    assert written["parameters"] == {
        "intercept": {"value": 2.0, "fixed": True},
        "ar.L1": {"value": 0.5, "fixed": False},
        "exog.feed": {"fixed": False},
    }
    reloaded = load_spec(path).of_kind("surrogate")[0]
    assert reloaded.free_parameters() == ["ar.L1", "exog.feed"]
    assert reloaded.config["history"] == HISTORY


# -- 0.1.1 -> 0.2.0 migration -------------------------------------------------


@pytest.mark.unit
def test_migrates_multilinear():
    """Coefficients become parameters; non-intercept keys become terms."""
    old = {
        "kind": "surrogate",
        "unit": "p.ro",
        "relation": "split_definition",
        "surrogate_type": "multilinear",
        "data": {
            "input_variables": {"feed": "m^3/hr", "temp": "K"},
            "output_variables": {"permeate": "m^3/hr"},
            "coefficients": {"feed": 0.4, "feed*temp": 0.01},
        },
        "provenance": {"rmse": 0.1},
    }
    new = migrate_surrogate_spec(old)
    assert "data" in old, "the input is not modified"
    assert new == {
        "kind": "surrogate",
        "unit": "p.ro",
        "relation": "split_definition",
        "surrogate_type": "multilinear",
        "provenance": {"rmse": 0.1},
        "config_version": 1,
        "config": {
            "input_variables": {"feed": "m^3/hr", "temp": "K"},
            "output_variables": {"permeate": "m^3/hr"},
            "terms": ["feed", "feed*temp"],
        },
        "parameters": {
            "intercept": {"value": 0.0, "fixed": True},
            "feed": {"value": 0.4, "fixed": True},
            "feed*temp": {"value": 0.01, "fixed": True},
        },
    }


@pytest.mark.unit
def test_migrates_constant_intensity():
    """The intensity becomes the one parameter; variables move to config."""
    new = migrate_surrogate_spec(
        {
            "surrogate_type": "constant_intensity",
            "data": {
                "coefficients": {"energy_intensity": 0.9},
                "input_variables": {"flow_out": "m^3/hr"},
                "output_variables": {"power_electrical": "kW"},
            },
        }
    )
    assert new["config"] == {
        "input_variables": {"flow_out": "m^3/hr"},
        "output_variables": {"power_electrical": "kW"},
    }
    assert new["parameters"] == {"energy_intensity": {"value": 0.9, "fixed": True}}


@pytest.mark.unit
def test_migrates_fitted_arima():
    """Order, deterministic term and history move to config; lags are named."""
    new = migrate_surrogate_spec(
        {
            "surrogate_type": "arima",
            "data": {
                "input_variables": {"feed": "kg/hr", "temp": "K"},
                "output_variables": {"gas": "m^3/hr"},
                "coefficients": {
                    "order": [2, 0, 1],
                    "intercept": 2.0,
                    "ar_coefs": [0.5, 0.1],
                    "ma_coefs": [0.2],
                    "exog_coefs": {"temp": 0.3, "feed": 0.8},
                },
                "history": HISTORY,
            },
        }
    )
    assert new["config"] == {
        "order": [2, 0, 1],
        "deterministic": "intercept",
        "input_variables": {"feed": "kg/hr", "temp": "K"},
        "output_variables": {"gas": "m^3/hr"},
        "history": HISTORY,
    }
    assert new["parameters"] == {
        "intercept": {"value": 2.0, "fixed": True},
        "ar.L1": {"value": 0.5, "fixed": True},
        "ar.L2": {"value": 0.1, "fixed": True},
        "ma.L1": {"value": 0.2, "fixed": True},
        "exog.feed": {"value": 0.8, "fixed": True},
        "exog.temp": {"value": 0.3, "fixed": True},
    }


@pytest.mark.unit
def test_migrates_regression_mode_arima_with_the_old_fixed_guesses():
    """Without history the old build fixed 1.0 guesses; the migration keeps them."""
    new = migrate_surrogate_spec(
        {
            "surrogate_type": "arima",
            "data": {
                "input_variables": {"feed": "kg/hr"},
                "output_variables": {"gas": "m^3/hr"},
                "coefficients": {"order": [1, 1, 1], "drift": 0.0},
            },
        }
    )
    assert new["config"]["deterministic"] == "drift"
    assert "history" not in new["config"]
    assert new["parameters"] == {
        "drift": {"value": 0.0, "fixed": True},
        "ar.L1": {"value": 1.0, "fixed": True},
        "ma.L1": {"value": 1.0, "fixed": True},
        "exog.feed": {"value": 1.0, "fixed": True},
    }


@pytest.mark.unit
def test_migrates_arima_without_a_deterministic_term():
    """No intercept or drift migrates to deterministic null."""
    new = migrate_surrogate_spec(
        {
            "surrogate_type": "arima",
            "data": {
                "input_variables": {},
                "output_variables": {"gas": "m^3/hr"},
                "coefficients": {"order": [0, 0, 0]},
                "history": HISTORY,
            },
        }
    )
    assert new["config"]["deterministic"] is None
    assert new["parameters"] == {}


@pytest.mark.unit
def test_migrates_a_one_input_positional_exog_list():
    """With one input a list cannot be mispaired, so it migrates."""
    new = migrate_surrogate_spec(
        {
            "surrogate_type": "arima",
            "data": {
                "input_variables": {"feed": "kg/hr"},
                "output_variables": {"gas": "m^3/hr"},
                "coefficients": {"order": [0, 0, 0], "exog_coefs": [0.8]},
                "history": HISTORY,
            },
        }
    )
    assert new["parameters"] == {"exog.feed": {"value": 0.8, "fixed": True}}


@pytest.mark.unit
def test_refuses_a_positional_exog_list_with_several_inputs():
    """A list with several inputs can't be paired reliably; the error shows the fix."""
    old = {
        "surrogate_type": "arima",
        "data": {
            "input_variables": {"feed": "kg/hr", "temp": "K"},
            "output_variables": {"gas": "m^3/hr"},
            "coefficients": {"order": [0, 0, 0], "exog_coefs": [0.8, 0.3]},
            "history": HISTORY,
        },
    }
    with pytest.raises(FlexConfigError, match=r"\{'feed': 0.8, 'temp': 0.3\}"):
        migrate_surrogate_spec(old)


@pytest.mark.unit
def test_refuses_an_arima_without_a_valid_order():
    """The order decides every parameter name, so it must be valid."""
    with pytest.raises(FlexConfigError, match=r"\[p, d, q\]"):
        migrate_surrogate_spec(
            {"surrogate_type": "arima", "data": {"coefficients": {"order": [1, 0]}}}
        )


@pytest.mark.unit
def test_migration_leaves_a_current_spec_unchanged():
    """A 0.2.0 spec passes through as an equal copy, so migrating is idempotent."""
    current = {
        "surrogate_type": "multilinear",
        "config": {"terms": []},
        "parameters": {"intercept": {"value": 1.0, "fixed": True}},
    }
    assert migrate_surrogate_spec(current) == current
    once = migrate_surrogate_spec(
        {"surrogate_type": "multilinear", "data": {"coefficients": {"intercept": 1.0}}}
    )
    assert migrate_surrogate_spec(once) == once


@pytest.mark.unit
def test_migration_keeps_a_stub_types_data_as_config():
    """Types with no pre-0.2.0 implementation carry their data over as config."""
    new = migrate_surrogate_spec({"surrogate_type": "quadratic", "data": {"a": 1}})
    assert new["config"] == {"a": 1}
    assert new["parameters"] == {}


@pytest.mark.unit
def test_load_spec_migrates_a_0_1_1_file(tmp_path):
    """A 0.1.1 file with several surrogates loads at 0.2.0, with a warning."""
    old = _flowsheet(
        {
            "kind": "surrogate",
            "unit": "p.ro",
            "relation": "split_definition",
            "surrogate_type": "multilinear",
            "data": {"coefficients": {"intercept": 1.0}},
        },
        {
            "kind": "surrogate",
            "unit": "p.ro",
            "surrogate_type": "arima",
            "data": {
                "input_variables": {"permeate": "m^3/hr"},
                "output_variables": {"power_electrical": "kW"},
                "coefficients": {
                    "order": [1, 0, 0],
                    "ar_coefs": [0.5],
                    "exog_coefs": {"permeate": 0.8},
                },
                "history": HISTORY,
            },
        },
        version="0.1.1",
    )
    path = tmp_path / "old.json"
    path.write_text(json.dumps(old))
    with pytest.warns(FlexMigrationWarning, match="2 surrogate element"):
        spec = load_spec(path)
    assert spec.schema_version == SCHEMA_VERSION
    power, split = sorted(spec.of_kind("surrogate"), key=lambda el: el.relation)
    assert power.relation == "power_electrical_relation"
    assert power.parameters["ar.L1"].value == 0.5
    assert power.parameters["exog.permeate"].value == 0.8
    assert power.free_parameters() == []
    assert split.parameters["intercept"].value == 1.0


@pytest.mark.unit
def test_load_spec_names_the_element_that_cannot_migrate():
    """A failed element migration says which unit and relation it was."""
    old = _flowsheet(
        {
            "kind": "surrogate",
            "unit": "p.ro",
            "surrogate_type": "arima",
            "data": {
                "input_variables": {"a": "kg/hr", "b": "K"},
                "output_variables": {"power_electrical": "kW"},
                "coefficients": {"order": [0, 0, 0], "exog_coefs": [1.0, 2.0]},
                "history": HISTORY,
            },
        },
        version="0.1.1",
    )
    with pytest.raises(FlexConfigError, match="p.ro.power_electrical_relation") as info:
        load_spec(old)
    assert info.value.field == "elements.2.data.coefficients.exog_coefs"


@pytest.mark.unit
def test_load_spec_rejects_the_old_data_field_at_0_2_0():
    """At 0.2.0, 'data' is an unknown field rather than silently migrated."""
    current = _flowsheet(
        {
            "kind": "surrogate",
            "unit": "p.ro",
            "surrogate_type": "multilinear",
            "data": {"coefficients": {"intercept": 1.0}},
        }
    )
    with pytest.raises(FlexConfigError, match="data"):
        load_spec(current)
