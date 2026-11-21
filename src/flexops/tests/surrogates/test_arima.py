"""Tests for the Pyomo ARIMA surrogate equation and regression helpers."""

from __future__ import annotations

import copy
import json
import logging
import warnings

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits

from flexcore.config.schema import SurrogateSpec, SurrogateType
from flexcore.exceptions import FlexConfigError
from flexops.core.ops_block import OpsBlock
from flexops.core.registration import CoefficientRegistry
from flexops.core.time_block import TimeBlock
from flexops.properties.simple_aqueous import SimpleAqueousFlow
from flexops.surrogates import ArimaSurrogate, surrogate_from_spec
from flexops.surrogates.arima import arima_parameter_names


def _make_unit(n_points: int = 5, start_date: str = "2025-01-01T00:00:00"):
    """Return a bare OpsBlock with an output and two possible ARIMAX inputs."""
    m = pyo.ConcreteModel()
    start = pd.Timestamp(start_date)
    end = start + pd.Timedelta(minutes=15 * n_points)
    m.time_block = TimeBlock(
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        time_step=15 * pyunits.min,
    )
    m.props = SimpleAqueousFlow(has_pressure=False)
    m.unit = OpsBlock(property_package=m.props)
    m.unit.add_stream_ports()
    m.unit.add_component(
        "biogas_m3_hour",
        pyo.Var(
            m.time_block.time_index,
            initialize=0.0,
            units=pyunits.m**3 / pyunits.hr,
        ),
    )
    m.unit.register_io_variable(m.unit.biogas_m3_hour, role="output")

    for name, units in (
        ("feed_volume_kg", pyunits.kg),
        ("TS_pct", pyunits.dimensionless),
    ):
        m.unit.add_component(
            name,
            pyo.Var(m.time_block.time_index, initialize=0.0, units=units),
        )
        m.unit.register_io_variable(m.unit.find_component(name), role="input")

    return m, m.unit


def _arima_spec(
    *,
    p: int = 1,
    d: int = 0,
    q: int = 0,
    n_exog: int = 0,
    deterministic: bool = True,
    eta_history: list[float] | None = None,
    eps_history: list[float] | None = None,
    history: bool = True,
    values: dict[str, float] | None = None,
    max_ar_coeff: float | None = None,
) -> dict:
    """Return a valid ARIMA spec dict: ``config`` plus ``parameters``.

    Every parameter is fixed at ``0.0`` unless ``values`` overrides it.
    ``history=False`` omits the history (regression mode).
    """
    exog_names = ["feed_volume_kg", "TS_pct"][:n_exog]
    input_variables = {
        name: "kg" if name == "feed_volume_kg" else "dimensionless"
        for name in exog_names
    }
    deterministic_name = ("intercept" if d == 0 else "drift") if deterministic else None
    config: dict[str, object] = {
        "order": [p, d, q],
        "deterministic": deterministic_name,
        "input_variables": input_variables,
        "output_variables": {"biogas_m3_hour": "m^3/hr"},
    }
    if max_ar_coeff is not None:
        config["max_ar_coeff"] = max_ar_coeff
    if history:
        history_prefix = max(p + d, q)
        eta_prefix = [0.0] * (p + d) if eta_history is None else list(eta_history)
        eps_prefix = [0.0] * q if eps_history is None else list(eps_history)
        if eta_prefix:
            eta_prefix = [eta_prefix[0]] * (
                history_prefix - len(eta_prefix)
            ) + eta_prefix
        else:
            eta_prefix = [0.0] * history_prefix
        eps_prefix = [0.0] * (history_prefix - len(eps_prefix)) + eps_prefix
        config["history"] = {
            "start_date": "2025-01-01T00:00:00",
            "time_step_seconds": 900.0,
            "eta_values": eta_prefix,
            "eps_values": eps_prefix,
        }
    names = arima_parameter_names([p, d, q], deterministic_name, exog_names)
    parameters = {name: {"value": 0.0, "fixed": True} for name in names}
    for name, value in (values or {}).items():
        parameters[name]["value"] = value
    return {"config": config, "parameters": parameters}


def _free(spec: dict, *prefixes: str, keep_value: bool = False) -> dict:
    """Free every parameter starting with one of ``prefixes``.

    Without ``keep_value`` the value is dropped, so the build uses the
    surrogate's initial guess.
    """
    for name, state in spec["parameters"].items():
        if name.startswith(prefixes):
            state["fixed"] = False
            if not keep_value:
                state.pop("value", None)
    return spec


def _add_swappable_relation(unit) -> None:
    """Register an empty relation for swap_relation integration tests."""
    unit.add_component(
        "biogas_m3_hour_relation",
        pyo.Constraint(
            unit.biogas_m3_hour.index_set(),
            rule=lambda _b, _t: pyo.Constraint.Skip,
        ),
    )
    unit.register_relation(
        unit.biogas_m3_hour_relation,
        target=unit.biogas_m3_hour,
    )


# -- validation ---------------------------------------------------------------


@pytest.mark.unit
def test_validate_accepts_clean_contract():
    surrogate = ArimaSurrogate(_arima_spec(p=1, d=1, q=1, n_exog=2))
    assert surrogate.surrogate_type is SurrogateType.ARIMA


@pytest.mark.unit
def test_make_spec_builds_the_documented_contract():
    """make_spec names every parameter and validates the result."""
    spec = ArimaSurrogate.make_spec(
        order=(1, 0, 1),
        deterministic="intercept",
        input_variables={"feed_volume_kg": "kg"},
        output_variables={"biogas_m3_hour": "m^3/hr"},
        values={"intercept": 1.0, "ar.L1": 0.5, "ma.L1": 0.2, "exog.feed_volume_kg": 3},
        fixed={"ma.L1": False},
    )
    assert list(spec.parameters) == [
        "intercept",
        "ar.L1",
        "ma.L1",
        "exog.feed_volume_kg",
    ]
    assert spec.free_parameters() == ["ma.L1"]
    assert "history" not in spec.config


@pytest.mark.unit
@pytest.mark.parametrize("missing", ["input_variables", "output_variables", "order"])
def test_validate_rejects_missing_required_config_field(missing):
    spec = _arima_spec()
    del spec["config"][missing]
    with pytest.raises(FlexConfigError, match=f"config.{missing}"):
        ArimaSurrogate(spec)


@pytest.mark.unit
def test_validate_accepts_missing_history_for_in_model_estimation():
    surrogate = ArimaSurrogate(_arima_spec(p=2, d=1, q=2, history=False))

    assert "history" not in surrogate.spec.config
    assert surrogate.config.history is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "legacy_key",
    ["_residuals", "init_values", "training_y_values", "training_start_date"],
)
def test_validate_rejects_unknown_config_keys(legacy_key):
    spec = _arima_spec()
    spec["config"][legacy_key] = []
    with pytest.raises(FlexConfigError, match=legacy_key):
        ArimaSurrogate(spec)


@pytest.mark.unit
def test_validate_rejects_the_pre_0_2_0_data_layout():
    """An old ``data`` dict is refused with a pointer to the migration."""
    old = {"data": {"coefficients": {"order": [1, 0, 0]}}}
    with pytest.raises(FlexConfigError, match="migrate_surrogate_spec"):
        ArimaSurrogate(old)


@pytest.mark.unit
def test_validate_rejects_d_above_one():
    spec = _arima_spec()
    spec["config"]["order"] = [1, 2, 0]
    with pytest.raises(FlexConfigError, match="d=0 or d=1"):
        ArimaSurrogate(spec)


@pytest.mark.unit
@pytest.mark.parametrize(
    "order", [[1, 0], [1, 0, 0, 1], [-1, 0, 0], [1.0, 0, 0], ["1", 0, 0]]
)
def test_validate_rejects_a_malformed_order(order):
    spec = _arima_spec()
    spec["config"]["order"] = order
    with pytest.raises(FlexConfigError, match="config.order"):
        ArimaSurrogate(spec)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda params: params.pop("ar.L1"), r"missing=\['ar.L1'\]"),
        (
            lambda params: params.update({"ma.L2": {"value": 0.2}}),
            r"unknown=\['ma.L2'\]",
        ),
        (lambda params: params.pop("exog.TS_pct"), r"missing=\['exog.TS_pct'\]"),
    ],
    ids=["missing-ar-lag", "extra-ma-lag", "missing-exog"],
)
def test_validate_rejects_parameters_that_do_not_match_the_order(change, message):
    """The order and inputs decide the parameter names exactly."""
    spec = _arima_spec(p=1, q=1, n_exog=2)
    change(spec["parameters"])
    with pytest.raises(FlexConfigError, match=message) as info:
        ArimaSurrogate(spec)
    assert info.value.field == "parameters"


def _exog_registry(block) -> dict[str, float]:
    """Return the block's exogenous coefficients keyed by registry name."""
    return {
        name: block.coefficients.value(name)
        for name in block.coefficients
        if name.startswith("exog.")
    }


@pytest.mark.unit
def test_exog_coefs_pair_by_name_whatever_the_input_key_order():
    """Reordering input_variables and parameters (as a JSON tool may) keeps pairing."""
    spec = _arima_spec(
        n_exog=2, values={"exog.feed_volume_kg": 0.5, "exog.TS_pct": 0.6}
    )
    reordered = json.loads(json.dumps(spec))
    reordered["config"]["input_variables"] = dict(
        reversed(spec["config"]["input_variables"].items())
    )
    reordered["parameters"] = dict(reversed(spec["parameters"].items()))
    expected = {"exog.feed_volume_kg": 0.5, "exog.TS_pct": 0.6}

    for variant in (spec, reordered):
        _m, unit = _make_unit()
        block, _body = ArimaSurrogate(variant).build(unit, unit.biogas_m3_hour)
        assert _exog_registry(block) == pytest.approx(expected)
    # The reordered inputs build the exogenous Var in their own order.
    assert block._exog_names == ["TS_pct", "feed_volume_kg"]
    assert pyo.value(block.exog_coefs[1]) == pytest.approx(0.6)


@pytest.mark.unit
def test_emitted_spec_rebuilds_with_reordered_input_variables():
    """A spec from get_surrogate_spec survives a key reorder of its inputs."""
    _m, unit = _make_unit()
    _add_swappable_relation(unit)
    spec = _arima_spec(
        p=1, n_exog=2, values={"exog.feed_volume_kg": 0.5, "exog.TS_pct": 0.6}
    )
    unit.swap_relation("biogas_m3_hour_relation", ArimaSurrogate(spec))
    emitted = unit.get_surrogate_spec("biogas_m3_hour_relation")
    assert isinstance(emitted, SurrogateSpec)
    assert emitted.parameters["exog.feed_volume_kg"].value == pytest.approx(0.5)
    assert emitted.parameters["exog.TS_pct"].value == pytest.approx(0.6)

    reordered = json.loads(emitted.model_dump_json())
    reordered["config"]["input_variables"] = dict(
        reversed(reordered["config"]["input_variables"].items())
    )
    _m2, unit2 = _make_unit()
    block, _body = ArimaSurrogate(reordered).build(unit2, unit2.biogas_m3_hour)
    assert _exog_registry(block) == pytest.approx(
        {"exog.feed_volume_kg": 0.5, "exog.TS_pct": 0.6}
    )


@pytest.mark.unit
def test_validate_rejects_a_list_value_for_a_scalar_coefficient():
    """Each exogenous coefficient is its own scalar; a list can't stand for several."""
    spec = _arima_spec(n_exog=2)
    spec["parameters"]["exog.feed_volume_kg"]["value"] = [0.5, 0.6]
    with pytest.raises(FlexConfigError, match=r"must have shape \(\)") as info:
        ArimaSurrogate(spec)
    assert info.value.field == "parameters.exog.feed_volume_kg"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "message"),
    [("x", "Invalid arima surrogate spec"), (True, "Invalid arima surrogate spec")],
)
def test_validate_rejects_non_numeric_parameter_values(value, message):
    spec = _arima_spec(n_exog=2)
    spec["parameters"]["exog.TS_pct"]["value"] = value
    with pytest.raises(FlexConfigError, match=message):
        ArimaSurrogate(spec)


@pytest.mark.unit
@pytest.mark.parametrize("prefix", ["ar.", "ma.", "exog."])
def test_validate_accepts_free_coefficients_without_values_for_regression(prefix):
    spec = _free(_arima_spec(p=2, d=1, q=2, n_exog=2, history=False), prefix)

    ArimaSurrogate(spec)


@pytest.mark.unit
@pytest.mark.parametrize("name", ["ar.L1", "ma.L1", "exog.feed_volume_kg"])
def test_validate_rejects_fixed_coefficients_without_values(name):
    """A fixed coefficient needs a value, with or without history."""
    spec = _arima_spec(p=1, q=1, n_exog=1)
    del spec["parameters"][name]["value"]

    with pytest.raises(FlexConfigError, match="fixed parameter needs a value"):
        ArimaSurrogate(spec)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "values"),
    [("eta_values", []), ("eps_values", [])],
)
def test_validate_rejects_history_prefix_length_mismatch(field, values):
    spec = _arima_spec(p=1, d=1, q=1)
    spec["config"]["history"][field] = values
    with pytest.raises(FlexConfigError, match=field):
        ArimaSurrogate(spec)


@pytest.mark.unit
def test_validate_rejects_history_shorter_than_max_p_d_q():
    """History must cover max(p+d, q), which is what build() slices.

    With q > p+d a check on p+d alone would let a too-short history pass
    validation and fail later inside build() with a misleading message.
    """
    spec = _arima_spec(p=0, d=0, q=2)
    spec["config"]["history"]["eta_values"] = []
    spec["config"]["history"]["eps_values"] = []

    with pytest.raises(FlexConfigError, match=r"max\(p\+d, q\)=2"):
        ArimaSurrogate(spec)


@pytest.mark.unit
def test_validate_accepts_history_equal_to_max_p_d_q():
    """A history of exactly max(p+d, q) values is sufficient."""
    spec = _arima_spec(p=0, d=0, q=2)
    assert len(spec["config"]["history"]["eta_values"]) == 2
    assert len(spec["config"]["history"]["eps_values"]) == 2

    ArimaSurrogate(spec)


@pytest.mark.unit
def test_validate_skips_history_length_rule_in_regression_mode():
    """Regression mode supplies no history at all, so the rule cannot fire."""
    ArimaSurrogate(_arima_spec(p=1, d=1, q=3, history=False))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value"),
    [("start_date", "not-a-date"), ("time_step_seconds", 0.0)],
)
def test_validate_rejects_a_malformed_history(field, value):
    spec = _arima_spec()
    spec["config"]["history"][field] = value
    with pytest.raises(FlexConfigError, match=f"config.history.{field}"):
        ArimaSurrogate(spec)


@pytest.mark.unit
def test_validate_rejects_intercept_for_d1():
    spec = _arima_spec(d=1)
    spec["config"]["deterministic"] = "intercept"
    spec["parameters"] = {
        ("intercept" if k == "drift" else k): v for k, v in spec["parameters"].items()
    }
    with pytest.raises(FlexConfigError, match="intercept.*d=1"):
        ArimaSurrogate(spec)


@pytest.mark.unit
def test_validate_rejects_drift_for_d0():
    spec = _arima_spec(d=0)
    spec["config"]["deterministic"] = "drift"
    spec["parameters"] = {
        ("drift" if k == "intercept" else k): v for k, v in spec["parameters"].items()
    }
    with pytest.raises(FlexConfigError, match="drift.*d=0"):
        ArimaSurrogate(spec)


@pytest.mark.unit
def test_validate_rejects_unparseable_units():
    spec = _arima_spec()
    spec["config"]["output_variables"]["biogas_m3_hour"] = "not-a-unit"
    with pytest.raises(FlexConfigError, match="unparsable units 'not-a-unit'"):
        ArimaSurrogate(spec)


@pytest.mark.unit
@pytest.mark.parametrize("bound", [1.5, 0.0, -0.2])
def test_max_ar_coeff_rejects_out_of_range_value(bound):
    with pytest.raises(FlexConfigError, match="max_ar_coeff"):
        ArimaSurrogate(_arima_spec(p=1, max_ar_coeff=bound))


@pytest.mark.unit
def test_max_ar_coeff_rejects_fixed_coefficient_above_bound():
    spec = _arima_spec(p=1, eta_history=[1.0], values={"ar.L1": 0.9}, max_ar_coeff=0.85)
    with pytest.raises(FlexConfigError, match="max_ar_coeff") as info:
        ArimaSurrogate(spec)
    assert info.value.field == "parameters.ar.L1"


@pytest.mark.unit
def test_max_ar_coeff_allows_a_free_coefficient_above_bound():
    """A free AR coefficient is bounded during solving, not refused."""
    spec = _free(
        _arima_spec(p=1, values={"ar.L1": 0.9}, max_ar_coeff=0.85),
        "ar.",
        keep_value=True,
    )
    ArimaSurrogate(spec)


@pytest.mark.unit
def test_max_ar_coeff_bounds_the_var_when_unfixed():
    _m, unit = _make_unit()
    spec = _arima_spec(p=1, eta_history=[1.0], values={"ar.L1": 0.5}, max_ar_coeff=0.85)
    surrogate = ArimaSurrogate(spec)
    block, _body = surrogate.build(unit, unit.biogas_m3_hour)
    block.ar_coefs.unfix()
    assert block.ar_coefs[1].lb == pytest.approx(-0.85)
    assert block.ar_coefs[1].ub == pytest.approx(0.85)


@pytest.mark.unit
def test_max_ar_coeff_caps_the_free_ar_guess():
    """A free AR lag without a value starts at min(1.0, max_ar_coeff)."""
    _m, unit = _make_unit()
    spec = _free(_arima_spec(p=1, history=False, max_ar_coeff=0.6), "ar.")
    block, _body = ArimaSurrogate(spec).build(unit, unit.biogas_m3_hour)
    assert pyo.value(block.ar_coefs[1]) == pytest.approx(0.6)
    assert not block.ar_coefs[1].fixed


# -- block and equations ------------------------------------------------------


@pytest.mark.unit
def test_build_creates_fixed_coefficients_state_and_innovations():
    m, unit = _make_unit()
    spec = _arima_spec(p=2, d=1, q=2, n_exog=1)
    block, _body = ArimaSurrogate(spec).build(unit, unit.biogas_m3_hour)

    assert isinstance(block.coefficients, CoefficientRegistry)
    assert len(block.coefficients) == 1 + 2 + 2 + 1
    assert all(var.fixed for _name, var in block.coefficients.items())
    assert all(block.initial_eta_history[h].fixed for h in block.initial_eta_history)
    assert all(block.initial_eps_history[h].fixed for h in block.initial_eps_history)
    assert all(block.eps[t].fixed and block.eps[t].value == 0 for t in block.eps)
    assert block.find_component("eps_constraint") is None
    assert block.find_component("innovation_square") is None


@pytest.mark.unit
def test_build_fixes_each_coefficient_as_the_spec_says():
    """Free parameters build free, fixed ones fixed, coefficient by coefficient."""
    _m, unit = _make_unit()
    spec = _arima_spec(p=2, q=1, n_exog=1, values={"ar.L2": 0.3})
    spec["parameters"]["ar.L2"]["fixed"] = False
    block, _body = ArimaSurrogate(spec).build(unit, unit.biogas_m3_hour)

    fixed = {name: var.fixed for name, var in block.coefficients.items()}
    assert fixed == {
        "intercept": True,
        "ar.L1": True,
        "ar.L2": False,
        "ma.L1": True,
        "exog.feed_volume_kg": True,
    }
    assert pyo.value(block.ar_coefs[2]) == pytest.approx(0.3)


@pytest.mark.unit
def test_build_without_initial_state_creates_unfixed_latent_state():
    _m, unit = _make_unit()
    spec = _arima_spec(p=2, d=1, q=2, history=False)

    block, _body = ArimaSurrogate(spec).build(unit, unit.biogas_m3_hour)

    assert len(block.initial_eta_history) == 3
    assert len(block.initial_eps_history) == 2
    assert all(
        not block.initial_eta_history[h].fixed
        and block.initial_eta_history[h].value == 0.0
        for h in block.initial_eta_history
    )
    assert all(
        not block.initial_eps_history[h].fixed
        and block.initial_eps_history[h].value == 0.0
        for h in block.initial_eps_history
    )
    assert all(var.fixed for _name, var in block.coefficients.items())
    assert all(block.eps[t].fixed for t in block.eps)


@pytest.mark.unit
def test_build_starts_free_regression_coefficients_from_the_guesses():
    """Free coefficients without values: AR, MA and exog at 1.0, drift at 0.0."""
    _m, unit = _make_unit()
    spec = _free(_arima_spec(p=2, d=1, q=2, n_exog=2, history=False), "")

    block, _body = ArimaSurrogate(spec).build(unit, unit.biogas_m3_hour)

    assert [pyo.value(block.ar_coefs[i]) for i in block.ar_coefs] == [1.0, 1.0]
    assert [pyo.value(block.ma_coefs[i]) for i in block.ma_coefs] == [1.0, 1.0]
    assert [pyo.value(block.exog_coefs[i]) for i in block.exog_coefs] == [1.0, 1.0]
    assert pyo.value(block.drift) == 0.0
    assert not any(var.fixed for _name, var in block.coefficients.items())


@pytest.mark.unit
def test_build_uses_supplied_regression_coefficient_guesses():
    _m, unit = _make_unit()
    spec = _arima_spec(
        p=2,
        q=2,
        n_exog=2,
        history=False,
        values={
            "ar.L1": 0.1,
            "ar.L2": 0.2,
            "ma.L1": 0.3,
            "ma.L2": 0.4,
            "exog.feed_volume_kg": 0.5,
            "exog.TS_pct": 0.6,
        },
    )
    _free(spec, "ar.", "ma.", "exog.", keep_value=True)

    block, _body = ArimaSurrogate(spec).build(unit, unit.biogas_m3_hour)

    assert [pyo.value(block.ar_coefs[i]) for i in block.ar_coefs] == [0.1, 0.2]
    assert [pyo.value(block.ma_coefs[i]) for i in block.ma_coefs] == [0.3, 0.4]
    assert [pyo.value(block.exog_coefs[i]) for i in block.exog_coefs] == [0.5, 0.6]
    assert not block.ar_coefs[1].fixed
    assert block.intercept.fixed


@pytest.mark.unit
def test_build_does_not_attach_components_to_unit():
    _m, unit = _make_unit()
    before = set(unit.component_map())

    ArimaSurrogate(_arima_spec()).build(unit, unit.biogas_m3_hour)

    assert set(unit.component_map()) == before


@pytest.mark.unit
def test_coefficients_and_eps_can_be_unfixed_and_refixed():
    _m, unit = _make_unit()
    block, _body = ArimaSurrogate(_arima_spec(p=1, q=1)).build(
        unit, unit.biogas_m3_hour
    )

    block.coefficients.unfix()
    block.eps.unfix()
    assert all(not var.fixed for _name, var in block.coefficients.items())
    assert all(not block.eps[t].fixed for t in block.eps)

    block.coefficients.fix()
    block.eps.fix(0.0)
    assert all(var.fixed for _name, var in block.coefficients.items())
    assert all(block.eps[t].fixed and block.eps[t].value == 0 for t in block.eps)


@pytest.mark.unit
def test_body_d0_matches_hand_calculation_and_history_indexing():
    _m, unit = _make_unit()
    spec = _arima_spec(
        p=2,
        q=2,
        n_exog=1,
        eta_history=[10.0, 20.0],
        eps_history=[1.0, 2.0],
        values={
            "intercept": 0.5,
            "ar.L1": 0.1,
            "ar.L2": 0.2,
            "ma.L1": 0.3,
            "ma.L2": 0.4,
            "exog.feed_volume_kg": 0.5,
        },
    )
    block, body = ArimaSurrogate(spec).build(unit, unit.biogas_m3_hour)

    unit.feed_volume_kg[0].set_value(4.0)
    block.eps[0].set_value(3.0)
    assert pyo.value(body(0)) == pytest.approx(10.5)

    unit.biogas_m3_hour[0].set_value(12.0)
    unit.feed_volume_kg[1].set_value(2.0)
    block.eps[1].set_value(-1.0)
    assert pyo.value(body(1)) == pytest.approx(7.2)


@pytest.mark.unit
def test_body_d1_matches_hand_calculation_and_history_indexing():
    _m, unit = _make_unit()
    spec = _arima_spec(
        p=2,
        d=1,
        q=1,
        n_exog=1,
        eta_history=[7.0, 9.0, 12.0],
        eps_history=[2.0],
        values={
            "drift": 0.5,
            "ar.L1": 0.2,
            "ar.L2": 0.1,
            "ma.L1": 0.4,
            "exog.feed_volume_kg": 0.5,
        },
    )
    block, body = ArimaSurrogate(spec).build(unit, unit.biogas_m3_hour)

    unit.feed_volume_kg[0].set_value(4.0)
    block.eps[0].set_value(1.0)
    assert pyo.value(body(0)) == pytest.approx(17.1)

    unit.biogas_m3_hour[0].set_value(17.1)
    unit.feed_volume_kg[1].set_value(0.0)
    block.eps[1].set_value(0.0)
    assert pyo.value(body(1)) == pytest.approx(16.92)


@pytest.mark.unit
def test_body_includes_current_innovation_exactly_once():
    _m, unit = _make_unit()
    block, body = ArimaSurrogate(_arima_spec(p=0, q=0)).build(unit, unit.biogas_m3_hour)

    block.eps[2].set_value(1.25)
    first = pyo.value(body(2))
    block.eps[2].set_value(3.75)
    second = pyo.value(body(2))
    assert second - first == pytest.approx(2.5)


@pytest.mark.unit
@pytest.mark.parametrize("d", [0, 1])
def test_body_supports_omitted_deterministic_term(d):
    _m, unit = _make_unit()
    surrogate = ArimaSurrogate(_arima_spec(p=0, d=d, q=0, deterministic=False))
    block, body = surrogate.build(unit, unit.biogas_m3_hour)
    assert block.find_component("intercept") is None
    assert block.find_component("drift") is None
    if d == 1:
        block.initial_eta_history[0].set_value(4.0)
    block.eps[0].set_value(1.5)

    assert pyo.value(body(0)) == pytest.approx(1.5 if d == 0 else 5.5)


@pytest.mark.unit
def test_body_d1_p0_is_random_walk_with_constant_drift():
    _m, unit = _make_unit()
    spec = _arima_spec(p=0, d=1, q=0, eta_history=[10.0], values={"drift": 0.75})
    block, body = ArimaSurrogate(spec).build(unit, unit.biogas_m3_hour)
    block.eps[0].set_value(-0.25)

    assert pyo.value(body(0)) == pytest.approx(10.5)


@pytest.mark.unit
def test_ma_history_used_before_local_horizon_then_local_eps_used():
    _m, unit = _make_unit()
    spec = _arima_spec(
        p=0,
        q=2,
        eps_history=[1.0, 2.0],
        values={"intercept": 0.0, "ma.L1": 0.5, "ma.L2": 0.25},
    )
    block, body = ArimaSurrogate(spec).build(unit, unit.biogas_m3_hour)
    block.eps[0].set_value(4.0)
    block.eps[1].set_value(8.0)
    block.eps[2].set_value(0.0)

    assert pyo.value(body(0)) == pytest.approx(5.25)
    assert pyo.value(body(1)) == pytest.approx(10.5)
    assert pyo.value(body(2)) == pytest.approx(5.0)


@pytest.mark.unit
def test_swap_relation_adds_one_equation_without_residual_constraint():
    _m, unit = _make_unit()
    _add_swappable_relation(unit)
    block = unit.swap_relation("biogas_m3_hour_relation", ArimaSurrogate(_arima_spec()))

    assert block.fitted.is_indexed()
    assert len(block.fitted) == len(unit.biogas_m3_hour)
    assert block.find_component("eps_constraint") is None


@pytest.mark.unit
def test_swap_relation_twice_uses_independent_latest_block():
    _m, unit = _make_unit()
    _add_swappable_relation(unit)
    first = unit.swap_relation(
        "biogas_m3_hour_relation",
        ArimaSurrogate(_arima_spec(p=0, values={"intercept": 1.0})),
    )
    second = unit.swap_relation(
        "biogas_m3_hour_relation",
        ArimaSurrogate(_arima_spec(p=0, values={"intercept": 2.0})),
    )

    assert not first.active
    assert second.active
    assert pyo.value(second.body(0)) == pytest.approx(2.0)


@pytest.mark.unit
def test_switch_back_returns_spec_from_reactivated_surrogate():
    """Switching to an older block must also restore its Surrogate instance.

    Otherwise get_surrogate_spec pairs the newest surrogate's config with the
    older block's coefficients.
    """
    _m, unit = _make_unit()
    _add_swappable_relation(unit)
    first_spec = _arima_spec(
        p=1, n_exog=1, values={"ar.L1": 0.3, "exog.feed_volume_kg": 0.7}
    )
    first = unit.swap_relation("biogas_m3_hour_relation", ArimaSurrogate(first_spec))
    unit.swap_relation(
        "biogas_m3_hour_relation", ArimaSurrogate(_arima_spec(p=1, n_exog=2))
    )

    unit.switch_surrogate_block(first.local_name)
    spec = unit.get_surrogate_spec("biogas_m3_hour_relation")

    assert spec.config["input_variables"] == first_spec["config"]["input_variables"]
    assert list(spec.parameters) == ["intercept", "ar.L1", "exog.feed_volume_kg"]
    assert spec.parameters["exog.feed_volume_kg"].value == pytest.approx(0.7)
    assert spec.parameters["ar.L1"].value == pytest.approx(0.3)


# -- regression objective ----------------------------------------------------


@pytest.mark.unit
def test_get_regression_objective_defaults_to_one_and_warns():
    _m, unit = _make_unit(n_points=3)
    block, _body = ArimaSurrogate(_arima_spec()).build(unit, unit.biogas_m3_hour)
    for t, value in enumerate((1.0, 2.0, 3.0)):
        block.eps[t].set_value(value)

    with pytest.warns(UserWarning, match="defaulting innovation_scale to 1.0"):
        expression = block.get_regression_objective()

    assert not isinstance(expression, pyo.Objective)
    assert pyo.value(expression) == pytest.approx(14.0)


@pytest.mark.unit
def test_unit_get_surrogate_objective_returns_active_arima_expression():
    _m, unit = _make_unit(n_points=3)
    _add_swappable_relation(unit)
    block = unit.swap_relation("biogas_m3_hour_relation", ArimaSurrogate(_arima_spec()))
    for t, value in enumerate((1.0, 2.0, 3.0)):
        block.eps[t].set_value(value)

    expression = unit.get_surrogate_objective()

    assert not isinstance(expression, pyo.Objective)
    assert pyo.value(expression) == pytest.approx(14.0)


@pytest.mark.unit
def test_unit_get_surrogate_objective_requires_active_surrogate():
    _m, unit = _make_unit(n_points=3)
    _add_swappable_relation(unit)

    with pytest.raises(FlexConfigError, match="has no active surrogate blocks"):
        unit.get_surrogate_objective()


@pytest.mark.unit
def test_get_regression_objective_explicit_scale_suppresses_warning():
    _m, unit = _make_unit(n_points=2)
    block, _body = ArimaSurrogate(_arima_spec()).build(unit, unit.biogas_m3_hour)
    block.eps[0].set_value(2.0)
    block.eps[1].set_value(4.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        expression = block.get_regression_objective(innovation_scale=2.0)

    assert caught == []
    assert pyo.value(expression) == pytest.approx(5.0)


@pytest.mark.unit
def test_get_regression_objective_supports_subset_and_indexed_scales():
    _m, unit = _make_unit(n_points=3)
    block, _body = ArimaSurrogate(_arima_spec()).build(unit, unit.biogas_m3_hour)
    block.eps[0].set_value(2.0)
    block.eps[1].set_value(99.0)
    block.eps[2].set_value(8.0)

    expression = block.get_regression_objective(
        innovation_scale={0: 1.0, 2: 4.0},
        time_index=[0, 2],
    )
    assert pyo.value(expression) == pytest.approx(8.0)


@pytest.mark.unit
def test_get_regression_objective_accepts_compatible_units():
    _m, unit = _make_unit(n_points=2)
    block, _body = ArimaSurrogate(_arima_spec()).build(unit, unit.biogas_m3_hour)
    block.eps[0].set_value(2.0)
    block.eps[1].set_value(4.0)

    expression = block.get_regression_objective(
        innovation_scale=2.0 * pyunits.m**3 / pyunits.hr
    )
    assert pyo.value(expression) == pytest.approx(5.0)


@pytest.mark.unit
def test_get_regression_objective_accepts_scalar_pyomo_param():
    m, unit = _make_unit(n_points=2)
    block, _body = ArimaSurrogate(_arima_spec()).build(unit, unit.biogas_m3_hour)
    block.eps[0].set_value(2.0)
    block.eps[1].set_value(4.0)
    m.innovation_scale = pyo.Param(initialize=2.0, units=pyunits.m**3 / pyunits.hr)

    expression = block.get_regression_objective(innovation_scale=m.innovation_scale)
    assert pyo.value(expression) == pytest.approx(5.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "scale",
    [0.0, -1.0, 2.0 * pyunits.Pa],
)
def test_get_regression_objective_rejects_invalid_scale(scale):
    _m, unit = _make_unit()
    block, _body = ArimaSurrogate(_arima_spec()).build(unit, unit.biogas_m3_hour)
    with pytest.raises(FlexConfigError, match="innovation_scale"):
        block.get_regression_objective(innovation_scale=scale)


@pytest.mark.unit
def test_get_regression_objective_rejects_unknown_time_index():
    _m, unit = _make_unit()
    block, _body = ArimaSurrogate(_arima_spec()).build(unit, unit.biogas_m3_hour)
    with pytest.raises(FlexConfigError, match="time_index"):
        block.get_regression_objective(innovation_scale=1.0, time_index=[99])


@pytest.mark.unit
def test_get_regression_objective_rejects_empty_time_index():
    _m, unit = _make_unit()
    block, _body = ArimaSurrogate(_arima_spec()).build(unit, unit.biogas_m3_hour)
    with pytest.raises(FlexConfigError, match="at least one"):
        block.get_regression_objective(innovation_scale=1.0, time_index=[])


@pytest.mark.unit
def test_get_regression_objective_does_not_change_fixation():
    _m, unit = _make_unit()
    block, _body = ArimaSurrogate(_arima_spec(p=1, q=1)).build(
        unit, unit.biogas_m3_hour
    )
    before = {var.name: var.fixed for var in block.component_data_objects(pyo.Var)}
    block.get_regression_objective(innovation_scale=1.0)
    after = {var.name: var.fixed for var in block.component_data_objects(pyo.Var)}
    assert after == before


# -- state extraction and reconstruction -------------------------------------


@pytest.mark.unit
def test_solved_history_extracts_final_state():
    _m, unit = _make_unit(n_points=3)
    spec = _arima_spec(p=2, q=1, eta_history=[-2.0, -1.0], eps_history=[-0.5])
    surrogate = ArimaSurrogate(spec)
    block, _body = surrogate.build(unit, unit.biogas_m3_hour)
    for t, value in enumerate((10.0, 11.0, 12.0)):
        unit.biogas_m3_hour[t].set_value(value)
    for t, value in enumerate((0.1, 0.2, 0.3)):
        block.eps[t].set_value(value)

    history = surrogate.solved_history(block, unit.biogas_m3_hour)

    assert set(history) == {
        "start_date",
        "time_step_seconds",
        "eta_values",
        "eps_values",
    }
    assert history["eta_values"] == pytest.approx([-2.0, -1.0, 10.0, 11.0, 12.0])
    assert history["eps_values"] == pytest.approx([0.0, -0.5, 0.1, 0.2, 0.3])
    assert history["time_step_seconds"] == pytest.approx(900.0)
    # solved_history only reports; the surrogate's own spec is unchanged.
    assert surrogate.spec.config["history"] == spec["config"]["history"]


@pytest.mark.unit
def test_update_state_records_history_and_read_spec_returns_it():
    """update_state rewrites config.history; read_spec then reports it."""
    _m, unit = _make_unit(n_points=3)
    spec = _arima_spec(p=2, q=1, eta_history=[-2.0, -1.0], eps_history=[-0.5])
    surrogate = ArimaSurrogate(spec)
    block, _body = surrogate.build(unit, unit.biogas_m3_hour)
    for t, value in enumerate((10.0, 11.0, 12.0)):
        unit.biogas_m3_hour[t].set_value(value)
    expected = surrogate.solved_history(block, unit.biogas_m3_hour)

    surrogate.update_state(block, unit.biogas_m3_hour)
    read = surrogate.read_spec(block, unit.biogas_m3_hour)

    assert read.config["history"] == expected
    assert surrogate.config.history.eta_values == pytest.approx(expected["eta_values"])
    assert read.parameters == ArimaSurrogate(spec).spec.parameters


@pytest.mark.unit
def test_unit_update_surrogate_state_then_get_surrogate_spec():
    """The unit-level calls are separate: only the update changes the spec."""
    _m, unit = _make_unit(n_points=2)
    _add_swappable_relation(unit)
    unit.swap_relation("biogas_m3_hour_relation", ArimaSurrogate(_arima_spec(p=1)))
    for t in range(2):
        unit.biogas_m3_hour[t].set_value(float(t + 1.0))

    before = unit.get_surrogate_spec()
    assert unit.get_surrogate_spec() == before
    unit.update_surrogate_state()
    after = unit.get_surrogate_spec()

    assert after.config["history"]["eta_values"] == pytest.approx([0.0, 1.0, 2.0])
    assert after.config["history"] != before.config["history"]
    assert after.parameters == before.parameters


@pytest.mark.unit
def test_solved_history_refuses_an_unsolved_block():
    _m, unit = _make_unit(n_points=2)
    surrogate = ArimaSurrogate(_arima_spec(p=1))
    block, _body = surrogate.build(unit, unit.biogas_m3_hour)
    unit.biogas_m3_hour[0].set_value(None)

    with pytest.raises(FlexConfigError, match="after a successful solve"):
        surrogate.update_state(block, unit.biogas_m3_hour)


@pytest.mark.unit
def test_read_spec_omits_zero_order_coefficients():
    """With p = q = 0 and no inputs only the intercept is a parameter."""
    _m, unit = _make_unit(n_points=2)
    surrogate = ArimaSurrogate(_arima_spec(p=0, q=0, n_exog=0))
    block, _body = surrogate.build(unit, unit.biogas_m3_hour)
    for t in range(2):
        unit.biogas_m3_hour[t].set_value(float(t + 1.0))
        block.eps[t].set_value(0.0)

    surrogate.update_state(block, unit.biogas_m3_hour)
    spec = surrogate.read_spec(block, unit.biogas_m3_hour)

    assert spec.config["order"] == [0, 0, 0]
    assert list(spec.parameters) == ["intercept"]
    assert block.find_component("ar_coefs") is None
    assert block.find_component("ma_coefs") is None
    assert block.find_component("exog_coefs") is None
    assert spec.config["history"]["eta_values"] == pytest.approx([1.0, 2.0])
    assert spec.config["history"]["eps_values"] == pytest.approx([0.0, 0.0])


def _build_capturing_warnings(surrogate, unit):
    """Build ``surrogate`` and return ``(block, body, warning_messages)``.

    ``flexcore.logger`` loggers do not propagate, so ``caplog`` cannot see
    them; a handler is attached to the module logger directly instead.
    """
    records: list[logging.LogRecord] = []
    capture = logging.Handler()
    capture.emit = records.append
    logger = logging.getLogger("flexops.surrogates.arima")
    logger.addHandler(capture)
    try:
        block, body = surrogate.build(unit, unit.biogas_m3_hour)
    finally:
        logger.removeHandler(capture)
    messages = [r.getMessage() for r in records if r.levelno >= logging.WARNING]
    return block, body, messages


@pytest.mark.unit
def test_build_forecasts_missing_history_before_future_start():
    _m, unit = _make_unit(n_points=2, start_date="2025-01-01T01:00:00")
    spec = _arima_spec(p=1, q=0, n_exog=0, values={"intercept": 1.0, "ar.L1": 0.5})
    spec["config"]["history"] = {
        "start_date": "2025-01-01T00:00:00",
        "time_step_seconds": 900.0,
        "eta_values": [10.0, 20.0, 21.0],
        "eps_values": [0.0, 0.0, 0.0],
    }

    block, _body, messages = _build_capturing_warnings(ArimaSurrogate(spec), unit)

    assert pyo.value(block.initial_eta_history[0]) == pytest.approx(6.75)
    assert len(messages) == 1
    assert "2 step(s) before the TimeBlock start" in messages[0]


@pytest.mark.unit
def test_build_does_not_warn_when_history_ends_at_future_start():
    _m, unit = _make_unit(n_points=2, start_date="2025-01-01T00:30:00")
    spec = _arima_spec(p=1, q=0, n_exog=0)
    spec["config"]["history"] = {
        "start_date": "2025-01-01T00:00:00",
        "time_step_seconds": 900.0,
        "eta_values": [10.0, 20.0, 21.0],
        "eps_values": [0.0, 0.0, 0.0],
    }

    block, _body, messages = _build_capturing_warnings(ArimaSurrogate(spec), unit)

    assert pyo.value(block.initial_eta_history[0]) == pytest.approx(21.0)
    assert messages == []


@pytest.mark.unit
def test_build_rejects_a_start_before_the_history():
    _m, unit = _make_unit(n_points=2, start_date="2024-12-31T23:00:00")
    with pytest.raises(FlexConfigError, match="starts before the stored history"):
        ArimaSurrogate(_arima_spec(p=1)).build(unit, unit.biogas_m3_hour)


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "order",
        "values",
        "history",
        "start_date",
        "expected_y",
        "expected_eps",
        "expected_body",
        "missing_steps",
    ),
    [
        pytest.param(
            # ARIMA(2,1,1): the first forecast step uses the stored eps 0.3;
            # the second uses the zero forecast innovation.
            #   step 1: 0.1 + 0.5*(5-4) + 0.2*(4-2) + 0.4*0.3 = 1.12 -> 6.12
            #   step 2: 0.1 + 0.5*(6.12-5) + 0.2*(5-4) + 0.4*0 = 0.86 -> 6.98
            [2, 1, 1],
            {"drift": 0.1, "ar.L1": 0.5, "ar.L2": 0.2, "ma.L1": 0.4},
            {"eta_values": [1.0, 2.0, 4.0, 5.0], "eps_values": [0.0, 0.0, 0.5, 0.3]},
            "2025-01-01T00:45:00",
            [5.0, 6.12, 6.98],
            [0.0],
            6.98 + 0.1 + 0.5 * (6.98 - 6.12) + 0.2 * (6.12 - 5.0),
            2,
            id="arima_2_1_1_gap",
        ),
        pytest.param(
            # Pure MA(2): 1.0 + 0.5*0.8 + 0.25*0.4 = 1.5.
            [0, 0, 2],
            {"intercept": 1.0, "ma.L1": 0.5, "ma.L2": 0.25},
            {"eta_values": [0.0, 0.0, 3.0], "eps_values": [0.2, 0.4, 0.8]},
            "2025-01-01T00:30:00",
            [],
            [0.8, 0.0],
            1.0 + 0.5 * 0.0 + 0.25 * 0.8,
            1,
            id="ma_2_gap",
        ),
        pytest.param(
            # Next horizon starts right after the last stored point.
            [2, 1, 1],
            {"drift": 0.1, "ar.L1": 0.5, "ar.L2": 0.2, "ma.L1": 0.4},
            {"eta_values": [1.0, 2.0, 4.0, 5.0], "eps_values": [0.0, 0.0, 0.5, 0.3]},
            "2025-01-01T00:15:00",
            [2.0, 4.0, 5.0],
            [0.3],
            5.0 + 0.1 + 0.5 * (5.0 - 4.0) + 0.2 * (4.0 - 2.0) + 0.4 * 0.3,
            0,
            id="arima_2_1_1_contiguous",
        ),
    ],
)
def test_build_initial_state_after_history(
    order,
    values,
    history,
    start_date,
    expected_y,
    expected_eps,
    expected_body,
    missing_steps,
):
    _m, unit = _make_unit(n_points=2, start_date=start_date)
    p, d, q = order
    spec = _arima_spec(p=p, d=d, q=q, n_exog=0, values=values)
    spec["config"]["history"] = {
        "start_date": "2025-01-01T00:00:00",
        "time_step_seconds": 900.0,
        **history,
    }

    block, body, messages = _build_capturing_warnings(ArimaSurrogate(spec), unit)

    assert [pyo.value(v) for v in block.initial_eta_history.values()] == (
        pytest.approx(expected_y)
    )
    assert [pyo.value(v) for v in block.initial_eps_history.values()] == (
        pytest.approx(expected_eps)
    )
    assert pyo.value(body(0)) == pytest.approx(expected_body)
    if missing_steps:
        assert len(messages) == 1
        assert f"{missing_steps} step(s) before" in messages[0]
    else:
        assert messages == []


@pytest.mark.unit
def test_validate_rejects_history_series_with_different_lengths():
    spec = _arima_spec(p=1, q=1)
    spec["config"]["history"]["eta_values"] = [0.0, 1.0, 2.0]
    spec["config"]["history"]["eps_values"] = [0.0, 0.5]

    with pytest.raises(FlexConfigError, match="same number of points"):
        ArimaSurrogate(spec)


@pytest.mark.unit
def test_solved_history_preserves_solved_eps_values_for_next_horizon():
    _m, unit = _make_unit(n_points=3)
    surrogate = ArimaSurrogate(_arima_spec(p=1, q=2, n_exog=1, history=False))
    block, _body = surrogate.build(unit, unit.biogas_m3_hour)
    for t, value in enumerate((0.5, -0.25, 0.75)):
        unit.biogas_m3_hour[t].set_value(float(t + 1.0))
        block.eps[t].set_value(value)

    history = surrogate.solved_history(block, unit.biogas_m3_hour)

    assert history["eps_values"] == pytest.approx([0.0, 0.0, 0.5, -0.25, 0.75])


@pytest.mark.unit
def test_update_state_turns_a_regression_spec_into_a_forecast_spec():
    """After a regression solve, update_state gives the spec its history."""
    _m, unit = _make_unit(n_points=3)
    surrogate = ArimaSurrogate(_arima_spec(p=1, history=False))
    block, _body = surrogate.build(unit, unit.biogas_m3_hour)
    for t in range(3):
        unit.biogas_m3_hour[t].set_value(float(t + 1.0))

    surrogate.update_state(block, unit.biogas_m3_hour)

    assert surrogate.config.history is not None
    ArimaSurrogate(surrogate.read_spec(block, unit.biogas_m3_hour))


@pytest.mark.unit
def test_recorded_state_roundtrip_preserves_equation():
    _m, unit = _make_unit(n_points=3)
    spec = _arima_spec(
        p=1,
        d=1,
        q=1,
        eta_history=[5.0, 7.0],
        eps_history=[0.25],
        values={"drift": 0.2, "ar.L1": 0.4, "ma.L1": 0.3},
    )
    surrogate = ArimaSurrogate(spec)
    block, body = surrogate.build(unit, unit.biogas_m3_hour)
    block.eps[0].set_value(0.5)
    first = pyo.value(body(0))
    unit.biogas_m3_hour[0].set_value(first)

    surrogate.update_state(block, unit.biogas_m3_hour)
    rebuilt = ArimaSurrogate(
        json.loads(surrogate.read_spec(block, unit.biogas_m3_hour).model_dump_json())
    )
    block2, body2 = rebuilt.build(unit, unit.biogas_m3_hour)
    block2.eps[0].set_value(0.0)

    expected = 7.0 + 0.2 + 0.4 * (7.0 - 5.0) + 0.3 * 0.25
    assert pyo.value(body2(0)) == pytest.approx(expected)


@pytest.mark.unit
def test_read_spec_reports_live_values_and_fixed_flags():
    """read_spec reads each parameter from the block, not the build spec."""
    _m, unit = _make_unit()
    surrogate = ArimaSurrogate(_arima_spec(p=1, n_exog=1))
    block, _body = surrogate.build(unit, unit.biogas_m3_hour)
    block.coefficients.set_value("ar.L1", 0.42)
    block.coefficients.unfix("exog.feed_volume_kg")

    spec = surrogate.read_spec(block, unit.biogas_m3_hour)

    assert spec.parameters["ar.L1"].value == pytest.approx(0.42)
    assert spec.parameters["ar.L1"].fixed is True
    assert spec.free_parameters() == ["exog.feed_volume_kg"]
    assert surrogate.spec.parameters["ar.L1"].value == 0.0


@pytest.mark.unit
def test_surrogate_from_spec_accepts_clean_contract():
    spec = SurrogateSpec(
        surrogate_type=SurrogateType.ARIMA,
        **copy.deepcopy(_arima_spec(p=1, q=1, n_exog=1)),
    )
    surrogate = surrogate_from_spec(spec)
    assert isinstance(surrogate, ArimaSurrogate)


# -- solver proof -------------------------------------------------------------


@pytest.mark.component
@pytest.mark.needs_ipopt
def test_in_model_regression_estimates_omitted_initial_state():
    m, unit = _make_unit(n_points=3)
    _add_swappable_relation(unit)
    spec = _arima_spec(p=1, history=False, values={"intercept": 1.0, "ar.L1": 0.5})
    block = unit.swap_relation("biogas_m3_hour_relation", ArimaSurrogate(spec))
    for t, value in enumerate((3.0, 2.5, 2.25)):
        unit.biogas_m3_hour[t].fix(value)

    block.eps.unfix()
    m.objective = pyo.Objective(
        expr=block.get_regression_objective(innovation_scale=1.0)
    )
    result = pyo.SolverFactory("ipopt").solve(m)
    pyo.assert_optimal_termination(result)

    assert pyo.value(block.initial_eta_history[0]) == pytest.approx(4.0)
    assert all(
        pyo.value(block.eps[t]) == pytest.approx(0.0, abs=1e-7) for t in block.time
    )


@pytest.mark.component
@pytest.mark.needs_ipopt
def test_direct_regression_recovers_conditional_least_squares_ar1():
    rng = np.random.default_rng(17)
    n = 30
    initial_y = 2.0
    true_intercept = 0.7
    true_phi = 0.35
    noise = rng.normal(0.0, 0.08, size=n)
    observed = np.empty(n)
    previous = initial_y
    for t in range(n):
        observed[t] = true_intercept + true_phi * previous + noise[t]
        previous = observed[t]

    regressors = np.column_stack(
        [np.ones(n), np.concatenate(([initial_y], observed[:-1]))]
    )
    expected, *_ = np.linalg.lstsq(regressors, observed, rcond=None)

    m, unit = _make_unit(n_points=n)
    _add_swappable_relation(unit)
    spec = _arima_spec(p=1, eta_history=[initial_y])
    block = unit.swap_relation("biogas_m3_hour_relation", ArimaSurrogate(spec))
    for t, value in enumerate(observed):
        unit.biogas_m3_hour[t].fix(float(value))

    block.coefficients.unfix()
    block.eps.unfix()
    m.objective = pyo.Objective(
        expr=block.get_regression_objective(innovation_scale=1.0)
    )
    result = pyo.SolverFactory("ipopt").solve(m)
    pyo.assert_optimal_termination(result)

    assert pyo.value(block.coefficients["intercept"]) == pytest.approx(
        expected[0], rel=1e-6
    )
    assert pyo.value(block.coefficients["ar.L1"]) == pytest.approx(
        expected[1], rel=1e-6
    )
