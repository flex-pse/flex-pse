"""Tests for the Pyomo ARIMA surrogate equation and regression helpers."""

from __future__ import annotations

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
        m.unit.register_io_variable(getattr(m.unit, name), role="input")

    return m, m.unit


def _arima_data(
    *,
    p: int = 1,
    d: int = 0,
    q: int = 0,
    n_exog: int = 0,
    deterministic: bool = True,
    y_history: list[float] | None = None,
    eps_history: list[float] | None = None,
) -> dict:
    """Return valid data for the clean ARIMA surrogate contract."""
    exog_names = ["feed_volume_kg", "TS_pct"][:n_exog]
    input_variables = {
        name: "kg" if name == "feed_volume_kg" else "dimensionless"
        for name in exog_names
    }
    coefficients: dict[str, object] = {
        "order": [p, d, q],
        "ar_coefs": [0.0] * p,
        "ma_coefs": [0.0] * q,
        "exog_coefs": [0.0] * n_exog,
    }
    if deterministic:
        coefficients["intercept" if d == 0 else "drift"] = 0.0
    history_prefix = max(p + d, q)
    y_prefix = [0.0] * (p + d) if y_history is None else list(y_history)
    eps_prefix = [0.0] * q if eps_history is None else list(eps_history)
    if y_prefix:
        y_prefix = [y_prefix[0]] * (history_prefix - len(y_prefix)) + y_prefix
    else:
        y_prefix = [0.0] * history_prefix
    eps_prefix = [0.0] * (history_prefix - len(eps_prefix)) + eps_prefix

    return {
        "input_variables": input_variables,
        "output_variables": {"biogas_m3_hour": "m^3/hr"},
        "coefficients": coefficients,
        "history": {
            "start_date": "2025-01-01T00:00:00",
            "time_step_seconds": 900.0,
            "y_values": y_prefix,
            "eps_values": eps_prefix,
        },
    }


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
    surrogate = ArimaSurrogate(_arima_data(p=1, d=1, q=1, n_exog=2))
    assert surrogate.surrogate_type is SurrogateType.ARIMA


@pytest.mark.unit
@pytest.mark.parametrize(
    "missing",
    ["input_variables", "output_variables", "coefficients"],
)
def test_validate_rejects_missing_required_key(missing):
    data = _arima_data()
    del data[missing]
    with pytest.raises(FlexConfigError, match=missing):
        ArimaSurrogate(data)


@pytest.mark.unit
def test_validate_accepts_missing_history_for_in_model_estimation():
    data = _arima_data(p=2, d=1, q=2)
    del data["history"]

    surrogate = ArimaSurrogate(data)

    assert "history" not in surrogate.data


@pytest.mark.unit
@pytest.mark.parametrize(
    "legacy_key",
    ["_residuals", "init_values", "training_y_values", "training_start_date"],
)
def test_validate_rejects_removed_training_keys(legacy_key):
    data = _arima_data()
    data[legacy_key] = []
    with pytest.raises(FlexConfigError, match=legacy_key):
        ArimaSurrogate(data)


@pytest.mark.unit
def test_validate_rejects_d_above_one():
    data = _arima_data()
    data["coefficients"]["order"] = [1, 2, 0]
    with pytest.raises(FlexConfigError, match="d=0 or d=1"):
        ArimaSurrogate(data)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "values", "message"),
    [
        ("ar_coefs", [], "ar_coefs"),
        ("ma_coefs", [0.1, 0.2], "ma_coefs"),
        ("exog_coefs", [0.1], "exog_coefs"),
    ],
)
def test_validate_rejects_coefficient_length_mismatch(field, values, message):
    data = _arima_data(p=1, q=1, n_exog=2)
    data["coefficients"][field] = values
    with pytest.raises(FlexConfigError, match=message):
        ArimaSurrogate(data)


@pytest.mark.unit
@pytest.mark.parametrize("field", ["ar_coefs", "ma_coefs", "exog_coefs"])
def test_validate_accepts_missing_coefficient_guesses_for_regression(field):
    data = _arima_data(p=2, d=1, q=2, n_exog=2)
    del data["history"]
    del data["coefficients"][field]

    ArimaSurrogate(data)


@pytest.mark.unit
@pytest.mark.parametrize("field", ["ar_coefs", "ma_coefs", "exog_coefs"])
def test_validate_rejects_missing_fitted_coefficients_for_forecast(field):
    data = _arima_data(p=1, q=1, n_exog=1)
    del data["coefficients"][field]

    with pytest.raises(FlexConfigError, match=field):
        ArimaSurrogate(data)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "values"),
    [("y_values", []), ("eps_values", [])],
)
def test_validate_rejects_history_prefix_length_mismatch(field, values):
    data = _arima_data(p=1, d=1, q=1)
    data["history"][field] = values
    with pytest.raises(FlexConfigError, match=field):
        ArimaSurrogate(data)


@pytest.mark.unit
def test_validate_rejects_history_shorter_than_max_p_d_q():
    """History must cover max(p+d, q), which is what build() slices.

    With q > p+d the old check only demanded p+d y_values, so a too-short
    history passed validation and failed later inside build() with a
    misleading "same number of modeled points" message.
    """
    data = _arima_data(p=0, d=0, q=2)
    data["history"]["y_values"] = []

    with pytest.raises(FlexConfigError, match=r"max\(p\+d, q\)=2"):
        ArimaSurrogate(data)


@pytest.mark.unit
def test_validate_accepts_history_equal_to_max_p_d_q():
    """A history of exactly max(p+d, q) values is sufficient."""
    data = _arima_data(p=0, d=0, q=2)
    assert len(data["history"]["y_values"]) == 2
    assert len(data["history"]["eps_values"]) == 2

    ArimaSurrogate(data)


@pytest.mark.unit
def test_validate_skips_history_length_rule_in_regression_mode():
    """Regression mode supplies no history at all, so the rule cannot fire."""
    data = _arima_data(p=1, d=1, q=3)
    del data["history"]

    ArimaSurrogate(data)


@pytest.mark.unit
def test_validate_rejects_intercept_for_d1():
    data = _arima_data(d=1)
    data["coefficients"]["intercept"] = 1.0
    with pytest.raises(FlexConfigError, match="intercept.*d=1"):
        ArimaSurrogate(data)


@pytest.mark.unit
def test_validate_rejects_drift_for_d0():
    data = _arima_data(d=0)
    data["coefficients"]["drift"] = 1.0
    with pytest.raises(FlexConfigError, match="drift.*d=0"):
        ArimaSurrogate(data)


@pytest.mark.unit
def test_validate_rejects_unparseable_units():
    data = _arima_data()
    data["output_variables"]["biogas_m3_hour"] = "not-a-unit"
    with pytest.raises(FlexConfigError, match="Could not parse"):
        ArimaSurrogate(data)


@pytest.mark.unit
def test_max_ar_coeff_rejects_out_of_range_argument():
    with pytest.raises(FlexConfigError, match="max_ar_coeff"):
        ArimaSurrogate(_arima_data(p=1), max_ar_coeff=1.5)


@pytest.mark.unit
def test_max_ar_coeff_rejects_fixed_coefficient_above_bound():
    data = _arima_data(p=1, y_history=[1.0])
    data["coefficients"]["ar_coefs"] = [0.9]
    with pytest.raises(FlexConfigError, match="max_ar_coeff"):
        ArimaSurrogate(data, max_ar_coeff=0.85)


@pytest.mark.unit
def test_max_ar_coeff_bounds_the_var_when_unfixed():
    _m, unit = _make_unit()
    data = _arima_data(p=1, y_history=[1.0])
    data["coefficients"]["ar_coefs"] = [0.5]
    surrogate = ArimaSurrogate(data, max_ar_coeff=0.85)
    block, _body = surrogate.build(unit, unit.biogas_m3_hour)
    block.ar_coefs.unfix()
    assert block.ar_coefs[1].lb == pytest.approx(-0.85)
    assert block.ar_coefs[1].ub == pytest.approx(0.85)


# -- block and equations ------------------------------------------------------


@pytest.mark.unit
def test_build_creates_fixed_coefficients_state_and_innovations():
    m, unit = _make_unit()
    data = _arima_data(p=2, d=1, q=2, n_exog=1)
    block, _body = ArimaSurrogate(data).build(unit, unit.biogas_m3_hour)

    assert isinstance(block.coefficients, CoefficientRegistry)
    assert len(block.coefficients) == 1 + 2 + 2 + 1
    assert all(var.fixed for _name, var in block.coefficients.items())
    assert all(block.initial_y_history[h].fixed for h in block.initial_y_history)
    assert all(block.initial_eps_history[h].fixed for h in block.initial_eps_history)
    assert all(block.eps[t].fixed and block.eps[t].value == 0 for t in block.eps)
    assert not hasattr(block, "eps_constraint")
    assert not hasattr(block, "innovation_square")


@pytest.mark.unit
def test_build_without_initial_state_creates_unfixed_latent_state():
    _m, unit = _make_unit()
    data = _arima_data(p=2, d=1, q=2)
    del data["history"]

    block, _body = ArimaSurrogate(data).build(unit, unit.biogas_m3_hour)

    assert len(block.initial_y_history) == 3
    assert len(block.initial_eps_history) == 2
    assert all(
        not block.initial_y_history[h].fixed and block.initial_y_history[h].value == 0.0
        for h in block.initial_y_history
    )
    assert all(
        not block.initial_eps_history[h].fixed
        and block.initial_eps_history[h].value == 0.0
        for h in block.initial_eps_history
    )
    assert all(var.fixed for _name, var in block.coefficients.items())
    assert all(block.eps[t].fixed for t in block.eps)


@pytest.mark.unit
def test_build_defaults_missing_regression_coefficient_guesses_to_one():
    _m, unit = _make_unit()
    data = _arima_data(p=2, d=1, q=2, n_exog=2)
    del data["history"]
    del data["coefficients"]["ar_coefs"]
    del data["coefficients"]["ma_coefs"]
    del data["coefficients"]["exog_coefs"]

    block, _body = ArimaSurrogate(data).build(unit, unit.biogas_m3_hour)

    assert [pyo.value(block.ar_coefs[i]) for i in block.ar_coefs] == [1.0, 1.0]
    assert [pyo.value(block.ma_coefs[i]) for i in block.ma_coefs] == [1.0, 1.0]
    assert [pyo.value(block.exog_coefs[i]) for i in block.exog_coefs] == [1.0, 1.0]


@pytest.mark.unit
def test_build_uses_supplied_regression_coefficient_guesses():
    _m, unit = _make_unit()
    data = _arima_data(p=2, q=2, n_exog=2)
    data["coefficients"].update(
        {
            "ar_coefs": [0.1, 0.2],
            "ma_coefs": [0.3, 0.4],
            "exog_coefs": [0.5, 0.6],
        }
    )
    del data["history"]

    block, _body = ArimaSurrogate(data).build(unit, unit.biogas_m3_hour)

    assert [pyo.value(block.ar_coefs[i]) for i in block.ar_coefs] == [0.1, 0.2]
    assert [pyo.value(block.ma_coefs[i]) for i in block.ma_coefs] == [0.3, 0.4]
    assert [pyo.value(block.exog_coefs[i]) for i in block.exog_coefs] == [0.5, 0.6]


@pytest.mark.unit
def test_build_does_not_attach_components_to_unit():
    _m, unit = _make_unit()
    before = set(unit.component_map())

    ArimaSurrogate(_arima_data()).build(unit, unit.biogas_m3_hour)

    assert set(unit.component_map()) == before


@pytest.mark.unit
def test_coefficients_and_eps_can_be_unfixed_and_refixed():
    _m, unit = _make_unit()
    block, _body = ArimaSurrogate(_arima_data(p=1, q=1)).build(
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
    data = _arima_data(
        p=2,
        q=2,
        n_exog=1,
        y_history=[10.0, 20.0],
        eps_history=[1.0, 2.0],
    )
    data["coefficients"].update(
        {
            "intercept": 0.5,
            "ar_coefs": [0.1, 0.2],
            "ma_coefs": [0.3, 0.4],
            "exog_coefs": [0.5],
        }
    )
    block, body = ArimaSurrogate(data).build(unit, unit.biogas_m3_hour)

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
    data = _arima_data(
        p=2,
        d=1,
        q=1,
        n_exog=1,
        y_history=[7.0, 9.0, 12.0],
        eps_history=[2.0],
    )
    data["coefficients"].update(
        {
            "drift": 0.5,
            "ar_coefs": [0.2, 0.1],
            "ma_coefs": [0.4],
            "exog_coefs": [0.5],
        }
    )
    block, body = ArimaSurrogate(data).build(unit, unit.biogas_m3_hour)

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
    block, body = ArimaSurrogate(_arima_data(p=0, q=0)).build(unit, unit.biogas_m3_hour)

    block.eps[2].set_value(1.25)
    first = pyo.value(body(2))
    block.eps[2].set_value(3.75)
    second = pyo.value(body(2))
    assert second - first == pytest.approx(2.5)


@pytest.mark.unit
@pytest.mark.parametrize("d", [0, 1])
def test_body_supports_omitted_deterministic_term(d):
    _m, unit = _make_unit()
    block, body = ArimaSurrogate(_arima_data(p=0, d=d, q=0, deterministic=False)).build(
        unit, unit.biogas_m3_hour
    )
    if d == 1:
        block.initial_y_history[0].set_value(4.0)
    block.eps[0].set_value(1.5)

    assert pyo.value(body(0)) == pytest.approx(1.5 if d == 0 else 5.5)


@pytest.mark.unit
def test_body_d1_p0_is_random_walk_with_constant_drift():
    _m, unit = _make_unit()
    data = _arima_data(p=0, d=1, q=0, y_history=[10.0])
    data["coefficients"]["drift"] = 0.75
    block, body = ArimaSurrogate(data).build(unit, unit.biogas_m3_hour)
    block.eps[0].set_value(-0.25)

    assert pyo.value(body(0)) == pytest.approx(10.5)


@pytest.mark.unit
def test_ma_history_used_before_local_horizon_then_local_eps_used():
    _m, unit = _make_unit()
    data = _arima_data(p=0, q=2, eps_history=[1.0, 2.0])
    data["coefficients"].update({"intercept": 0.0, "ma_coefs": [0.5, 0.25]})
    block, body = ArimaSurrogate(data).build(unit, unit.biogas_m3_hour)
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
    block = unit.swap_relation("biogas_m3_hour_relation", ArimaSurrogate(_arima_data()))

    assert block.fitted.is_indexed()
    assert len(block.fitted) == len(unit.biogas_m3_hour)
    assert not hasattr(block, "eps_constraint")


@pytest.mark.unit
def test_swap_relation_twice_uses_independent_latest_block():
    _m, unit = _make_unit()
    _add_swappable_relation(unit)
    first_data = _arima_data(p=0)
    first_data["coefficients"]["intercept"] = 1.0
    first = unit.swap_relation("biogas_m3_hour_relation", ArimaSurrogate(first_data))
    second_data = _arima_data(p=0)
    second_data["coefficients"]["intercept"] = 2.0
    second = unit.swap_relation("biogas_m3_hour_relation", ArimaSurrogate(second_data))

    assert not first.active
    assert second.active
    assert pyo.value(second.body(0)) == pytest.approx(2.0)


# -- regression objective ----------------------------------------------------


@pytest.mark.unit
def test_get_regression_objective_defaults_to_one_and_warns():
    _m, unit = _make_unit(n_points=3)
    block, _body = ArimaSurrogate(_arima_data()).build(unit, unit.biogas_m3_hour)
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
    block = unit.swap_relation("biogas_m3_hour_relation", ArimaSurrogate(_arima_data()))
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
    block, _body = ArimaSurrogate(_arima_data()).build(unit, unit.biogas_m3_hour)
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
    block, _body = ArimaSurrogate(_arima_data()).build(unit, unit.biogas_m3_hour)
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
    block, _body = ArimaSurrogate(_arima_data()).build(unit, unit.biogas_m3_hour)
    block.eps[0].set_value(2.0)
    block.eps[1].set_value(4.0)

    expression = block.get_regression_objective(
        innovation_scale=2.0 * pyunits.m**3 / pyunits.hr
    )
    assert pyo.value(expression) == pytest.approx(5.0)


@pytest.mark.unit
def test_get_regression_objective_accepts_scalar_pyomo_param():
    m, unit = _make_unit(n_points=2)
    block, _body = ArimaSurrogate(_arima_data()).build(unit, unit.biogas_m3_hour)
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
    block, _body = ArimaSurrogate(_arima_data()).build(unit, unit.biogas_m3_hour)
    with pytest.raises(FlexConfigError, match="innovation_scale"):
        block.get_regression_objective(innovation_scale=scale)


@pytest.mark.unit
def test_get_regression_objective_rejects_unknown_time_index():
    _m, unit = _make_unit()
    block, _body = ArimaSurrogate(_arima_data()).build(unit, unit.biogas_m3_hour)
    with pytest.raises(FlexConfigError, match="time_index"):
        block.get_regression_objective(innovation_scale=1.0, time_index=[99])


@pytest.mark.unit
def test_get_regression_objective_rejects_empty_time_index():
    _m, unit = _make_unit()
    block, _body = ArimaSurrogate(_arima_data()).build(unit, unit.biogas_m3_hour)
    with pytest.raises(FlexConfigError, match="at least one"):
        block.get_regression_objective(innovation_scale=1.0, time_index=[])


@pytest.mark.unit
def test_get_regression_objective_does_not_change_fixation():
    _m, unit = _make_unit()
    block, _body = ArimaSurrogate(_arima_data(p=1, q=1)).build(
        unit, unit.biogas_m3_hour
    )
    before = {var.name: var.fixed for var in block.component_data_objects(pyo.Var)}
    block.get_regression_objective(innovation_scale=1.0)
    after = {var.name: var.fixed for var in block.component_data_objects(pyo.Var)}
    assert after == before


# -- state extraction and reconstruction -------------------------------------


@pytest.mark.unit
def test_get_surrogate_spec_extracts_final_state():
    _m, unit = _make_unit(n_points=3)
    data = _arima_data(p=2, q=1, y_history=[-2.0, -1.0], eps_history=[-0.5])
    surrogate = ArimaSurrogate(data)
    block, _body = surrogate.build(unit, unit.biogas_m3_hour)
    for t, value in enumerate((10.0, 11.0, 12.0)):
        unit.biogas_m3_hour[t].set_value(value)
    for t, value in enumerate((0.1, 0.2, 0.3)):
        block.eps[t].set_value(value)

    state = surrogate.get_surrogate_spec(block, unit.biogas_m3_hour)

    assert set(state) == {
        "input_variables",
        "output_variables",
        "coefficients",
        "history",
    }
    assert state["history"]["y_values"] == pytest.approx([-2.0, -1.0, 10.0, 11.0, 12.0])
    assert state["history"]["eps_values"] == pytest.approx([0.0, -0.5, 0.1, 0.2, 0.3])
    assert state["history"]["time_step_seconds"] == pytest.approx(900.0)


@pytest.mark.unit
def test_get_surrogate_spec_omits_zero_order_coefficient_keys():
    _m, unit = _make_unit(n_points=2)
    surrogate = ArimaSurrogate(_arima_data(p=0, q=0, n_exog=0))
    block, _body = surrogate.build(unit, unit.biogas_m3_hour)
    for t in range(2):
        unit.biogas_m3_hour[t].set_value(float(t + 1.0))
        block.eps[t].set_value(0.0)

    state = surrogate.get_surrogate_spec(block, unit.biogas_m3_hour)

    assert state["coefficients"]["order"] == [0, 0, 0]
    assert "ar_coefs" not in state["coefficients"]
    assert "ma_coefs" not in state["coefficients"]
    assert "exog_coefs" not in state["coefficients"]
    assert state["history"]["y_values"] == pytest.approx([1.0, 2.0])
    assert state["history"]["eps_values"] == pytest.approx([0.0, 0.0])


@pytest.mark.unit
def test_build_forecasts_missing_history_before_future_start():
    _m, unit = _make_unit(n_points=2, start_date="2025-01-01T01:00:00")
    data = _arima_data(p=1, q=0, n_exog=0)
    data["coefficients"].update({"intercept": 1.0, "ar_coefs": [0.5]})
    data["history"] = {
        "start_date": "2025-01-01T00:00:00",
        "time_step_seconds": 900.0,
        "y_values": [10.0, 20.0, 21.0],
        "eps_values": [0.0, 0.0, 0.0],
    }

    block, _body = ArimaSurrogate(data).build(unit, unit.biogas_m3_hour)

    assert pyo.value(block.initial_y_history[0]) == pytest.approx(6.75)


@pytest.mark.unit
def test_get_surrogate_spec_preserves_solved_eps_values_for_next_horizon():
    _m, unit = _make_unit(n_points=3)
    data = _arima_data(p=1, q=2, n_exog=1)
    del data["history"]
    surrogate = ArimaSurrogate(data)
    block, _body = surrogate.build(unit, unit.biogas_m3_hour)
    for t, value in enumerate((0.5, -0.25, 0.75)):
        unit.biogas_m3_hour[t].set_value(float(t + 1.0))
        block.eps[t].set_value(value)

    state = surrogate.get_surrogate_spec(block, unit.biogas_m3_hour)

    assert state["history"]["eps_values"] == pytest.approx([0.0, 0.0, 0.5, -0.25, 0.75])


@pytest.mark.unit
def test_get_surrogate_spec_roundtrip_preserves_equation():
    _m, unit = _make_unit(n_points=3)
    data = _arima_data(p=1, d=1, q=1, y_history=[5.0, 7.0], eps_history=[0.25])
    data["coefficients"].update({"drift": 0.2, "ar_coefs": [0.4], "ma_coefs": [0.3]})
    surrogate = ArimaSurrogate(data)
    block, body = surrogate.build(unit, unit.biogas_m3_hour)
    block.eps[0].set_value(0.5)
    first = pyo.value(body(0))
    unit.biogas_m3_hour[0].set_value(first)

    state = surrogate.get_surrogate_spec(block, unit.biogas_m3_hour)
    rebuilt = ArimaSurrogate(state)
    block2, body2 = rebuilt.build(unit, unit.biogas_m3_hour)
    block2.eps[0].set_value(0.0)

    expected = 7.0 + 0.2 + 0.4 * (7.0 - 5.0) + 0.3 * 0.25
    assert pyo.value(body2(0)) == pytest.approx(expected)


@pytest.mark.unit
def test_surrogate_from_spec_accepts_clean_contract():
    spec = SurrogateSpec(
        surrogate_type=SurrogateType.ARIMA,
        data=_arima_data(p=1, q=1, n_exog=1),
    )
    surrogate = surrogate_from_spec(spec)
    assert isinstance(surrogate, ArimaSurrogate)


# -- solver proof -------------------------------------------------------------


@pytest.mark.component
@pytest.mark.needs_ipopt
def test_in_model_regression_estimates_omitted_initial_state():
    m, unit = _make_unit(n_points=3)
    _add_swappable_relation(unit)
    data = _arima_data(p=1)
    data["coefficients"].update({"intercept": 1.0, "ar_coefs": [0.5]})
    del data["history"]
    block = unit.swap_relation("biogas_m3_hour_relation", ArimaSurrogate(data))
    for t, value in enumerate((3.0, 2.5, 2.25)):
        unit.biogas_m3_hour[t].fix(value)

    block.eps.unfix()
    m.objective = pyo.Objective(
        expr=block.get_regression_objective(innovation_scale=1.0)
    )
    result = pyo.SolverFactory("ipopt").solve(m)
    pyo.assert_optimal_termination(result)

    assert pyo.value(block.initial_y_history[0]) == pytest.approx(4.0)
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
    data = _arima_data(p=1, y_history=[initial_y])
    block = unit.swap_relation("biogas_m3_hour_relation", ArimaSurrogate(data))
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
