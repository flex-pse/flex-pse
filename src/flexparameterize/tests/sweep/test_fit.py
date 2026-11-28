"""Tests for fit_sweep on synthetic sweep data (no solver)."""

import numpy as np
import pandas as pd
import pytest

from flexcore.config.schema import SurrogateType
from flexcore.exceptions import FlexDataError
from flexparameterize.config import config_hash
from flexparameterize.sweep import fit_sweep
from flexparameterize.tests.sweep.helpers import toy_config, toy_frame

pytest.importorskip("sklearn")

UNIT = "facility.plant"
RELATION = "power_electrical_relation"


def _spec(config, data):
    """The single spec fit_sweep returns for the toy config."""
    return fit_sweep(config, data)[UNIT][RELATION]


@pytest.mark.unit
def test_fit_returns_apply_to_model_shape():
    """The result is keyed by unit, then relation, and holds a multilinear spec."""
    specs = fit_sweep(toy_config(), toy_frame())

    assert list(specs) == [UNIT]
    assert list(specs[UNIT]) == [RELATION]
    assert specs[UNIT][RELATION].surrogate_type is SurrogateType.MULTILINEAR


@pytest.mark.unit
def test_fit_recovers_coefficients_in_output_units():
    """Coefficients are keyed by flex-pse names and converted from W to kW."""
    spec = _spec(toy_config(), toy_frame())

    expected = {
        "intercept": 0.1,
        "flow_out": 50.0,
        "outlet_state.pressure": 1e-6,
        "flow_out*outlet_state.pressure": 2e-3,
    }
    for key, value in expected.items():
        assert spec.parameters[key].value == pytest.approx(value, rel=1e-8)
    assert spec.config["input_variables"] == {
        "flow_out": "m^3/s",
        "outlet_state.pressure": "Pa",
    }
    assert spec.config["output_variables"] == {"power_electrical": "kW"}


@pytest.mark.unit
def test_fit_records_provenance():
    """Provenance holds the hashes, versions, fitted ranges, metrics and counts."""
    config, data = toy_config(), toy_frame()

    provenance = _spec(config, data).provenance

    assert provenance["config_hash"] == config_hash(config)
    assert len(provenance["data_hash"]) == 64
    assert {"flex-pse", "pyomo", "idaes-pse"} <= set(provenance["versions"])
    train = data[data.index % 5 != 0]
    assert provenance["input_ranges"]["flow_out"] == [train.a.min(), train.a.max()]
    assert provenance["metrics"]["holdout_r2"] == pytest.approx(1.0)
    assert (provenance["n_train"], provenance["n_holdout"]) == (28, 8)
    assert provenance["n_failed"] == 0


@pytest.mark.unit
def test_data_hash_tracks_the_data():
    """Equal data hashes equal; a changed value changes the hash."""
    config, data = toy_config(), toy_frame()
    changed = data.copy()
    changed.loc[3, "power_electrical"] += 1e-9

    first = _spec(config, data).provenance["data_hash"]
    assert _spec(config, data.copy()).provenance["data_hash"] == first
    assert _spec(config, changed).provenance["data_hash"] != first


@pytest.mark.unit
def test_failed_rows_are_excluded_from_the_fit():
    """A failed row with a garbage output does not move the coefficients."""
    data = toy_frame()
    data.loc[1, ["power_electrical", "solve_successful"]] = [1e9, False]

    spec = _spec(toy_config(), data)

    assert spec.parameters["flow_out"].value == pytest.approx(50.0, rel=1e-8)
    assert spec.provenance["n_failed"] == 1


@pytest.mark.unit
def test_too_many_failures_raise():
    """More failed rows than max_failed_fraction allows raises before fitting."""
    data = toy_frame()
    data.loc[:4, "solve_successful"] = False

    with pytest.raises(FlexDataError, match="max_failed_fraction"):
        fit_sweep(toy_config(), data)


@pytest.mark.unit
def test_holdout_miss_raises():
    """Noise beyond the acceptance limits raises with the holdout numbers."""
    data = toy_frame()
    noise = np.random.default_rng(3).normal(0.0, 50.0, size=len(data))
    data["power_electrical"] += noise

    with pytest.raises(FlexDataError, match="holdout"):
        fit_sweep(toy_config(), data)


@pytest.mark.unit
def test_holdout_is_by_row_position_and_ignores_failures():
    """Holdout rows are positions with pos % k == r, chosen before failures drop."""
    data = toy_frame()
    data.loc[1, "solve_successful"] = False

    provenance = _spec(toy_config(), data).provenance

    assert (provenance["n_train"], provenance["n_holdout"]) == (27, 8)


@pytest.mark.unit
def test_concatenated_frames_fit():
    """Rows from several runs concatenate, with their own index columns, and fit."""
    data = toy_frame()
    extra = data.iloc[[7]].assign(index=0)

    spec = _spec(toy_config(), pd.concat([data, extra]))

    assert spec.provenance["n_train"] + spec.provenance["n_holdout"] == 37


@pytest.mark.unit
def test_missing_column_raises():
    """Data without a configured column raises naming it."""
    with pytest.raises(FlexDataError, match="'b'"):
        fit_sweep(toy_config(), toy_frame().drop(columns="b"))


@pytest.mark.unit
def test_relative_rmse_limit_raises():
    """A holdout relative RMSE above max_rel_rmse raises even when R^2 is unchecked."""
    data = toy_frame()
    data["power_electrical"] += np.random.default_rng(5).normal(0.0, 50.0, len(data))
    loose = {"max_rel_error": 10.0, "max_rel_rmse": 1e-6}

    with pytest.raises(FlexDataError, match="relative RMSE"):
        fit_sweep(toy_config(acceptance=loose), data)


@pytest.mark.unit
def test_default_acceptance_passes_a_close_fit():
    """Noise of about 1% passes the default 10% limits and records the RMSE."""
    data = toy_frame()
    data["power_electrical"] *= 1 + np.random.default_rng(5).normal(
        0.0, 0.01, len(data)
    )
    raw = toy_config().model_dump(mode="json", exclude={"acceptance"})
    config = toy_config().model_validate(raw)

    spec = _spec(config, data)

    assert spec.provenance["metrics"]["holdout_rel_rmse"] < 0.1


@pytest.mark.unit
def test_output_name_is_the_spec_output_variable():
    """The spec's output is output_name, whatever key labels the output."""
    raw = toy_config().model_dump(mode="json")["outputs"]["power_electrical"]
    config = toy_config(outputs={"z_power": raw | {"output_name": "power_thermal"}})
    data = toy_frame().rename(columns={"power_electrical": "z_power"})

    spec = _spec(config, data)

    assert spec.config["output_variables"] == {"power_thermal": "kW"}
    assert spec.parameters["flow_out"].value == pytest.approx(50.0, rel=1e-8)


@pytest.mark.unit
def test_output_name_defaults_to_electrical_power():
    """Without output_name, the spec's output is power_electrical, not the key."""
    raw = toy_config().model_dump(mode="json")["outputs"]["power_electrical"]
    raw.pop("output_name", None)
    config = toy_config(outputs={"z_power": raw})
    data = toy_frame().rename(columns={"power_electrical": "z_power"})

    assert _spec(config, data).config["output_variables"] == {"power_electrical": "kW"}
