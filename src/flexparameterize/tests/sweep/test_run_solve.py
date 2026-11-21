"""Tests for run_sweep and solve_point against a toy Pyomo model."""

import pandas as pd
import pyomo.environ as pyo
import pytest

from flexcore.exceptions import FlexConfigError
from flexparameterize.apply import apply_to_model
from flexparameterize.sweep import fit_sweep, run_sweep, solve_point
from flexparameterize.tags import TagMap
from flexparameterize.tests.helpers import build_plant, evaluate_data
from flexparameterize.tests.sweep.helpers import (
    build_toy_model,
    initialize_toy,
    optimize_toy,
    toy_config,
    toy_output,
)

pytest.importorskip("parameter_sweep")
pytest.importorskip("sklearn")

TOY = (build_toy_model, initialize_toy, optimize_toy)
GRID = {"type": "grid", "n": 3, "seed": 0}


def _input(path: str, units: str) -> dict:
    """The toy config's inputs with ``a`` replaced by another path and units."""
    raw = toy_config().model_dump(mode="json")["inputs"]
    return raw | {"a": raw["a"] | {"path": path, "units": units}}


@pytest.mark.component
@pytest.mark.needs_ipopt
def test_sweep_columns_and_values():
    """Every point solves and records the toy output for its inputs."""
    data = run_sweep(toy_config(), *TOY)

    assert list(data.columns) == [
        "index",
        "a",
        "b",
        "power_electrical",
        "solve_successful",
    ]
    assert len(data) == 20
    assert data.solve_successful.all()
    expected = toy_output(data.a, data.b)
    assert data.power_electrical.to_numpy() == pytest.approx(expected.to_numpy())


@pytest.mark.component
@pytest.mark.needs_ipopt
def test_same_seed_gives_identical_data():
    """Two sweeps with the same config return identical frames."""
    pd.testing.assert_frame_equal(
        run_sweep(toy_config(), *TOY), run_sweep(toy_config(), *TOY)
    )


@pytest.mark.component
@pytest.mark.needs_ipopt
def test_lhs_rows_are_sorted_by_the_first_input():
    """Latin hypercube points come back ordered by the first input."""
    data = run_sweep(toy_config(), *TOY)

    assert data.a.is_monotonic_increasing


@pytest.mark.component
@pytest.mark.needs_ipopt
def test_grid_gives_n_to_the_d_points():
    """A grid of n points per input sweeps every combination."""
    data = run_sweep(toy_config(sampler=GRID), *TOY)

    assert len(data) == 9
    assert sorted(set(data.a)) == pytest.approx([1e-3, 3e-3, 5e-3])


@pytest.mark.component
@pytest.mark.needs_ipopt
def test_solve_point_reproduces_a_sweep_row():
    """A point solved alone gives the row it got inside the sweep."""
    row = run_sweep(toy_config(), *TOY).iloc[7]

    point = solve_point(toy_config(), *TOY, {"a": row.a, "b": row.b})

    assert len(point) == 1
    assert (point.a[0], point.b[0]) == (row.a, row.b)
    assert point.power_electrical[0] == pytest.approx(row.power_electrical, rel=1e-10)


@pytest.mark.component
@pytest.mark.needs_ipopt
def test_bounds_sweep_stays_inside_the_box():
    """A bounds box replaces the config ranges for one sweep."""
    data = run_sweep(toy_config(), *TOY, bounds={"a": (2e-3, 3e-3)})

    assert data.a.between(2e-3, 3e-3).all()
    assert data.b.between(1e5, 5e5).all()


@pytest.mark.component
@pytest.mark.needs_ipopt
def test_context_is_applied_after_build():
    """Context values are set on the built model before every solve."""
    data = run_sweep(toy_config(sampler=GRID, context={"offset": 7.0}), *TOY)

    expected = toy_output(data.a, data.b) + 7.0
    assert data.power_electrical.to_numpy() == pytest.approx(expected.to_numpy())


@pytest.mark.component
@pytest.mark.needs_ipopt
def test_sweep_then_fit_recovers_coefficients():
    """Sweep rows fit back to the toy model's coefficients."""
    spec = fit_sweep(toy_config(), run_sweep(toy_config(), *TOY))["facility.plant"][
        "power_electrical_relation"
    ]

    assert spec.data["coefficients"]["flow_out"] == pytest.approx(50.0, rel=1e-6)
    assert spec.data["coefficients"]["flow_out*outlet_state.pressure"] == pytest.approx(
        2e-3, rel=1e-6
    )


@pytest.mark.component
@pytest.mark.needs_ipopt
def test_sweep_and_point_rows_fit_together():
    """Rows from a sweep and a single point concatenate and fit."""
    sweep = run_sweep(toy_config(), *TOY)
    point = solve_point(toy_config(), *TOY, {"a": 4.2e-3, "b": 1.7e5})

    specs = fit_sweep(toy_config(), pd.concat([sweep, point]))

    provenance = specs["facility.plant"]["power_electrical_relation"].provenance
    assert provenance["n_train"] + provenance["n_holdout"] == 21


@pytest.mark.component
@pytest.mark.needs_ipopt
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"inputs": _input("missing", "m^3/s")}, "missing"),
        ({"inputs": _input("z", "W")}, "fixed"),
        ({"inputs": _input("a", "m^3/hr")}, "units"),
        ({"context": {"nowhere": 1.0}}, "nowhere"),
    ],
    ids=["unknown_path", "unfixed_var", "units_mismatch", "unknown_context_path"],
)
def test_model_mismatches_raise(overrides, message):
    """Paths and units that do not match the built model raise before solving."""
    with pytest.raises(FlexConfigError, match=message):
        run_sweep(toy_config(**overrides), *TOY)


@pytest.mark.component
@pytest.mark.needs_ipopt
def test_fitted_spec_matches_the_model_on_a_flex_pse_unit():
    """Sweep, fit and apply: the unit's relation reproduces the toy model."""
    data = run_sweep(toy_config(), *TOY)
    specs = fit_sweep(toy_config(), data)
    m, unit = build_plant(has_pressure=True)

    apply_to_model(m, evaluate_data(unit), TagMap({}), surrogates=specs)

    for row in data.iloc[::5].itertuples():
        unit.flow_out[0].set_value(row.a * 3600.0)
        unit.outlet_state.pressure[0].set_value(row.b)
        unit.power_electrical[0].set_value(0.0)
        body = pyo.value(unit.surrogate_power_electrical.fitted[0].body)
        assert -body == pytest.approx(row.power_electrical / 1000.0, rel=1e-6)
