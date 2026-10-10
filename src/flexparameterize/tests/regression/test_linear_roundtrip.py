"""One-fit/two-consumers checks (R10) for a *fitted* LinearRegressor spec.

Minimal deltas on the existing hand-built-spec tests
(``test_apply_swaps_energy_relation_to_multilinear``,
``test_apply_swaps_a_named_relation`` in ``test_apply.py``): the swap
mechanics themselves are already covered there and are not re-asserted here
beyond "it swapped and the numbers match."
"""

import pandas as pd
import pyomo.environ as pyo
import pytest

pytest.importorskip("sklearn")

from flexcore.config.io import dump_model_config, load_model_config  # noqa: E402
from flexcore.config.schema import PropertyPackageSpec  # noqa: E402
from flexops.core.build import build_model  # noqa: E402
from flexparameterize.apply import apply_to_model  # noqa: E402
from flexparameterize.emit import emit_model_config  # noqa: E402
from flexparameterize.regression.linear import LinearRegressor  # noqa: E402
from flexparameterize.tags import TagMap  # noqa: E402
from flexparameterize.tests.helpers import build_plant, evaluate_data  # noqa: E402

ALIASED = TagMap({})
PROBES = ((2.0, 1.0e5), (5.0, 2.0e5), (9.0, 2.5e5), (13.0, 3.5e5), (18.0, 4.5e5))
"""tuple: 5 (flow, pressure) probe points shared by both consumer tests."""

INPUT_UNITS = {"flow_out": "m^3/hr", "outlet_state.pressure": "Pa"}


def _fit_linear_spec(unit):
    """Fit a linear (no cross term) power draw against flow and pressure.

    Direct evaluation (no solver): the unit's flow/pressure/power Vars are
    set to a known linear relationship, then read back into the fitted
    regressor's own ``X``/``y`` columns, named exactly as
    :class:`~flexops.surrogates.multilinear.MultilinearSurrogate` resolves
    them on the unit (Pitfall 8: column names double as coefficient keys).
    """
    tb = unit.model().time_block
    flows = [2.0 + i for i in range(tb.n_points)]
    pressures = [1.0e5 + 2.0e4 * i for i in range(tb.n_points)]
    powers = [
        0.4 * flow + 1e-5 * pressure + 1.0
        for flow, pressure in zip(flows, pressures, strict=True)
    ]
    for i, t in enumerate(tb.time_index):
        unit.flow_out[t].set_value(flows[i])
        unit.outlet_state.pressure[t].set_value(pressures[i])
        unit.power_electrical[t].set_value(powers[i])

    X = pd.DataFrame({"flow_out": flows, "outlet_state.pressure": pressures})
    y = pd.DataFrame({"power_electrical": powers})
    regressor = LinearRegressor().fit(X, y, input_units=INPUT_UNITS, output_units="kW")
    spec = regressor.to_surrogate_spec()
    return regressor, spec


def _predicted(regressor, flow: float, pressure: float) -> float:
    """The regressor's own prediction at one (flow, pressure) probe point."""
    return (
        regressor.coefficients["flow_out"] * flow
        + regressor.coefficients["outlet_state.pressure"] * pressure
        + regressor.coefficients["intercept"]
    )


@pytest.mark.component
def test_linear_fit_emit_rebuild_predictions(tmp_path):
    """A fitted spec, emitted and rebuilt, reproduces the fit's own predictions."""
    m, unit = build_plant(has_pressure=True)
    regressor, spec = _fit_linear_spec(unit)

    cfg = emit_model_config(unit, spec, {"data_source": "linear roundtrip test"})
    cfg = cfg.model_copy(
        update={
            "properties": {
                "properties": PropertyPackageSpec(
                    property_class="SimpleAqueousFlow", options={"has_pressure": True}
                )
            }
        }
    )
    path = tmp_path / "linear.json"
    dump_model_config(cfg, path)
    rebuilt = build_model(load_model_config(path)).facility.plant

    for flow, pressure in PROBES:
        rebuilt.flow_out[0].set_value(flow)
        rebuilt.outlet_state.pressure[0].set_value(pressure)
        rebuilt.power_electrical[0].set_value(0.0)
        assert pyo.value(
            rebuilt.surrogate_power_electrical.fitted[0].body
        ) == pytest.approx(-_predicted(regressor, flow, pressure), rel=1e-6)


@pytest.mark.component
def test_linear_surrogate_spec_applies_in_place():
    """The same fitted spec through apply_to_model reproduces the same predictions."""
    m, unit = build_plant(has_pressure=True)
    data = evaluate_data(unit)
    regressor, spec = _fit_linear_spec(unit)

    report = apply_to_model(m, data, ALIASED, surrogates={unit.name: spec})

    assert report.swapped_relations == {unit.name: ["power_electrical_relation"]}
    for flow, pressure in PROBES:
        unit.flow_out[0].set_value(flow)
        unit.outlet_state.pressure[0].set_value(pressure)
        unit.power_electrical[0].set_value(0.0)
        assert pyo.value(
            unit.surrogate_power_electrical.fitted[0].body
        ) == pytest.approx(-_predicted(regressor, flow, pressure), rel=1e-6)
