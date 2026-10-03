"""Fit a LumpedReactor composition from plant-style tags and swap it in place."""

import pandas as pd
import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits
from pyomo.opt import assert_optimal_termination

pytest.importorskip("sklearn")

from flexcore.solvers import get_solver  # noqa: E402
from flexops.properties.simple_aqueous import SimpleAqueousFlow  # noqa: E402
from flexops.surrogates import surrogate_from_spec  # noqa: E402
from flexops.testing import dummy_time_block  # noqa: E402
from flexops.unit_models import LumpedReactor  # noqa: E402
from flexparameterize.regression.linear import LinearRegressor  # noqa: E402


@pytest.mark.integration
@pytest.mark.needs_highs
def test_fitted_composition_relation_replaces_constant_and_tracks_temperature():
    """composition = a + b * reactor_temperature, fitted then swapped."""
    m = dummy_time_block(6)
    m.warm = SimpleAqueousFlow(has_temperature=True)
    m.unit = LumpedReactor(
        property_package=m.warm,
        has_thermal_dynamics=True,
        thermal_gain=0.2 * pyunits.K / pyunits.kW,
        energy_intensity=1.0 * pyunits.kWh / pyunits.m**3,
        compositions={"ch4": 0.6},
    )
    unit = m.unit

    temperatures = [305.0, 308.0, 310.0, 312.0, 315.0, 318.0]
    compositions = [0.4 + 0.001 * (temp - 300.0) for temp in temperatures]
    regressor = LinearRegressor().fit(
        pd.DataFrame({"reactor_temperature": temperatures}),
        pd.DataFrame({"outlet_composition_ch4": compositions}),
        input_units={"reactor_temperature": "K"},
        output_units="dimensionless",
    )
    unit.swap_relation(
        "outlet_composition_ch4_relation",
        surrogate_from_spec(regressor.to_surrogate_spec()),
    )
    assert not unit.outlet_composition_ch4_relation.active

    for t, flow in zip(m.time_block.time_index, [10, 20, 40, 40, 80, 80], strict=True):
        unit.flow_in_feed[t].fix(flow)
        unit.flow_out_product[t].fix(flow)
        unit.inlet_feed_state.temperature[t].fix(300.0)
    m.obj = pyo.Objective(expr=0)
    assert_optimal_termination(get_solver(model=m, prefer="highs").solve(m))

    for t in m.time_block.time_index:
        temperature = pyo.value(unit.reactor_temperature[t])
        assert pyo.value(unit.outlet_composition_ch4[t]) == pytest.approx(
            0.4 + 0.001 * (temperature - 300.0), rel=1e-6
        )
