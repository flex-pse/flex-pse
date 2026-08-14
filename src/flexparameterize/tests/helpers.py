"""Shared model and data builders for the flexparameterize tests.

Builds a one-unit plant on a ``TimeBlock`` with a known energy intensity, and
exports noise-free ``(flow, power)`` data by **direct evaluation** of the
unit's energy relation -- no solver is involved, so the tests using these stay
inside the component budget.
"""

import datetime
import math

import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import units as pyunits

from flexops.core.plant_block import PlantBlock
from flexops.core.time_block import TimeBlock
from flexops.properties.simple_aqueous import SimpleAqueousFlow
from flexops.unit_models import ConstantEnergyIntensityModel
from flexparameterize.regression import ConstantIntensityRegressor
from flexparameterize.tags import model_alias

INTENSITY = 0.42
"""float: the known truth, kWh/m^3, every test regresses back out of the data."""

N_POINTS = 24
"""int: number of 15-minute time points the fixture horizon spans."""


def build_plant(
    unit_class=ConstantEnergyIntensityModel,
    *,
    intensity=INTENSITY,
    has_pressure: bool = False,
):
    """Build a one-unit plant whose energy intensity is known.

    Args:
        unit_class: The flexops unit-model class to build (anything taking an
            ``energy_intensity`` construction option).
        intensity: The unit's energy intensity in kWh/m^3.
        has_pressure: Whether the property package carries a pressure state, so
            a relationship can be keyed to outlet pressure as well as flow.

    Returns:
        ``(model, unit)`` -- the model carrying ``time_block``, ``properties``
        and ``facility``, and the built ``facility.plant`` unit.
    """
    m = pyo.ConcreteModel()
    end = datetime.datetime(2025, 1, 1) + datetime.timedelta(minutes=15 * N_POINTS)
    m.time_block = TimeBlock(
        start_date="2025-01-01",
        end_date=end.isoformat(),
        time_step=15 * pyunits.min,
    )
    m.properties = SimpleAqueousFlow(has_pressure=has_pressure)
    m.facility = PlantBlock(time_block=m.time_block)
    m.facility.plant = unit_class(
        property_package=m.properties,
        energy_intensity=intensity * pyunits.kWh / pyunits.m**3,
    )
    return m, m.facility.plant


def evaluate_data(unit) -> pd.DataFrame:
    """Return noise-free flow/power data for ``unit``, keyed by model alias.

    Sets a deterministic, strictly positive flow profile on the unit and
    computes the matching power straight from its constant-intensity relation
    (direct evaluation, no solver). Columns are the aliases
    :func:`~flexparameterize.tags.model_alias` gives the unit's registered IO
    variables; the index is the ``TimeBlock``'s ``datetime_index``.

    Args:
        unit: A built unit with ``flow_in``/``flow_out``, ``power_electrical``
            and ``energy_intensity``.

    Returns:
        The exported data frame.
    """
    tb = unit.model().time_block
    flows = [10.0 + 5.0 * math.sin(i) for i in range(tb.n_points)]
    intensity = pyo.value(unit.energy_intensity)
    powers = [intensity * flow for flow in flows]
    for i, t in enumerate(tb.time_index):
        unit.flow_in[t].set_value(flows[i])
        unit.flow_out[t].set_value(flows[i])
        unit.power_electrical[t].set_value(powers[i])
    return pd.DataFrame(
        {
            model_alias(unit.inlet_state.flow_vol_phase): flows,
            model_alias(unit.outlet_state.flow_vol_phase): flows,
            model_alias(unit.power_electrical): powers,
        },
        index=tb.datetime_index,
    )


def fit_intensity(unit, data) -> ConstantIntensityRegressor:
    """Fit ``unit``'s energy intensity from aliased ``data``, the emit-path way.

    The caller-driven half of the pipeline: pick the unit's outlet-flow and
    power columns out of the aliased frame and hand them to the regressor. The
    outlet is the basis the unit's own relation is metered on, and
    ``apply_to_model`` resolves the same pair for itself.

    Args:
        unit: A built unit whose outlet flow and ``power_electrical`` appear in
            ``data`` under their model aliases.
        data: The aliased data frame.

    Returns:
        The fitted regressor.
    """
    return ConstantIntensityRegressor().fit(
        data[[model_alias(unit.outlet_state.flow_vol_phase)]],
        data[[model_alias(unit.power_electrical)]],
    )
