"""dumps: a built design model pickles and plain pickle.loads reads it back."""

import pickle

import pyomo.environ as pyo
import pytest

from flexcore.pickling import dumps
from flexdesign import build_design_model
from flexdesign.tests.fixtures import pump_battery_solar_study


@pytest.mark.unit
def test_design_model_round_trips_through_dumps():
    model = build_design_model(pump_battery_solar_study())

    loaded = pickle.loads(dumps(model))

    assert sorted(loaded.capacity) == sorted(model.capacity)
    assert sum(1 for _ in loaded.component_data_objects(pyo.Constraint)) == sum(
        1 for _ in model.component_data_objects(pyo.Constraint)
    )
