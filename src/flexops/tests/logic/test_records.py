"""Every public logic helper records the spec element that would rebuild it."""

import json

import pytest
from pydantic import TypeAdapter
from pyomo.environ import units as pyunits

import flexops.logic as logic
from flexcore.config.schema import UnitCommitmentConfig
from flexcore.config.spec import Element
from flexops.logic._records import LogicRecord, suppress_logic_records
from flexops.testing import dummy_time_block
from flexops.unit_models import BatteryModel, Pump

NOT_RECORDED = {"relax": "solve strategy", "unrelax": "solve strategy"}
"""Public helpers with no spec form, and why."""

KW = pyunits.kW


def two_pumps(with_status: bool = True):
    """Return a model with pumps x and y, with statuses built unrecorded."""
    m = dummy_time_block(4)
    m.x = Pump(property_package=m.properties)
    m.y = Pump(property_package=m.properties)
    if with_status:
        with suppress_logic_records():
            for pump in (m.x, m.y):
                logic.add_status(pump, pump.power_electrical, 0 * KW, 10 * KW)
    return m


def all_records(m) -> list[LogicRecord]:
    """Return the records on the model and on each pump."""
    return [
        *m.__dict__.get("_flex_logic", []),
        *m.x.__dict__.get("_flex_logic", []),
        *m.y.__dict__.get("_flex_logic", []),
    ]


CALLS = {
    "add_status": lambda m: logic.add_status(
        m.x, m.x.flow_in, 0 * pyunits.m**3 / pyunits.hr, 5 * pyunits.m**3 / pyunits.hr
    ),
    "add_startup_shutdown": lambda m: logic.add_startup_shutdown(
        m.x, m.x.status, min_uptime=2
    ),
    "add_dwell": lambda m: logic.add_dwell(m.x.flow_in, 2),
    "add_ramp_rate": lambda m: logic.add_ramp_rate(
        m.x.power_electrical, ramp_up=1.0, ramp_rate_units=KW / pyunits.hr, window=2
    ),
    "add_bypass": lambda m: logic.add_bypass(
        m.x, m.x.flow_in, 1 * pyunits.m**3 / pyunits.hr
    ),
    "add_startup_delay": lambda m: logic.add_startup_delay(m.y, m.x, 1),
    "add_conditional": lambda m: logic.add_conditional(m.x, m.y, then="off"),
    "register_parallel_group": lambda m: logic.register_parallel_group([m.x, m.y]),
}
KINDS = {
    "add_status": "status",
    "add_startup_shutdown": "startup_shutdown",
    "add_dwell": "dwell",
    "add_ramp_rate": "ramp",
    "add_bypass": "bypass",
    "add_startup_delay": "startup_delay",
    "add_conditional": "conditional",
    "register_parallel_group": "parallel_group",
}


@pytest.mark.unit
def test_every_logic_helper_is_recorded_or_excused():
    assert set(logic.__all__) == set(CALLS) | set(NOT_RECORDED)
    assert all(NOT_RECORDED.values())


@pytest.mark.unit
@pytest.mark.parametrize("helper", sorted(CALLS))
def test_helper_leaves_one_record_that_validates_as_an_element(helper):
    m = two_pumps(with_status=helper != "add_status")
    CALLS[helper](m)

    (record,) = all_records(m)
    assert record.kind == KINDS[helper]
    json.dumps(record.fields)
    unit = {} if record.owner is m else {"unit": "x"}
    TypeAdapter(Element).validate_python({"kind": record.kind, **unit, **record.fields})


@pytest.mark.unit
def test_unit_level_records_sit_on_the_unit_and_cross_unit_on_the_model():
    m = two_pumps()
    logic.add_dwell(m.x.flow_in, 2)
    logic.add_conditional(m.x, m.y)

    assert [r.kind for r in m.x._flex_logic] == ["dwell"]
    assert [r.kind for r in m._flex_logic] == ["conditional"]
    assert m._flex_logic[0].fields == {"unit": "x", "implies": "y", "state": "on"}


@pytest.mark.unit
def test_parallel_group_records_once():
    m = two_pumps()
    logic.register_parallel_group([m.x, m.y], variables=["flow_in"])

    assert [r.kind for r in all_records(m)] == ["parallel_group"]
    assert all_records(m)[0].fields == {
        "units": ["x", "y"],
        "variables": ["flow_in"],
        "order_status": True,
    }


@pytest.mark.unit
def test_battery_status_not_recorded():
    m = dummy_time_block(3)
    m.b = BatteryModel(
        capacity=10 * pyunits.kWh, unit_commitment=UnitCommitmentConfig(status=True)
    )

    assert m.b.find_component("status") is not None
    assert "_flex_logic" not in m.b.__dict__


@pytest.mark.unit
def test_suppress_logic_records_restores_recording():
    m = two_pumps()
    with suppress_logic_records():
        logic.add_dwell(m.x.flow_in, 2)
    logic.add_dwell(m.y.flow_in, 2)

    assert "_flex_logic" not in m.x.__dict__
    assert [r.kind for r in m.y._flex_logic] == ["dwell"]
