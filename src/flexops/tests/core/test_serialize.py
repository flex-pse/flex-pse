"""Tests for flexops.core.serialize: construction values and units to spec data."""

import datetime
import enum
import math

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits

from flexcore.config.schema import UnitCommitmentConfig
from flexcore.exceptions import FlexConfigError
from flexops.core.serialize import to_jsonable, unit_model_class_name, units_to_str
from flexops.core.units import parse_units
from flexops.costing import currency_units
from flexops.properties import SimpleAqueousFlow
from flexops.testing import dummy_time_block
from flexops.unit_models import Pump, SISOBlock


class Color(enum.Enum):
    RED = "red"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("units", "expected"),
    [
        (pyunits.kWh / pyunits.m**3, "kWh/m^3"),
        (pyunits.m**3 / pyunits.hr, "m^3/h"),
        (pyunits.kW, "kW"),
        (pyunits.Pa, "Pa"),
        (pyunits.min, "min"),
        (pyunits.kg * pyunits.m / pyunits.s**2, "kg*m/s^2"),
        (pyunits.kg / pyunits.m**2 / pyunits.s, "kg/m^2/s"),
        (1 / pyunits.s, "1/s"),
        (pyunits.J / pyunits.kg / pyunits.K, "J/kg/K"),
    ],
)
def test_units_to_str_writes_a_string_parse_units_reads_back(units, expected):
    text = units_to_str(units)

    assert text == expected
    assert math.isclose(
        pyunits.convert_value(1.0, from_units=parse_units(text), to_units=units), 1.0
    )


@pytest.mark.unit
def test_units_to_str_handles_currency():
    assert units_to_str(currency_units("USD") / pyunits.kWh) == "USD/kWh"


@pytest.mark.unit
def test_units_to_str_error_names_where():
    with pytest.raises(FlexConfigError, match="pump.flow") as info:
        units_to_str(pyunits.m**0.5, where="pump.flow")

    assert info.value.field == "pump.flow"


@pytest.mark.unit
@pytest.mark.parametrize("value", [None, True, False, "polarization", 3, 2.5])
def test_to_jsonable_passes_plain_values_through(value):
    out = to_jsonable(value, where="opt")

    assert out == value
    assert type(out) is type(value)


@pytest.mark.unit
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_to_jsonable_rejects_non_finite_numbers(value):
    with pytest.raises(FlexConfigError, match="opt"):
        to_jsonable(value, where="opt")


@pytest.mark.unit
def test_to_jsonable_writes_enum_value():
    assert to_jsonable(Color.RED, where="opt") == "red"


@pytest.mark.unit
def test_to_jsonable_writes_quantity_with_units():
    assert to_jsonable(0.5 * pyunits.kWh / pyunits.m**3, where="opt") == {
        "value": 0.5,
        "units": "kWh/m^3",
    }


@pytest.mark.unit
def test_to_jsonable_writes_dimensionless_expression_as_number():
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=0.4, mutable=True)

    assert to_jsonable(2 * m.p, where="opt") == pytest.approx(0.8)


@pytest.mark.unit
def test_to_jsonable_writes_param_and_var_data_with_units():
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=2.0, units=pyunits.kW, mutable=True)
    m.v = pyo.Var(initialize=3.0, units=pyunits.Pa)

    assert to_jsonable(m.p, where="opt") == {"value": 2.0, "units": "kW"}
    assert to_jsonable(m.v, where="opt") == {"value": 3.0, "units": "Pa"}


@pytest.mark.unit
def test_to_jsonable_writes_indexed_param_in_index_order():
    m = pyo.ConcreteModel()
    m.s = pyo.Set(initialize=[2, 0, 1], ordered=True)
    m.p = pyo.Param(m.s, initialize={2: 1.0, 0: 2.0, 1: 3.0}, units=pyunits.kW)

    assert to_jsonable(m.p, where="opt") == {"value": [1.0, 2.0, 3.0], "units": "kW"}


@pytest.mark.unit
def test_to_jsonable_converts_tuples_lists_and_dicts_recursively():
    value = {"a": (1, Color.RED), "b": [0.5 * pyunits.kW]}

    assert to_jsonable(value, where="opt") == {
        "a": [1, "red"],
        "b": [{"value": 0.5, "units": "kW"}],
    }


@pytest.mark.unit
def test_to_jsonable_rejects_non_string_dict_keys():
    with pytest.raises(FlexConfigError, match="opt"):
        to_jsonable({1: 2}, where="opt")


@pytest.mark.unit
def test_to_jsonable_dumps_pydantic_models():
    assert to_jsonable(UnitCommitmentConfig(status=False), where="opt") == (
        UnitCommitmentConfig(status=False).model_dump(mode="json", by_alias=True)
    )


@pytest.mark.unit
def test_to_jsonable_writes_dates_as_iso_strings():
    assert to_jsonable(datetime.date(2025, 1, 2), where="opt") == "2025-01-02"
    assert (
        to_jsonable(datetime.datetime(2025, 1, 2, 3, 4), where="opt")
        == "2025-01-02T03:04:00"
    )


@pytest.mark.unit
def test_to_jsonable_writes_a_known_package_as_a_reference():
    m = pyo.ConcreteModel()
    m.gas = SimpleAqueousFlow()

    assert to_jsonable({"feed": m.gas}, where="opt", packages={"gas": m.gas}) == {
        "feed": {"$package": "gas"}
    }


@pytest.mark.unit
def test_to_jsonable_rejects_anything_else_naming_the_option():
    with pytest.raises(
        FlexConfigError, match="plant.pump.thing holds a object"
    ) as info:
        to_jsonable(object(), where="plant.pump.thing")

    assert info.value.field == "plant.pump.thing"


@pytest.mark.unit
def test_to_jsonable_rejects_an_unknown_package_block():
    m = pyo.ConcreteModel()
    m.gas = SimpleAqueousFlow()

    with pytest.raises(FlexConfigError, match="opt"):
        to_jsonable(m.gas, where="opt")


@pytest.mark.unit
def test_unit_model_class_name_strips_data_suffix_from_instance_and_class():
    m = dummy_time_block(3)
    m.pump = Pump(property_package=m.properties)

    assert unit_model_class_name(m.pump) == "Pump"
    assert unit_model_class_name(SISOBlock) == "SISOBlock"


@pytest.mark.unit
def test_unit_model_class_name_rejects_non_unit():
    with pytest.raises(FlexConfigError):
        unit_model_class_name(object())
