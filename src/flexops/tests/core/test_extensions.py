"""Tests for flexops.extension and resolve_extension."""

import importlib.metadata

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits

import flexops
from flexcore.exceptions import FlexConfigError
from flexops.core.extensions import resolve_extension
from flexops.testing import dummy_time_block
from flexops.tests import sample_extensions
from flexops.unit_models import Pump


def pump_model():
    """Return a model with one pump ``x``."""
    m = dummy_time_block(3)
    m.x = Pump(property_package=m.properties)
    return m


@pytest.mark.unit
def test_calling_an_extension_records_it_and_runs_it():
    m = pump_model()

    sample_extensions.cap_power(m, unit="x", max_kw=5.0)

    assert m.x.find_component("power_cap") is not None
    assert m._flex_extensions == [
        {
            "entry_point": "flexops.tests.sample_extensions:cap_power",
            "options": {"unit": "x", "max_kw": 5.0},
            "name": "cap_power",
        }
    ]


@pytest.mark.unit
def test_logic_an_extension_builds_is_not_recorded_as_logic():
    m = dummy_time_block(3)
    m.x = flexops.ConstantEnergyIntensityModel(
        property_package=m.properties,
        energy_intensity=0.5 * pyunits.kWh / pyunits.m**3,
    )

    sample_extensions.status_on_flow(m, unit="x", max_flow=5.0)

    assert m.x.find_component("status") is not None
    assert m.x.__dict__.get("_flex_logic", []) == []
    assert [r["name"] for r in m._flex_extensions] == ["status_on_flow"]


@pytest.mark.unit
def test_repeated_calls_get_numbered_names_and_a_name_can_be_pinned():
    m = pump_model()
    m.y = Pump(property_package=m.properties)
    m.z = Pump(property_package=m.properties)

    sample_extensions.cap_power(m, unit="x", max_kw=5.0)
    sample_extensions.cap_power(m, unit="y", max_kw=5.0)
    sample_extensions.cap_power(m, unit="z", max_kw=5.0, _name="z_cap")

    assert [r["name"] for r in m._flex_extensions] == [
        "cap_power",
        "cap_power_2",
        "z_cap",
    ]
    assert "_name" not in m._flex_extensions[2]["options"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "options",
    [{"unit": "x", "max_kw": 5 * pyunits.kW}, {"unit": ("x",), "max_kw": 5.0}],
    ids=["quantity", "tuple"],
)
def test_options_must_be_plain_json(options):
    m = pump_model()

    with pytest.raises(FlexConfigError, match="cap_power"):
        sample_extensions.cap_power(m, **options)


@pytest.mark.unit
def test_resolve_module_and_function():
    assert (
        resolve_extension("flexops.tests.sample_extensions:minimize_power")
        is sample_extensions.minimize_power
    )


@pytest.mark.unit
def test_resolve_dotted_qualname():
    assert resolve_extension("flexops:extension") is flexops.extension
    assert (
        resolve_extension("pyomo.environ:ConcreteModel.model")
        is pyo.ConcreteModel.model
    )


@pytest.mark.unit
def test_resolve_entry_point_name(monkeypatch):
    point = importlib.metadata.EntryPoint(
        name="plain",
        value="flexops.tests.sample_extensions:plain_function",
        group="flexpse.extensions",
    )
    calls = []

    def entry_points(**selection):
        calls.append(selection)
        return importlib.metadata.EntryPoints([point])

    monkeypatch.setattr(importlib.metadata, "entry_points", entry_points)

    assert resolve_extension("plain") is sample_extensions.plain_function
    assert calls == [{"group": "flexpse.extensions", "name": "plain"}]


@pytest.mark.unit
@pytest.mark.parametrize(
    "entry_point",
    ["no_such_extension", "no_such_pkg.mod:fn", "flexops.tests.sample_extensions:nope"],
)
def test_unknown_extension_asks_whether_its_package_is_installed(entry_point):
    with pytest.raises(FlexConfigError, match="installed") as excinfo:
        resolve_extension(entry_point)

    assert entry_point in str(excinfo.value)
