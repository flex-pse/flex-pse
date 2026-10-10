"""build_model: the config-driven path equals the imperative one (§2.3, R3)."""

import json
import warnings
from pathlib import Path

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits
from pyomo.network import Arc, Port
from pyomo.opt import assert_optimal_termination

from flexcore.config.io import load_model_config, load_spec
from flexcore.config.spec import Connection, DispatchElement
from flexcore.exceptions import FlexConfigError
from flexcore.solvers import get_solver
from flexops import (
    BatteryModel,
    ConstantEnergyIntensityModel,
    FlexCosting,
    PlantBlock,
    SimpleAqueousFlow,
    SimpleGasFlow,
    Tank,
    TimeBlock,
    build_model,
)
from flexops.core import stages as stages_module
from flexops.core.build import parse_quantity, parse_units
from flexops.core.ops_block import OpsBlock
from flexops.core.stages import BuildContext, _apply_dispatch, _build_connection
from flexops.properties import PROPERTY_PACKAGES

_FIXTURES = Path(__file__).parent.parent / "fixtures"
_CONFIG = _FIXTURES / "plant_config_demo.json"


def _hand_built() -> pyo.ConcreteModel:
    """Build the twin of ``plant_config_demo.json`` imperatively."""
    m = pyo.ConcreteModel()
    m.time_block = TimeBlock(
        start_date="2025-07-08", end_date="2025-07-09", time_step=1 * pyunits.hr
    )
    m.properties = SimpleAqueousFlow()
    m.costing = FlexCosting(
        time_block=m.time_block, tariff_file=str(_FIXTURES / "tariff_tou_demo.json")
    )
    m.demo = PlantBlock(time_block=m.time_block)
    m.demo.tank = Tank(
        property_package=m.properties,
        max_volume=1000.0 * pyunits.m**3,
        initial_volume=500.0 * pyunits.m**3,
    )
    m.demo.surrogate = ConstantEnergyIntensityModel(
        property_package=m.properties,
        energy_intensity=0.5 * pyunits.kWh / pyunits.m**3,
        costing_package=m.costing,
    )
    m.demo.battery = BatteryModel(
        capacity=10.0 * pyunits.kWh, costing_package=m.costing
    )
    m.demo.arc_0 = Arc(source=m.demo.tank.outlet, destination=m.demo.surrogate.inlet)
    m.costing.cost_process()
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
    return m


def _solve(model) -> float:
    """Expand arcs, solve, and return the objective value."""
    pyo.TransformationFactory("network.expand_arcs").apply_to(model)
    results = get_solver(model=model, prefer="highs").solve(model)
    assert_optimal_termination(results)
    return pyo.value(model.objective)


@pytest.mark.component
@pytest.mark.needs_highs
def test_build_model_matches_hand_built(monkeypatch):
    """Config-driven and imperative builds solve to the same objective."""
    monkeypatch.chdir(_FIXTURES)
    from_config = build_model(load_model_config(_CONFIG))
    assert from_config.demo.tank.find_component("holdup") is not None
    assert from_config.demo.surrogate.find_component("power_electrical") is not None

    assert _solve(from_config) == pytest.approx(_solve(_hand_built()), rel=1e-6)


@pytest.mark.unit
def test_build_model_bad_config_raises():
    """A malformed unit config errors with the offending field path in the message."""
    bad = {
        "schema_version": "0.0.1",
        "time": {
            "start_date": "2025-07-08",
            "end_date": "2025-07-09",
            "time_step": "1 hr",
        },
        "costing": {
            "energy_prices": {"electrical": {"value": 0.1, "units": "USD/kWh"}}
        },
        "plant": {
            "name": "demo",
            "units": {"surrogate": {"construction_options": {}}},
        },
    }
    with pytest.raises(FlexConfigError) as excinfo:
        build_model(bad)
    assert "plant.units.surrogate.unit_model_class" in str(excinfo.value)


@pytest.mark.unit
def test_build_model_network_branch(monkeypatch):
    """A config with 'network' instead of 'plant' builds a NetworkBlock of plants."""
    monkeypatch.chdir(_FIXTURES)
    cfg = json.loads(_CONFIG.read_text())
    plant = cfg.pop("plant")
    cfg["network"] = {"name": "net", "plants": {"demo": plant}}

    model = build_model(cfg)

    assert model.net.demo.tank.find_component("holdup") is not None
    assert model.net.demo.surrogate.find_component("power_electrical") is not None


@pytest.mark.unit
def test_build_connection_bad_port_raises():
    """A connection endpoint that does not resolve to a port is a config error."""
    m = pyo.ConcreteModel()
    m.time_block = TimeBlock(
        start_date="2025-01-01", end_date="2025-01-01T01:00", time_step=15 * pyunits.min
    )
    m.unit = OpsBlock()
    m.unit.outlet = pyo.Var()
    m.unit.outlet_port = Port(initialize={"x": m.unit.outlet})

    context = BuildContext(base_dir=None, units={"unit": m.unit})
    connection = Connection(port="outlet_port", to="unit.nope")
    with pytest.raises(FlexConfigError, match="has no port"):
        _build_connection(m, "unit", connection, context)


@pytest.mark.unit
def test_parse_units_multi_factor_and_exponent():
    """Compact unit strings parse into the expected dimensioned quantity."""
    assert pyo.value(
        pyunits.convert(1 * parse_units("kWh/m^3"), pyunits.kWh / pyunits.m**3)
    ) == pytest.approx(1.0)
    assert pyo.value(
        pyunits.convert(1 * parse_units("m^3/hr"), pyunits.m**3 / pyunits.hr)
    ) == pytest.approx(1.0)
    assert pyo.value(
        pyunits.convert(1 * parse_units("kWh*/hr"), pyunits.kWh / pyunits.hr)
    ) == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("kg/m^2/s", pyunits.kg / pyunits.m**2 / pyunits.s),
        ("J/K/kg", pyunits.J / pyunits.K / pyunits.kg),
        ("1/s", 1 / pyunits.s),
        ("1/m^3", 1 / pyunits.m**3),
    ],
)
def test_parse_units_reads_chained_division_and_unit_numerator(text, expected):
    """Every '/' after the first divides again, and a bare '1' is dimensionless."""
    assert pyo.value(pyunits.convert(1 * parse_units(text), expected)) == (
        pytest.approx(1.0)
    )


@pytest.mark.unit
def test_parse_units_unknown_token_becomes_currency():
    """A token pyomo does not know is registered as a currency.

    Uses a token ("XYZ") no other test in the suite registers, so this stays
    deterministic regardless of test order (unlike "USD", which FlexCosting's
    default currency registers globally the first time any test builds one).
    """
    assert str(parse_units("XYZ")) == "XYZ"


@pytest.mark.unit
def test_parse_units_bad_token_raises():
    """A token that is not a name-and-exponent pair is a config error."""
    with pytest.raises(FlexConfigError, match="Could not parse"):
        parse_units("1min")


@pytest.mark.unit
def test_parse_quantity_mapping_and_string_forms():
    """Both the {value, units} mapping and the '<number> <units>' string parse."""
    from_mapping = parse_quantity({"value": 1500.0, "units": "L"})
    from_string = parse_quantity("1.5 m^3")
    assert pyo.value(pyunits.convert(from_mapping, pyunits.m**3)) == pytest.approx(1.5)
    assert pyo.value(pyunits.convert(from_string, pyunits.m**3)) == pytest.approx(1.5)


@pytest.mark.unit
def test_parse_quantity_non_numeric_string_strict_raises():
    """strict=True (the default) rejects a string with no numeric magnitude."""
    with pytest.raises(FlexConfigError, match="Could not read"):
        parse_quantity("polarization")


@pytest.mark.unit
def test_parse_quantity_non_numeric_string_lenient_passthrough():
    """strict=False returns a non-quantity string unchanged (an enum member name)."""
    assert parse_quantity("polarization", strict=False) == "polarization"


@pytest.mark.unit
def test_parse_quantity_passes_through_non_dict_non_string_values():
    """A value that is neither a {value, units} mapping nor a string is untouched."""
    assert parse_quantity(5.0) == 5.0
    assert parse_quantity(None) is None


@pytest.mark.unit
def test_apply_dispatch_noop_without_dispatch_elements():
    """A spec with no dispatch elements leaves every variable free."""
    m = pyo.ConcreteModel()
    m.time_block = TimeBlock(
        start_date="2025-01-01", end_date="2025-01-01T01:00", time_step=15 * pyunits.min
    )
    m.unit = OpsBlock()
    m.unit.power_electrical = pyo.Var(m.time_block.time_index, units=pyunits.kW)
    stages_module.stage_state(m, load_spec(_demo_dict()), BuildContext(base_dir=None))
    assert not m.unit.power_electrical[0].fixed


@pytest.mark.unit
def test_apply_dispatch_unknown_variable_raises():
    """A dispatched variable that is not on the unit is a config error."""
    m = pyo.ConcreteModel()
    m.time_block = TimeBlock(
        start_date="2025-01-01", end_date="2025-01-01T01:00", time_step=15 * pyunits.min
    )
    m.unit = OpsBlock()
    element = DispatchElement(
        kind="dispatch", unit="unit", variable="nope", values=[1.0]
    )
    with pytest.raises(FlexConfigError, match="not on"):
        _apply_dispatch(m, element, BuildContext(base_dir=None))


@pytest.mark.unit
def test_apply_dispatch_applies_series(tmp_path):
    """A JSON dispatch series fixes the named var, coercing string keys to ints."""
    m = pyo.ConcreteModel()
    m.time_block = TimeBlock(
        start_date="2025-01-01", end_date="2025-01-01T01:00", time_step=15 * pyunits.min
    )
    m.unit = OpsBlock()
    m.unit.power_electrical = pyo.Var(m.time_block.time_index, units=pyunits.kW)

    series_file = tmp_path / "series.json"
    series_file.write_text(json.dumps({"0": 1.0, "1": 2.0, "2": 3.0, "3": 4.0}))
    element = DispatchElement(
        kind="dispatch",
        unit="unit",
        variable="power_electrical",
        values=json.loads(series_file.read_text()),
    )

    _apply_dispatch(m, element, BuildContext(base_dir=None))

    for t in m.time_block.time_index:
        assert m.unit.power_electrical[t].fixed
        assert pyo.value(m.unit.power_electrical[t]) == pytest.approx(t + 1.0)


def _demo_dict(**overrides) -> dict:
    """The demo config as a dict, with top-level keys overridden."""
    return {**json.loads(_CONFIG.read_text()), **overrides}


def _prices_only() -> dict:
    """A costing section that needs no tariff file."""
    return {"energy_prices": {"electrical": {"value": 0.1, "units": "USD/kWh"}}}


@pytest.mark.unit
def test_build_model_gas_package():
    """A SimpleGasFlow entry in properties builds as a named gas block."""
    cfg = _demo_dict(
        schema_version="0.0.4",
        costing=_prices_only(),
        properties={"gas": {"property_class": "SimpleGasFlow"}},
        plant={
            "name": "p",
            "units": {
                "battery": {
                    "unit_model_class": "BatteryModel",
                    "construction_options": {
                        "capacity": {"value": 10.0, "units": "kWh"}
                    },
                    "property_package": None,
                }
            },
        },
    )

    model = build_model(cfg)

    assert isinstance(model.gas, SimpleGasFlow)


@pytest.mark.unit
def test_build_model_unknown_property_class():
    """An unknown property_class lists the registered package names."""
    cfg = _demo_dict(
        schema_version="0.0.4",
        properties={"properties": {"property_class": "Nope"}},
    )

    with pytest.raises(FlexConfigError) as excinfo:
        build_model(cfg)
    assert all(name in str(excinfo.value) for name in PROPERTY_PACKAGES)


@pytest.mark.unit
def test_unit_property_package_none():
    """property_package null builds the unit with no property package."""
    cfg = _demo_dict(
        schema_version="0.0.4",
        costing=_prices_only(),
        plant={
            "name": "p",
            "units": {
                "battery": {
                    "unit_model_class": "BatteryModel",
                    "construction_options": {
                        "capacity": {"value": 10.0, "units": "kWh"}
                    },
                    "property_package": None,
                }
            },
        },
    )

    model = build_model(cfg)

    assert model.p.battery.config.property_package is None


@pytest.mark.unit
def test_unit_costing_false():
    """costing false leaves one unit uncosted while its neighbor stays costed."""
    cfg = _demo_dict(
        schema_version="0.0.4",
        costing=_prices_only(),
        plant={
            "name": "p",
            "units": {
                "free": {
                    "unit_model_class": "BatteryModel",
                    "construction_options": {
                        "capacity": {"value": 10.0, "units": "kWh"}
                    },
                    "property_package": None,
                    "costing": False,
                },
                "billed": {
                    "unit_model_class": "BatteryModel",
                    "construction_options": {
                        "capacity": {"value": 10.0, "units": "kWh"}
                    },
                    "property_package": None,
                },
            },
        },
    )

    model = build_model(cfg)

    assert model.p.free.config.costing_package is None
    assert model.p.billed.config.costing_package is model.costing


def _write_demo_in(directory: Path, **overrides) -> Path:
    """Write the demo config and its tariff into ``directory``; return the config."""
    directory.mkdir()
    (directory / "tariff_tou_demo.json").write_text(
        (_FIXTURES / "tariff_tou_demo.json").read_text()
    )
    path = directory / "config.json"
    path.write_text(json.dumps(_demo_dict(**overrides)))
    return path


@pytest.mark.unit
def test_tariff_path_resolves_against_config_dir(tmp_path, monkeypatch):
    """A tariff beside the config loads even when the working directory is elsewhere."""
    path = _write_demo_in(tmp_path / "sub")
    monkeypatch.chdir(tmp_path)

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Path ")
        model = build_model(load_model_config(path))

    assert model.costing is not None


@pytest.mark.unit
def test_relative_path_cwd_fallback_warns(tmp_path, monkeypatch):
    """A tariff found only relative to the working directory warns and still builds."""
    path = _write_demo_in(tmp_path / "sub")
    (tmp_path / "sub" / "tariff_tou_demo.json").rename(
        tmp_path / "tariff_tou_demo.json"
    )
    monkeypatch.chdir(tmp_path)

    with pytest.warns(DeprecationWarning, match="relative to the config file"):
        build_model(load_model_config(path))


@pytest.mark.unit
def test_dispatch_path_resolves_against_config_dir(tmp_path, monkeypatch):
    """An external-dispatch series beside the config loads from another cwd."""
    plant = json.loads(_CONFIG.read_text())["plant"]
    plant["units"]["surrogate"]["external_dispatch"] = {
        "variable": "power_electrical",
        "source": "series.json",
    }
    path = _write_demo_in(tmp_path / "sub", plant=plant)
    (tmp_path / "sub" / "series.json").write_text(
        json.dumps({str(t): 1.0 for t in range(24)})
    )
    monkeypatch.chdir(tmp_path)

    model = build_model(load_model_config(path))

    assert model.demo.surrogate.power_electrical[0].fixed
