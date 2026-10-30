"""build_model: the config-driven path equals the imperative one."""

import json
from pathlib import Path

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits
from pyomo.network import Arc
from pyomo.opt import assert_optimal_termination

from flexcore.config.io import load_spec
from flexcore.config.spec import DispatchElement, UnfixElement
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
from flexops.core.fixing import apply_state
from flexops.core.ops_block import OpsBlock
from flexops.core.stages import BuildContext
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
def test_build_model_matches_hand_built():
    """Config-driven and imperative builds solve to the same objective."""
    from_config = build_model(_CONFIG)
    assert from_config.demo.tank.find_component("holdup") is not None
    assert from_config.demo.surrogate.find_component("power_electrical") is not None

    assert _solve(from_config) == pytest.approx(_solve(_hand_built()), rel=1e-6)


@pytest.mark.unit
def test_build_model_export_prices_split_electrical_power():
    """Config export_prices reach FlexCosting and build the import/export split."""
    cfg = _demo_dict()
    costing = _element(cfg, "costing")
    del costing["tariff_source"]
    costing["energy_prices"] = {"electrical": {"value": 0.1, "units": "USD/kWh"}}
    costing["export_prices"] = {"electrical": {"value": 0.05, "units": "USD/kWh"}}
    m = build_model(cfg)
    assert m.costing.opex.find_component("export_electrical") is not None


@pytest.mark.unit
def test_build_model_bad_config_raises():
    """A malformed unit element errors with the offending field path in the message."""
    bad = _demo_dict()
    del _element(bad, "unit")["unit_model_class"]
    with pytest.raises(FlexConfigError) as excinfo:
        build_model(bad)
    assert "unit.unit_model_class" in str(excinfo.value)


@pytest.mark.unit
def test_build_model_applies_and_prices_configured_degradation():
    """A degradation element is built on its unit and billed through costing."""
    cfg = _demo_dict()
    cfg["elements"].append(
        {
            "kind": "degradation",
            "unit": "demo.surrogate",
            "name": "wear",
            "terms": [
                {"kind": "variation", "variable": "power_electrical", "price": 1.5}
            ],
        }
    )

    model = build_model(cfg)

    unit = model.demo.surrogate
    assert unit.find_component("wear_rate") is not None
    assert pyo.value(unit.wear_0_price) == 1.5
    assert model.costing.opex.find_component("scalar_cost_surrogate_wear") is not None


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
def test_parse_quantity_list_of_mappings_gives_a_list_of_quantities():
    """A list of {value, units} mappings (a per-period series) parses item-wise."""
    parsed = parse_quantity(
        [{"value": 1.0, "units": "kW"}, {"value": 2, "units": "MW"}]
    )
    assert [pyo.value(pyunits.convert(q, pyunits.kW)) for q in parsed] == [
        pytest.approx(1.0),
        pytest.approx(2000.0),
    ]


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
        apply_state(m, [element])


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

    apply_state(m, [element])

    for t in m.time_block.time_index:
        assert m.unit.power_electrical[t].fixed
        assert pyo.value(m.unit.power_electrical[t]) == pytest.approx(t + 1.0)


@pytest.mark.unit
def test_build_keeps_fix_report():
    """build_model keeps the state stage's FixReport on the model."""
    spec = load_spec(_CONFIG)
    spec.elements.append(
        UnfixElement(kind="unfix", variable="demo.surrogate.energy_intensity")
    )

    model = build_model(spec)

    report = model._flex_fix_report
    assert report.unfixed == ["demo.surrogate.energy_intensity"]
    assert report.skipped == []
    assert not model.demo.surrogate.energy_intensity.fixed


def _demo_dict() -> dict:
    """The demo spec as a dict, its tariff path made absolute."""
    spec = json.loads(_CONFIG.read_text())
    _element(spec, "costing")["tariff_source"] = str(_FIXTURES / "tariff_tou_demo.json")
    return spec


def _element(spec: dict, kind: str) -> dict:
    """Return the first element of ``kind`` in a spec dict."""
    return next(el for el in spec["elements"] if el["kind"] == kind)


@pytest.mark.unit
def test_build_model_gas_package():
    """A SimpleGasFlow property_package element builds as a named gas block."""
    cfg = _demo_dict()
    cfg["elements"].append(
        {"kind": "property_package", "name": "gas", "property_class": "SimpleGasFlow"}
    )
    for unit in cfg["elements"]:
        if unit["kind"] == "unit" and unit.get("property_package", "auto") == "auto":
            unit["property_package"] = "properties"

    model = build_model(cfg)

    assert isinstance(model.gas, SimpleGasFlow)


@pytest.mark.unit
def test_build_model_unknown_property_class():
    """An unknown property_class lists the registered package names."""
    cfg = _demo_dict()
    _element(cfg, "property_package")["property_class"] = "Nope"

    with pytest.raises(FlexConfigError) as excinfo:
        build_model(cfg)
    assert all(name in str(excinfo.value) for name in PROPERTY_PACKAGES)


@pytest.mark.unit
def test_unit_without_package_or_costing():
    """A null property_package or costing_package leaves that unit without one."""
    model = build_model(_CONFIG)

    assert model.demo.battery.config.property_package is None
    assert model.demo.tank.config.costing_package is None
    assert model.demo.battery.config.costing_package is model.costing


def _write_demo_in(directory: Path) -> Path:
    """Write the demo spec and its tariff into ``directory``; return the spec."""
    directory.mkdir()
    (directory / "tariff_tou_demo.json").write_text(
        (_FIXTURES / "tariff_tou_demo.json").read_text()
    )
    path = directory / "config.json"
    path.write_text(_CONFIG.read_text())
    return path


@pytest.mark.unit
def test_tariff_path_resolves_against_config_dir(tmp_path, monkeypatch):
    """A tariff beside the config loads even when the working directory is elsewhere."""
    path = _write_demo_in(tmp_path / "sub")
    monkeypatch.chdir(tmp_path)

    model = build_model(path)

    assert model.costing is not None


@pytest.mark.unit
def test_dispatch_source_resolves_against_config_dir(tmp_path, monkeypatch):
    """A dispatch series in a $source file beside the config loads from another cwd."""
    path = _write_demo_in(tmp_path / "sub")
    spec = json.loads(path.read_text())
    spec["elements"].append(
        {
            "kind": "dispatch",
            "unit": "demo.surrogate",
            "variable": "power_electrical",
            "values": {"$source": "series.json"},
        }
    )
    path.write_text(json.dumps(spec))
    (tmp_path / "sub" / "series.json").write_text(json.dumps([1.0] * 24))
    monkeypatch.chdir(tmp_path)

    model = build_model(path)

    assert model.demo.surrogate.power_electrical[0].fixed
