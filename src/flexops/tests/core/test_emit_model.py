"""Tests for flexops.emit_model: a live model -> the flat spec it was built from."""

import gzip
import json
import shutil
import warnings
from pathlib import Path

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits
from pyomo.network import Arc

import flexops as fo
from flexcore.config.io import dump_spec
from flexcore.exceptions import FlexConfigError, FlexEmitWarning
from flexops import apply_spec, build_model, emit_model
from flexops.core.stages import INTENSITY_PARAMETER
from flexops.costing import currency_units, load_tariff
from flexops.surrogates import MultilinearSurrogate
from flexops.testing import assert_models_equivalent

FIXTURES = Path(__file__).parent.parent / "fixtures"
USD_PER_KWH = currency_units("USD") / pyunits.kWh
INTENSITY = 0.5 * pyunits.kWh / pyunits.m**3
SURROGATE_DATA = {
    "input_variables": {"flow_out": "m^3/hr"},
    "output_variables": {"power_electrical": "kW"},
    "coefficients": {"flow_out": 0.8, "intercept": 0.0},
}


def base_model(n_hours: int = 3, **costing_options) -> pyo.ConcreteModel:
    """Return a model with time, one package, one costing block and a plant ``p``."""
    m = pyo.ConcreteModel(name="site")
    m.time_block = fo.TimeBlock(
        start_date="2025-01-01",
        end_date=f"2025-01-01T{n_hours:02d}:00",
        time_step=1 * pyunits.hr,
    )
    m.properties = fo.SimpleAqueousFlow()
    costing_options.setdefault("energy_prices", {"electrical": 0.1 * USD_PER_KWH})
    m.costing = fo.FlexCosting(time_block=m.time_block, **costing_options)
    m.p = fo.PlantBlock(time_block=m.time_block)
    return m


def emit_quietly(model, **kwargs):
    """Process costing as build_model does, then emit; fail on any FlexEmitWarning."""
    if model.costing.find_component("aggregate_operating_cost") is None:
        model.costing.cost_process()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FlexEmitWarning)
        return emit_model(model, **kwargs)


def elements(spec, kind: str) -> dict:
    """Return the spec's elements of ``kind`` keyed by name (or target)."""
    found = {}
    for el in spec.of_kind(kind):
        key = getattr(el, "name", None)
        if kind in ("surrogate", "dispatch"):
            key = (el.unit, el.relation if kind == "surrogate" else el.variable)
        found[key] = el
    return found


@pytest.mark.unit
def test_emits_time_package_costing_plant_and_units():
    m = base_model()
    m.p.tank = fo.Tank(property_package=m.properties)
    m.p.pump = fo.ConstantEnergyIntensityModel(
        property_package=m.properties,
        energy_intensity=INTENSITY,
        costing_package=m.costing,
    )

    spec = emit_quietly(m)

    (time,) = spec.of_kind("time")
    assert (time.start_date, time.end_date, time.time_step) == (
        "2025-01-01",
        "2025-01-01T03:00",
        "1 h",
    )
    assert set(elements(spec, "property_package")) == {"properties"}
    assert elements(spec, "costing")["costing"].energy_prices["electrical"].units == (
        "USD/kWh"
    )
    assert set(elements(spec, "plant")) == {"p"}
    units = elements(spec, "unit")
    assert units["p.tank"].unit_model_class == "Tank"
    assert units["p.tank"].costing_package is None
    assert units["p.pump"].costing_package == "auto"
    assert units["p.pump"].property_package == "auto"
    assert units["p.pump"].construction_options == {
        "energy_intensity": {"value": 0.5, "units": "kWh/m^3"}
    }


@pytest.mark.unit
def test_emits_units_on_the_model_and_in_a_network():
    m = base_model()
    m.root_tank = fo.Tank(property_package=m.properties)
    m.net = fo.NetworkBlock(time_block=m.time_block)
    m.net.a = fo.PlantBlock(time_block=m.time_block)
    m.net.a.tank = fo.Tank(property_package=m.properties)

    spec = emit_quietly(m)

    assert set(elements(spec, "network")) == {"net"}
    assert set(elements(spec, "plant")) == {"p", "net.a"}
    assert set(elements(spec, "unit")) == {"root_tank", "net.a.tank"}


@pytest.mark.unit
def test_names_the_package_when_there_are_several():
    m = base_model()
    m.gas = fo.SimpleGasFlow()
    m.p.tank = fo.Tank(property_package=m.properties)

    spec = emit_quietly(m)

    assert elements(spec, "unit")["p.tank"].property_package == "properties"


@pytest.mark.unit
def test_connection_carries_port_destination_name_and_doc():
    m = base_model()
    m.p.tank = fo.Tank(property_package=m.properties)
    m.p.pump = fo.ConstantEnergyIntensityModel(
        property_package=m.properties, energy_intensity=INTENSITY
    )
    m.p.feed = Arc(source=m.p.tank.outlet, destination=m.p.pump.inlet, doc="Feed line.")

    (connection,) = elements(emit_quietly(m), "unit")["p.tank"].connections

    assert (connection.port, connection.to, connection.name, connection.doc) == (
        "outlet",
        "p.pump.inlet",
        "feed",
        "Feed line.",
    )
    assert connection.directed and connection.index is None


@pytest.mark.unit
def test_undirected_arc_is_written_on_its_first_port():
    m = base_model()
    m.p.tank = fo.Tank(property_package=m.properties)
    m.p.pump = fo.ConstantEnergyIntensityModel(
        property_package=m.properties, energy_intensity=INTENSITY
    )
    m.p.link = Arc(ports=(m.p.tank.outlet, m.p.pump.inlet))

    (connection,) = elements(emit_quietly(m), "unit")["p.tank"].connections

    assert not connection.directed
    assert connection.to == "p.pump.inlet"


@pytest.mark.unit
def test_indexed_unit_over_a_list_emits_its_implicit_set():
    m = base_model()
    m.p.tank = fo.Tank(["x", "y"], property_package=m.properties)

    spec = emit_quietly(m)

    assert elements(spec, "set")["p.tank_index"].values == ["x", "y"]
    assert elements(spec, "unit")["p.tank"].index == "p.tank_index"


@pytest.mark.unit
def test_indexed_arc_becomes_a_templated_connection():
    m = base_model()
    m.p.trains = pyo.Set(initialize=[0, 1], ordered=True)
    m.p.tank = fo.Tank(m.p.trains, property_package=m.properties)
    m.p.mix = fo.Mixer(property_package=m.properties, inlet_names=("t0", "t1"))
    m.p.to_mix = Arc(
        m.p.trains,
        rule=lambda b, i: (b.tank[i].outlet, b.mix.find_component(f"inlet_t{i}")),
    )

    spec = emit_quietly(m)

    (connection,) = elements(spec, "unit")["p.tank"].connections
    assert (connection.port, connection.to) == ("outlet", "p.mix.inlet_t{i}")
    assert connection.index is None
    assert elements(spec, "set")["p.trains"].values == [0, 1]


@pytest.mark.unit
def test_indexed_arc_from_a_scalar_unit_names_its_set():
    m = base_model()
    m.p.trains = pyo.Set(initialize=["a", "b"], ordered=True)
    m.p.split = fo.Splitter(property_package=m.properties, outlet_names=("a", "b"))
    m.p.tank = fo.Tank(m.p.trains, property_package=m.properties)
    m.p.fan = Arc(
        m.p.trains,
        rule=lambda b, i: {
            "source": b.split.find_component(f"outlet_{i}"),
            "destination": b.tank[i].inlet,
        },
    )

    (connection,) = elements(emit_quietly(m), "unit")["p.split"].connections

    assert (connection.port, connection.to, connection.index) == (
        "outlet_{i}",
        "p.tank[{i}].inlet",
        "p.trains",
    )


@pytest.mark.unit
def test_ambiguous_arc_template_is_an_error():
    m = base_model()
    m.p.trains = pyo.Set(initialize=[0, 1], ordered=True)
    m.p.tank = fo.Tank(m.p.trains, property_package=m.properties)
    m.p.mix = fo.Mixer(property_package=m.properties, inlet_names=("10", "11"))
    m.p.to_mix = Arc(
        m.p.trains,
        rule=lambda b, i: (b.tank[i].outlet, b.mix.find_component(f"inlet_1{i}")),
    )

    with pytest.raises(FlexConfigError, match=r"to_mix\[\d\]"):
        emit_model(m, check=False)


@pytest.mark.unit
def test_indexed_members_that_differ_are_an_error_naming_both():
    m = base_model()
    m.p.trains = pyo.Set(initialize=[0, 1, 2], ordered=True)
    m.p.ro = fo.ConstantEnergyIntensityModel(
        m.p.trains,
        property_package=m.properties,
        energy_intensity=INTENSITY,
        initialize={
            2: {
                "property_package": m.properties,
                "energy_intensity": 0.6 * pyunits.kWh / pyunits.m**3,
            }
        },
    )

    with pytest.raises(FlexConfigError, match=r"p\.ro\[0\] and p\.ro\[2\] differ"):
        emit_model(m, check=False)


@pytest.mark.unit
def test_unitless_price_is_an_error():
    m = base_model(energy_prices={"electrical": 0.1})

    with pytest.raises(FlexConfigError, match="needs units"):
        emit_model(m, check=False)


@pytest.mark.unit
def test_tariff_object_is_an_error():
    tariff = load_tariff(str(FIXTURES / "tariff_tou_demo.json"))
    m = base_model(energy_prices=None, tariff=tariff)

    with pytest.raises(FlexConfigError, match="tariff_file"):
        emit_model(m, check=False)


@pytest.mark.unit
def test_tariff_file_is_written_relative_to_the_spec_directory(tmp_path):
    shutil.copy(FIXTURES / "tariff_tou_demo.json", tmp_path)
    m = base_model(
        energy_prices=None, tariff_file=str(tmp_path / "tariff_tou_demo.json")
    )

    spec = emit_quietly(m, relative_to=tmp_path)

    assert elements(spec, "costing")["costing"].tariff_source == "tariff_tou_demo.json"


@pytest.mark.unit
def test_custom_objective_warns_naming_it():
    m = base_model()
    m.p.tank = fo.Tank(property_package=m.properties)
    m.custom = pyo.Objective(expr=m.p.tank.volume[0])

    with pytest.warns(FlexEmitWarning, match="custom"):
        spec = emit_model(m)

    assert spec.of_kind("objective") == []


@pytest.mark.unit
def test_cost_objective_is_emitted():
    m = base_model()
    m.p.pump = fo.ConstantEnergyIntensityModel(
        property_package=m.properties,
        energy_intensity=INTENSITY,
        costing_package=m.costing,
    )
    m.costing.cost_process()
    m.obj = pyo.Objective(expr=m.costing.aggregate_operating_cost, sense=pyo.maximize)

    (objective,) = emit_quietly(m).of_kind("objective")

    assert (objective.name, objective.sense, objective.costing_package) == (
        "obj",
        "maximize",
        "auto",
    )


@pytest.mark.unit
def test_extra_constraint_warns_from_the_check():
    m = base_model()
    m.p.tank = fo.Tank(property_package=m.properties)
    m.p.extra = pyo.Constraint(expr=m.p.tank.volume[0] >= 1)
    m.costing.cost_process()

    with pytest.warns(FlexEmitWarning, match=r"p\.extra"):
        emit_model(m)


@pytest.mark.unit
def test_check_warns_when_the_rebuild_fails(tmp_path):
    tariff = tmp_path / "tariff_tou_demo.json"
    shutil.copy(FIXTURES / "tariff_tou_demo.json", tariff)
    m = base_model(energy_prices=None, tariff_file=str(tariff))
    tariff.unlink()

    with pytest.warns(FlexEmitWarning, match="check was skipped"):
        emit_model(m)


@pytest.mark.unit
def test_custom_surrogate_object_warns():
    m = base_model()
    m.p.pump = fo.Pump(property_package=m.properties)
    m.p.pump.swap_relation(
        "power_electrical_relation", MultilinearSurrogate(SURROGATE_DATA)
    )

    with pytest.warns(FlexEmitWarning, match="custom Surrogate object"):
        emit_model(m, check=False)


@pytest.mark.unit
def test_spec_built_surrogate_swapped_in_code_is_emitted():
    m = base_model()
    m.p.pump = fo.Pump(property_package=m.properties)
    apply_spec(
        m,
        [
            {
                "kind": "surrogate",
                "unit": "p.pump",
                "surrogate_type": "multilinear",
                "data": SURROGATE_DATA,
            }
        ],
    )

    surrogate = elements(emit_quietly(m), "surrogate")[
        ("p.pump", "power_electrical_relation")
    ]

    assert surrogate.data == SURROGATE_DATA


@pytest.mark.unit
def test_constant_intensity_applied_with_apply_spec_is_emitted(tmp_path):
    m = base_model()
    m.p.pump = fo.ConstantEnergyIntensityModel(
        property_package=m.properties, energy_intensity=INTENSITY
    )
    apply_spec(
        m,
        [
            {
                "kind": "surrogate",
                "unit": "p.pump",
                "surrogate_type": "constant_intensity",
                "data": {"coefficients": {INTENSITY_PARAMETER: 0.7}},
            }
        ],
    )

    spec = emit_quietly(m)

    (surrogate,) = spec.of_kind("surrogate")
    assert surrogate.data == {"coefficients": {INTENSITY_PARAMETER: 0.7}}
    dump_spec(spec, tmp_path / "spec.json")
    assert_models_equivalent(m, build_model(tmp_path / "spec.json"))


@pytest.mark.unit
def test_external_dispatch_is_emitted():
    m = base_model()
    m.p.battery = fo.BatteryModel(capacity=10 * pyunits.kWh)
    m.p.battery.set_external_dispatch(
        m.p.battery.power_charge, {0: 1.0, 1: 2.0, 2: 0.0}, fix=False
    )

    dispatch = elements(emit_quietly(m), "dispatch")[("p.battery", "power_charge")]

    assert (dispatch.values, dispatch.fix) == ([1.0, 2.0, 0.0], False)


@pytest.mark.unit
def test_long_list_goes_to_a_gzipped_sidecar(tmp_path):
    m = base_model(energy_prices={"electrical": [0.1 * USD_PER_KWH] * 3})

    spec = emit_quietly(m, data_dir=tmp_path / "data", inline_limit=2)

    ref = elements(spec, "costing")["costing"].energy_prices["electrical"].value
    assert ref.source == "data/costing.energy_prices.electrical.json.gz"
    data = json.loads(gzip.decompress((tmp_path / ref.source).read_bytes()))
    assert data == [0.1, 0.1, 0.1]


@pytest.mark.unit
def test_long_list_without_data_dir_warns_and_stays_inline():
    m = base_model(energy_prices={"electrical": [0.1 * USD_PER_KWH] * 3})

    with pytest.warns(FlexEmitWarning, match="data_dir"):
        spec = emit_model(m, inline_limit=2)

    price = elements(spec, "costing")["costing"].energy_prices["electrical"]
    assert price.value == [0.1, 0.1, 0.1]


@pytest.mark.unit
def test_emit_deterministic(tmp_path):
    m = base_model()
    m.p.tank = fo.Tank(property_package=m.properties)
    m.p.pump = fo.ConstantEnergyIntensityModel(
        property_package=m.properties,
        energy_intensity=INTENSITY,
        costing_package=m.costing,
    )
    m.p.feed = Arc(source=m.p.tank.outlet, destination=m.p.pump.inlet)

    dump_spec(emit_quietly(m), tmp_path / "a.json")
    dump_spec(emit_quietly(m), tmp_path / "b.json")

    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


@pytest.mark.unit
def test_digestor_package_options_become_package_references():
    m = base_model()
    m.biogas = fo.SimpleGasFlow()
    m.sludge = fo.SimpleAqueousFlow()
    m.p.digestor = fo.Digestor(
        inlet_packages={"feed": m.properties},
        biogas_property_package=m.biogas,
        sludge_property_package=m.sludge,
    )

    unit = elements(emit_quietly(m), "unit")["p.digestor"]

    assert unit.construction_options["inlet_packages"] == {
        "feed": {"$package": "properties"}
    }
    assert unit.construction_options["biogas_property_package"] == {
        "$package": "biogas"
    }
    assert unit.construction_options["sludge_property_package"] == {
        "$package": "sludge"
    }
    assert unit.property_package is None


@pytest.mark.unit
def test_unprocessed_costing_warns():
    m = base_model()

    with pytest.warns(FlexEmitWarning, match=r"costing has not run cost_process\(\)"):
        emit_model(m, check=False)
