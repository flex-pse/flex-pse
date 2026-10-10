"""Every flexops unit-model class round-trips through emit_model and build_model."""

import warnings

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits

import flexops as fo
from flexcore.config.io import dump_spec
from flexcore.config.schema import UnitCommitmentConfig
from flexcore.exceptions import FlexEmitWarning
from flexops import build_model, emit_model, unit_models
from flexops.testing import assert_models_equivalent


def plant_model(gas: bool = False):
    """Return a 3-point model with one property package and an empty plant."""
    m = pyo.ConcreteModel(name="contract")
    m.time_block = fo.TimeBlock(
        start_date="2025-01-01", end_date="2025-01-01T00:45", time_step=15 * pyunits.min
    )
    m.properties = fo.SimpleGasFlow() if gas else fo.SimpleAqueousFlow()
    m.plant = fo.PlantBlock(time_block=m.time_block)
    return m


def simple(unit_class, **options):
    """Return a builder placing ``unit_class(property_package, **options)``."""

    def build():
        m = plant_model()
        m.plant.unit = unit_class(property_package=m.properties, **options)
        return m

    return build


def build_battery():
    m = plant_model()
    m.plant.unit = fo.BatteryModel(
        capacity=10 * pyunits.kWh,
        unit_commitment=UnitCommitmentConfig(status=False),
    )
    return m


def build_combustor():
    m = plant_model(gas=True)
    m.plant.unit = fo.Combustor(property_package=m.properties, inlet_names=("fuel",))
    return m


def build_generic_renewables():
    m = plant_model()
    m.plant.unit = fo.GenericRenewables(
        capacity=10 * pyunits.kW, capacity_factor=[0.2, 0.5, 0.8]
    )
    return m


def build_digestor():
    m = plant_model()
    m.biogas = fo.SimpleGasFlow()
    m.sludge = fo.SimpleAqueousFlow()
    m.plant.unit = fo.Digestor(
        inlet_packages={"feed": m.properties},
        biogas_property_package=m.biogas,
        sludge_property_package=m.sludge,
    )
    return m


CONTRACT_BUILDERS = {
    "BatteryModel": build_battery,
    "Combustor": build_combustor,
    "ConstantEnergyIntensityModel": simple(fo.ConstantEnergyIntensityModel),
    "DIDOBlock": simple(fo.DIDOBlock),
    "Digestor": build_digestor,
    "Exchanger": simple(fo.Exchanger),
    "Feed": simple(fo.Feed, outlet_names=("a",)),
    "GenericRenewables": build_generic_renewables,
    "Mixer": simple(fo.Mixer, inlet_names=("a", "b")),
    "Product": simple(fo.Product, inlet_names=("a",)),
    "Pump": simple(fo.Pump),
    "ReverseOsmosis": simple(fo.ReverseOsmosis),
    "SIDOBlock": simple(fo.SIDOBlock),
    "SISOBlock": simple(fo.SISOBlock),
    "Splitter": simple(fo.Splitter, outlet_names=("a", "b")),
    "Tank": simple(fo.Tank),
}


@pytest.mark.unit
def test_every_unit_class_has_a_contract_builder():
    assert set(CONTRACT_BUILDERS) == set(unit_models.__all__)


@pytest.mark.unit
@pytest.mark.parametrize("class_name", sorted(CONTRACT_BUILDERS))
def test_unit_class_round_trips(class_name, tmp_path):
    model = CONTRACT_BUILDERS[class_name]()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FlexEmitWarning)
        spec = emit_model(model)
    dump_spec(spec, tmp_path / "spec.json")

    assert_models_equivalent(model, build_model(tmp_path / "spec.json"))
