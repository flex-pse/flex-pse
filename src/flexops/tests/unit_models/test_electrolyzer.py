"""Harness-driven and hand tests for the Electrolyzer family."""

import dataclasses

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits

from flexcore.exceptions import FlexConfigError
from flexops.properties.simple_aqueous import SimpleAqueousFlow
from flexops.properties.simple_gas import SimpleGasFlow
from flexops.surrogates import MultilinearSurrogate
from flexops.testing import UnitModelTestHarness, dummy_time_block
from flexops.testing.harness import _solve_with_inputs_fixed
from flexops.unit_models import CO2Electrolyzer, Electrolyzer, WaterElectrolyzer
from flexops.unit_models.electrolyzer import (
    ElectrochemicalProduct,
    ElectrolyzerTechnology,
    ProductPhase,
)

FARADAY = 96485.33212  # C/mol
GAS_CONSTANT = 8.314462618153241  # J/(mol K), Pyomo's pyunits.R
T_OP = 333.15  # K
P_OP = 101325.0  # Pa
MOLAR_VOLUME = GAS_CONSTANT * T_OP / P_OP * 3600  # (m^3/hr) per (mol/s)
WATER_VOLUME = 0.018015 / 1000.0 * 3600  # (m^3/hr) per (mol/s) at 1000 kg/m^3

N_CELLS = 10
CURRENT = 1000.0  # A
ELECTRONS = N_CELLS * CURRENT / FARADAY  # mol e-/s

CO_PRODUCT = ElectrochemicalProduct(
    name="CO",
    electrons=2,
    faradaic_efficiency=0.9,
    phase=ProductPhase.GAS,
    co2_per_mol=1.0,
    water_per_mol=0.0,
    molar_mass=28.01 * pyunits.g / pyunits.mol,
)
H2_SIDE_PRODUCT = ElectrochemicalProduct(
    name="H2",
    electrons=2,
    faradaic_efficiency=0.1,
    phase=ProductPhase.GAS,
    co2_per_mol=0.0,
    water_per_mol=1.0,
    molar_mass=2.016 * pyunits.g / pyunits.mol,
)
FORMATE_PRODUCT = ElectrochemicalProduct(
    name="HCOOH",
    electrons=2,
    faradaic_efficiency=0.5,
    phase=ProductPhase.LIQUID,
    co2_per_mol=1.0,
    water_per_mol=1.0,
    molar_mass=46.03 * pyunits.g / pyunits.mol,
)

_COMMON = dict(
    n_cells=N_CELLS,
    operating_temperature=T_OP * pyunits.K,
    operating_pressure=P_OP * pyunits.Pa,
    separator_volume=10 * pyunits.m**3,
    initial_liquid_volume=5 * pyunits.m**3,
)


def _water(n: int = 3, **kwargs):
    """Build a WaterElectrolyzer on an ``n``-point time block, current set."""
    m = dummy_time_block(n)
    m.gas = SimpleGasFlow()
    m.unit = WaterElectrolyzer(
        liquid_property_package=m.properties,
        gas_property_package=m.gas,
        **{**_COMMON, **kwargs},
    )
    m.unit.current[:].set_value(CURRENT)
    return m, m.unit


def _co2(n: int = 3, **kwargs):
    """Build a CO2Electrolyzer on an ``n``-point time block, current set."""
    m = dummy_time_block(n)
    m.gas = SimpleGasFlow()
    m.unit = CO2Electrolyzer(
        liquid_property_package=m.properties,
        gas_property_package=m.gas,
        **{**_COMMON, **kwargs},
    )
    m.unit.current[:].set_value(CURRENT)
    return m, m.unit


def _registered(unit, role: str) -> set[str]:
    """Return the local names of the IO variables registered in ``role``."""
    return {
        rec.var.local_name for rec in unit._io_registry.io_variables if rec.role == role
    }


def _relations(unit) -> set[str]:
    """Return the names of the unit's registered (swappable) relations."""
    return {rec.name for rec in unit._io_registry.relations}


def _parameters(unit) -> set[str]:
    """Return the names of the unit's registered process parameters."""
    return {rec.name for rec in unit._io_registry.parameters}


# -- harness -----------------------------------------------------------------


class TestWaterElectrolyzerPEM(UnitModelTestHarness):
    """Fixing current and make-up water determines every flow and the power."""

    expected_dof = 0
    expected_solution = {
        "production_H2[1]": ELECTRONS / 2,
        "power_electrical[1]": N_CELLS * 2.0 * CURRENT * 1.1 / 1000.0,
        "outlet_cathode_gas_state.flow_vol_phase[1,Vap]": ELECTRONS / 2 * MOLAR_VOLUME,
        "outlet_anode_gas_state.flow_vol_phase[1,Vap]": ELECTRONS / 4 * MOLAR_VOLUME,
        "outlet_cathode_gas_state.temperature[1]": T_OP,
    }

    def configure(self):
        m, unit = _water(cell_voltage=2.0 * pyunits.V, bop_fraction=0.1)
        unit.inlet_water_state.flow_vol_phase[:, "Liq"].set_value(0.01)
        return m, unit


class TestCO2Electrolyzer(UnitModelTestHarness):
    """CO + H2 slate with crossover; CO2 feed follows from single-pass conversion."""

    expected_dof = 0
    expected_solution = {
        "production_CO[1]": 0.9 * ELECTRONS / 2,
        "production_H2[1]": 0.1 * ELECTRONS / 2,
        "inlet_co2_state.flow_vol_phase[1,Vap]": (
            0.9 * ELECTRONS / 2 + 0.25 * ELECTRONS
        )
        / 0.5
        * MOLAR_VOLUME,
    }

    def configure(self):
        m, unit = _co2(single_pass_conversion=0.5, co2_crossover_per_electron=0.25)
        unit.inlet_water_state.flow_vol_phase[:, "Liq"].set_value(0.01)
        return m, unit


# -- structure ---------------------------------------------------------------


@pytest.mark.unit
def test_water_electrolyzer_builds_water_inlet_and_two_gas_outlets():
    _, unit = _water()
    assert unit.find_component("inlet_water") is not None
    assert unit.find_component("outlet_cathode_gas") is not None
    assert unit.find_component("outlet_anode_gas") is not None
    assert unit.find_component("inlet_co2") is None
    assert unit.find_component("outlet_liquid") is None


@pytest.mark.unit
def test_has_liquid_outlet_builds_the_liquid_outlet_port():
    _, unit = _water(has_liquid_outlet=True, bleed_fraction=0.05)
    assert unit.find_component("outlet_liquid") is not None
    assert "bleed_fraction" in _parameters(unit)


@pytest.mark.unit
def test_co2_electrolyzer_adds_a_co2_inlet():
    _, unit = _co2()
    assert unit.find_component("inlet_co2") is not None


@pytest.mark.unit
def test_generic_electrolyzer_accepts_a_custom_product_table():
    m = dummy_time_block(3)
    m.gas = SimpleGasFlow()
    m.unit = Electrolyzer(
        liquid_property_package=m.properties,
        gas_property_package=m.gas,
        products=[CO_PRODUCT, H2_SIDE_PRODUCT],
    )
    assert {"faradaic_relation_CO", "faradaic_relation_H2"} <= _relations(m.unit)


@pytest.mark.unit
def test_products_accept_plain_dicts():
    _, unit = _co2(
        products=[
            {
                "name": "CO",
                "electrons": 2,
                "faradaic_efficiency": 0.95,
                "phase": "gas",
                "co2_per_mol": 1.0,
                "water_per_mol": 0.0,
                "molar_mass": 28.01 * pyunits.g / pyunits.mol,
            }
        ]
    )
    assert pyo.value(unit.faradaic_efficiency_CO) == pytest.approx(0.95)


# -- registration ------------------------------------------------------------


@pytest.mark.unit
def test_power_and_faradaic_relations_are_registered_swappable():
    _, unit = _water()
    assert _relations(unit) == {"power_electrical_relation", "faradaic_relation_H2"}


@pytest.mark.unit
def test_co2_electrolyzer_registers_one_faradaic_relation_per_product():
    _, unit = _co2()
    assert _relations(unit) == {
        "power_electrical_relation",
        "faradaic_relation_CO",
        "faradaic_relation_H2",
    }


@pytest.mark.unit
def test_scalar_coefficients_are_fixed_regressable_parameters():
    _, unit = _co2()
    expected = {
        "cell_voltage",
        "bop_fraction",
        "faradaic_efficiency_CO",
        "faradaic_efficiency_H2",
        "single_pass_conversion",
        "co2_crossover_per_electron",
    }
    assert expected <= _parameters(unit)
    for name in expected:
        assert unit.find_component(name).fixed


@pytest.mark.unit
def test_current_and_make_up_water_are_inputs_power_is_output():
    _, unit = _water()
    assert {"current", "flow_vol_phase"} <= _registered(unit, "input")
    assert {"power_electrical", "liquid_volume"} <= _registered(unit, "output")
    inputs = [rec.var for rec in unit._io_registry.io_variables if rec.role == "input"]
    assert any(var is unit.inlet_water_state.flow_vol_phase for var in inputs)


# -- technology defaults -----------------------------------------------------


@pytest.mark.unit
def test_technology_selects_default_cell_voltage_and_temperature():
    m = dummy_time_block(3)
    m.gas = SimpleGasFlow()
    m.pem = WaterElectrolyzer(
        liquid_property_package=m.properties, gas_property_package=m.gas
    )
    m.aem = WaterElectrolyzer(
        liquid_property_package=m.properties,
        gas_property_package=m.gas,
        technology=ElectrolyzerTechnology.AEM,
    )
    assert pyo.value(m.pem.cell_voltage) != pyo.value(m.aem.cell_voltage)
    assert pyo.value(m.pem.outlet_cathode_gas_state.temperature[0]) != pyo.value(
        m.aem.outlet_cathode_gas_state.temperature[0]
    )


@pytest.mark.unit
def test_explicit_cell_voltage_overrides_the_technology_default():
    _, unit = _water(
        technology=ElectrolyzerTechnology.AEM, cell_voltage=2.3 * pyunits.V
    )
    assert pyo.value(unit.cell_voltage) == pytest.approx(2.3)


# -- hand-computed relation bodies -------------------------------------------


def _assert_satisfied(constraint) -> None:
    """Every member of ``constraint`` has a zero residual at current values."""
    for idx in constraint:
        body = constraint[idx]
        assert pyo.value(body.body) == pytest.approx(
            pyo.value(body.upper), rel=1e-9, abs=1e-12
        )


@pytest.mark.unit
def test_faradaic_and_gas_balances_hold_on_hand_computed_values():
    _, unit = _water()
    for t in unit.current:
        unit.production_H2[t].set_value(ELECTRONS / 2)
        unit.outlet_cathode_gas_state.flow_vol_phase[t, "Vap"].set_value(
            ELECTRONS / 2 * MOLAR_VOLUME
        )
        unit.outlet_anode_gas_state.flow_vol_phase[t, "Vap"].set_value(
            ELECTRONS / 4 * MOLAR_VOLUME
        )
    _assert_satisfied(unit.faradaic_relation_H2)
    _assert_satisfied(unit.cathode_gas_balance)
    _assert_satisfied(unit.anode_gas_balance)


@pytest.mark.unit
def test_liquid_holdup_tracks_make_up_minus_consumption():
    m, unit = _water(n=4)
    dt = 0.25  # hr
    make_up = 0.02  # m^3/hr
    consumed = ELECTRONS / 2 * WATER_VOLUME
    volume = 5.0
    for t in m.time_block.time_index:
        unit.production_H2[t].set_value(ELECTRONS / 2)
        unit.inlet_water_state.flow_vol_phase[t, "Liq"].set_value(make_up)
        if t > 0:
            volume += dt * (make_up - consumed)
        unit.liquid_volume[t].set_value(volume)
    assert len(unit.liquid_holdup) == 3
    _assert_satisfied(unit.liquid_holdup)


@pytest.mark.unit
def test_liquid_outlet_carries_bleed_and_liquid_products():
    _, unit = _co2(
        products=[
            dataclasses.replace(CO_PRODUCT, faradaic_efficiency=0.4),
            FORMATE_PRODUCT,
        ],
        has_liquid_outlet=True,
        bleed_fraction=0.1,
    )
    formate = 0.5 * ELECTRONS / 2
    consumed = formate * WATER_VOLUME
    formate_volume = formate * 0.04603 / 1000.0 * 3600
    for t in unit.current:
        unit.production_HCOOH[t].set_value(formate)
        unit.outlet_liquid_state.flow_vol_phase[t, "Liq"].set_value(
            0.1 * consumed + formate_volume
        )
    _assert_satisfied(unit.liquid_outlet_balance)


@pytest.mark.unit
def test_co2_feed_closes_the_carbon_balance():
    _, unit = _co2(single_pass_conversion=0.5, co2_crossover_per_electron=0.25)
    co = 0.9 * ELECTRONS / 2
    crossover = 0.25 * ELECTRONS
    feed = (co + crossover) / 0.5
    for t in unit.current:
        unit.production_CO[t].set_value(co)
        unit.production_H2[t].set_value(0.1 * ELECTRONS / 2)
        unit.inlet_co2_state.flow_vol_phase[t, "Vap"].set_value(feed * MOLAR_VOLUME)
    _assert_satisfied(unit.co2_feed_balance)
    t = 0
    # carbon in == carbon to product + unreacted (cathode) + crossover (anode)
    assert pyo.value(unit.co2_unreacted[t]) == pytest.approx(feed - co - crossover)


@pytest.mark.unit
def test_waste_heat_is_current_times_overpotential_above_thermoneutral():
    _, unit = _water(cell_voltage=2.0 * pyunits.V)
    expected_kw = N_CELLS * CURRENT * (2.0 - 1.48) / 1000.0
    assert pyo.value(unit.waste_heat[0]) == pytest.approx(expected_kw)


# -- config errors -----------------------------------------------------------


@pytest.mark.unit
def test_rejects_faradaic_efficiencies_summing_above_one():
    over = dataclasses.replace(CO_PRODUCT, faradaic_efficiency=0.95)
    with pytest.raises(FlexConfigError, match="faradaic"):
        _co2(products=[over, H2_SIDE_PRODUCT])


@pytest.mark.unit
def test_rejects_duplicate_product_names():
    with pytest.raises(FlexConfigError, match="unique"):
        _co2(products=[H2_SIDE_PRODUCT, H2_SIDE_PRODUCT])


@pytest.mark.unit
def test_rejects_liquid_product_without_liquid_outlet():
    with pytest.raises(FlexConfigError, match="has_liquid_outlet"):
        _co2(products=[FORMATE_PRODUCT])


@pytest.mark.unit
@pytest.mark.parametrize("conversion", [0.0, 1.5])
def test_rejects_single_pass_conversion_outside_unit_interval(conversion):
    with pytest.raises(FlexConfigError, match="single_pass_conversion"):
        _co2(single_pass_conversion=conversion)


@pytest.mark.unit
def test_rejects_initial_liquid_volume_outside_the_level_window():
    with pytest.raises(FlexConfigError, match="initial_liquid_volume"):
        _water(initial_liquid_volume=9.5 * pyunits.m**3)


@pytest.mark.unit
def test_rejects_missing_gas_property_package():
    m = dummy_time_block(3)
    with pytest.raises(FlexConfigError, match="gas_property_package"):
        m.unit = WaterElectrolyzer(liquid_property_package=SimpleAqueousFlow())


@pytest.mark.unit
def test_per_port_gas_packages_override_the_shared_gas_package():
    m = dummy_time_block(3)
    m.gas = SimpleGasFlow()
    m.h2 = SimpleGasFlow()
    m.co2 = SimpleGasFlow()
    m.unit = CO2Electrolyzer(
        liquid_property_package=m.properties,
        gas_property_package=m.gas,
        cathode_gas_property_package=m.h2,
        co2_property_package=m.co2,
    )
    assert m.unit.outlet_cathode_gas_state.params is m.h2
    assert m.unit.outlet_anode_gas_state.params is m.gas
    assert m.unit.inlet_co2_state.params is m.co2


@pytest.mark.unit
def test_per_port_gas_packages_replace_the_shared_gas_package():
    m = dummy_time_block(3)
    m.h2 = SimpleGasFlow()
    m.o2 = SimpleGasFlow()
    m.unit = WaterElectrolyzer(
        liquid_property_package=m.properties,
        cathode_gas_property_package=m.h2,
        anode_gas_property_package=m.o2,
    )
    assert m.unit.outlet_cathode_gas_state.params is m.h2
    assert m.unit.outlet_anode_gas_state.params is m.o2


@pytest.mark.unit
def test_rejects_missing_co2_property_package():
    m = dummy_time_block(3)
    m.gas = SimpleGasFlow()
    with pytest.raises(FlexConfigError, match="co2_property_package"):
        m.unit = CO2Electrolyzer(
            liquid_property_package=m.properties,
            cathode_gas_property_package=m.gas,
            anode_gas_property_package=m.gas,
        )


@pytest.mark.unit
def test_electrolyzers_are_in_the_unit_model_registry():
    from flexops import unit_models

    for name in ("Electrolyzer", "WaterElectrolyzer", "CO2Electrolyzer"):
        assert name in unit_models.__all__


# -- surrogate swap ----------------------------------------------------------


@pytest.mark.component
def test_power_relation_swaps_to_a_fitted_polarization_surrogate():
    m, unit = _water()
    unit.inlet_water_state.flow_vol_phase[:, "Liq"].set_value(0.01)
    unit.swap_relation(
        "power_electrical_relation",
        MultilinearSurrogate(
            {
                "input_variables": {"current": "A"},
                "output_variables": {"power_electrical": "kW"},
                "coefficients": {"intercept": 1.0, "current": 0.02},
            }
        ),
    )
    _solve_with_inputs_fixed(m, unit)
    assert pyo.value(unit.power_electrical[1]) == pytest.approx(1.0 + 0.02 * CURRENT)
