"""Harness-driven and hand tests for the Electrolyzer family."""

import dataclasses
import logging

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
    OXYGEN,
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
ELECTRODE_AREA = 500.0  # cm^2
CURRENT_DENSITY = 2.0  # A/cm^2
CURRENT = CURRENT_DENSITY * ELECTRODE_AREA  # A
ELECTRONS = N_CELLS * CURRENT / FARADAY  # mol e-/s

CO_PRODUCT = ElectrochemicalProduct(
    name="CO",
    electrons=2,
    faradaic_efficiency=0.9,
    phase=ProductPhase.GAS,
    molar_mass=28.01 * pyunits.g / pyunits.mol,
    reactants={"CO2": 1.0},
)
H2_SIDE_PRODUCT = ElectrochemicalProduct(
    name="H2",
    electrons=2,
    faradaic_efficiency=0.1,
    phase=ProductPhase.GAS,
    molar_mass=2.016 * pyunits.g / pyunits.mol,
    reactants={"H2O": 1.0},
)
FORMATE_PRODUCT = ElectrochemicalProduct(
    name="HCOOH",
    electrons=2,
    faradaic_efficiency=0.5,
    phase=ProductPhase.LIQUID,
    molar_mass=46.03 * pyunits.g / pyunits.mol,
    reactants={"CO2": 1.0, "H2O": 1.0},
)
CL2_PRODUCT = ElectrochemicalProduct(
    name="Cl2",
    electrons=2,
    faradaic_efficiency=1.0,
    phase=ProductPhase.GAS,
    molar_mass=70.90 * pyunits.g / pyunits.mol,
    reactants={"NaCl": 2.0},
)
NACLO_PRODUCT = ElectrochemicalProduct(
    name="NaClO",
    electrons=2,
    faradaic_efficiency=0.5,
    phase=ProductPhase.LIQUID,
    molar_mass=74.44 * pyunits.g / pyunits.mol,
    reactants={"NaCl": 1.0},
)

_COMMON = dict(
    n_cells=N_CELLS,
    electrode_area=ELECTRODE_AREA * pyunits.cm**2,
    operating_temperature=T_OP * pyunits.K,
    operating_pressure=P_OP * pyunits.Pa,
    separator_volume=10 * pyunits.m**3,
    initial_liquid_volume=5 * pyunits.m**3,
)


def _water(n: int = 3, **kwargs):
    """Build a WaterElectrolyzer on an ``n``-point time block, j set."""
    m = dummy_time_block(n)
    m.gas = SimpleGasFlow()
    m.unit = WaterElectrolyzer(
        liquid_property_package=m.properties,
        gas_property_package=m.gas,
        **{**_COMMON, **kwargs},
    )
    m.unit.current_density[:].set_value(CURRENT_DENSITY)
    return m, m.unit


def _co2(n: int = 3, **kwargs):
    """Build a CO2Electrolyzer on an ``n``-point time block, j set."""
    m = dummy_time_block(n)
    m.gas = SimpleGasFlow()
    m.unit = CO2Electrolyzer(
        liquid_property_package=m.properties,
        gas_property_package=m.gas,
        **{**_COMMON, **kwargs},
    )
    m.unit.current_density[:].set_value(CURRENT_DENSITY)
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
    """Fixing current density and make-up water determines every flow and power."""

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
                "molar_mass": 28.01 * pyunits.g / pyunits.mol,
                "reactants": {"CO2": 1.0},
            }
        ],
        balance_product=None,
    )
    assert pyo.value(unit.faradaic_efficiency_CO) == pytest.approx(0.95)


# -- registration ------------------------------------------------------------


@pytest.mark.unit
def test_power_and_faradaic_relations_are_registered_swappable():
    _, unit = _water()
    assert _relations(unit) == {
        "power_electrical_relation",
        "faradaic_relation_H2",
        "faradaic_relation_O2",
    }


@pytest.mark.unit
def test_co2_electrolyzer_registers_a_faradaic_relation_per_non_balance_product():
    _, unit = _co2()
    assert _relations(unit) == {
        "power_electrical_relation",
        "faradaic_relation_CO",
        "faradaic_relation_O2",
    }


@pytest.mark.unit
def test_scalar_coefficients_are_fixed_regressable_parameters():
    _, unit = _co2()
    expected = {
        "cell_voltage",
        "bop_fraction",
        "faradaic_efficiency_CO",
        "single_pass_conversion",
        "co2_crossover_per_electron",
    }
    assert expected <= _parameters(unit)
    for name in expected:
        assert unit.find_component(name).fixed


@pytest.mark.unit
def test_current_density_and_make_up_water_are_inputs_power_is_output():
    _, unit = _water()
    assert {"current_density", "flow_vol_phase"} <= _registered(unit, "input")
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
        unit.production_O2[t].set_value(ELECTRONS / 4)
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
        balance_product=None,
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
def test_balance_product_takes_the_charge_the_other_products_leave():
    _, unit = _co2()
    for t in unit.current:
        unit.production_CO[t].set_value(0.9 * ELECTRONS / 2)
        unit.production_H2[t].set_value(0.1 * ELECTRONS / 2)
    _assert_satisfied(unit.charge_balance_H2)
    assert unit.find_component("faradaic_relation_H2") is None
    assert unit.find_component("faradaic_efficiency_H2") is None


@pytest.mark.unit
def test_rejects_balance_product_not_in_the_cathode_products():
    with pytest.raises(FlexConfigError, match="balance_product"):
        _co2(balance_product="CH4")


@pytest.mark.unit
def test_no_mass_transfer_coefficient_builds_no_transport_limit():
    _, unit = _co2()
    assert unit.find_component("co2_transport_limit") is None
    assert unit.find_component("co2_local_concentration") is None


@pytest.mark.unit
def test_mass_transfer_coefficient_builds_fixed_transport_parameters():
    _, unit = _co2(mass_transfer_coefficient=1e-4 * pyunits.m / pyunits.s)
    assert {"mass_transfer_coefficient", "co2_bulk_concentration"} <= _parameters(unit)
    assert unit.mass_transfer_coefficient.fixed
    assert pyo.value(unit.co2_bulk_concentration) == pytest.approx(34.0)


@pytest.mark.unit
def test_local_co2_reaches_zero_at_the_transport_limit():
    flux = 0.9 * ELECTRONS / 2  # CO2 to CO, no crossover
    k_m = flux / (34.0 * N_CELLS * ELECTRODE_AREA * 1e-4)  # m/s
    _, unit = _co2(mass_transfer_coefficient=k_m)
    for t in unit.current:
        unit.production_CO[t].set_value(flux)
    assert pyo.value(unit.co2_local_concentration[0]) == pytest.approx(0.0, abs=1e-9)
    _assert_satisfied(unit.co2_transport_limit)
    unit.production_CO[0].set_value(flux / 2)
    assert pyo.value(unit.co2_local_concentration[0]) == pytest.approx(17.0)


@pytest.mark.unit
@pytest.mark.parametrize("k_m", [0.0, -1e-4])
def test_rejects_non_positive_mass_transfer_coefficient(k_m):
    with pytest.raises(FlexConfigError, match="mass_transfer_coefficient"):
        _co2(mass_transfer_coefficient=k_m)


@pytest.mark.unit
def test_waste_heat_is_current_times_overpotential_above_thermoneutral():
    _, unit = _water(cell_voltage=2.0 * pyunits.V)
    expected_kw = N_CELLS * CURRENT * (2.0 - 1.48) / 1000.0
    assert pyo.value(unit.waste_heat[0]) == pytest.approx(expected_kw)


# -- current density and ohmic loss ------------------------------------------


@pytest.mark.unit
def test_current_is_current_density_times_electrode_area():
    _, unit = _water()
    assert isinstance(unit.current_density, pyo.Var)
    assert pyo.value(unit.current[0]) == pytest.approx(CURRENT)
    assert unit.current_density[0].ub == pytest.approx(2.0)


@pytest.mark.unit
def test_rated_current_density_bounds_the_operating_variable():
    _, unit = _water(rated_current_density=1.5 * pyunits.A / pyunits.cm**2)
    assert unit.current_density[0].ub == pytest.approx(1.5)


@pytest.mark.unit
def test_no_ohmic_loss_keeps_power_linear_in_current_density():
    _, unit = _water()
    assert unit.config.ohmic_loss is False
    assert unit.find_component("area_specific_resistance") is None
    assert unit.power_electrical_relation[0].body.polynomial_degree() == 1


@pytest.mark.unit
def test_ohmic_loss_adds_asr_times_current_density_to_cell_voltage():
    _, unit = _water(
        cell_voltage=2.0 * pyunits.V,
        ohmic_loss=True,
        area_specific_resistance=0.1 * pyunits.ohm * pyunits.cm**2,
    )
    voltage = 2.0 + 0.1 * CURRENT_DENSITY
    assert unit.area_specific_resistance.fixed
    assert "area_specific_resistance" in _parameters(unit)
    assert pyo.value(unit.operating_voltage[0]) == pytest.approx(voltage)
    assert unit.power_electrical_relation[0].body.polynomial_degree() == 2
    unit.power_electrical[:].set_value(N_CELLS * voltage * CURRENT / 1000.0)
    _assert_satisfied(unit.power_electrical_relation)
    expected_heat = N_CELLS * CURRENT * (voltage - 1.48) / 1000.0
    assert pyo.value(unit.waste_heat[0]) == pytest.approx(expected_heat)


@pytest.mark.unit
def test_ohmic_loss_warns_about_solve_complexity(caplog):
    with caplog.at_level(logging.WARNING, logger="flexops.unit_models.electrolyzer"):
        _water(ohmic_loss=True)
    assert "quadratic" in caplog.text.lower()


# -- dissolved CO2 -----------------------------------------------------------


@pytest.mark.unit
def test_dissolved_co2_fraction_splits_unreacted_co2_out_of_the_cathode_gas():
    _, unit = _co2(single_pass_conversion=0.5, co2_dissolved_fraction=0.3)
    co = 0.9 * ELECTRONS / 2
    h2 = 0.1 * ELECTRONS / 2
    unreacted = co  # conversion 0.5, no crossover
    for t in unit.current:
        unit.production_CO[t].set_value(co)
        unit.production_H2[t].set_value(h2)
        unit.outlet_cathode_gas_state.flow_vol_phase[t, "Vap"].set_value(
            (co + h2 + 0.7 * unreacted) * MOLAR_VOLUME
        )
    assert pyo.value(unit.co2_dissolved[0]) == pytest.approx(0.3 * unreacted)
    _assert_satisfied(unit.cathode_gas_balance)


@pytest.mark.unit
@pytest.mark.parametrize("fraction", [-0.1, 1.1])
def test_rejects_co2_dissolved_fraction_outside_unit_interval(fraction):
    with pytest.raises(FlexConfigError, match="co2_dissolved_fraction"):
        _co2(co2_dissolved_fraction=fraction)


# -- anode products ----------------------------------------------------------


@pytest.mark.unit
def test_default_anode_product_is_oxygen():
    _, unit = _water()
    assert isinstance(unit.production_O2, pyo.Var)
    assert pyo.value(unit.faradaic_efficiency_O2) == pytest.approx(1.0)
    assert "anode_products" not in unit.config


@pytest.mark.unit
def test_generic_electrolyzer_accepts_a_custom_anode_product_table():
    m = dummy_time_block(3)
    m.gas = SimpleGasFlow()
    m.unit = Electrolyzer(
        liquid_property_package=m.properties,
        gas_property_package=m.gas,
        anode_products=[CL2_PRODUCT],
        **_COMMON,
    )
    unit = m.unit
    unit.current_density[:].set_value(CURRENT_DENSITY)
    for t in unit.current:
        unit.production_Cl2[t].set_value(ELECTRONS / 2)
        unit.outlet_anode_gas_state.flow_vol_phase[t, "Vap"].set_value(
            ELECTRONS / 2 * MOLAR_VOLUME
        )
    _assert_satisfied(unit.faradaic_relation_Cl2)
    _assert_satisfied(unit.anode_gas_balance)


@pytest.mark.unit
def test_rejects_anode_faradaic_efficiencies_summing_above_one():
    with pytest.raises(FlexConfigError, match="anode"):
        _co2(anode_products=[CL2_PRODUCT, OXYGEN])


@pytest.mark.unit
def test_liquid_anode_product_leaves_through_liquid_outlet():
    m = dummy_time_block(3)
    m.gas = SimpleGasFlow()
    m.unit = Electrolyzer(
        liquid_property_package=m.properties,
        gas_property_package=m.gas,
        anode_products=[
            dataclasses.replace(OXYGEN, faradaic_efficiency=0.5),
            NACLO_PRODUCT,
        ],
        has_liquid_outlet=True,
        **_COMMON,
    )
    unit = m.unit
    unit.current_density[:].set_value(CURRENT_DENSITY)
    hypochlorite = 0.5 * ELECTRONS / 2
    for t in unit.current:
        unit.production_O2[t].set_value(0.5 * ELECTRONS / 4)
        unit.production_NaClO[t].set_value(hypochlorite)
        unit.outlet_anode_gas_state.flow_vol_phase[t, "Vap"].set_value(
            0.5 * ELECTRONS / 4 * MOLAR_VOLUME
        )
        unit.outlet_liquid_state.flow_vol_phase[t, "Liq"].set_value(
            hypochlorite * 0.07444 / 1000.0 * 3600
        )
    _assert_satisfied(unit.anode_gas_balance)
    _assert_satisfied(unit.liquid_outlet_balance)


@pytest.mark.unit
def test_rejects_liquid_anode_product_without_liquid_outlet():
    with pytest.raises(FlexConfigError, match="has_liquid_outlet"):
        _co2(anode_products=[NACLO_PRODUCT])


@pytest.mark.unit
def test_builds_a_consumption_expression_per_reactant():
    _, unit = _co2(anode_products=[CL2_PRODUCT])
    for t in unit.current:
        unit.production_CO[t].set_value(0.9 * ELECTRONS / 2)
        unit.production_Cl2[t].set_value(ELECTRONS / 2)
    assert pyo.value(unit.consumption_NaCl[0]) == pytest.approx(ELECTRONS)
    assert pyo.value(unit.consumption_CO2[0]) == pytest.approx(0.9 * ELECTRONS / 2)


@pytest.mark.unit
def test_negative_stoichiometry_produces_the_reactant():
    m = dummy_time_block(3)
    m.gas = SimpleGasFlow()
    m.unit = Electrolyzer(
        liquid_property_package=m.properties,
        gas_property_package=m.gas,
        products=[
            dataclasses.replace(
                H2_SIDE_PRODUCT, faradaic_efficiency=1.0, reactants={"H2O": 2.0}
            )
        ],
        anode_products=[dataclasses.replace(OXYGEN, reactants={"H2O": -2.0})],
        **_COMMON,
    )
    unit = m.unit
    for t in unit.current:
        unit.production_H2[t].set_value(ELECTRONS / 2)
        unit.production_O2[t].set_value(ELECTRONS / 4)
    # 2 H2O per H2 at the cathode, 2 H2O back per O2 at the anode: 1 per H2 net
    assert pyo.value(unit.consumption_H2O[0]) == pytest.approx(ELECTRONS / 2)
    assert pyo.value(unit.water_consumption[0]) == pytest.approx(
        ELECTRONS / 2 * WATER_VOLUME
    )


@pytest.mark.unit
def test_rejects_reactant_name_that_is_not_an_identifier():
    bad = dataclasses.replace(CO_PRODUCT, reactants={"CO-2": 1.0})
    with pytest.raises(FlexConfigError, match="reactant"):
        _co2(products=[bad, H2_SIDE_PRODUCT])


@pytest.mark.unit
def test_rejects_product_names_shared_between_cathode_and_anode():
    with pytest.raises(FlexConfigError, match="unique"):
        _co2(anode_products=[dataclasses.replace(OXYGEN, name="CO")])


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


@pytest.mark.component
def test_swapped_faradaic_relation_leaves_the_remaining_charge_to_hydrogen():
    m, unit = _co2()
    unit.inlet_water_state.flow_vol_phase[:, "Liq"].set_value(0.01)
    co = 0.5 * ELECTRONS / 2
    unit.swap_relation(
        "faradaic_relation_CO",
        MultilinearSurrogate(
            {
                "input_variables": {"current": "A"},
                "output_variables": {"production_CO": "mol/s"},
                "coefficients": {"intercept": 0.0, "current": co / CURRENT},
            }
        ),
    )
    _solve_with_inputs_fixed(m, unit)
    assert pyo.value(unit.production_CO[1]) == pytest.approx(co)
    assert pyo.value(unit.production_H2[1]) == pytest.approx(ELECTRONS / 2 - co)
