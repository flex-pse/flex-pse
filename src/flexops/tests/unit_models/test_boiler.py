"""Boiler(OpsBlockData): fired and heat-recovery steam raising with a thermal lag."""

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits
from pyomo.network import Arc, Port
from pyomo.opt import assert_optimal_termination

from flexcore import nomenclature as nm
from flexcore.config.schema import (
    SurrogateSpec,
    SurrogateType,
    UnitCommitmentConfig,
    UnitConfig,
)
from flexcore.exceptions import FlexConfigError, FlexSolverError
from flexcore.solvers import get_solver
from flexops.properties.simple_aqueous import SimpleAqueousFlow
from flexops.properties.simple_gas import SimpleGasFlow
from flexops.surrogates import MultilinearSurrogate
from flexops.testing import UnitModelTestHarness, dummy_gas_time_block
from flexops.unit_models import Boiler, Combustor
from flexops.unit_models.powergeneration.boiler import BoilerType

_KWH_PER_M3 = pyunits.kWh / pyunits.m**3
_NATURAL_GAS_HV = 10.5
_BIOGAS_HV = 6.0
_EFFICIENCY = 0.9
_TAU_S = 300.0
_DT_S = 900.0
_STEAM_HEAT_CONTENT = 3.4
_DENSITY_RATIO = 0.0054
_GAS_HEAT_CONTENT = 0.15
_NO_UC = UnitCommitmentConfig(status=False)


def _model(n: int = 3):
    """Build an ``n``-point time block with gas, water, and steam packages."""
    m = dummy_gas_time_block(n)
    m.water = SimpleAqueousFlow()
    m.steam = SimpleGasFlow()
    return m


def _boiler(n: int = 3, **kwargs):
    """Build a Boiler on :func:`_model`; defaults to a natural-gas fired boiler."""
    m = _model(n)
    options = {
        "feedwater_property_package": m.water,
        "steam_property_package": m.steam,
        "unit_commitment": _NO_UC,
    }
    if kwargs.get("boiler_type") is not BoilerType.HEAT_RECOVERY:
        options["utility_fuel_source"] = "natural_gas"
        options["heating_values"] = {"natural_gas": _NATURAL_GAS_HV * _KWH_PER_M3}
    options.update(kwargs)
    m.unit = Boiler(**options)
    return m, m.unit


def _hrsg(n: int = 3, **kwargs):
    """Build an unfired heat-recovery Boiler on :func:`_model`."""
    m = _model(n)
    return _boiler(
        n,
        boiler_type=BoilerType.HEAT_RECOVERY,
        hot_gas_property_package=m.properties,
        **kwargs,
    )


def _solve(m) -> None:
    """Solve ``m`` and assert optimality, skipping when no solver is installed."""
    try:
        solver = get_solver(model=m)
    except FlexSolverError as exc:
        pytest.skip(str(exc))
    assert_optimal_termination(solver.solve(m))


def _fix_dispatch(unit, heat_output, initial_heat_output) -> None:
    """Fix heat output per step, zero dissipation, and set the initial heat output."""
    unit.initial_heat_output.set_value(initial_heat_output)
    for t, value in enumerate(heat_output):
        unit.heat_output[t].fix(value)
        unit.heat_dissipated[t].fix(0.0)


def _firing_rate(unit, t) -> float:
    """Return the solved fuel heat input at ``t`` (kW)."""
    return pyo.value(unit.utility_flow_natural_gas[t]) * _NATURAL_GAS_HV


def _port_names(unit) -> set[str]:
    """Return the local names of the unit's ports."""
    return {p.local_name for p in unit.component_objects(Port, descend_into=False)}


def _relation_names(unit) -> set[str]:
    """Return the names of the unit's registered relations."""
    return {record.name for record in unit._io_registry.relations}


class TestBoilerFired(UnitModelTestHarness):
    """Natural-gas fired boiler held at a 6000 kW steady state.

    Fuel is fixed at 6000/0.9/10.5 m^3/hr, so heat_absorbed is 6000 kW and,
    starting from 6000 kW with no dissipation, heat_output stays at 6000 kW.
    """

    expected_dof = 0
    expected_solution = {
        "heat_output[2]": 6000.0,
        "flow_out[2]": 6000.0 / _STEAM_HEAT_CONTENT,
        "flow_feedwater[2]": 6000.0 / _STEAM_HEAT_CONTENT * _DENSITY_RATIO,
        "power_thermal[2]": -6000.0,
    }

    def configure(self):
        m, unit = _boiler(3)
        unit.initial_heat_output.set_value(6000.0)
        for t in m.time_block.time_index:
            unit.utility_flow_natural_gas[t].set_value(
                6000.0 / _EFFICIENCY / _NATURAL_GAS_HV
            )
            unit.heat_dissipated[t].fix(0.0)
        return m, unit


class TestBoilerHeatRecovery(UnitModelTestHarness):
    """Unfired HRSG on 40000 m^3/hr of hot gas: 0.15 * 40000 = 6000 kW."""

    expected_dof = 0
    expected_solution = {
        "heat_output[2]": 6000.0,
        "heat_recovered[2]": 6000.0,
        "outlet_gas_state.flow_vol_phase[2,Vap]": 40000.0,
        "outlet_gas_state.temperature[2]": 423.0,
    }

    def configure(self):
        m, unit = _hrsg(3)
        unit.initial_heat_output.set_value(6000.0)
        for t in m.time_block.time_index:
            unit.flow_hot_gas[t].set_value(40000.0)
            unit.heat_dissipated[t].fix(0.0)
        return m, unit


# -- build, ports, registration -------------------------------------------


@pytest.mark.unit
def test_fired_boiler_has_feedwater_and_steam_ports_only_with_utility_fuel():
    """A utility-only fired boiler has no fuel or gas ports."""
    _, unit = _boiler()
    assert _port_names(unit) == {"inlet_feedwater", "outlet_steam"}


@pytest.mark.unit
def test_fired_boiler_builds_one_port_per_fuel_inlet():
    """Fuel inlets become ``inlet_<name>`` ports alongside utility fuel."""
    m = _model()
    _, unit = _boiler(
        fuel_property_package=m.properties,
        fuel_inlet_names=("biogas",),
        utility_fuel_source="natural_gas",
        heating_values={
            "biogas": _BIOGAS_HV * _KWH_PER_M3,
            "natural_gas": _NATURAL_GAS_HV * _KWH_PER_M3,
        },
    )
    assert _port_names(unit) == {"inlet_biogas", "inlet_feedwater", "outlet_steam"}


@pytest.mark.unit
def test_fired_boiler_builds_with_inlet_fuel_only():
    """A fired boiler may take all its fuel through an inlet port."""
    m = _model()
    _, unit = _boiler(
        fuel_property_package=m.properties,
        fuel_inlet_names=("biogas",),
        utility_fuel_source=None,
        heating_values={"biogas": _BIOGAS_HV * _KWH_PER_M3},
    )
    assert unit._io_registry.fuel == []
    assert unit.find_component("flow_in_biogas") is not None


@pytest.mark.unit
def test_heat_recovery_boiler_has_hot_gas_ports():
    """An HRSG adds a hot-gas inlet and a cooled-gas outlet."""
    _, unit = _hrsg()
    assert _port_names(unit) == {
        "inlet_hot_gas",
        "outlet_gas",
        "inlet_feedwater",
        "outlet_steam",
    }


@pytest.mark.unit
def test_registered_relations_per_boiler_type():
    """Only an HRSG registers ``heat_recovered_relation``."""
    shared = {
        "heat_absorbed_relation",
        "steam_flow_relation",
        "steam_pressure_relation",
        "power_electrical_relation",
    }
    _, fired = _boiler()
    _, hrsg = _hrsg()
    assert _relation_names(fired) == shared
    assert _relation_names(hrsg) == shared | {"heat_recovered_relation"}


@pytest.mark.unit
def test_regressable_parameters_are_registered():
    """Every physical scalar is a registered, regressable process parameter."""
    _, unit = _boiler()
    regressable = {r.name for r in unit._io_registry.parameters if r.regressable}
    assert {
        "heating_value_natural_gas",
        "efficiency",
        "no_load_loss",
        "max_firing_rate",
        "time_constant",
        "steam_heat_content",
        "steam_to_water_density_ratio",
        "steam_pressure",
        "steam_temperature",
        "energy_intensity",
    } <= regressable
    not_regressable = {
        r.name for r in unit._io_registry.parameters if not r.regressable
    }
    assert "initial_heat_output" in not_regressable


@pytest.mark.unit
def test_utility_fuel_is_registered_for_costing():
    """Each utility fuel registers its volumetric flow under its own name."""
    _, unit = _boiler()
    records = unit._io_registry.fuel
    assert [r.fuel_name for r in records] == ["natural_gas"]
    assert records[0].var is unit.utility_flow_natural_gas


@pytest.mark.unit
def test_power_records_are_thermal_export_and_electrical_aux():
    """Steam is a thermal export carrying its temperature; aux draw is electrical."""
    _, unit = _boiler(steam_temperature=813 * pyunits.K)
    by_kind = {r.kind: r for r in unit._io_registry.power}
    assert set(by_kind) == {nm.PowerKind.THERMAL, nm.PowerKind.ELECTRICAL}
    assert pyo.value(by_kind[nm.PowerKind.THERMAL].temperature) == pytest.approx(813)


@pytest.mark.unit
def test_status_exists_only_when_unit_commitment_is_on():
    """``status`` gates heat output when UC status is enabled."""
    _, without = _boiler()
    _, with_status = _boiler(unit_commitment=UnitCommitmentConfig(status=True))
    assert without.find_component("status") is None
    assert with_status.find_component("status") is not None
    assert with_status.find_component("status_min_link") is not None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("relation_name", "inputs", "output"),
    [
        (
            "heat_absorbed_relation",
            {"utility_flow_natural_gas": "m^3/hr"},
            {"heat_absorbed": "kW"},
        ),
        ("steam_flow_relation", {"heat_output": "kW"}, {"flow_out": "m^3/hr"}),
        ("steam_pressure_relation", {"heat_output": "kW"}, {"pressure_out": "Pa"}),
    ],
)
def test_swap_relation_deactivates_without_deleting(relation_name, inputs, output):
    """A multilinear swap deactivates the original relation in place."""
    _, unit = _boiler()
    original = unit.find_component(relation_name)
    surrogate = MultilinearSurrogate(
        {
            "input_variables": inputs,
            "output_variables": output,
            "coefficients": {"intercept": 1.0, next(iter(inputs)): 2.0},
        }
    )
    unit.swap_relation(relation_name, surrogate)
    assert unit.find_component(relation_name) is original
    assert not original.active


@pytest.mark.unit
def test_config_surrogate_swaps_power_electrical_relation():
    """A config-file surrogate replaces the aux electrical relation."""
    spec = SurrogateSpec(
        surrogate_type=SurrogateType.MULTILINEAR,
        data={
            "input_variables": {"flow_out": "m^3/hr"},
            "output_variables": {"power_electrical": "kW"},
            "coefficients": {"intercept": 5.0, "flow_out": 0.01},
        },
    )
    _, unit = _boiler(
        flexops_config=UnitConfig(unit_model_class="Boiler", surrogate=spec)
    )
    assert not unit.power_electrical_relation.active
    assert unit.heat_absorbed_relation.active


@pytest.mark.unit
def test_firing_rate_limit_caps_overfiring():
    """Fuel heat input above ``max_firing_rate`` violates the limit."""
    _, unit = _boiler(max_firing_rate=9000 * pyunits.kW)
    unit.utility_flow_natural_gas[1].set_value(9629.63 / _NATURAL_GAS_HV)
    assert pyo.value(unit.firing_rate_limit[1].body) > pyo.value(
        unit.firing_rate_limit[1].upper
    )


# -- config validation ----------------------------------------------------


@pytest.mark.unit
def test_rejects_plain_string_boiler_type():
    """``boiler_type`` must be a ``BoilerType`` member, not its value string."""
    with pytest.raises(FlexConfigError) as excinfo:
        _boiler(boiler_type="fired")
    assert excinfo.value.field == "boiler_type"


@pytest.mark.unit
def test_rejects_fired_boiler_without_fuel():
    """A fired boiler needs at least one fuel source."""
    with pytest.raises(FlexConfigError):
        _boiler(utility_fuel_source=None, heating_values=None)


@pytest.mark.unit
@pytest.mark.parametrize(
    "option",
    [{"gas_heat_content": 0.2 * _KWH_PER_M3}, {"stack_temperature": 400 * pyunits.K}],
)
def test_rejects_heat_recovery_options_on_a_fired_boiler(option):
    """Heat-recovery-only options are rejected under ``BoilerType.FIRED``."""
    with pytest.raises(FlexConfigError) as excinfo:
        _boiler(**option)
    assert excinfo.value.field == next(iter(option))


@pytest.mark.unit
@pytest.mark.parametrize(
    "option", [{"efficiency": 0.8}, {"max_firing_rate": 5000 * pyunits.kW}]
)
def test_rejects_firing_options_without_fuel(option):
    """Firing options are rejected on an unfired HRSG."""
    with pytest.raises(FlexConfigError) as excinfo:
        _hrsg(**option)
    assert excinfo.value.field == next(iter(option))


@pytest.mark.unit
def test_rejects_overlapping_fuel_names():
    """A fuel may not be both an inlet and a utility."""
    m = _model()
    with pytest.raises(FlexConfigError):
        _boiler(
            fuel_property_package=m.properties,
            fuel_inlet_names=("natural_gas",),
            utility_fuel_source="natural_gas",
        )


@pytest.mark.unit
@pytest.mark.parametrize("name", ["feedwater", "hot_gas"])
def test_rejects_reserved_fuel_names(name):
    """Fuel names may not collide with the boiler's own port names."""
    m = _model()
    with pytest.raises(FlexConfigError) as excinfo:
        _boiler(
            fuel_property_package=m.properties,
            fuel_inlet_names=(name,),
            utility_fuel_source=None,
            heating_values={name: _BIOGAS_HV * _KWH_PER_M3},
        )
    assert excinfo.value.field == "fuel_inlet_names"


@pytest.mark.unit
@pytest.mark.parametrize(
    "heating_values",
    [{}, {"natural_gas": 10.5 * _KWH_PER_M3, "coal": 8.0 * _KWH_PER_M3}],
)
def test_rejects_missing_or_unknown_heating_values(heating_values):
    """Every fuel source needs a heating value, and only fuel sources may have one."""
    with pytest.raises(FlexConfigError) as excinfo:
        _boiler(heating_values=heating_values)
    assert excinfo.value.field == "heating_values"


@pytest.mark.unit
@pytest.mark.parametrize(
    "option",
    [
        {"efficiency": 1.2},
        {"time_constant": -1 * pyunits.s},
        {"min_heat_output": 12000 * pyunits.kW},
    ],
)
def test_rejects_out_of_range_options(option):
    """Efficiency outside (0, 1], a negative lag, or min > max raises."""
    with pytest.raises(FlexConfigError):
        _boiler(**option)


@pytest.mark.unit
def test_rejects_missing_fuel_property_package_with_fuel_inlets():
    """Fuel inlet ports need a fuel property package."""
    with pytest.raises(FlexConfigError) as excinfo:
        _boiler(
            fuel_inlet_names=("biogas",),
            utility_fuel_source=None,
            heating_values={"biogas": _BIOGAS_HV * _KWH_PER_M3},
        )
    assert excinfo.value.field == "fuel_property_package"


# -- solved behaviour -----------------------------------------------------


@pytest.mark.component
def test_fired_steady_state_fuel_matches_heat_over_efficiency():
    """At steady state, fuel heat input is (Q + no_load_loss) / efficiency."""
    m, unit = _boiler(no_load_loss=200 * pyunits.kW)
    _fix_dispatch(unit, [6000.0] * 3, 6000.0)
    _solve(m)
    for t in range(3):
        assert _firing_rate(unit, t) == pytest.approx((6000.0 + 200.0) / _EFFICIENCY)


@pytest.mark.component
def test_step_up_requires_overfiring_by_the_lag():
    """Raising load needs an extra tau * dQ / dt of absorbed heat at the step."""
    m, unit = _boiler()
    _fix_dispatch(unit, [6000.0, 8000.0, 8000.0], 6000.0)
    _solve(m)
    overfire = _TAU_S * (8000.0 - 6000.0) / _DT_S
    assert _firing_rate(unit, 0) == pytest.approx(6000.0 / _EFFICIENCY)
    assert _firing_rate(unit, 1) == pytest.approx((8000.0 + overfire) / _EFFICIENCY)
    assert _firing_rate(unit, 2) == pytest.approx(8000.0 / _EFFICIENCY)


@pytest.mark.component
def test_zero_time_constant_is_static():
    """With no lag, a step needs no overfiring."""
    m, unit = _boiler(time_constant=0 * pyunits.s)
    _fix_dispatch(unit, [6000.0, 8000.0, 8000.0], 6000.0)
    _solve(m)
    assert _firing_rate(unit, 1) == pytest.approx(8000.0 / _EFFICIENCY)


@pytest.mark.component
def test_shutdown_is_feasible_through_heat_dissipation():
    """Dropping to zero output dumps stored heat rather than going infeasible."""
    m, unit = _boiler(
        min_heat_output=0 * pyunits.kW,
        unit_commitment=UnitCommitmentConfig(status=True),
    )
    unit.initial_heat_output.set_value(8000.0)
    unit.heat_output[0].fix(0.0)
    unit.status[0].fix(0)
    for t in (1, 2):
        unit.heat_output[t].fix(0.0)
    m.objective = pyo.Objective(expr=sum(unit.utility_flow_natural_gas.values()))
    _solve(m)
    assert _firing_rate(unit, 0) == pytest.approx(0.0, abs=1e-6)
    assert pyo.value(unit.heat_dissipated[0]) == pytest.approx(_TAU_S * 8000.0 / _DT_S)


@pytest.mark.component
def test_duct_fired_hrsg_adds_firing_to_recovered_heat():
    """Duct firing adds efficiency * fuel heat to the recovered heat."""
    m, unit = _hrsg(
        utility_fuel_source="natural_gas",
        heating_values={"natural_gas": _NATURAL_GAS_HV * _KWH_PER_M3},
    )
    _fix_dispatch(unit, [8000.0] * 3, 8000.0)
    for t in range(3):
        unit.flow_hot_gas[t].fix(40000.0)
    _solve(m)
    recovered = _GAS_HEAT_CONTENT * 40000.0
    for t in range(3):
        assert pyo.value(unit.heat_recovered[t]) == pytest.approx(recovered)
        assert _firing_rate(unit, t) == pytest.approx(
            (8000.0 - recovered) / _EFFICIENCY
        )


@pytest.mark.component
def test_hrsg_fed_by_a_combustor_arc_solves():
    """A Combustor's flue gas drives an HRSG through an arc."""
    m = _model()
    m.chp = Combustor(
        property_package=m.properties,
        inlet_names=("fuel",),
        flue_gas_temperature=750 * pyunits.K,
    )
    m.unit = Boiler(
        boiler_type=BoilerType.HEAT_RECOVERY,
        hot_gas_property_package=m.properties,
        feedwater_property_package=m.water,
        steam_property_package=m.steam,
        unit_commitment=_NO_UC,
        min_heat_output=0 * pyunits.kW,
    )
    m.flue = Arc(source=m.chp.outlet, destination=m.unit.inlet_hot_gas)
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)
    m.unit.initial_heat_output.set_value(0.0)
    for t in m.time_block.time_index:
        m.chp.flow_in_fuel[t].fix(2000.0)
        m.chp.inlet_fuel_state.pressure[t].fix(101325.0)
        m.chp.inlet_fuel_state.temperature[t].fix(300.0)
        m.unit.heat_dissipated[t].fix(0.0)
    _solve(m)
    flue = (1 + 9.5) * 2000.0
    assert pyo.value(m.unit.flow_hot_gas[2]) == pytest.approx(flue)
    assert pyo.value(m.unit.outlet_gas_state.flow_vol_phase[2, "Vap"]) == (
        pytest.approx(flue)
    )


@pytest.mark.component
def test_steam_outlet_and_aux_power_values():
    """Outlet pressure/temperature follow their parameters; aux scales with steam."""
    m, unit = _boiler(aux_energy_intensity=0.02 * _KWH_PER_M3)
    _fix_dispatch(unit, [6000.0] * 3, 6000.0)
    _solve(m)
    flow = 6000.0 / _STEAM_HEAT_CONTENT
    assert pyo.value(unit.pressure_out[0]) == pytest.approx(10e5)
    assert pyo.value(unit.outlet_steam_state.temperature[0]) == pytest.approx(453.0)
    assert pyo.value(unit.power_electrical[0]) == pytest.approx(0.02 * flow)


@pytest.mark.component
@pytest.mark.parametrize("hrsg", [False, True])
def test_defaults_are_feasible_with_status(hrsg):
    """Both types solve from a cold start with UC status on."""
    build = _hrsg if hrsg else _boiler
    m, unit = build(unit_commitment=UnitCommitmentConfig(status=True))
    if hrsg:
        for t in m.time_block.time_index:
            unit.flow_hot_gas[t].fix(40000.0)
    m.objective = pyo.Objective(expr=sum(unit.heat_output.values()), sense=pyo.maximize)
    _solve(m)


@pytest.mark.component
def test_high_pressure_parameter_set_solves():
    """A 100-bar, 813 K boiler is just a parameter set."""
    m, unit = _boiler(
        steam_pressure=100 * pyunits.bar,
        steam_temperature=813 * pyunits.K,
        steam_heat_content=20.4 * _KWH_PER_M3,
        steam_to_water_density_ratio=0.036,
    )
    _fix_dispatch(unit, [6000.0] * 3, 6000.0)
    _solve(m)
    assert pyo.value(unit.pressure_out[0]) == pytest.approx(100e5)
    assert pyo.value(unit.flow_out[0]) == pytest.approx(6000.0 / 20.4)
    assert pyo.value(unit.flow_feedwater[0]) == pytest.approx(6000.0 / 20.4 * 0.036)
