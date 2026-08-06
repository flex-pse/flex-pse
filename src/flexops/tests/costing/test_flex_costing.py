"""Unit-tier tests for FlexCosting: opex/capex structure, fuels, scalar costs.

These tests exercise the wrapper — indexed per-carrier power aggregation, the
``opex`` block (electricity + fuel + fixed + scalar operating cost), the empty
``capex`` placeholder, annualization, the capex-in-objective-only-in-design-mode
rule, ``report_cost``'s categorized breakdown, the DR container no-op, modes, and
the construction-order invariant. None of them invoke a solver (that is
``test_load_shifting_component.py``); derived Vars are propagated through their
defining equality constraints via :func:`_propagate`. The tariff *math* itself is
EECO.
"""

from pathlib import Path

import numpy as np
import pyomo.environ as pyo
import pytest
from pyomo.core.base.units_container import InconsistentUnitsError, UnitsError
from pyomo.environ import units as pyunits
from pyomo.network import Arc
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.check_units import assert_units_consistent, assert_units_equivalent

import flexops as fo
from flexcore import nomenclature as nm
from flexcore.exceptions import FlexConfigError, FlexDataError
from flexcore.solvers import ProblemClass, classify
from flexops.core.registration import FuelUsageRecord, IORegistry, PowerRecord
from flexops.core.time_block import TimeBlock
from flexops.costing import (
    CapitalCostBreakdown,
    CostReport,
    FlexCosting,
    OperatingCostBreakdown,
    currency_units,
    evaluate_fuel_cost,
    load_tariff,
    merge_tariffs,
    monthly_scale_factor,
)
from flexops.properties.simple_aqueous import SimpleAqueousFlow
from flexops.unit_models import Pump, Tank

_FIXTURES = Path(__file__).parent.parent / "fixtures"
_TARIFF_JSON = _FIXTURES / "tariff_tou_demo.json"
_DR_JSON = _FIXTURES / "dr_events_demo.json"


def _two_utility_tariff():
    """A flat (no-tier, no-demand) tariff with one electric and one gas charge."""
    records = [
        {
            "utility": "electric",
            "type": "energy",
            "name": "allday",
            "month_start": 1,
            "month_end": 12,
            "weekday_start": 0,
            "weekday_end": 6,
            "hour_start": 0,
            "hour_end": 24,
            "basic_charge_limit (metric)": 0,
            "charge (metric)": 0.10,
            "units": "$/kWh",
        },
        {
            "utility": "gas",
            "type": "energy",
            "name": "allday",
            "month_start": 1,
            "month_end": 12,
            "weekday_start": 0,
            "weekday_end": 6,
            "hour_start": 0,
            "hour_end": 24,
            "basic_charge_limit (metric)": 0,
            "charge (metric)": 0.50,
            "units": "$/m3",
        },
    ]
    return load_tariff(records)


def _time_model() -> pyo.ConcreteModel:
    """A model with a 24-hour, hourly TimeBlock over 2025-07-08 + properties."""
    m = pyo.ConcreteModel()
    m.time_block = TimeBlock(
        start_date="2025-07-08", end_date="2025-07-09", time_step=1 * pyunits.hr
    )
    m.properties = SimpleAqueousFlow()
    return m


def _pump_tank_costing(
    *,
    costing_first: bool = True,
    pump_first: bool = True,
    fixed_operating_cost: float = 0.0,
    dr_event_file=None,
    tariff=None,
    run_cost_process: bool = True,
    energy_prices=None,
    no_tariff: bool = False,
    **costing_kwargs,
) -> pyo.ConcreteModel:
    """Build a Pump -> Arc -> Tank system with a FlexCosting block.

    ``costing_first`` / ``pump_first`` permute the component-creation order (the
    construction-order invariant). When costing is created after the units they
    are built with ``costing_package=None`` (aggregation pulls from the model, so
    the association is not required). ``tariff`` overrides the default electric
    demo tariff with a pre-loaded rate_data object; ``no_tariff=True`` passes no
    tariff at all (pricing then comes from ``energy_prices``). ``energy_prices``
    may be the mapping itself, or a callable taking the model and returning it, so
    a price that is a Pyomo component can be attached to this model first. Extra
    ``costing_kwargs`` go straight to the FlexCosting constructor.
    """
    m = _time_model()

    def add_costing() -> None:
        kwargs = dict(
            time_block=m.time_block,
            fixed_operating_cost=fixed_operating_cost,
            dr_event_file=dr_event_file,
            **costing_kwargs,
        )
        if energy_prices is not None:
            kwargs["energy_prices"] = (
                energy_prices(m) if callable(energy_prices) else energy_prices
            )
        if tariff is not None:
            kwargs["tariff"] = tariff
        elif not no_tariff:
            kwargs["tariff_file"] = str(_TARIFF_JSON)
        m.costing = FlexCosting(**kwargs)

    def add_units() -> None:
        cp = getattr(m, "costing", None)
        pump = Pump(
            property_package=m.properties,
            energy_intensity=0.5 * pyunits.kWh / pyunits.m**3,
            costing_package=cp,
        )
        tank = Tank(
            property_package=m.properties,
            max_volume=1000 * pyunits.m**3,
            initial_volume=200 * pyunits.m**3,
        )
        if pump_first:
            m.pump, m.tank = pump, tank
        else:
            m.tank, m.pump = tank, pump
        m.arc = Arc(source=m.pump.outlet, destination=m.tank.inlet)
        pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    if costing_first:
        add_costing()
        add_units()
    else:
        add_units()
        add_costing()

    if run_cost_process:
        m.costing.cost_process()
    return m


def _set_power(m: pyo.ConcreteModel, profile: dict[int, float]) -> None:
    """Set the pump's ``power_electrical`` values directly (no solve)."""
    for t, val in profile.items():
        m.pump.power_electrical[t].set_value(val)


def _propagate(costing, passes: int = 8) -> None:
    """Solve every ``var == expr`` defining constraint for its Var, no solver.

    Walks the costing block (and its sub-blocks) for each Var that has a sibling
    constraint named ``eq_<var_local_name>`` and computes the Var from it via
    :func:`calculate_variable_from_constraint`. Repeated ``passes`` propagate
    values along the dependency chain regardless of component order. EECO's own
    internal cost Vars carry no ``eq_`` sibling and are left at their init values
    (matching the pre-conversion Expression behavior).
    """
    pairs = []
    for var in costing.component_objects(pyo.Var, descend_into=True, sort=False):
        con = var.parent_block().component(f"eq_{var.local_name}")
        if con is None:
            continue
        for idx in var:
            pairs.append((var[idx], con[idx]))
    for _ in range(passes):
        for v, c in pairs:
            calculate_variable_from_constraint(v, c)


def _add_fuel_usage(
    m,
    fuel_name: str,
    values: dict[int, float],
    *,
    units=pyunits.m**3 / pyunits.hr,
    tag: str = "",
):
    """Attach a bare block carrying a registered volumetric fuel-usage Var, fixed."""
    blk = pyo.Block()
    setattr(m, f"burner_{fuel_name}{tag}", blk)
    var = pyo.Var(m.time_block.time_index, initialize=0.0, units=units)
    blk.add_component(f"{nm.FUEL_USAGE}_{fuel_name}", var)
    blk._io_registry = IORegistry()
    blk._io_registry.fuel.append(
        FuelUsageRecord(
            var=var,
            name=f"{nm.FUEL_USAGE}_{fuel_name}",
            fuel_name=fuel_name,
        )
    )
    for t, val in values.items():
        var[t].set_value(val)
    return var


def _add_thermal_draw(m, tag: str, temperature, values: dict[int, float]):
    """Attach a bare block carrying a registered thermal-duty Var (kW) at T, fixed."""
    blk = pyo.Block()
    setattr(m, f"thermal_{tag}", blk)
    var = pyo.Var(m.time_block.time_index, initialize=0.0, units=pyunits.kW)
    blk.add_component(nm.POWER_THERMAL, var)
    blk._io_registry = IORegistry()
    blk._io_registry.power.append(
        PowerRecord(
            var=var,
            name=nm.POWER_THERMAL,
            kind=nm.PowerKind.THERMAL,
            temperature=temperature,
        )
    )
    for t, val in values.items():
        var[t].set_value(val)
    return var


@pytest.mark.unit
def test_config_exclusivity():
    """At most one tariff source, and some pricing source, or FlexConfigError.

    Both tariff_file and tariff -> error (they are alternatives). Neither, with no
    energy_prices either -> error (nothing prices the model). Neither, but with
    energy_prices -> valid: a flat price needs no tariff.
    """
    m = _time_model()
    with pytest.raises(FlexConfigError, match="tariff"):
        m.costing = FlexCosting(time_block=m.time_block)  # neither, unpriced

    m2 = _time_model()
    with pytest.raises(FlexConfigError, match="tariff"):
        m2.costing = FlexCosting(
            time_block=m2.time_block,
            tariff_file=str(_TARIFF_JSON),
            tariff=load_tariff(_TARIFF_JSON),  # both
        )

    m3 = _time_model()
    m3.costing = FlexCosting(  # neither, but priced -> valid
        time_block=m3.time_block,
        energy_prices={"electrical": 0.12 * currency_units("USD") / pyunits.kWh},
    )
    assert m3.costing.find_component("dr") is not None


@pytest.mark.unit
def test_fo_exports_flexcosting():
    """FlexCosting is reachable as fo.FlexCosting (API-freeze name)."""
    assert fo.FlexCosting is FlexCosting


@pytest.mark.unit
def test_construct_before_units():
    """FlexCosting builds and cost_process runs on a bare TimeBlock model."""
    m = _time_model()
    m.costing = FlexCosting(time_block=m.time_block, tariff_file=str(_TARIFF_JSON))
    m.costing.cost_process()

    assert m.costing.find_component("aggregate_electrical_power") is not None
    assert m.costing.find_component("aggregate_power") is not None
    assert m.costing.find_component("opex") is not None
    assert m.costing.find_component("capex") is not None
    _propagate(m.costing)
    # Empty registry -> the 0*kW placeholder body -> zero everywhere.
    for t in m.time_block.time_index:
        assert pyo.value(m.costing.aggregate_electrical_power[t]) == pytest.approx(0.0)


@pytest.mark.unit
def test_aggregate_electrical_power():
    """aggregate_electrical_power sums the registered units' power_electrical."""
    m = _pump_tank_costing()
    profile = {t: float(t) for t in m.time_block.time_index}
    _set_power(m, profile)
    _propagate(m.costing)
    for t in (0, 5, 16, 23):
        expected = pyo.value(m.pump.power_electrical[t])
        assert pyo.value(m.costing.aggregate_electrical_power[t]) == pytest.approx(
            expected
        )
        assert pyo.value(m.costing.aggregate_power[t, "electrical"]) == pytest.approx(
            expected
        )


@pytest.mark.unit
def test_opex_block_line_items():
    """The opex block exposes electricity/fuel/fixed/scalar and their sum."""
    m = _pump_tank_costing(fixed_operating_cost=250.0)
    _set_power(m, {t: 100.0 for t in m.time_block.time_index})
    _propagate(m.costing)
    opex = m.costing.opex

    assert pyo.value(opex.fuel_cost) == pytest.approx(0.0)  # no fuel unit
    assert pyo.value(opex.scalar_cost) == pytest.approx(0.0)  # no scalar cost
    assert pyo.value(opex.fixed_operating_cost) == pytest.approx(250.0)
    assert pyo.value(opex.total_operating_cost) == pytest.approx(
        pyo.value(opex.electricity_cost)
        + pyo.value(opex.fuel_cost)
        + pyo.value(opex.fixed_operating_cost)
        + pyo.value(opex.scalar_cost)
    )


@pytest.mark.unit
def test_fixed_operating_cost_flows_through():
    """fixed_operating_cost adds to the opex total, distinct from the tariff charge."""
    profile = {t: 100.0 for t in range(24)}

    m0 = _pump_tank_costing(fixed_operating_cost=0.0)
    _set_power(m0, profile)
    _propagate(m0.costing)
    m1 = _pump_tank_costing(fixed_operating_cost=1234.0)
    _set_power(m1, profile)
    _propagate(m1.costing)

    delta = pyo.value(m1.costing.aggregate_operating_cost) - pyo.value(
        m0.costing.aggregate_operating_cost
    )
    assert delta == pytest.approx(1234.0)
    # The fixed operating cost is NOT part of the (EECO) electricity charge.
    assert pyo.value(m1.costing.opex.electricity_cost) == pytest.approx(
        pyo.value(m0.costing.opex.electricity_cost)
    )


@pytest.mark.unit
def test_operating_cost_is_eeco_total():
    """aggregate_operating_cost maps EECO's total, not a re-derived one."""
    m = _pump_tank_costing(fixed_operating_cost=0.0)
    _set_power(m, {t: 100.0 for t in range(24)})
    _propagate(m.costing)
    # EECO built its own electric_* components on the opex block.
    assert any(
        v.local_name.startswith("electric_")
        for v in m.costing.opex.component_objects(pyo.Var)
    )
    # With no fuel/scalar and no fixed cost, the aggregate is EECO's electric total.
    assert pyo.value(m.costing.aggregate_operating_cost) == pytest.approx(
        pyo.value(m.costing.opex.electricity_cost)
    )


@pytest.mark.unit
def test_operating_costs_carry_tariff_currency():
    """Cost Vars carry the tariff currency (USD); power Vars carry kW."""
    m = _pump_tank_costing(fixed_operating_cost=100.0)
    assert str(m.costing.base_currency) == "USD"  # from the tariff sheet's "$"
    for cost in (
        m.costing.opex.electricity_cost,
        m.costing.opex.fuel_cost,
        m.costing.opex.fixed_operating_cost,
        m.costing.opex.scalar_cost,
        m.costing.opex.total_operating_cost,
        m.costing.aggregate_operating_cost,
        m.costing.capex.total_capital_cost,
        m.costing.aggregate_capital_cost,
        m.costing.total_cost,
    ):
        assert str(pyunits.get_units(cost)) == "USD"
    # Power aggregates are kW; annualized cost is USD per year.
    assert str(pyunits.get_units(m.costing.aggregate_power)) == "kW"
    assert "USD" in str(pyunits.get_units(m.costing.annualized_cost))


@pytest.mark.unit
def test_capex_block_empty():
    """The capex block is an empty placeholder: total_capital_cost == 0."""
    m = _pump_tank_costing()
    _propagate(m.costing)
    assert pyo.value(m.costing.capex.total_capital_cost) == pytest.approx(0.0)
    assert pyo.value(m.costing.aggregate_capital_cost) == pytest.approx(0.0)


@pytest.mark.unit
def test_capex_excluded_from_operations_objective():
    """aggregate_operating_cost == opex total; total_cost = operating + capital."""
    m = _pump_tank_costing()
    _set_power(m, {t: 100.0 for t in range(24)})
    _propagate(m.costing)
    assert pyo.value(m.costing.aggregate_operating_cost) == pytest.approx(
        pyo.value(m.costing.opex.total_operating_cost)
    )
    assert pyo.value(m.costing.total_cost) == pytest.approx(
        pyo.value(m.costing.aggregate_operating_cost)
        + pyo.value(m.costing.aggregate_capital_cost)
    )


@pytest.mark.unit
def test_annualized_cost():
    """annualized_cost scales opex to a year; CRF matches the config formula."""
    m = _pump_tank_costing(fixed_operating_cost=8760.0)
    _set_power(m, {t: 100.0 for t in range(24)})
    _propagate(m.costing)

    horizon_years = pyo.value(pyunits.convert(m.time_block.horizon, pyunits.year))
    op = pyo.value(m.costing.aggregate_operating_cost)
    # capex is 0 in v0, so annualized_cost == operating / horizon_years.
    assert pyo.value(m.costing.annualized_cost) == pytest.approx(op / horizon_years)

    i, n = 0.08, 20.0
    expected_crf = i * (1 + i) ** n / ((1 + i) ** n - 1)
    assert pyo.value(m.costing.capital_recovery_factor) == pytest.approx(expected_crf)


@pytest.mark.unit
def test_power_units_normalized():
    """A power var in MW normalizes to kW; a non-power var raises loudly."""
    m = _pump_tank_costing(tariff=_two_utility_tariff(), run_cost_process=False)
    # An MW-denominated thermal duty aggregates in kW (x1000).
    _add_thermal_draw(m, "hot", 400 * pyunits.K, {t: 10.0 for t in range(24)})
    m.costing.cost_process()
    _propagate(m.costing)
    assert pyo.value(m.costing.aggregate_power[0, "thermal@400K"]) == pytest.approx(
        10.0
    )

    # A non-power (volumetric) var registered as electrical must raise at aggregation.
    m2 = _pump_tank_costing(run_cost_process=False)
    blk = pyo.Block()
    m2.bad = blk
    bad = pyo.Var(m2.time_block.time_index, units=pyunits.m**3 / pyunits.hr)
    blk.add_component(nm.POWER_ELECTRICAL, bad)
    blk._io_registry = IORegistry()
    blk._io_registry.power.append(
        PowerRecord(var=bad, name=nm.POWER_ELECTRICAL, kind=nm.PowerKind.ELECTRICAL)
    )
    with pytest.raises(InconsistentUnitsError):
        m2.costing.cost_process()


@pytest.mark.unit
def test_fuel_usage_units_normalized():
    """A fuel flow in L/min normalizes to m3/hr; a non-volumetric flow raises."""
    m = _pump_tank_costing(tariff=_two_utility_tariff(), run_cost_process=False)
    # 1000 L/min = 60 m^3/hr.
    _add_fuel_usage(
        m,
        "gas",
        {t: 1000.0 for t in range(24)},
        units=pyunits.L / pyunits.min,
    )
    m.costing.cost_process()
    _propagate(m.costing)
    assert pyo.value(m.costing.aggregate_fuel_usage[0, "gas"]) == pytest.approx(60.0)

    # A fuel usage var that is not a volumetric rate must raise at aggregation.
    m2 = _pump_tank_costing(tariff=_two_utility_tariff(), run_cost_process=False)
    _add_fuel_usage(m2, "natural_gas", {t: 1.0 for t in range(24)}, units=pyunits.kW)
    with pytest.raises(InconsistentUnitsError):
        m2.costing.cost_process()


@pytest.mark.unit
def test_fuel_usage_aggregated_and_billed():
    """Registered fuel flows sum per fuel and are billed via EECO's gas leg."""
    m = _pump_tank_costing(tariff=_two_utility_tariff(), run_cost_process=False)
    # Two burners on the same fuel: the aggregate is their sum, one EECO leg.
    _add_fuel_usage(m, "natural_gas", {t: 6.0 for t in range(24)}, tag="_a")
    _add_fuel_usage(m, "natural_gas", {t: 4.0 for t in range(24)}, tag="_b")
    m.costing.cost_process()
    _propagate(m.costing)

    for t in (0, 12, 23):
        assert pyo.value(
            m.costing.aggregate_fuel_usage[t, "natural_gas"]
        ) == pytest.approx(10.0)
    # Fuel is a volumetric flow, never a power carrier.
    assert "natural_gas" not in {c for _t, c in m.costing.aggregate_power}
    # The fuel leg wired through add_fuel_cost: its own sub-block + EECO gas_* comps.
    assert m.costing.opex.find_component("fuel_natural_gas") is not None
    assert m.costing.opex.find_component("fuel_cost_natural_gas") is not None
    assert any(
        v.local_name.startswith("gas_")
        for v in m.costing.opex.fuel_natural_gas.component_objects(pyo.Var)
    )


def _tariff_and_fuel_model():
    """A tariff-priced model with one EECO electric leg and one EECO fuel leg."""
    m = _pump_tank_costing(tariff=_two_utility_tariff(), run_cost_process=False)
    _add_fuel_usage(m, "gas", {t: 10.0 for t in range(24)})
    m.costing.cost_process()
    return m


@pytest.mark.unit
def test_eeco_normalization_vars_are_dimensionless():
    """Both EECO normalization Vars carry no units — EECO bills bare magnitudes."""
    opex = _tariff_and_fuel_model().costing.opex

    assert_units_equivalent(
        opex.eeco_aggregate_electrical_power[0], pyunits.dimensionless
    )
    assert_units_equivalent(
        opex.fuel_gas.eeco_aggregate_fuel_usage[0], pyunits.dimensionless
    )


@pytest.mark.unit
def test_opex_constraints_are_unit_consistent():
    """Every constraint under ``opex`` — including EECO's own — is unit-consistent.

    ``_assert_cost_units_consistent`` only checks flex-pse's ``eq_*`` cost
    constraints, so this covers what it cannot see: the constraints EECO builds
    inside its own components. EECO treats the series it is handed as bare
    numbers (``converted[t] == series[t] * factor``), so handing it a Var carrying
    kW or m3/hr makes those constraints dimensionally inconsistent.
    """
    opex = _tariff_and_fuel_model().costing.opex

    inconsistent = []
    for con in opex.component_data_objects(
        pyo.Constraint, active=True, descend_into=True
    ):
        try:
            assert_units_consistent(con)
        except UnitsError:
            inconsistent.append(con.name)
    assert inconsistent == []


@pytest.mark.unit
def test_no_fuel_in_model_leaves_fuel_cost_zero():
    """With no registered fuel flow, no leg is built and fuel_cost is 0."""
    m = _pump_tank_costing(tariff=_two_utility_tariff())
    _set_power(m, {t: 10.0 for t in range(24)})
    _propagate(m.costing)

    assert len(m.costing.aggregate_fuel_usage) == 0
    assert pyo.value(m.costing.opex.fuel_cost) == pytest.approx(0.0)


@pytest.mark.unit
def test_report_cost_fuel_is_post_hoc_on_volume_series():
    """report_cost recomputes the fuel bill from the realized m3/hr aggregate."""
    m = _pump_tank_costing(tariff=_two_utility_tariff(), run_cost_process=False)
    _add_fuel_usage(m, "natural_gas", {t: 10.0 for t in range(24)})
    m.costing.cost_process()
    _set_power(m, {t: 10.0 for t in range(24)})
    _propagate(m.costing)

    report = m.costing.report_cost(m)
    expected = evaluate_fuel_cost(
        np.full(len(m.time_block.time_index), 10.0),
        _two_utility_tariff(),
        dt_hours=1.0,
        time_index=m.time_block.datetime_index,
    )
    # 24 h x 10 m3/hr x $0.50/m3 = $120.
    assert expected == pytest.approx(120.0)
    assert report.operating.fuel == pytest.approx(expected)


@pytest.mark.unit
def test_therm_priced_tariff_billed_on_volume_series():
    """A therm-priced tariff changes nothing: flex-pse assumes no heating value.

    The gas row's ``charge (metric)`` is read by EECO as $/m3 whatever its
    ``units`` string says (``tariff_currency_units`` reads that column only for the
    ``$`` symbol), and EECO applies its own CH4 heating value only for a
    therm-denominated bare ``charge`` column. Either way flex-pse hands EECO the
    metered volumetric flow, unconverted.
    """
    tariff = load_tariff(
        [
            {
                "utility": "electric",
                "type": "energy",
                "name": "allday",
                "month_start": 1,
                "month_end": 12,
                "weekday_start": 0,
                "weekday_end": 6,
                "hour_start": 0,
                "hour_end": 24,
                "basic_charge_limit (metric)": 0,
                "charge (metric)": 0.10,
                "units": "$/kWh",
            },
            {
                "utility": "gas",
                "type": "energy",
                "name": "allday",
                "month_start": 1,
                "month_end": 12,
                "weekday_start": 0,
                "weekday_end": 6,
                "hour_start": 0,
                "hour_end": 24,
                "basic_charge_limit (metric)": 0,
                "charge (metric)": 1.20,
                "units": "$/therm",
            },
        ]
    )
    m = _pump_tank_costing(tariff=tariff, run_cost_process=False)
    _add_fuel_usage(m, "natural_gas", {t: 10.0 for t in range(24)})  # m3/hr
    m.costing.cost_process()
    _propagate(m.costing)

    usage = m.costing.aggregate_fuel_usage
    assert_units_equivalent(usage[0, "natural_gas"], pyunits.m**3 / pyunits.hr)
    assert pyo.value(usage[0, "natural_gas"]) == pytest.approx(10.0)


@pytest.mark.unit
def test_thermal_aggregated_by_temperature():
    """Thermal duties at different temperatures are separate carriers, never mixed."""
    m = _pump_tank_costing(run_cost_process=False)
    _add_thermal_draw(m, "lo", 350 * pyunits.K, {t: 10.0 for t in range(24)})
    _add_thermal_draw(m, "hi", 400 * pyunits.K, {t: 20.0 for t in range(24)})
    _add_thermal_draw(m, "lo2", 350 * pyunits.K, {t: 5.0 for t in range(24)})
    m.costing.cost_process()
    _propagate(m.costing)

    lo = "thermal@350K"
    hi = "thermal@400K"
    assert pyo.value(m.costing.aggregate_power[0, lo]) == pytest.approx(15.0)  # 10 + 5
    assert pyo.value(m.costing.aggregate_power[0, hi]) == pytest.approx(20.0)
    # The temperature-blind total sums all thermal buckets.
    assert pyo.value(m.costing.aggregate_thermal_power[0]) == pytest.approx(35.0)


@pytest.mark.unit
def test_register_scalar_cost():
    """A non-energy scalar cost (price x quantity) enters the opex total, no EECO."""
    m = _pump_tank_costing(run_cost_process=False)
    # A water-withdrawal flow, m^3/hr, priced at $2.5/m^3.
    m.water = pyo.Var(m.time_block.time_index, units=pyunits.m**3 / pyunits.hr)
    for t in m.time_block.time_index:
        m.water[t].set_value(3.0)
    m.costing.register_scalar_cost(
        "water", m.water, price=2.5, quantity_units=pyunits.m**3 / pyunits.hr
    )
    m.costing.cost_process()
    _propagate(m.costing)

    dt_hours = pyo.value(pyunits.convert(m.time_block.dt, pyunits.hr))
    expected = 2.5 * 3.0 * 24 * dt_hours
    assert pyo.value(m.costing.opex.scalar_cost) == pytest.approx(expected)
    assert pyo.value(m.costing.opex.total_operating_cost) == pytest.approx(
        pyo.value(m.costing.opex.electricity_cost)
        + pyo.value(m.costing.opex.fuel_cost)
        + pyo.value(m.costing.opex.fixed_operating_cost)
        + expected
    )


@pytest.mark.unit
def test_scalar_cost_not_via_eeco():
    """Scalar costs never build EECO components."""
    m = _pump_tank_costing(run_cost_process=False)
    m.chem = pyo.Var(m.time_block.time_index, units=pyunits.kg / pyunits.hr)
    for t in m.time_block.time_index:
        m.chem[t].set_value(1.0)
    m.costing.register_scalar_cost(
        "chem", m.chem, price=4.0, quantity_units=pyunits.kg / pyunits.hr
    )
    m.costing.cost_process()
    # No gas_* EECO components appear (scalar costs are not routed through EECO).
    assert not any(
        v.local_name.startswith("gas_")
        for v in m.costing.opex.component_objects(pyo.Var)
    )


@pytest.mark.unit
def test_register_scalar_cost_unit_attribution():
    """An optional unit= is stored on the spec for later per-unit attribution."""
    m = _pump_tank_costing(run_cost_process=False)
    m.water = pyo.Var(m.time_block.time_index, units=pyunits.m**3 / pyunits.hr)

    # Default: no unit association.
    plain = m.costing.register_scalar_cost(
        "water", m.water, price=2.5, quantity_units=pyunits.m**3 / pyunits.hr
    )
    assert plain.unit is None

    # An attributed cost records the owning unit block verbatim.
    attributed = m.costing.register_scalar_cost(
        "water_pump",
        m.water,
        price=2.5,
        quantity_units=pyunits.m**3 / pyunits.hr,
        unit=m.pump,
    )
    assert attributed.unit is m.pump


@pytest.mark.unit
def test_report_cost_breakdown_shape():
    """report_cost returns a categorized CostReport with v0 zero placeholders."""
    m = _pump_tank_costing(fixed_operating_cost=500.0, dr_event_file=str(_DR_JSON))
    profile = {t: 100.0 for t in range(24)}
    _set_power(m, profile)
    _propagate(m.costing)

    report = m.costing.report_cost(m)
    assert isinstance(report, CostReport)
    assert isinstance(report.operating, OperatingCostBreakdown)
    assert isinstance(report.capital, CapitalCostBreakdown)
    # Every number in the report is a magnitude in this one currency.
    assert report.currency == "USD"  # from the tariff sheet's "$"

    assert report.operating.fuel == pytest.approx(0.0)
    assert report.operating.fixed == pytest.approx(500.0)
    assert report.operating.scalar == pytest.approx(0.0)
    # DR is containers-only: a loaded DR file produces no credit.
    assert report.operating.dr_revenue == pytest.approx(0.0)
    assert report.capital.by_component == {}
    assert report.capital.total == pytest.approx(0.0)

    assert report.operating.total == pytest.approx(
        report.operating.electricity
        + report.operating.fuel
        + report.operating.fixed
        + report.operating.scalar
        - report.operating.dr_revenue
    )
    assert report.total == pytest.approx(report.operating.total + report.capital.total)


@pytest.mark.unit
def test_mode_toggles():
    """Design/operations modes are idempotent no-ops over empty registries; LP both."""
    m = _pump_tank_costing()
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)

    m.costing.set_design_mode()
    m.costing.set_design_mode()  # idempotent
    assert classify(m) is ProblemClass.LP  # empty capex -> no nonlinearity

    m.costing.set_operations_mode()
    m.costing.set_operations_mode()  # idempotent
    assert classify(m) is ProblemClass.LP


@pytest.mark.unit
def test_construction_order_permutation():
    """aggregate_operating_cost is identical across component-creation orders."""
    profile = {t: 100.0 for t in range(24)}

    values = []
    for costing_first in (True, False):
        for pump_first in (True, False):
            m = _pump_tank_costing(costing_first=costing_first, pump_first=pump_first)
            _set_power(m, profile)
            _propagate(m.costing)
            values.append(pyo.value(m.costing.aggregate_operating_cost))

    for v in values[1:]:
        assert v == pytest.approx(values[0], rel=1e-12)


@pytest.mark.unit
def test_dr_container_loads_noop():
    """A loaded DR file populates the container and builds no DR constraints."""
    from flexops.costing import DRConfig

    m_dr = _pump_tank_costing(dr_event_file=str(_DR_JSON))
    assert isinstance(m_dr.costing.dr, DRConfig)
    assert m_dr.costing.dr.program is not None

    m_no = _pump_tank_costing()
    # No DR constraints: the active-constraint count is unchanged, LP both.
    n_dr = len(list(m_dr.component_data_objects(pyo.Constraint, active=True)))
    n_no = len(list(m_no.component_data_objects(pyo.Constraint, active=True)))
    assert n_dr == n_no
    assert classify(m_dr) is ProblemClass.LP


@pytest.mark.unit
def test_model_classifies_lp():
    """The built pump+tank+costing model classifies LP."""
    m = _pump_tank_costing()
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
    assert classify(m) is ProblemClass.LP


# --------------------------------------------------------------------------- #
# Flat (scalar) energy prices: pricing without a tariff / without EECO
# --------------------------------------------------------------------------- #
_USD = currency_units("USD")


def _gas_only_tariff():
    """The flat two-utility tariff with its electric rows dropped."""
    tariff = _two_utility_tariff()
    return tariff[tariff["utility"] == "gas"].reset_index(drop=True)


@pytest.mark.unit
def test_flat_electricity_price_without_tariff():
    """A flat $/kWh price bills electricity natively, with no tariff and no EECO."""
    price = 0.12 * _USD / pyunits.kWh
    m = _pump_tank_costing(no_tariff=True, energy_prices={"electrical": price})
    _set_power(m, {t: 100.0 for t in range(24)})
    _propagate(m.costing)

    # 24 h x 100 kW x $0.12/kWh = $288.00
    assert pyo.value(m.costing.opex.electricity_cost) == pytest.approx(288.0)
    assert pyo.value(m.costing.aggregate_operating_cost) == pytest.approx(288.0)
    # The EECO normalization Var only exists on the tariff path.
    assert m.costing.opex.find_component("eeco_aggregate_electrical_power") is None


@pytest.mark.unit
def test_flat_fuel_price_with_tariff_electricity():
    """A fuel priced flat and electricity priced by tariff bill on separate legs."""
    m = _pump_tank_costing(
        tariff=_two_utility_tariff(),
        energy_prices={"gas": 0.25 * _USD / pyunits.m**3},
        run_cost_process=False,
    )
    _add_fuel_usage(m, "gas", {t: 10.0 for t in range(24)})
    m.costing.cost_process()
    _set_power(m, {t: 100.0 for t in range(24)})
    _propagate(m.costing)

    # The flat leg is a native constraint, so it propagates without a solver:
    # 24 h x 10 m3/hr x $0.25/m3 = $60.00 (not the tariff's $0.50/m3).
    assert pyo.value(m.costing.opex.fuel_cost_gas) == pytest.approx(60.0)
    # A flat-priced fuel has no EECO sub-block, so no normalization Var either.
    assert m.costing.opex.find_component("fuel_gas") is None
    # The EECO leg's in-objective cost is built from EECO's own Vars, which have
    # no eq_ sibling to propagate through, so read the tariff leg post-hoc:
    # 24 h x 100 kW x $0.10/kWh = $240.00.
    report = m.costing.report_cost(m)
    assert report.operating.electricity == pytest.approx(240.0)
    assert report.operating.fuel == pytest.approx(60.0)


@pytest.mark.unit
def test_flat_price_overrides_tariff_for_that_carrier():
    """A flat price wins over a tariff that also covers the carrier."""
    m = _pump_tank_costing(
        tariff=_two_utility_tariff(),  # prices electric at $0.10/kWh
        energy_prices={"electrical": 0.20 * _USD / pyunits.kWh},
    )
    _set_power(m, {t: 100.0 for t in range(24)})
    _propagate(m.costing)

    # The flat $0.20/kWh is used, not the tariff's $0.10/kWh.
    assert pyo.value(m.costing.opex.electricity_cost) == pytest.approx(480.0)
    assert m.costing.opex.find_component("eeco_aggregate_electrical_power") is None


@pytest.mark.unit
def test_registered_draw_with_no_pricing_source_raises():
    """A carrier with draws but neither a price nor tariff coverage errors by name."""
    m = _pump_tank_costing(tariff=_gas_only_tariff(), run_cost_process=False)
    with pytest.raises(FlexConfigError, match="electrical"):
        m.costing.cost_process()


@pytest.mark.unit
def test_flat_price_wrong_dimension_raises():
    """A price whose units do not reconcile with the carrier raises loudly."""
    m = _pump_tank_costing(
        no_tariff=True,
        energy_prices={"electrical": 0.12 * _USD / pyunits.m**3},  # not a $/energy
        run_cost_process=False,
    )
    with pytest.raises((FlexConfigError, InconsistentUnitsError)):
        m.costing.cost_process()


@pytest.mark.unit
def test_bare_electricity_price_infers_per_kwh():
    """A bare number prices electricity in currency/kWh, the metered quantity."""
    m = _pump_tank_costing(no_tariff=True, energy_prices={"electrical": 0.12})
    _set_power(m, {t: 100.0 for t in range(24)})
    _propagate(m.costing)

    # Same bill as the explicit 0.12 * USD/kWh: 24 h x 100 kW x $0.12/kWh = $288.
    assert pyo.value(m.costing.opex.electricity_cost) == pytest.approx(288.0)


@pytest.mark.unit
def test_bare_fuel_price_infers_per_cubic_meter():
    """A bare number prices a fuel in currency/m3, the metered quantity."""
    m = _pump_tank_costing(
        tariff=_two_utility_tariff(),
        energy_prices={"gas": 0.25},
        run_cost_process=False,
    )
    _add_fuel_usage(m, "gas", {t: 10.0 for t in range(24)})
    m.costing.cost_process()
    _propagate(m.costing)

    # 24 h x 10 m3/hr x $0.25/m3 = $60.00.
    assert pyo.value(m.costing.opex.fuel_cost_gas) == pytest.approx(60.0)


@pytest.mark.unit
def test_report_cost_flat_priced_is_native():
    """report_cost recomputes a flat-priced carrier natively (no EECO call)."""
    m = _pump_tank_costing(
        no_tariff=True, energy_prices={"electrical": 0.12 * _USD / pyunits.kWh}
    )
    _set_power(m, {t: 100.0 for t in range(24)})
    _propagate(m.costing)

    report = m.costing.report_cost(m)
    assert report.operating.electricity == pytest.approx(288.0)
    assert report.operating.fuel == 0.0
    assert report.total == pytest.approx(report.operating.total)


@pytest.mark.unit
def test_report_cost_currency_follows_configured_basis():
    """With no tariff the report is labeled with the configured currency, not USD."""
    eur = currency_units("EUR")
    m = _pump_tank_costing(
        no_tariff=True,
        currency="EUR",
        energy_prices={"electrical": 0.12 * eur / pyunits.kWh},
    )
    _set_power(m, {t: 100.0 for t in range(24)})
    _propagate(m.costing)

    report = m.costing.report_cost(m)
    assert report.currency == "EUR"
    assert report.currency == str(m.costing.base_currency)


@pytest.mark.unit
def test_flat_priced_model_needs_no_eeco(monkeypatch):
    """With every carrier flat-priced, build/cost/report never touch eeco."""
    from flexops.costing import opex as opex_mod

    monkeypatch.setattr(opex_mod, "_HAS_EECO", False)
    m = _pump_tank_costing(
        no_tariff=True, energy_prices={"electrical": 0.12 * _USD / pyunits.kWh}
    )
    _set_power(m, {t: 100.0 for t in range(24)})
    _propagate(m.costing)
    assert m.costing.report_cost(m).operating.electricity == pytest.approx(288.0)


@pytest.mark.unit
def test_tariff_path_without_eeco_raises_install_hint(monkeypatch):
    """A tariff path with eeco unavailable errors naming both remedies."""
    from flexops.costing import opex as opex_mod

    monkeypatch.setattr(opex_mod, "_HAS_EECO", False)
    with pytest.raises(FlexConfigError, match="eeco"):
        load_tariff(_TARIFF_JSON)


@pytest.mark.unit
def test_priced_leg_rejects_foreign_time_index():
    """A priced series on a time set other than the TimeBlock's errors loudly."""
    m = _pump_tank_costing(run_cost_process=False)
    m.other_time = pyo.Set(initialize=range(5), ordered=True)
    m.other_flow = pyo.Var(
        m.other_time, initialize=1.0, units=pyunits.m**3 / pyunits.hr
    )
    m.costing.register_scalar_cost(
        "water", m.other_flow, 1.0, pyunits.m**3 / pyunits.hr
    )
    with pytest.raises(FlexConfigError, match="time"):
        m.costing.cost_process()


# --------------------------------------------------------------------------- #
# Time-varying flat prices: an array, or a Pyomo indexed component, of length T
# --------------------------------------------------------------------------- #
# A price of $0.10/kWh over the first half of the horizon and $0.20/kWh over the
# second, against a constant 100 kW draw: 100 kW x (12 h x $0.10 + 12 h x $0.20).
_HALVES = [0.10] * 12 + [0.20] * 12
_HALVES_COST = 360.0


def _add_price_param(m, *, units=_USD / pyunits.kWh, foreign=False, values=_HALVES):
    """Attach an indexed price Param to ``m`` and return it.

    ``foreign=True`` puts it on its own Set rather than the TimeBlock's
    ``time_index``, which is what lets ``values`` be a length other than T.
    """
    if foreign:
        m.price_index = pyo.Set(initialize=range(len(values)), ordered=True)
        index = m.price_index
    else:
        index = m.time_block.time_index
    m.price = pyo.Param(
        index, initialize=dict(enumerate(values)), units=units, mutable=True
    )
    return m.price


@pytest.mark.unit
def test_time_varying_electricity_price_from_list():
    """A length-T list of bare prices bills sum_t price[t] x power[t] x dt."""
    m = _pump_tank_costing(no_tariff=True, energy_prices={"electrical": _HALVES})
    _set_power(m, {t: 100.0 for t in range(24)})
    _propagate(m.costing)

    assert pyo.value(m.costing.opex.electricity_cost) == pytest.approx(_HALVES_COST)


@pytest.mark.unit
def test_time_varying_electricity_price_from_ndarray_with_units():
    """A numpy array carrying units is billed per period, like a bare list."""
    prices = np.array(_HALVES) * _USD / pyunits.kWh
    m = _pump_tank_costing(no_tariff=True, energy_prices={"electrical": prices})
    _set_power(m, {t: 100.0 for t in range(24)})
    _propagate(m.costing)

    assert pyo.value(m.costing.opex.electricity_cost) == pytest.approx(_HALVES_COST)


@pytest.mark.unit
def test_time_varying_price_from_param_on_time_index():
    """An indexed Param on the TimeBlock's own time index is billed per period."""
    m = _pump_tank_costing(
        no_tariff=True,
        energy_prices=lambda mdl: {"electrical": _add_price_param(mdl)},
    )
    _set_power(m, {t: 100.0 for t in range(24)})
    _propagate(m.costing)

    assert pyo.value(m.costing.opex.electricity_cost) == pytest.approx(_HALVES_COST)


@pytest.mark.unit
def test_time_varying_price_from_unitless_param_infers_units():
    """A dimensionless indexed Param gets the carrier's per-quantity units."""
    m = _pump_tank_costing(
        no_tariff=True,
        energy_prices=lambda mdl: {"electrical": _add_price_param(mdl, units=None)},
    )
    _set_power(m, {t: 100.0 for t in range(24)})
    _propagate(m.costing)

    assert pyo.value(m.costing.opex.electricity_cost) == pytest.approx(_HALVES_COST)


@pytest.mark.unit
def test_time_varying_price_from_param_on_foreign_index_of_same_length():
    """A price Param on its own T-member index set is accepted, read in order.

    Unlike the costed *series*, an exogenous price is not required to live on the
    TimeBlock's index set — only to have one value per time point.
    """
    m = _pump_tank_costing(
        no_tariff=True,
        energy_prices=lambda mdl: {"electrical": _add_price_param(mdl, foreign=True)},
    )
    _set_power(m, {t: 100.0 for t in range(24)})
    _propagate(m.costing)

    assert pyo.value(m.costing.opex.electricity_cost) == pytest.approx(_HALVES_COST)


@pytest.mark.unit
def test_time_varying_price_from_indexed_var():
    """A fixed indexed Var can serve as the price series."""

    def prices(mdl):
        mdl.price_var = pyo.Var(
            mdl.time_block.time_index,
            initialize=dict(enumerate(_HALVES)),
            units=_USD / pyunits.kWh,
        )
        mdl.price_var.fix()
        return {"electrical": mdl.price_var}

    m = _pump_tank_costing(no_tariff=True, energy_prices=prices)
    _set_power(m, {t: 100.0 for t in range(24)})
    _propagate(m.costing)

    assert pyo.value(m.costing.opex.electricity_cost) == pytest.approx(_HALVES_COST)


@pytest.mark.unit
def test_time_varying_fuel_price_from_list():
    """A fuel priced by a length-T list bills per period in currency/m3."""
    prices = [0.20] * 12 + [0.30] * 12
    m = _pump_tank_costing(
        tariff=_two_utility_tariff(),
        energy_prices={"gas": prices},
        run_cost_process=False,
    )
    _add_fuel_usage(m, "gas", {t: 10.0 for t in range(24)})
    m.costing.cost_process()
    _propagate(m.costing)

    # 10 m3/hr x (12 h x $0.20 + 12 h x $0.30) = $60.00.
    assert pyo.value(m.costing.opex.fuel_cost_gas) == pytest.approx(60.0)


@pytest.mark.unit
@pytest.mark.parametrize("length", [23, 25])
def test_time_varying_price_wrong_length_raises(length):
    """A price array that is not length T raises at FlexCosting construction."""
    with pytest.raises(FlexConfigError, match="24"):
        _pump_tank_costing(no_tariff=True, energy_prices={"electrical": [0.1] * length})


@pytest.mark.unit
def test_time_varying_price_wrong_index_cardinality_raises():
    """An indexed price whose index set has other than T members raises."""
    with pytest.raises(FlexConfigError, match="24"):
        _pump_tank_costing(
            no_tariff=True,
            energy_prices=lambda mdl: {
                "electrical": _add_price_param(mdl, foreign=True, values=[0.1] * 25)
            },
        )


@pytest.mark.unit
def test_energy_prices_mapping_value_rejected():
    """A mapping price is rejected: iterating it would silently cost its keys.

    Pyomo wraps a ConfigValue domain's exception in ValueError (as for the other
    declared domains), so the FlexConfigError message is asserted through it.
    """
    m = _time_model()
    with pytest.raises(ValueError, match="mapping"):
        m.costing = FlexCosting(
            time_block=m.time_block,
            energy_prices={"electrical": dict(enumerate(_HALVES))},
        )


@pytest.mark.unit
def test_time_varying_price_report_cost_is_native():
    """report_cost recomputes a per-period-priced carrier without EECO."""
    m = _pump_tank_costing(no_tariff=True, energy_prices={"electrical": _HALVES})
    _set_power(m, {t: 100.0 for t in range(24)})
    _propagate(m.costing)

    report = m.costing.report_cost(m)
    assert report.operating.electricity == pytest.approx(_HALVES_COST)
    assert report.operating.fuel == 0.0


@pytest.mark.unit
def test_time_varying_price_wrong_dimension_raises():
    """A per-period price whose units do not reconcile with the carrier raises."""
    prices = np.array(_HALVES) * _USD / pyunits.m**3  # not a $/energy
    m = _pump_tank_costing(
        no_tariff=True,
        energy_prices={"electrical": prices},
        run_cost_process=False,
    )
    with pytest.raises((FlexConfigError, InconsistentUnitsError)):
        m.costing.cost_process()


@pytest.mark.unit
def test_time_varying_price_opex_constraints_are_unit_consistent():
    """Every opex constraint stays unit-consistent under per-period prices."""
    m = _pump_tank_costing(
        no_tariff=True,
        energy_prices={"electrical": _HALVES, "gas": [0.2] * 24},
        run_cost_process=False,
    )
    _add_fuel_usage(m, "gas", {t: 10.0 for t in range(24)})
    m.costing.cost_process()

    inconsistent = []
    for con in m.costing.opex.component_data_objects(
        pyo.Constraint, active=True, descend_into=True
    ):
        try:
            assert_units_consistent(con)
        except UnitsError:
            inconsistent.append(con.name)
    assert inconsistent == []


# --------------------------------------------------------------------------- #
# Several tariff files -> one merged frame, indexed by utility
# --------------------------------------------------------------------------- #
def _electric_only_records():
    """Records list holding only the flat electric energy charge."""
    return [
        r
        for r in _two_utility_tariff().to_dict("records")
        if r["utility"] == "electric"
    ]


def _gas_only_records():
    """Records list holding only the flat gas energy charge."""
    return [
        r for r in _two_utility_tariff().to_dict("records") if r["utility"] == "gas"
    ]


@pytest.mark.unit
def test_merge_tariffs_sequence():
    """A sequence of sources merges into one frame carrying both utilities."""
    merged = merge_tariffs([_electric_only_records(), _gas_only_records()])
    assert set(merged["utility"]) == {"electric", "gas"}
    assert len(merged) == 2


@pytest.mark.unit
def test_merge_tariffs_mapping_assigns_by_utility():
    """A mapping assigns each source to a utility and drops its stray rows."""
    # Each source is the FULL two-utility tariff; the mapping keeps only the
    # rows for the utility it was assigned to, so no charge is duplicated.
    both = _two_utility_tariff()
    merged = merge_tariffs({"electric": both, "gas": both})
    assert sorted(merged["utility"]) == ["electric", "gas"]
    assert len(merged) == 2


@pytest.mark.unit
def test_merge_tariffs_mapping_missing_utility_raises():
    """A source contributing no rows for its assigned utility errors by name."""
    with pytest.raises((FlexConfigError, FlexDataError), match="gas"):
        merge_tariffs({"gas": _electric_only_records()})


@pytest.mark.unit
def test_merge_tariffs_duplicate_charge_raises():
    """The same charge from two sources errors rather than silently doubling."""
    with pytest.raises((FlexConfigError, FlexDataError), match="[Dd]uplicate"):
        merge_tariffs([_electric_only_records(), _electric_only_records()])


@pytest.mark.unit
def test_merged_tariff_bills_both_legs():
    """A merged electric+gas tariff bills electricity and fuel from one frame."""
    merged = merge_tariffs(
        {"electric": _electric_only_records(), "gas": _gas_only_records()}
    )
    m = _pump_tank_costing(tariff=merged, run_cost_process=False)
    _add_fuel_usage(m, "natural_gas", {t: 10.0 for t in range(24)})
    m.costing.cost_process()
    _set_power(m, {t: 100.0 for t in range(24)})
    _propagate(m.costing)

    assert m.costing._tariff_utilities == {"electric", "gas"}
    # Both legs are EECO legs, so read them post-hoc (see the note in
    # test_flat_fuel_price_with_tariff_electricity).
    report = m.costing.report_cost(m)
    assert report.operating.electricity == pytest.approx(240.0)
    assert report.operating.fuel == pytest.approx(120.0)


# --------------------------------------------------------------------------- #
# Annualization: interest + discount -> one effective rate
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_crf_interest_unset_matches_discount_rate():
    """interest_rate=None leaves the CRF exactly as the discount rate alone."""
    m = _pump_tank_costing()
    i, n = 0.08, 20.0
    expected = i * (1 + i) ** n / ((1 + i) ** n - 1)
    assert pyo.value(m.costing.capital_recovery_factor) == pytest.approx(expected)
    assert pyo.value(m.costing.effective_rate) == pytest.approx(0.08)


@pytest.mark.unit
def test_crf_combines_interest_and_discount():
    """Both rates given -> effective rate (1+i)/(1+d)-1 drives the CRF."""
    i, d, n = 0.10, 0.03, 25.0
    m = _pump_tank_costing(interest_rate=i, discount_rate=d, lifetime_years=n)
    r = (1 + i) / (1 + d) - 1
    expected = r * (1 + r) ** n / ((1 + r) ** n - 1)
    assert pyo.value(m.costing.effective_rate) == pytest.approx(r)
    assert pyo.value(m.costing.capital_recovery_factor) == pytest.approx(expected)


@pytest.mark.unit
def test_crf_zero_effective_rate_is_straight_line():
    """Equal interest and discount -> zero effective rate -> CRF == 1/lifetime."""
    m = _pump_tank_costing(interest_rate=0.05, discount_rate=0.05, lifetime_years=20.0)
    assert pyo.value(m.costing.effective_rate) == pytest.approx(0.0)
    assert pyo.value(m.costing.capital_recovery_factor) == pytest.approx(1.0 / 20.0)


@pytest.mark.unit
def test_crf_rejects_degenerate_rates():
    """An effective rate <= -1 and a non-positive lifetime both error by field."""
    with pytest.raises(FlexConfigError, match="rate"):
        _pump_tank_costing(interest_rate=-1.0, discount_rate=0.5)

    with pytest.raises(FlexConfigError, match="lifetime"):
        _pump_tank_costing(lifetime_years=0.0)


# --------------------------------------------------------------------------- #
# Sub-monthly prorating of monthly demand + fixed charges
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_monthly_scale_factor_partial_and_full_month():
    """The scale factor is horizon / calendar-month length, clamped at 1.0."""
    import pandas as pd

    july_day = pd.date_range("2025-07-08", periods=24, freq="h")
    assert monthly_scale_factor(july_day, 1.0) == pytest.approx(24.0 / 744.0)

    full_july = pd.date_range("2025-07-01", periods=744, freq="h")
    assert monthly_scale_factor(full_july, 1.0) == pytest.approx(1.0)

    # 30-day month -> a different denominator than July's 31 days.
    nov_day = pd.date_range("2025-11-05", periods=24, freq="h")
    assert monthly_scale_factor(nov_day, 1.0) == pytest.approx(24.0 / 720.0)


@pytest.mark.unit
def test_monthly_scale_factor_handles_december():
    """December works (a naive month+1 rollover would raise)."""
    import pandas as pd

    dec_day = pd.date_range("2025-12-10", periods=24, freq="h")
    assert monthly_scale_factor(dec_day, 1.0) == pytest.approx(24.0 / 744.0)


@pytest.mark.unit
def test_prorating_scales_demand_and_fixed_charges():
    """On a 24-h horizon the monthly demand + customer charges are prorated.

    Read post-hoc, so the demand charges are real numbers rather than the
    in-objective EECO Vars (which no solver has touched here).
    """
    m_on = _pump_tank_costing(prorate_monthly_charges=True)
    m_off = _pump_tank_costing(prorate_monthly_charges=False)
    for m in (m_on, m_off):
        _set_power(m, {t: 100.0 for t in range(24)})
        _propagate(m.costing)

    on = m_on.costing.report_cost(m_on).operating.electricity
    off = m_off.costing.report_cost(m_off).operating.electricity
    # Prorating only ever reduces the bill on a sub-monthly horizon.
    assert on < off
    # Only the monthly-assessed charges scale: $150 customer + $40.50/kW demand at
    # a flat 100 kW. The energy charge is untouched, so it cancels in the diff.
    scale = 24.0 / 744.0
    monthly_assessed = 150.0 + (21.5 + 19.0) * 100.0
    assert off - on == pytest.approx(monthly_assessed * (1.0 - scale), rel=1e-6)


@pytest.mark.unit
def test_prorating_leaves_the_energy_charge_alone():
    """Prorating touches demand and fixed charges only, never energy."""
    m = _pump_tank_costing(prorate_monthly_charges=True)
    _set_power(m, {t: 100.0 for t in range(24)})
    _propagate(m.costing)

    scale = 24.0 / 744.0
    # Summer weekday (2025-07-08): 5 peak hours @ $0.18 + 19 off-peak @ $0.09.
    energy = 100.0 * (5 * 0.18 + 19 * 0.09)
    expected = energy + (150.0 + 40.5 * 100.0) * scale
    assert m.costing.report_cost(m).operating.electricity == pytest.approx(expected)
