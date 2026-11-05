"""build_design_model: the merged structure, checked without a solve."""

from pathlib import Path

import pyomo.environ as pyo
import pytest
from pyomo.repn import generate_standard_repn
from pyomo.util.check_units import assert_units_consistent

import flexops
from flexcore.exceptions import FlexConfigError
from flexcore.solvers import ProblemClass, classify
from flexdesign import build_design_model
from flexdesign.tests.fixtures import pump_battery_solar_study


@pytest.fixture(scope="module")
def model():
    """Return the merged model of the shared study."""
    return build_design_model(pump_battery_solar_study())


def _with_capacity(**fields):
    """Return the shared study with its solar decision's fields replaced."""
    spec = pump_battery_solar_study()
    for name, value in fields.items():
        setattr(spec.capacities[2], name, value)
    return spec


@pytest.mark.unit
def test_one_horizon_per_year_and_month(model):
    """The model holds one block per year and month, each with its own time block."""
    assert list(model.horizon.keys()) == [
        (2026, 1),
        (2026, 7),
        (2031, 1),
        (2031, 7),
    ]
    assert model.horizon[2031, 7].time_block.n_points == 24


@pytest.mark.unit
def test_every_active_constraint_is_linear(model):
    """Assembly leaves no nonlinear row, including in the links and batteries."""
    for con in model.component_data_objects(pyo.Constraint, active=True):
        assert generate_standard_repn(con.body).is_linear(), con.name


@pytest.mark.unit
def test_link_equates_each_horizon_var_with_the_shared_size(model):
    """Each link's body is zero only when a horizon's Var equals the size."""
    block = model.capacity["solar"]
    block.size.set_value(10.0)
    model.horizon[2026, 7].pv.capacity.set_value(10.0)
    assert pyo.value(block.link[2026, 7].body) == pytest.approx(0.0)
    model.horizon[2026, 7].pv.capacity.set_value(7.0)
    assert abs(pyo.value(block.link[2026, 7].body)) == pytest.approx(3.0)


@pytest.mark.unit
def test_exactly_one_objective_is_active(model):
    """Each horizon's own objective is off, leaving the annualized one."""
    active = list(model.component_data_objects(pyo.Objective, active=True))
    assert [objective.name for objective in active] == ["annualized_cost"]


@pytest.mark.unit
def test_every_capacity_links_every_horizon(model):
    """Each decision has one equality per horizon."""
    for name in ("battery_energy", "battery_power", "solar"):
        assert len(model.capacity[name].link) == 4


@pytest.mark.unit
def test_only_listed_capacities_are_unfixed():
    """Listed Vars are free in every horizon, and unlisted sizing Vars stay fixed."""
    spec = pump_battery_solar_study()
    spec.capacities = spec.capacities[2:]
    m = build_design_model(spec)
    for key in m.horizon:
        assert not m.horizon[key].pv.capacity.fixed
        assert m.horizon[key].battery.capacity.fixed
        assert m.horizon[key].battery.power_capacity.fixed


@pytest.mark.unit
def test_units_are_consistent(model):
    """The whole merged model passes the units check."""
    assert_units_consistent(model)


@pytest.mark.unit
def test_merged_model_classifies_as_lp(model):
    """Free capacities leave the merged model linear."""
    assert classify(model) is ProblemClass.LP


@pytest.mark.unit
@pytest.mark.parametrize(
    "variable, message",
    [
        ("battery.not_there", "not found"),
        ("battery.charge", "scalar"),
        ("battery.charge_init", "Var"),
    ],
)
def test_capacity_path_must_name_a_scalar_var(variable, message):
    """An unknown path, an indexed Var or a non-Var raises."""
    with pytest.raises(FlexConfigError, match=message):
        build_design_model(_with_capacity(variable=variable))


@pytest.mark.unit
def test_upper_above_the_vars_own_bound_raises():
    """A sizing range the unit can't honor raises rather than silently capping."""
    spec = pump_battery_solar_study()
    spec.capacities[1].upper = 3000.0
    with pytest.raises(FlexConfigError, match="bounds"):
        build_design_model(spec)


@pytest.mark.unit
def test_non_cyclic_storage_raises():
    """A fixed initial state would let each month drain free energy."""
    document = pump_battery_solar_study().model_dump(by_alias=True)
    battery = next(
        el for el in document["base"]["elements"] if el.get("name") == "battery"
    )
    battery["construction_options"]["cyclic"] = False
    spec = pump_battery_solar_study(base=document["base"])
    with pytest.raises(FlexConfigError, match="initial states"):
        build_design_model(spec)


@pytest.mark.unit
def test_horizons_with_different_effective_rates_raise():
    """One study has one rate, so a change can't vary it by horizon."""
    change = {"element": "costing", "years": [2031], "values": {"discount_rate": 0.02}}
    document = pump_battery_solar_study().model_dump(by_alias=True)
    spec = pump_battery_solar_study(changes=[*document["changes"], change])
    with pytest.raises(FlexConfigError, match="effective rate"):
        build_design_model(spec)


@pytest.mark.unit
def test_series_of_the_wrong_length_raises():
    """A month's series must have one value per time point."""
    change = {
        "element": "pv",
        "months": [7],
        "values": {"construction_options": {"capacity_factor": [0.5] * 10}},
    }
    document = pump_battery_solar_study().model_dump(by_alias=True)
    spec = pump_battery_solar_study(changes=[*document["changes"], change])
    with pytest.raises(FlexConfigError, match="capacity_factor has 10 values"):
        build_design_model(spec)


@pytest.mark.unit
def test_tariff_study_assembles_with_linked_capacities():
    """A tariff-billed study builds and links like a natively priced one."""
    tariff = Path(flexops.__file__).parent / "tests/fixtures/tariff_tou_demo.json"
    costing = {
        "kind": "costing",
        "name": "costing",
        "tariff_source": str(tariff),
        "discount_rate": 0.07,
    }
    m = build_design_model(pump_battery_solar_study(costing=costing))
    assert len(m.capacity["solar"].link) == 4
    assert_units_consistent(m)
