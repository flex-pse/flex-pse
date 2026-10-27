"""Server: one server's IT power, PDU loss and heat split."""

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits
from pyomo.opt import assert_optimal_termination

import flexops as fo
from flexcore.config.io import dump_spec
from flexcore.config.schema import UnitCommitmentConfig
from flexcore.exceptions import FlexConfigError, FlexEmitWarning
from flexcore.solvers import ProblemClass, classify, get_solver
from flexops.costing import currency_units
from flexops.logic import relax
from flexops.testing import UnitModelTestHarness, dummy_time_block
from flexops.unit_models import Server


def server_options(n: int = 3, **overrides) -> dict:
    """Return Server options for an n-step build: 4-12 kW, 0.9 kg/s airflow."""
    options = {
        "utilization": [0.5] * n,
        "idle_power": 4.0 * pyunits.kW,
        "peak_power": 12.0 * pyunits.kW,
        "fan_rated_power": 0.3 * pyunits.kW,
        "rated_airflow": 1.0 * pyunits.kg / pyunits.s,
        "airflow": [0.9] * n,
    }
    options.update(overrides)
    return options


def solve(m) -> None:
    """Solve ``m`` with the facade-selected solver and assert optimality."""
    if not list(m.component_objects(pyo.Objective, active=True)):
        m.objective = pyo.Objective(expr=0)
    assert_optimal_termination(get_solver(model=m).solve(m))


class TestServer(UnitModelTestHarness):
    """One server at 50% utilization with a 2% PDU loss."""

    expected_dof = 0
    it_power = 4.0 + 8.0 * 0.5 + 0.3 * 0.9**3
    expected_solution = {
        "it_power[0]": it_power,
        "power_electrical[0]": it_power * 1.02,
        "heat_output[0]": it_power,
    }

    def configure(self):
        m = dummy_time_block(3)
        m.unit = Server(**server_options())
        return m, m.unit


@pytest.mark.unit
def test_server_rejects_utilization_outside_unit_interval():
    m = dummy_time_block(3)
    with pytest.raises(FlexConfigError) as err:
        m.unit = Server(**server_options(utilization=[0.5, 1.2, 0.5]))
    assert err.value.field == "utilization"


@pytest.mark.unit
def test_server_rejects_utilization_of_wrong_length():
    m = dummy_time_block(3)
    with pytest.raises(FlexConfigError) as err:
        m.unit = Server(**server_options(utilization=[0.5, 0.5]))
    assert err.value.field == "utilization"


@pytest.mark.unit
def test_server_rejects_live_source_missing_a_step():
    m = dummy_time_block(3)
    m.steps = pyo.Set(initialize=[0, 1])
    m.source = pyo.Param(m.steps, initialize=0.5, mutable=True)
    with pytest.raises(FlexConfigError) as err:
        m.unit = Server(**server_options(utilization=m.source))
    assert err.value.field == "utilization"


@pytest.mark.unit
def test_server_live_var_requires_linear_curve():
    m = dummy_time_block(3)
    m.source = pyo.Var(m.time_block.time_index, bounds=(0, 1), initialize=0.5)
    with pytest.raises(FlexConfigError) as err:
        m.unit = Server(
            **server_options(utilization=m.source, utilization_exponent=1.4)
        )
    assert err.value.field == "utilization_exponent"


@pytest.mark.unit
def test_server_data_utilization_is_a_fixed_input():
    m = dummy_time_block(3)
    m.unit = Server(**server_options(utilization=[0.2, 0.4, 0.6]))
    assert all(m.unit.utilization[t].fixed for t in range(3))
    assert m.unit.utilization[2].value == pytest.approx(0.6)
    assert m.unit.find_component("utilization_definition") is None


@pytest.mark.unit
def test_server_live_source_is_referenced_not_copied():
    m = dummy_time_block(3)
    m.source = pyo.Param(m.time_block.time_index, initialize=0.5, mutable=True)
    m.unit = Server(**server_options(utilization=m.source))
    assert not m.unit.utilization[0].fixed
    m.source[1] = 0.8
    m.unit.utilization[1].set_value(0.8)
    definition = m.unit.utilization_definition[1]
    assert pyo.value(definition.body) == pytest.approx(pyo.value(definition.upper))


@pytest.mark.unit
def test_server_nonlinear_curve_is_precomputed_from_data():
    m = dummy_time_block(3)
    m.unit = Server(**server_options(utilization_exponent=2.0))
    assert pyo.value(m.unit.utilization_curve[0]) == pytest.approx(0.25)


@pytest.mark.unit
def test_server_linear_curve_uses_utilization_directly():
    m = dummy_time_block(3)
    m.unit = Server(**server_options())
    assert m.unit.find_component("utilization_curve") is None


@pytest.mark.unit
def test_server_update_parameters_refreshes_power_curve():
    m = dummy_time_block(3)
    m.unit = Server(**server_options(utilization_exponent=2.0))
    m.unit.update_parameters({"utilization_exponent": 3.0})
    assert pyo.value(m.unit.utilization_curve[0]) == pytest.approx(0.125)


@pytest.mark.unit
def test_server_fan_power_follows_cube_law():
    m = dummy_time_block(3)
    m.unit = Server(**server_options(airflow=[0.5, 1.0, 0.9]))
    assert pyo.value(m.unit.fan_power[0]) == pytest.approx(0.3 * 0.5**3)
    assert pyo.value(m.unit.fan_power[1]) == pytest.approx(0.3)


@pytest.mark.unit
def test_server_without_fans_needs_no_airflow():
    m = dummy_time_block(3)
    m.unit = Server(
        utilization=[0.5] * 3, idle_power=4 * pyunits.kW, peak_power=12 * pyunits.kW
    )
    assert pyo.value(m.unit.fan_power[0]) == pytest.approx(0.0)


@pytest.mark.unit
def test_server_fans_require_airflow_and_rated_airflow():
    m = dummy_time_block(3)
    with pytest.raises(FlexConfigError) as err:
        m.a = Server(**server_options(rated_airflow=None))
    assert err.value.field == "rated_airflow"
    with pytest.raises(FlexConfigError) as err:
        m.b = Server(**server_options(airflow=None))
    assert err.value.field == "airflow"


@pytest.mark.unit
def test_server_rejects_non_positive_airflow():
    m = dummy_time_block(3)
    with pytest.raises(FlexConfigError) as err:
        m.unit = Server(**server_options(airflow=[0.9, 0.0, 0.9]))
    assert err.value.field == "airflow"


@pytest.mark.unit
def test_server_rejects_peak_below_idle():
    m = dummy_time_block(3)
    with pytest.raises(FlexConfigError) as err:
        m.unit = Server(**server_options(peak_power=3 * pyunits.kW))
    assert err.value.field == "peak_power"


@pytest.mark.unit
def test_server_rejects_liquid_fraction_outside_unit_interval():
    m = dummy_time_block(3)
    with pytest.raises(FlexConfigError) as err:
        m.unit = Server(**server_options(liquid_fraction=1.5))
    assert err.value.field == "liquid_fraction"


@pytest.mark.unit
def test_server_default_is_lp():
    m = dummy_time_block(3)
    m.unit = Server(**server_options())
    assert classify(m) is ProblemClass.LP


@pytest.mark.unit
def test_server_status_requires_rated_power():
    m = dummy_time_block(3)
    with pytest.raises(FlexConfigError) as err:
        m.unit = Server(
            **server_options(unit_commitment=UnitCommitmentConfig(status=True))
        )
    assert err.value.field == "rated_power"


@pytest.mark.unit
def test_server_live_source_emits_as_data(tmp_path):
    m = dummy_time_block(3)
    m.plant = fo.PlantBlock(time_block=m.time_block)
    m.source = pyo.Var(m.time_block.time_index, initialize={0: 0.2, 1: 0.4, 2: 0.6})
    m.source.fix()
    m.plant.server = Server(**server_options(utilization=m.source))
    with pytest.warns(FlexEmitWarning, match="utilization_definition"):
        spec = fo.emit_model(m)
    dump_spec(spec, tmp_path / "spec.json")
    rebuilt = fo.build_model(tmp_path / "spec.json").plant.server
    assert [rebuilt.utilization[t].value for t in range(3)] == pytest.approx(
        [0.2, 0.4, 0.6]
    )


@pytest.mark.component
def test_server_heat_splits_between_air_and_liquid():
    m = dummy_time_block(3)
    m.unit = Server(**server_options(liquid_fraction=0.3))
    solve(m)
    heat = pyo.value(m.unit.heat_output[0])
    assert heat == pytest.approx(pyo.value(m.unit.it_power[0]))
    assert pyo.value(m.unit.air_heat[0]) == pytest.approx(0.7 * heat)
    assert pyo.value(m.unit.liquid_heat[0]) == pytest.approx(0.3 * heat)


@pytest.mark.component
def test_server_status_gates_whole_draw():
    status = UnitCommitmentConfig(status=True)
    m = dummy_time_block(3)
    m.gated = Server(
        **server_options(unit_commitment=status, rated_power=20 * pyunits.kW)
    )
    m.free = Server(**server_options())
    m.gated.status[0].fix(0)
    m.gated.status[1].fix(1)
    m.gated.status[2].fix(1)
    solve(m)
    assert pyo.value(m.gated.power_electrical[0]) == pytest.approx(0.0, abs=1e-8)
    assert pyo.value(m.gated.heat_output[0]) == pytest.approx(0.0, abs=1e-8)
    for name in ("power_electrical", "heat_output"):
        gated, free = m.gated.find_component(name), m.free.find_component(name)
        assert pyo.value(gated[1]) == pytest.approx(pyo.value(free[1]))


@pytest.mark.component
def test_server_relax_gives_lp_whose_objective_bounds_the_milp():
    def build():
        m = dummy_time_block(3)
        m.costing = fo.FlexCosting(
            time_block=m.time_block,
            energy_prices={"electrical": 0.1 * currency_units("USD") / pyunits.kWh},
        )
        m.plant = fo.PlantBlock(time_block=m.time_block)
        m.plant.server = Server(
            **server_options(
                unit_commitment=UnitCommitmentConfig(status=True),
                rated_power=20 * pyunits.kW,
                costing_package=m.costing,
            )
        )
        m.must_run = pyo.Constraint(expr=sum(m.plant.server.status.values()) >= 1.5)
        m.costing.cost_process()
        m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
        return m

    milp, lp = build(), build()
    relax(lp.plant.server)
    assert classify(milp) is ProblemClass.MILP
    assert classify(lp) is ProblemClass.LP
    solve(milp)
    solve(lp)
    assert pyo.value(lp.objective) <= pyo.value(milp.objective) + 1e-8


@pytest.mark.component
def test_server_live_source_drives_power():
    prices = [0.1, 0.5, 0.1]

    def build(source_kind: str):
        m = dummy_time_block(3)
        m.costing = fo.FlexCosting(
            time_block=m.time_block,
            energy_prices={
                "electrical": [p * currency_units("USD") / pyunits.kWh for p in prices]
            },
        )
        m.plant = fo.PlantBlock(time_block=m.time_block)
        if source_kind == "data":
            utilization = [0.5] * 3
        else:
            m.work = pyo.Var(m.time_block.time_index, bounds=(0, 1), initialize=0.5)
            m.work_done = pyo.Constraint(expr=sum(m.work.values()) == 1.5)
            utilization = m.work
        m.plant.server = Server(
            **server_options(utilization=utilization, costing_package=m.costing)
        )
        m.costing.cost_process()
        m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
        return m

    data, fixed_live, free_live = build("data"), build("live"), build("live")
    fixed_live.work.fix(0.5)
    for m in (data, fixed_live, free_live):
        solve(m)
    for t in range(3):
        assert pyo.value(fixed_live.plant.server.power_electrical[t]) == pytest.approx(
            pyo.value(data.plant.server.power_electrical[t])
        )
    assert pyo.value(free_live.work[1]) == pytest.approx(0.0, abs=1e-8)
    assert pyo.value(free_live.objective) < pyo.value(data.objective)
