"""Tests for SpeciesReactor: RTD compartments with swappable feed and kinetics."""

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits
from pyomo.opt import assert_optimal_termination

from flexcore.exceptions import FlexConfigError
from flexcore.solvers import get_solver
from flexops.surrogates import MultilinearSurrogate
from flexops.testing import UnitModelTestHarness, dummy_time_block
from flexops.unit_models import SpeciesReactor


class TestSpeciesReactor(UnitModelTestHarness):
    """Fixing the feed flows and initial holdups determines the trajectory."""

    expected_dof = 0

    def configure(self):
        m = dummy_time_block(4)
        m.unit = SpeciesReactor(
            property_package=m.properties,
            inlet_names=("a", "b"),
            outlet_names=("purge", "c"),
            n_compartments=3,
            rtd_fractions=(0.5, 0.3, 0.2),
            yields={("c", "a"): 1.0, ("c", "b"): 0.5},
        )
        return m, m.unit


def _reactor(n: int = 6, **kwargs):
    """Build a fresh SpeciesReactor on an ``n``-point dummy_time_block."""
    m = dummy_time_block(n)
    m.unit = SpeciesReactor(property_package=m.properties, **kwargs)
    return m, m.unit


def _solve_with_feed(m, unit, feeds: dict[str, list[float]]):
    """Fix each named inlet flow to its series, then solve the LP."""
    for name, series in feeds.items():
        flow = unit.find_component(f"flow_in_{name}")
        for t, value in zip(m.time_block.time_index, series, strict=True):
            flow[t].fix(value)
    m.obj = pyo.Objective(expr=0)
    assert_optimal_termination(get_solver(model=m, prefer="highs").solve(m))


@pytest.mark.unit
def test_holdup_indexed_over_species_compartments_and_time():
    m, unit = _reactor(
        n=4, inlet_names=("a", "b"), outlet_names=("c",), n_compartments=3
    )
    assert len(unit.holdup) == 3 * 3 * 4


@pytest.mark.unit
@pytest.mark.parametrize("fractions", [(0.5, 0.4, 0.2), (0.5, 0.5)])
def test_rtd_fractions_must_sum_to_one_with_one_per_compartment(fractions):
    m = dummy_time_block(3)
    with pytest.raises(FlexConfigError, match="rtd_fractions"):
        m.unit = SpeciesReactor(
            property_package=m.properties, n_compartments=3, rtd_fractions=fractions
        )


@pytest.mark.unit
def test_species_names_shared_between_inlet_and_outlet_rejected():
    m = dummy_time_block(3)
    with pytest.raises(FlexConfigError, match="species"):
        m.unit = SpeciesReactor(
            property_package=m.properties, inlet_names=("a",), outlet_names=("a",)
        )


@pytest.mark.unit
def test_yield_naming_unknown_species_rejected():
    m = dummy_time_block(3)
    with pytest.raises(FlexConfigError, match="yields"):
        m.unit = SpeciesReactor(
            property_package=m.properties, yields={("product", "nope"): 1.0}
        )


@pytest.mark.unit
def test_rtd_rate_yield_and_compartment_time_are_regressable():
    _, unit = _reactor(inlet_names=("a",), outlet_names=("c", "d"), n_compartments=3)
    regressable = {r.name for r in unit._io_registry.parameters if r.regressable}
    assert {
        "rtd_fraction_1",
        "rtd_fraction_2",
        "compartment_time",
        "rate_constant_a",
        "yield_c_a",
        "yield_d_a",
    } <= regressable
    assert "rtd_fraction_3" not in regressable


@pytest.mark.unit
def test_feed_and_kinetics_relations_registered_with_time_only_targets():
    _, unit = _reactor(inlet_names=("a",), outlet_names=("c",), n_compartments=2)
    relations = {r.name: r for r in unit._io_registry.relations}
    expected = {
        "side_feed_a_1_relation",
        "reaction_rate_a_1_relation",
        "reaction_rate_a_2_relation",
        "reaction_rate_c_1_relation",
        "reaction_rate_c_2_relation",
    }
    assert expected <= set(relations)
    assert "side_feed_a_2_relation" not in relations
    for name in expected:
        assert relations[name].target.index_set().dimen == 1


@pytest.mark.unit
def test_conservation_equations_are_not_swappable():
    _, unit = _reactor()
    relations = {r.name for r in unit._io_registry.relations}
    assert not {"holdup_balance", "feed_closure", "outlet_flow_eq"} & relations


@pytest.mark.component
@pytest.mark.needs_highs
def test_single_compartment_step_matches_backward_euler_by_hand():
    m, unit = _reactor(n=4, n_compartments=1, rate_constants={"feed": 0.0})
    _solve_with_feed(m, unit, {"feed": [100.0] * 4})

    # tau_c = 1 hr, dt = 0.25 hr: h[t] = (h[t-1] + dt*F) / (1 + dt/tau_c).
    expected = [0.0, 20.0, 36.0, 48.8]
    for t, h in enumerate(expected):
        assert pyo.value(unit.holdup["feed", 1, t]) == pytest.approx(h)
        assert pyo.value(unit.flow_out_product[t]) == pytest.approx(h)


@pytest.mark.component
@pytest.mark.needs_highs
def test_inert_feed_conserves_flow_and_holdup_at_steady_state():
    m, unit = _reactor(
        n=4,
        n_compartments=3,
        residence_time=3 * pyunits.hr,
        rtd_fractions=(0.6, 0.3, 0.1),
        rate_constants={"feed": 0.0},
    )
    steady = {1: 60.0, 2: 90.0, 3: 100.0}
    for i, h in steady.items():
        unit.initial_holdup["feed", i] = h
    _solve_with_feed(m, unit, {"feed": [100.0] * 4})
    for t in m.time_block.time_index:
        assert pyo.value(unit.flow_out_product[t]) == pytest.approx(100.0)
        for i, h in steady.items():
            assert pyo.value(unit.holdup["feed", i, t]) == pytest.approx(h)


@pytest.mark.component
@pytest.mark.needs_highs
def test_channeling_responds_faster_than_back_mixing():
    outlet_at_first_step = {}
    for label, fractions in (("back_mixing", (1, 0, 0)), ("channeling", (0, 0, 1))):
        m, unit = _reactor(
            n=3, n_compartments=3, rtd_fractions=fractions, rate_constants={"feed": 0.0}
        )
        _solve_with_feed(m, unit, {"feed": [100.0] * 3})
        outlet_at_first_step[label] = pyo.value(unit.flow_out_product[1])
    assert outlet_at_first_step["channeling"] > outlet_at_first_step["back_mixing"]


@pytest.mark.component
@pytest.mark.needs_highs
def test_single_compartment_steady_conversion_matches_cstr():
    k, y, tau, fin = 2.0, 0.5, 1.0, 100.0
    m, unit = _reactor(
        n=3,
        outlet_names=("purge", "product"),
        n_compartments=1,
        rate_constants={"feed": k},
        yields={("product", "feed"): y},
    )
    feed_ss = fin * tau / (1 + k * tau)
    unit.initial_holdup["feed", 1] = feed_ss
    unit.initial_holdup["product", 1] = tau * y * k * feed_ss
    _solve_with_feed(m, unit, {"feed": [fin] * 3})
    for t in m.time_block.time_index:
        assert pyo.value(unit.flow_out_product[t]) == pytest.approx(
            y * k * tau / (1 + k * tau) * fin
        )
        assert pyo.value(unit.flow_out_purge[t]) == pytest.approx(feed_ss / tau)


@pytest.mark.component
@pytest.mark.needs_highs
def test_kinetics_relation_swaps_without_touching_conservation():
    m, unit = _reactor(n=3, n_compartments=1)
    surrogate = MultilinearSurrogate(
        {
            "input_variables": {"holdup_feed_1": "m^3"},
            "output_variables": {"reaction_rate_product_1": "m^3/hr"},
            "coefficients": {"intercept": 0.0, "holdup_feed_1": 0.5},
        }
    )
    unit.swap_relation("reaction_rate_product_1_relation", surrogate)

    assert not unit.reaction_rate_product_1_relation.active
    assert unit.holdup_balance.active
    _solve_with_feed(m, unit, {"feed": [100.0] * 3})
    for t in m.time_block.time_index:
        assert pyo.value(unit.reaction_rate["product", 1, t]) == pytest.approx(
            0.5 * pyo.value(unit.holdup["feed", 1, t])
        )


@pytest.mark.integration
@pytest.mark.needs_ipopt
def test_rtd_kinetics_and_yields_recovered_from_step_response_data():
    """Fit E, tau_c, k, and Y to data generated by a known reactor."""
    config = dict(
        inlet_names=("a",),
        outlet_names=("purge", "c"),
        n_compartments=2,
    )
    truth = {
        "rtd_fraction_1": 0.7,
        "compartment_time": 0.5,
        "rate_constant_a": 1.5,
        "yield_c_a": 0.8,
    }
    feed = [0, 50, 50, 50, 50, 120, 120, 120, 120, 120, 30, 30, 30, 30, 30, 30]
    n = len(feed)

    m, unit = _reactor(n=n, **config)
    unit.update_parameters(truth)
    _solve_with_feed(m, unit, {"a": feed})
    data = {
        name: [pyo.value(unit.find_component(f"flow_out_{name}")[t]) for t in range(n)]
        for name in ("purge", "c")
    }

    fit_m, fit = _reactor(n=n, **config)
    fit.update_parameters({name: value * 1.3 for name, value in truth.items()})
    for t, value in enumerate(feed):
        fit.flow_in_a[t].fix(value)
    for name in truth:
        fit.find_component(name).unfix()
    fit_m.obj = pyo.Objective(
        expr=sum(
            (fit.find_component(f"flow_out_{name}")[t] - data[name][t]) ** 2
            for name in data
            for t in range(n)
        )
    )
    assert_optimal_termination(get_solver(model=fit_m, prefer="ipopt").solve(fit_m))
    for name, value in truth.items():
        assert pyo.value(fit.find_component(name)) == pytest.approx(value, rel=1e-3)
