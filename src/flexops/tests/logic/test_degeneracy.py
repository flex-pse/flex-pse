"""Tests for manual parallel-train hierarchy declaration (architecture §3.5, R8).

``register_parallel_group`` operates over units the caller lists, and those
units live under a ``PlantBlock``/``NetworkBlock``; a bare ``pyo.Block`` stands
in for one here.
"""

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits
from pyomo.network import Arc

from flexcore.exceptions import FlexConfigError
from flexops.logic import add_status, register_parallel_group
from flexops.testing import dummy_time_block
from flexops.unit_models import Pump, ReverseOsmosis


def _satisfied(con):
    """Whether a ConstraintData holds at the Vars' current values."""
    body = pyo.value(con.body)
    lower, upper = con.lower, con.upper
    ok = True
    if lower is not None:
        ok = ok and pyo.value(lower) <= body + 1e-9
    if upper is not None:
        ok = ok and body <= pyo.value(upper) + 1e-9
    return ok


def _plant_pump_and_ro_trains(n: int = 4):
    """One Pump feeding three parallel ReverseOsmosis trains via Arcs."""
    m = dummy_time_block(n)
    m.plant = pyo.Block()
    m.plant.pump = Pump(property_package=m.properties)
    for i in range(3):
        m.plant.add_component(f"ro{i}", ReverseOsmosis(property_package=m.properties))
    for i in range(3):
        ro = m.plant.component(f"ro{i}")
        m.plant.add_component(
            f"pump_to_ro{i}", Arc(source=m.plant.pump.outlet, destination=ro.inlet)
        )
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)
    return m


@pytest.mark.unit
def test_register_parallel_group_orders_lower_index_first():
    """Registering 3 RO trains chains ro0.status >= ro1.status >= ro2.status."""
    m = _plant_pump_and_ro_trains()
    for i in range(3):
        ro = m.plant.component(f"ro{i}")
        add_status(ro, ro.power_electrical, 0.0, 100.0)

    cons = register_parallel_group([m.plant.ro0, m.plant.ro1, m.plant.ro2])

    # Each pair's constraint lands on the *later* unit, so ro0 carries none.
    assert len(cons) == 2
    assert cons[0] is m.plant.ro1.conditional
    assert cons[1] is m.plant.ro2.conditional
    assert not hasattr(m.plant.ro0, "conditional")

    # ro0 on, ro1 off: the canonical order (a train runs only if its
    # predecessor runs).
    m.plant.ro0.status[0].set_value(1)
    m.plant.ro1.status[0].set_value(0)
    m.plant.ro2.status[0].set_value(0)
    assert _satisfied(m.plant.ro1.conditional[0])
    assert _satisfied(m.plant.ro2.conditional[0])

    # ro0 off, ro1 on: violates the ro0 -> ro1 link.
    m.plant.ro0.status[0].set_value(0)
    m.plant.ro1.status[0].set_value(1)
    assert not _satisfied(m.plant.ro1.conditional[0])


@pytest.mark.unit
def test_register_parallel_group_orders_scalar_variables():
    """A scalar Var (RO energy intensity) gets one ordering Constraint per pair."""
    m = _plant_pump_and_ro_trains()
    for i in range(3):
        ro = m.plant.component(f"ro{i}")
        add_status(ro, ro.power_electrical, 0.0, 100.0)

    register_parallel_group(
        [m.plant.ro0, m.plant.ro1, m.plant.ro2], variables=["energy_intensity"]
    )

    assert not m.plant.ro1.energy_intensity_ordering.is_indexed()
    assert not hasattr(m.plant.ro0, "energy_intensity_ordering")

    # intensity0 >= intensity1 holds.
    m.plant.ro0.energy_intensity.set_value(6.0)
    m.plant.ro1.energy_intensity.set_value(4.0)
    m.plant.ro2.energy_intensity.set_value(4.0)
    assert _satisfied(m.plant.ro1.energy_intensity_ordering)
    assert _satisfied(m.plant.ro2.energy_intensity_ordering)

    # intensity0 < intensity1 violates it.
    m.plant.ro1.energy_intensity.set_value(8.0)
    assert not _satisfied(m.plant.ro1.energy_intensity_ordering)


@pytest.mark.unit
def test_register_parallel_group_orders_time_indexed_variables():
    """A time-indexed Var (RO permeate flow) gets one ordering Constraint per t."""
    m = _plant_pump_and_ro_trains()
    for i in range(3):
        ro = m.plant.component(f"ro{i}")
        add_status(ro, ro.power_electrical, 0.0, 100.0)

    register_parallel_group(
        [m.plant.ro0, m.plant.ro1, m.plant.ro2], variables=["permeate"]
    )

    assert set(m.plant.ro1.permeate_ordering) == set(m.time_block.time_index)

    m.plant.ro0.permeate[0].set_value(10.0)
    m.plant.ro1.permeate[0].set_value(5.0)
    assert _satisfied(m.plant.ro1.permeate_ordering[0])

    m.plant.ro1.permeate[0].set_value(20.0)
    assert not _satisfied(m.plant.ro1.permeate_ordering[0])


@pytest.mark.unit
def test_register_parallel_group_rejects_unknown_variable():
    """A variable name that does not resolve on every unit -> FlexConfigError."""
    m = _plant_pump_and_ro_trains()
    for i in range(2):
        ro = m.plant.component(f"ro{i}")
        add_status(ro, ro.power_electrical, 0.0, 100.0)

    with pytest.raises(FlexConfigError):
        register_parallel_group(
            [m.plant.ro0, m.plant.ro1], variables=["split_fraction"]
        )


@pytest.mark.component
@pytest.mark.needs_highs
def test_registered_group_makes_solve_deterministic():
    """A manually registered group also selects the lower-indexed train."""
    from flexcore.solvers import get_solver

    m = dummy_time_block(1)
    m.plant = pyo.Block()
    m.plant.u0 = Pump(property_package=m.properties)
    m.plant.u1 = Pump(property_package=m.properties)

    # Same construction as test_symmetry_breaking_makes_solve_deterministic:
    # two solver-equivalent optima absent symmetry breaking.
    flows = {}
    for name in ("u0", "u1"):
        unit = getattr(m.plant, name)
        flow = pyo.Reference(unit.inlet_state.flow_vol_phase[:, "Liq"])
        flows[name] = flow
        add_status(
            unit,
            flow,
            50.0 * pyunits.m**3 / pyunits.hr,
            300.0 * pyunits.m**3 / pyunits.hr,
        )

    register_parallel_group([m.plant.u0, m.plant.u1])

    m.demand = pyo.Constraint(expr=flows["u0"][0] + flows["u1"][0] == 50.0)

    @m.Objective(sense=pyo.minimize)
    def total_power(b):
        return b.plant.u0.power_electrical[0] + b.plant.u1.power_electrical[0]

    results = get_solver(model=m, prefer="highs").solve(m)
    pyo.assert_optimal_termination(results)

    assert pyo.value(m.plant.u0.status[0]) == pytest.approx(1.0)
    assert pyo.value(m.plant.u1.status[0]) == pytest.approx(0.0)


@pytest.mark.unit
def test_register_parallel_group_requires_at_least_two_units():
    """A single-unit list is not a group -> FlexConfigError."""
    m = _plant_pump_and_ro_trains()
    add_status(m.plant.ro0, m.plant.ro0.power_electrical, 0.0, 100.0)

    with pytest.raises(FlexConfigError):
        register_parallel_group([m.plant.ro0])


@pytest.mark.unit
def test_register_parallel_group_rejects_duplicate_units():
    """The same unit repeated in the list -> FlexConfigError."""
    m = _plant_pump_and_ro_trains()
    for i in range(2):
        ro = m.plant.component(f"ro{i}")
        add_status(ro, ro.power_electrical, 0.0, 100.0)

    with pytest.raises(FlexConfigError):
        register_parallel_group([m.plant.ro0, m.plant.ro1, m.plant.ro0])


@pytest.mark.unit
def test_order_status_false_orders_units_without_a_status_var():
    """A group carrying no status Var at all still gets its Vars ordered."""
    m = _plant_pump_and_ro_trains()  # deliberately no add_status on any train

    cons = register_parallel_group(
        [m.plant.ro0, m.plant.ro1, m.plant.ro2],
        variables=["recovery"],
        order_status=False,
    )

    assert cons == []
    for i in range(3):
        assert not hasattr(m.plant.component(f"ro{i}"), "status")

    # The one name chains across the whole group: ro0 >= ro1 >= ro2.
    assert not hasattr(m.plant.ro0, "recovery_ordering")
    m.plant.ro0.recovery[0].set_value(0.6)
    m.plant.ro1.recovery[0].set_value(0.5)
    m.plant.ro2.recovery[0].set_value(0.4)
    assert _satisfied(m.plant.ro1.recovery_ordering[0])
    assert _satisfied(m.plant.ro2.recovery_ordering[0])

    m.plant.ro2.recovery[0].set_value(0.9)
    assert not _satisfied(m.plant.ro2.recovery_ordering[0])


@pytest.mark.unit
def test_order_status_true_rejects_units_without_a_status_var():
    """The default path needs a status Var -- exactly what order_status frees."""
    m = _plant_pump_and_ro_trains()

    with pytest.raises(FlexConfigError):
        register_parallel_group([m.plant.ro0, m.plant.ro1], variables=["recovery"])


@pytest.mark.unit
def test_order_status_false_leaves_status_unconstrained():
    """Units that do carry status keep their on/off order free."""
    m = _plant_pump_and_ro_trains()
    for i in range(3):
        ro = m.plant.component(f"ro{i}")
        add_status(ro, ro.power_electrical, 0.0, 100.0)

    before = {id(c) for c in m.component_data_objects(pyo.Constraint, active=True)}
    cons = register_parallel_group(
        [m.plant.ro0, m.plant.ro1, m.plant.ro2],
        variables=["recovery"],
        order_status=False,
    )
    added = [
        c
        for c in m.component_data_objects(pyo.Constraint, active=True)
        if id(c) not in before
    ]

    # The *only* thing registration added is the recovery ordering: nothing new
    # touches status, so ro0 off while ro1 on -- which the default path forbids
    # -- is left feasible.
    assert cons == []
    assert {c.parent_component().local_name for c in added} == {"recovery_ordering"}
    assert len(added) == 2 * len(m.time_block.time_index)
    for i in range(3):
        assert not hasattr(m.plant.component(f"ro{i}"), "conditional")


@pytest.mark.unit
def test_order_status_false_still_orders_variables():
    """Releasing the status order leaves the continuous ordering fully in force."""
    m = _plant_pump_and_ro_trains()
    for i in range(3):
        ro = m.plant.component(f"ro{i}")
        add_status(ro, ro.power_electrical, 0.0, 100.0)

    register_parallel_group(
        [m.plant.ro0, m.plant.ro1, m.plant.ro2],
        variables=["recovery"],
        order_status=False,
    )

    m.plant.ro0.recovery[0].set_value(0.6)
    m.plant.ro1.recovery[0].set_value(0.4)
    m.plant.ro2.recovery[0].set_value(0.4)
    assert _satisfied(m.plant.ro1.recovery_ordering[0])
    assert _satisfied(m.plant.ro2.recovery_ordering[0])

    m.plant.ro1.recovery[0].set_value(0.8)
    assert not _satisfied(m.plant.ro1.recovery_ordering[0])


@pytest.mark.unit
def test_order_status_false_without_variables_is_rejected():
    """Ordering neither status nor any variable orders nothing -> FlexConfigError."""
    m = _plant_pump_and_ro_trains()
    for i in range(2):
        ro = m.plant.component(f"ro{i}")
        add_status(ro, ro.power_electrical, 0.0, 100.0)

    with pytest.raises(FlexConfigError):
        register_parallel_group([m.plant.ro0, m.plant.ro1], order_status=False)


@pytest.mark.component
@pytest.mark.needs_highs
def test_order_status_false_orders_a_degenerate_split_deterministically():
    """Ordering a flow alone breaks a split symmetry, with no UC in the model."""
    from flexcore.solvers import get_solver

    m = dummy_time_block(1)
    m.plant = pyo.Block()
    m.plant.u0 = Pump(property_package=m.properties)
    m.plant.u1 = Pump(property_package=m.properties)

    # No add_status anywhere: neither pump carries a status Var, so the default
    # path could not register this group at all.
    register_parallel_group(
        [m.plant.u0, m.plant.u1], variables=["flow_in"], order_status=False
    )

    m.demand = pyo.Constraint(
        expr=m.plant.u0.flow_in[0] + m.plant.u1.flow_in[0] == 50.0
    )

    # Minimizing u0's flow alone would push the whole duty onto u1 (0/50); the
    # ordering u0 >= u1 forbids that, leaving the even split as the unique
    # optimum.
    @m.Objective(sense=pyo.minimize)
    def lead_flow(b):
        return b.plant.u0.flow_in[0]

    results = get_solver(model=m, prefer="highs").solve(m)
    pyo.assert_optimal_termination(results)

    assert pyo.value(m.plant.u0.flow_in[0]) == pytest.approx(25.0, rel=1e-6)
    assert pyo.value(m.plant.u1.flow_in[0]) == pytest.approx(25.0, rel=1e-6)
