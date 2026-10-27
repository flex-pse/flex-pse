"""Product(OpsBlockData): a boundary sink -- N inlets, zero outlets (§3.2, §3.4)."""

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits
from pyomo.network import Port

from flexcore.exceptions import FlexConfigError
from flexops.core.registration import BoundaryKind
from flexops.testing import (
    UnitModelTestHarness,
    dummy_gas_time_block,
    dummy_time_block,
)
from flexops.unit_models import Product

_INFLOW = {0: 2.0, 1: 3.0, 2: 4.0}


def _product(n: int = 3, **kwargs):
    """Build a Product on an ``n``-point aqueous ``dummy_time_block``."""
    m = dummy_time_block(n)
    m.potable = Product(property_package=m.properties, **kwargs)
    return m, m.potable


def _gas_product(n: int = 3, **kwargs):
    """Build a Product on an ``n``-point ``dummy_gas_time_block``."""
    m = dummy_gas_time_block(n)
    m.flue = Product(property_package=m.properties, **kwargs)
    return m, m.flue


def _set_inflows(unit, names, profile) -> None:
    """Set each named inlet's flow to ``profile`` (the harness fixes them)."""
    for name in names:
        flow = unit.find_component(f"flow_in_{name}")
        for t, value in profile.items():
            flow[t].set_value(value)


def _registered(unit, role: str) -> set[str]:
    """Return ``{"<state block>.<var>"}`` for IO variables in ``role``."""
    return {
        f"{rec.var.parent_block().local_name}.{rec.name}"
        for rec in unit._io_registry.io_variables
        if rec.role == role
    }


# -- harness ----------------------------------------------------------------


class TestProductAqueous(UnitModelTestHarness):
    """One aqueous inlet: the arriving flow is the registered input.

    ``add_stream_ports`` registers an inlet port's flow as ``role="input"``,
    so fixing the registered inputs pins the arriving stream and
    ``eq_delivery`` determines ``delivery`` -- zero degrees of freedom.
    """

    expected_dof = 0
    expected_solution = {"delivery[0]": 2.0, "delivery[2]": 4.0}

    def configure(self):
        m, unit = _product(3, inlet_names=("a",))
        _set_inflows(unit, ("a",), _INFLOW)
        return m, unit


class TestProductAqueousTwoInlets(UnitModelTestHarness):
    """Two aqueous inlets aggregate into one delivery, with no blending."""

    expected_dof = 0
    expected_solution = {"delivery[0]": 4.0, "delivery[2]": 8.0}

    def configure(self):
        m, unit = _product(3, inlet_names=("a", "b"))
        _set_inflows(unit, ("a", "b"), _INFLOW)
        return m, unit


class TestProductGas(UnitModelTestHarness):
    """Two gas inlets: the intensive states stay independent and untied."""

    expected_dof = 0
    expected_solution = {"delivery[1]": 6.0}

    def configure(self):
        m, unit = _gas_product(3, inlet_names=("a", "b"))
        _set_inflows(unit, ("a", "b"), _INFLOW)
        return m, unit


class TestProductHorizonDemand(UnitModelTestHarness):
    """A horizon-basis limit is degree-of-freedom neutral.

    ``delivery_total`` is one free scalar Var, and ``eq_delivery_total`` is one
    scalar equality determining it, so the accounting is unchanged from
    :class:`TestProductAqueous`. The total is ``(2 + 3 + 4) m**3/hr * 15 min``.
    """

    expected_dof = 0
    expected_solution = {"delivery[0]": 2.0, "delivery_total": 2.25}

    def configure(self):
        m, unit = _product(
            3,
            inlet_names=("a",),
            max_demand=100 * pyunits.m**3,
            demand_basis="horizon",
        )
        _set_inflows(unit, ("a",), _INFLOW)
        return m, unit


# -- topology ---------------------------------------------------------------


@pytest.mark.unit
def test_product_builds_one_port_per_inlet_name_and_no_outlet():
    """A sink has ``inlet_<name>`` ports only -- never an outlet."""
    _, unit = _product(3, inlet_names=("municipal", "reuse"))

    ports = {p.local_name for p in unit.component_objects(Port, descend_into=False)}
    assert ports == {"inlet_municipal", "inlet_reuse"}
    for port in ports:
        assert len(list(unit.find_component(port).values())) > 0
    assert unit.find_component("outlet") is None
    assert unit.find_component("outlet_state") is None


@pytest.mark.unit
def test_product_declares_no_power():
    """A boundary block has no energy relation -- it draws and exports nothing."""
    _, unit = _product(3)
    assert unit._io_registry.power == []
    assert unit._io_registry.fuel == []
    assert unit.find_component("power_electrical") is None
    assert unit.find_component("power_thermal") is None


@pytest.mark.unit
def test_product_does_not_blend_composition():
    """No state ties: blending is Mixer's job, and it is bilinear."""
    _, unit = _gas_product(3, inlet_names=("a", "b"))
    for state_var in ("pressure", "temperature"):
        assert unit.find_component(f"inlet_state_equality_{state_var}") is None

    outputs = _registered(unit, "output")
    for name in ("a", "b"):
        for state_var in ("pressure", "temperature"):
            assert f"inlet_{name}_state.{state_var}" in outputs


# -- delivery balance -------------------------------------------------------


@pytest.mark.unit
def test_product_delivery_balance_body():
    """``delivery[t]`` equals the sum of the inlet flows -- nothing else."""
    _, unit = _product(3, inlet_names=("a", "b", "c"))
    for t in range(3):
        unit.delivery[t].fix(6.0)
        unit.flow_in_a[t].fix(1.0)
        unit.flow_in_b[t].fix(2.0)
        unit.flow_in_c[t].fix(3.0)
        assert pyo.value(unit.eq_delivery[t].body) == pytest.approx(0.0, abs=1e-9)
        unit.flow_in_c[t].fix(4.0)
        assert pyo.value(unit.eq_delivery[t].body) == pytest.approx(-1.0, abs=1e-9)


@pytest.mark.unit
def test_product_delivery_is_a_var_so_it_can_be_dispatched():
    """``delivery`` is a fixable Var, not an Expression."""
    m, unit = _product(3)
    assert isinstance(unit.delivery, pyo.Var)
    unit.set_external_dispatch(unit.delivery, _INFLOW)
    for t in m.time_block.time_index:
        assert unit.delivery[t].fixed
        assert pyo.value(unit.delivery[t]) == pytest.approx(_INFLOW[t])


# -- demand limits ----------------------------------------------------------


@pytest.mark.unit
def test_product_limits_are_built_only_when_configured():
    """Limit Params/Constraints exist iff their config option was given."""
    _, bare = _product(3)
    for name in (
        "delivery_min",
        "delivery_max",
        "delivery_min_limit",
        "delivery_max_limit",
    ):
        assert bare.find_component(name) is None

    _, bounded = _product(
        3,
        min_demand=1 * pyunits.m**3 / pyunits.hr,
        max_demand=9 * pyunits.m**3 / pyunits.hr,
    )
    for name in (
        "delivery_min",
        "delivery_max",
        "delivery_min_limit",
        "delivery_max_limit",
    ):
        assert bounded.find_component(name) is not None


@pytest.mark.unit
def test_product_limit_bodies_and_mutable_params():
    """The limits bound ``delivery``, and rewriting a Param moves the bound."""
    _, unit = _product(3, min_demand=2 * pyunits.m**3 / pyunits.hr)
    for t in range(3):
        assert pyo.value(unit.delivery_min[t]) == pytest.approx(2.0)
        unit.delivery[t].fix(5.0)
        limit = unit.delivery_min_limit[t]
        assert pyo.value(limit.body) == pytest.approx(5.0)
        assert pyo.value(limit.lower) == pytest.approx(2.0)

    # The Param is mutable, so rewriting it moves the bound in place.
    unit.delivery_min[1].set_value(4.0)
    assert pyo.value(unit.delivery_min_limit[1].lower) == pytest.approx(4.0)
    assert pyo.value(unit.delivery_min_limit[0].lower) == pytest.approx(2.0)


@pytest.mark.unit
def test_product_composition_limits_are_per_inlet():
    """``add_time_limits`` bounds one named inlet's state, not a blend."""
    from flexops.unit_models.base._boundary import add_time_limits

    _, unit = _gas_product(3, inlet_names=("a", "b"))
    # On-demand properties are built lazily through StateBlockData.__getattr__,
    # so a state-block property is resolved with getattr, not find_component.
    state_name = "pressure"
    pressure = getattr(unit.inlet_a_state, state_name)
    add_time_limits(unit, pressure, "inlet_a_pressure", upper=200000 * pyunits.Pa)
    assert unit.find_component("inlet_a_pressure_max") is not None
    assert unit.find_component("inlet_b_pressure_max") is None
    for t in range(3):
        unit.inlet_a_state.pressure[t].fix(101325.0)
        limit = unit.inlet_a_pressure_max_limit[t]
        assert pyo.value(limit.body) == pytest.approx(101325.0)
        assert pyo.value(limit.upper) == pytest.approx(200000.0)


# -- horizon-basis demand limits --------------------------------------------


@pytest.mark.unit
def test_product_horizon_basis_builds_a_scalar_total():
    """``demand_basis="horizon"`` bounds one scalar total, not each period."""
    _, unit = _product(3, max_demand=100 * pyunits.m**3, demand_basis="horizon")

    assert isinstance(unit.delivery_total, pyo.Var)
    assert not unit.delivery_total.is_indexed()
    assert unit.find_component("eq_delivery_total") is not None
    assert not unit.delivery_max.is_indexed()
    assert not unit.delivery_max_limit.is_indexed()


@pytest.mark.unit
def test_product_period_basis_is_the_default_and_builds_no_total():
    """Without ``demand_basis`` the limits stay per-period, as before."""
    _, unit = _product(3, max_demand=9 * pyunits.m**3 / pyunits.hr)

    assert unit.find_component("delivery_total") is None
    assert unit.find_component("eq_delivery_total") is None
    assert unit.delivery_max.is_indexed()


@pytest.mark.unit
def test_product_horizon_total_is_the_time_integral_of_delivery():
    """``delivery_total`` equals ``sum_t delivery[t] * dt`` -- 15-minute steps."""
    _, unit = _product(3, max_demand=100 * pyunits.m**3, demand_basis="horizon")
    for t, value in _INFLOW.items():
        unit.delivery[t].fix(value)

    # (2 + 3 + 4) m**3/hr * 0.25 hr == 2.25 m**3.
    unit.delivery_total.set_value(2.25)
    assert pyo.value(unit.eq_delivery_total.body) == pytest.approx(0.0, abs=1e-9)
    unit.delivery_total.set_value(3.0)
    assert pyo.value(unit.eq_delivery_total.body) == pytest.approx(0.75, abs=1e-9)


@pytest.mark.unit
def test_product_horizon_limit_params_are_scalar_and_mutable():
    """The horizon limits bound the total, and rewriting a Param moves them."""
    _, unit = _product(
        3,
        min_demand=1 * pyunits.m**3,
        max_demand=9 * pyunits.m**3,
        demand_basis="horizon",
    )
    unit.delivery_total.set_value(5.0)
    assert pyo.value(unit.delivery_max_limit.body) == pytest.approx(5.0)
    assert pyo.value(unit.delivery_max_limit.upper) == pytest.approx(9.0)
    assert pyo.value(unit.delivery_min_limit.lower) == pytest.approx(1.0)

    # The Param is mutable, so rewriting it moves the bound in place.
    unit.delivery_max.set_value(6.0)
    assert pyo.value(unit.delivery_max_limit.upper) == pytest.approx(6.0)


@pytest.mark.unit
def test_product_horizon_limits_are_converted_to_one_basis():
    """Limits given in mixed units land on the upper limit's basis."""
    from pyomo.util.check_units import assert_units_consistent

    _, unit = _product(
        3,
        min_demand=1000 * pyunits.L,
        max_demand=5 * pyunits.m**3,
        demand_basis="horizon",
    )
    assert_units_consistent(unit)
    assert pyo.value(unit.delivery_min) == pytest.approx(1.0)
    assert pyo.value(unit.delivery_max) == pytest.approx(5.0)


@pytest.mark.unit
def test_product_horizon_basis_leaves_delivery_time_indexed():
    """The metered Var is untouched: costing, aggregation and dispatch still work."""
    m, unit = _product(3, max_demand=100 * pyunits.m**3, demand_basis="horizon")

    assert unit.delivery.is_indexed()
    assert unit.delivery.index_set() is m.time_block.time_index
    (record,) = unit._io_registry.boundary
    assert record.kind is BoundaryKind.PRODUCT
    assert record.var is unit.delivery

    unit.set_external_dispatch(unit.delivery, _INFLOW)
    for t in m.time_block.time_index:
        assert unit.delivery[t].fixed


@pytest.mark.unit
def test_product_demand_basis_alone_builds_nothing():
    """A basis with no limit configured is not a limit -- nothing is built."""
    _, unit = _product(3, demand_basis="horizon")
    for name in ("delivery_total", "eq_delivery_total", "delivery_min", "delivery_max"):
        assert unit.find_component(name) is None


# -- boundary registration --------------------------------------------------


@pytest.mark.unit
def test_product_registers_its_delivery_as_a_product_boundary_flow():
    """One boundary record, keyed PRODUCT, carrying the delivery Var."""
    _, unit = _product(3)
    (record,) = unit._io_registry.boundary
    assert record.kind is BoundaryKind.PRODUCT
    assert record.var is unit.delivery
    assert record.name == "delivery"


@pytest.mark.unit
def test_product_resource_name_defaults_to_the_block_local_name():
    """``resource_name=None`` falls back to the Pyomo block name."""
    _, unit = _product(3)
    (record,) = unit._io_registry.boundary
    assert record.resource == "potable"


@pytest.mark.unit
def test_product_explicit_resource_name_overrides_without_renaming_the_block():
    """The aggregation key is independent of the block name."""
    _, unit = _product(3, resource_name="potable_water")
    (record,) = unit._io_registry.boundary
    assert record.resource == "potable_water"
    assert unit.local_name == "potable"


# -- config rejection -------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("inlet_names", [(), ("a", "a"), ("a", ""), ("a", 1)])
def test_product_rejects_bad_inlet_names(inlet_names):
    """Empty, duplicated, or non-string ``inlet_names`` raise, naming the field."""
    with pytest.raises(FlexConfigError) as excinfo:
        _product(3, inlet_names=inlet_names)
    assert excinfo.value.field == "inlet_names"


@pytest.mark.unit
def test_product_rejects_a_rate_limit_on_the_horizon_basis():
    """A rate given on the horizon basis is a units error, not a silent rescale."""
    with pytest.raises(FlexConfigError) as excinfo:
        _product(
            3,
            max_demand=100 * pyunits.m**3 / pyunits.hr,
            demand_basis="horizon",
        )
    assert excinfo.value.field == "max_demand"


@pytest.mark.unit
def test_product_rejects_a_quantity_limit_on_the_period_basis():
    """The mirror slip: a horizon quantity left on the default period basis."""
    with pytest.raises(FlexConfigError) as excinfo:
        _product(3, max_demand=2400 * pyunits.m**3)
    assert excinfo.value.field == "max_demand"


@pytest.mark.unit
def test_product_rejects_an_unknown_demand_basis():
    """``demand_basis`` is one of the two declared bases, naming the field."""
    with pytest.raises(FlexConfigError) as excinfo:
        _product(3, demand_basis="daily")
    assert excinfo.value.field == "demand_basis"


@pytest.mark.unit
def test_product_price_without_a_costing_package_raises():
    """A price that could not be billed is rejected, never silently dropped."""
    with pytest.raises(FlexConfigError) as excinfo:
        _product(3, price=-0.5)
    assert excinfo.value.field == "price"


@pytest.mark.unit
def test_product_is_in_the_unit_model_registry():
    """``Product`` is reachable from ``flexops`` and its unit-model registry."""
    import flexops
    from flexops import unit_models

    assert "Product" in unit_models.__all__
    assert flexops.Product is Product
