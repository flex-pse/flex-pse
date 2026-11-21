"""ReverseOsmosis: feed -> permeate + brine (§3.4, R6)."""

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits

from flexcore.config.schema import SurrogateSpec, SurrogateType
from flexcore.exceptions import FlexConfigError
from flexops import SimpleAqueousFlow, TimeBlock
from flexops.surrogates import surrogate_from_spec
from flexops.testing import UnitModelTestHarness, dummy_time_block
from flexops.unit_models import ReverseOsmosis


class TestReverseOsmosis(UnitModelTestHarness):
    """RO skid: feed -> permeate (outlet_a) + brine (outlet_b)."""

    expected_dof = 0

    def configure(self):
        m = dummy_time_block(3)
        m.unit = ReverseOsmosis(property_package=m.properties)
        return m, m.unit


@pytest.mark.unit
def test_reverseosmosis_outlet_semantics():
    """The skid's ``recovery`` is its permeate fraction of the feed."""
    m = dummy_time_block(3)
    m.unit = ReverseOsmosis(property_package=m.properties, recovery=0.45)
    for t in m.time_block.time_index:
        m.unit.feed[t].fix(4.0)
        m.unit.permeate[t].fix(1.8)
        m.unit.brine[t].fix(2.2)
        assert pyo.value(m.unit.split_definition[t].body) == pytest.approx(
            1.8 - 0.45 * 4.0, abs=1e-9
        )
        assert pyo.value(m.unit.split_mass_balance[t].body) == pytest.approx(
            0.0, abs=1e-9
        )


@pytest.mark.component
def test_recovery_relation_is_swappable():
    """A richer recovery model replaces the split definition in place.

    ``split_definition`` (determining ``permeate``, RO's name for
    ``flow_out_a``) is registered as a swappable relation (M10, extended
    beyond energy relationships). Conservation (``split_mass_balance``) is
    never registered and so is untouched by the swap.
    """
    m = pyo.ConcreteModel()
    m.time_block = TimeBlock(
        start_date="2025-01-01", end_date="2025-01-01T00:45", time_step=15 * pyunits.min
    )
    m.properties = SimpleAqueousFlow(has_pressure=True)
    m.unit = ReverseOsmosis(property_package=m.properties)

    m.unit.swap_relation(
        "split_definition",
        surrogate_from_spec(
            SurrogateSpec(
                surrogate_type=SurrogateType.MULTILINEAR,
                data={
                    "input_variables": {
                        "feed": "m^3/hr",
                        "inlet_state.pressure": "Pa",
                    },
                    "output_variables": {"permeate": "m^3/hr"},
                    "coefficients": {
                        "intercept": 0.3,
                        "feed": 0.01,
                        "inlet_state.pressure": 1e-6,
                        "feed*inlet_state.pressure": 1e-7,
                    },
                },
            )
        ),
    )

    m.unit.inlet_state.pressure[0].set_value(3.0e5)
    m.unit.feed[0].set_value(10.0)
    m.unit.permeate[0].set_value(0.0)

    assert m.unit.split_definition[0].active is False
    assert m.unit.split_mass_balance[0].active is True
    # permeate - (0.3 + 0.01*10 + 1e-6*3e5 + 1e-7*10*3e5) == -(0.3+0.1+0.3+0.3)
    assert pyo.value(m.unit.surrogate_split_definition.fitted[0].body) == pytest.approx(
        -1.0
    )


@pytest.mark.unit
def test_reverseosmosis_renames_the_split_fraction_to_recovery():
    """``recovery`` replaces the inherited ``split_fraction`` — it is not an alias."""
    m = dummy_time_block(3)
    m.unit = ReverseOsmosis(property_package=m.properties)

    assert pyo.value(m.unit.recovery[0]) == pytest.approx(0.45)
    assert m.unit.find_component("split_fraction") is None
    with pytest.raises(ValueError, match="split_fraction"):
        m.rejected = ReverseOsmosis(property_package=m.properties, split_fraction=0.45)


@pytest.mark.unit
def test_reverseosmosis_renames_the_flows_to_feed_permeate_brine():
    """``feed``/``permeate``/``brine`` replace ``flow_in``/``flow_out_a``/``_b``."""
    m = dummy_time_block(3)
    m.unit = ReverseOsmosis(property_package=m.properties)

    assert m.unit.find_component("flow_in") is None
    assert m.unit.find_component("flow_out_a") is None
    assert m.unit.find_component("flow_out_b") is None
    for name in ("feed", "permeate", "brine"):
        assert m.unit.find_component(name) is not None

    # Ports are unaffected by the flow rename.
    for port in ("inlet", "outlet_a", "outlet_b"):
        assert m.unit.find_component(port) is not None


@pytest.mark.unit
def test_reverseosmosis_recovery_window_bounds_the_recovery():
    """``recovery_min``/``_max`` override the default seawater window."""
    m = dummy_time_block(3)
    m.unit = ReverseOsmosis(
        property_package=m.properties,
        recovery=0.4,
        recovery_min=0.3,
        recovery_max=0.6,
    )

    for t in m.time_block.time_index:
        assert m.unit.recovery[t].bounds == (0.3, 0.6)
        assert m.unit.recovery[t].fixed
        assert pyo.value(m.unit.recovery[t]) == pytest.approx(0.4)


@pytest.mark.unit
def test_reverseosmosis_recovery_is_time_indexed_and_not_regressable():
    """``recovery[t]`` is a per-step Var, kept out of the regressable parameters."""
    m = dummy_time_block(3)
    m.unit = ReverseOsmosis(property_package=m.properties)

    assert list(m.unit.recovery.index_set()) == list(m.time_block.time_index)
    regressable = {r.name for r in m.unit._io_registry.parameters if r.regressable}
    assert "recovery" not in regressable
    assert "energy_intensity" in regressable


@pytest.mark.unit
def test_reverseosmosis_recovery_can_differ_by_step():
    """Each step's split definition reads that step's recovery."""
    m = dummy_time_block(3)
    m.unit = ReverseOsmosis(property_package=m.properties, recovery=0.45)
    m.unit.recovery[1].fix(0.55)
    for t in (0, 1):
        m.unit.feed[t].set_value(10.0)
        m.unit.permeate[t].set_value(0.0)

    assert pyo.value(m.unit.split_definition[0].body) == pytest.approx(-4.5)
    assert pyo.value(m.unit.split_definition[1].body) == pytest.approx(-5.5)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("options", "field"),
    [
        ({"recovery": 0.9, "recovery_max": 0.6}, "recovery"),
        ({"recovery_min": 0.7, "recovery_max": 0.6}, "recovery_min"),
    ],
)
def test_reverseosmosis_rejects_an_unusable_recovery_window(options, field):
    """An inverted window, or a setpoint outside it, is rejected by name."""
    m = dummy_time_block(3)
    with pytest.raises(FlexConfigError) as excinfo:
        m.unit = ReverseOsmosis(property_package=m.properties, **options)
    assert excinfo.value.field == field


@pytest.mark.unit
def test_reverseosmosis_registers_its_naming_convention_at_build():
    """The mapping RO passes up to SIDO's build is the one the unit resolves through."""
    m = dummy_time_block(3)
    m.unit = ReverseOsmosis(property_package=m.properties)

    assert m.unit._component_names == {
        "flow_in": "feed",
        "flow_out_a": "permeate",
        "flow_out_b": "brine",
        "split_fraction": "recovery",
    }


@pytest.mark.unit
def test_reverseosmosis_energy_intensity_is_per_permeate():
    """``energy_intensity`` is per m^3 of permeate, not of feed."""
    m = dummy_time_block(3)
    m.unit = ReverseOsmosis(
        property_package=m.properties,
        recovery=0.5,
        energy_intensity=4.0 * pyunits.kWh / pyunits.m**3,
    )
    t = m.time_block.time_index.first()
    m.unit.feed[t].fix(100.0)
    m.unit.permeate[t].fix(50.0)
    m.unit.power_electrical[t].fix(200.0)  # 4.0 * permeate, not 4.0 * feed

    assert pyo.value(m.unit.power_electrical_relation[t].body) == pytest.approx(
        0.0, abs=1e-9
    )
