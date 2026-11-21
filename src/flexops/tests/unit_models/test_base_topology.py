"""SIDOBlock/DIDOBlock topology bases: ports and mass-balance bodies (§3.4)."""

import pyomo.environ as pyo
import pytest
from idaes.core import declare_process_block_class
from pyomo.network import Port

from flexops.testing import dummy_time_block
from flexops.unit_models.base import DIDOBlock, SIDOBlock, SISOBlock
from flexops.unit_models.base.siso import SISOBlockData


@pytest.mark.unit
def test_sido_mass_balance_bodies():
    """1 inlet / 2 outlet ports; the split balance is satisfied by a hand split."""
    m = dummy_time_block(3)
    m.unit = SIDOBlock(property_package=m.properties, split_fraction=0.25)

    ports = {p.local_name for p in m.unit.component_objects(Port)}
    assert ports == {"inlet", "outlet_a", "outlet_b"}

    for t in m.time_block.time_index:
        m.unit.flow_in[t].fix(8.0)
        m.unit.flow_out_a[t].fix(2.0)
        m.unit.flow_out_b[t].fix(6.0)
        assert pyo.value(m.unit.split_mass_balance[t].body) == pytest.approx(
            8.0 - 2.0 - 6.0, abs=1e-9
        )
        assert pyo.value(m.unit.split_definition[t].body) == pytest.approx(
            2.0 - 0.25 * 8.0, abs=1e-9
        )


@pytest.mark.unit
def test_sido_split_fraction_is_a_fixed_physical_fraction():
    """The generic topology bounds the split only to a physical fraction."""
    m = dummy_time_block(3)
    m.unit = SIDOBlock(property_package=m.properties, split_fraction=0.4)

    assert m.unit.split_fraction.bounds == (0.0, 1.0)
    assert m.unit.split_fraction.fixed
    assert pyo.value(m.unit.split_fraction) == pytest.approx(0.4)


@pytest.mark.unit
def test_sido_takes_no_split_window_options():
    """A narrower window is a physical subclass's business, not the base's."""
    m = dummy_time_block(3)
    with pytest.raises(ValueError, match="split_fraction_min"):
        m.unit = SIDOBlock(property_package=m.properties, split_fraction_min=0.3)


@pytest.mark.unit
def test_dido_mass_balance_bodies():
    """2 inlet / 2 outlet ports; both coupled per-stream balances check by hand."""
    m = dummy_time_block(3)
    m.unit = DIDOBlock(property_package=m.properties, transfer_fraction=0.1)

    ports = {p.local_name for p in m.unit.component_objects(Port)}
    assert ports == {"inlet_a", "inlet_b", "outlet_a", "outlet_b"}

    for t in m.time_block.time_index:
        m.unit.flow_in_a[t].fix(10.0)
        m.unit.flow_in_b[t].fix(4.0)
        m.unit.flow_out_a[t].fix(9.0)
        m.unit.flow_out_b[t].fix(5.0)
        assert pyo.value(m.unit.mass_balance_a[t].body) == pytest.approx(
            9.0 - (10.0 - 0.1 * 10.0), abs=1e-9
        )
        assert pyo.value(m.unit.mass_balance_b[t].body) == pytest.approx(
            5.0 - (4.0 + 0.1 * 10.0), abs=1e-9
        )


@pytest.mark.unit
def test_component_names_default_to_the_topology_vocabulary():
    """A base builds its own generic role names when no naming_dict is passed."""
    m = dummy_time_block(3)
    m.siso = SISOBlock(property_package=m.properties)
    m.sido = SIDOBlock(property_package=m.properties)
    m.dido = DIDOBlock(property_package=m.properties)

    for name in ("flow_in", "flow_out"):
        assert m.siso.find_component(name) is not None
    for name in ("flow_in", "flow_out_a", "flow_out_b", "split_fraction"):
        assert m.sido.find_component(name) is not None
    for name in ("flow_in_a", "flow_in_b", "flow_out_a", "flow_out_b"):
        assert m.dido.find_component(name) is not None


@declare_process_block_class("RenamedSISOBlock")
class RenamedSISOBlockData(SISOBlockData):
    """A SISO subclass renaming one role, the way a physical unit does."""

    def build(self) -> None:
        """Build the SISO base with ``flow_in`` renamed to ``raw``."""
        super().build(naming_dict={**SISOBlockData._component_names, "flow_in": "raw"})


@pytest.mark.unit
def test_a_subclass_naming_dict_renames_only_the_roles_it_overrides():
    """The spread default carries the untouched roles; ports are never renamed."""
    m = dummy_time_block(3)
    # RenamedSISOBlock is injected into this module by the decorator above.
    m.unit = RenamedSISOBlock(property_package=m.properties)  # noqa: F821

    assert m.unit.find_component("raw") is not None
    assert m.unit.find_component("flow_in") is None
    assert m.unit.find_component("flow_out") is not None

    ports = {p.local_name for p in m.unit.component_objects(Port)}
    assert ports == {"inlet", "outlet"}


@pytest.mark.unit
def test_sido_split_fraction_stays_a_regressable_scalar():
    """The generic SIDO split is one scalar Var, registered as regressable."""
    m = dummy_time_block(3)
    m.unit = SIDOBlock(property_package=m.properties, split_fraction=0.4)

    assert not m.unit.split_fraction.is_indexed()
    regressable = {r.name for r in m.unit._io_registry.parameters if r.regressable}
    assert "split_fraction" in regressable
