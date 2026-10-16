"""Spec extensions used by the tests and the round-trip corpus."""

import pyomo.environ as pyo
from pyomo.environ import units as pyunits

import flexops
from flexops import logic


@flexops.extension
def cap_power(model, *, unit: str, max_kw: float):
    """Cap a unit's electrical power at every time point."""
    block = model.find_component(unit)
    block.power_cap = pyo.Constraint(
        block.power_electrical.index_set(),
        rule=lambda b, t: b.power_electrical[t] <= max_kw,
    )


@flexops.extension
def terminal_level(model, *, tank: str, fraction: float):
    """Require a tank to end the horizon at least ``fraction`` full."""
    block = model.find_component(tank)
    last = block.volume.index_set().last()
    block.terminal_level = pyo.Constraint(
        expr=block.volume[last] >= fraction * block.capacity
    )


@flexops.extension
def minimize_power(model, *, units: list[str]):
    """Make the objective the total electrical energy of the named units."""
    blocks = [model.find_component(name) for name in units]
    model.power_objective = pyo.Objective(
        expr=sum(sum(b.power_electrical.values()) for b in blocks)
    )


def plain_function(model):
    """An undecorated function, for entry-point resolution tests."""
    return model


@flexops.extension
def status_on_flow(model, *, unit: str, max_flow: float):
    """Add an on/off status to a unit's inlet flow, in m^3/hr."""
    block = model.find_component(unit)
    flow = pyunits.m**3 / pyunits.hr
    logic.add_status(block, block.flow_in, 0 * flow, max_flow * flow)
