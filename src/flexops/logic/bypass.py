"""Bypass-stream constraints around a unit.

Smallest useful v0 form (documented choice): a fraction of a flow may bypass
the unit's energy relation entirely. Rewiring Ports/Arcs for a physical bypass
stream is explicitly out of scope for v0 -- this only introduces the
``treated_flow`` quantity a unit's own energy relation should consume in place
of the raw flow.
"""

from typing import Any

import pyomo.environ as pyo
from pyomo.environ import units as pyunits

from flexops.core.serialize import to_jsonable
from flexops.logic._records import record, variable_path


def add_bypass(unit: Any, flow_var: pyo.Var, bypass_max: Any) -> pyo.Var:
    """Attach ``bypass_flow[t]`` and the ``treated_flow[t]`` it defines.

    Adds:

    * ``bypass_flow[t]`` -- Var, same units as ``flow_var``, bounded
      ``[0, bypass_max]``.
    * ``treated_flow[t]`` -- Var, same units as ``flow_var``; the quantity a
      unit's energy relation should consume instead of the raw flow.
    * ``treated_flow_eq[t]``: ``treated_flow[t] == flow_var[t] -
      bypass_flow[t]``.

    Args:
        unit: The unit block to attach the bypass structure to.
        flow_var: The time-indexed flow ``Var`` being (partially) bypassed.
        bypass_max: Upper bound on ``bypass_flow[t]``, a units-carrying
            quantity matching ``flow_var``.

    Returns:
        The attached ``treated_flow`` Var.
    """
    record(
        unit,
        "bypass",
        variable=variable_path(flow_var, unit),
        max=to_jsonable(bypass_max, where="add_bypass.bypass_max"),
    )
    tb = unit._find_time_block()
    ref = next(iter(flow_var.values())) if flow_var.is_indexed() else flow_var
    units = pyunits.get_units(ref)
    bypass_max_val = pyo.value(pyunits.convert(bypass_max, units))

    unit.bypass_flow = pyo.Var(
        tb.time_index,
        bounds=(0.0, bypass_max_val),
        units=units,
        doc="Flow bypassing the unit's energy relation.",
    )
    unit.treated_flow = pyo.Var(
        tb.time_index,
        units=units,
        doc="Flow the unit's energy relation consumes (flow_var minus bypass).",
    )

    @unit.Constraint(
        tb.time_index, doc="treated_flow[t] == flow_var[t] - bypass_flow[t]."
    )
    def treated_flow_eq(_b, t):
        return unit.treated_flow[t] == flow_var[t] - unit.bypass_flow[t]

    return unit.treated_flow
