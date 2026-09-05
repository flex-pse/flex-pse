# Pass through, bypass, and the arc layer

The words *pass through* and *bypass* name three distinct things in
flex-pse. All three concern flow moving between an inlet and an outlet,
which makes them easy to confuse. This page fixes the vocabulary.

## Pass through, copying a property inside a unit (available now)

A single unit model must be **well posed on its own**. With its inlet
states fixed, its outlet states must be fully determined (degrees of
freedom equal 0). For every state variable a unit does *not* otherwise
govern, the outlet simply equals the inlet.
`OpsBlockData.add_pass_through_constraints(inlet, outlet, *,
exclude_vars=())` builds exactly those equalities, `outlet_var[idx] ==
inlet_var[idx]` over each variable's full index set, and skips any
variable that's already fully fixed (an inlet held at a known pressure,
say, where the constraint would be redundant).

The `allow_pass_through` config slot gates this behavior. `SISOBlock` (and
its subclasses `Pump`/`Tank`) override the base default to `True`, so the
flow topology is well posed out of the box. The base `OpsBlock` default is
`False`. Each generated constraint takes the name `pass_through_{name}_eq`,
for example `pass_through_flow_vol_phase_eq`.

Subclasses that genuinely govern a variable exclude it and wire their own
relationship instead.

- `Pump` (hydraulic relation) excludes `pressure`. A pump *raises* pressure
  between its ports rather than passing it through.
- `Tank` excludes the flow basis variable. Its holdup difference equation
  governs flow instead.

Pass through is **not** a physical stream. It's bookkeeping that copies
properties the unit doesn't otherwise govern, just enough to close the
unit.

## Bypass, a stream that diverts flow

A **bypass** is a real physical stream, a fraction of a flow routed
*around* a unit's energy relation instead of through it.
`flexops.logic.bypass.add_bypass(unit, flow_var, bypass_max)` attaches
three things.

- `bypass_flow[t]`, a `Var` bounded `[0, bypass_max]`, the diverted flow.
- `treated_flow[t]`, the quantity the unit's energy relation should
  consume in place of the raw flow.
- `treated_flow_eq[t]`, which reads
  `treated_flow[t] == flow_var[t] - bypass_flow[t]`.

This changes what the unit actually processes, so it's a genuinely
different kind of behavior than pass through. `add_bypass` only introduces
the `treated_flow` quantity. Rewiring Ports/Arcs to physically route a
stream around a whole unit block is a flowsheet level concern, described
next.

## The arc and topology layer

Neither of the above connects one unit to another. Connectivity at the
flowsheet level, arcs between unit ports, network conservation, and any
routing across a plant, lives one level up, in `PlantBlock` (a collection
of units) and `NetworkBlock` (a collection of plants). See
{doc}`../reference/flexops/core` for how they compose units and plants
over standard IDAES/Pyomo `Arc`s. A bypass at the flowsheet level (routing
a stream around a whole unit block by rewiring arcs, rather than the
`treated_flow` substitution described above) belongs there.

## Summary

| Term | Scope | Builds |
| --- | --- | --- |
| Pass through | Within one unit | `pass_through_{name}_eq` (`outlet == inlet`) |
| Bypass stream | Around one unit's energy relation | `bypass_flow` / `treated_flow` |
| Arc layer | Between units (flowsheet) | Ports/Arcs, conservation |

Reserve **bypass** for the stream that diverts flow. Use **pass through**
for copying a property inside one unit.
