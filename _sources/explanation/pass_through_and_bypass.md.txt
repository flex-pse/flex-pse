# Pass-through, bypass, and the arc layer

The words *pass-through* and *bypass* name three distinct things in flex-pse.
They are easy to confuse because all three concern flow moving between an
inlet and an outlet, so this page fixes the vocabulary.

## Pass-through — intra-unit property copying (available now)

A single unit model must be **well-posed on its own**: with its inlet states
fixed, its outlet states must be determined (degrees of freedom == 0). For every
state variable a unit does *not* otherwise govern, the outlet simply equals the
inlet. `OpsBlockData.add_pass_through_constraints(inlet, outlet, *,
exclude_vars=())` builds exactly those equalities — `outlet_var[idx] ==
inlet_var[idx]` over each variable's full index set — skipping any variable that
is already fully fixed (e.g. `dens_mass` under `fixed_density=True`, where the
constraint would be redundant).

It is gated by the `allow_pass_through` config slot. `SISOBlock` (and its
subclasses `Pump`/`Tank`) override the base default to `True` so the flow
topology is well-posed out of the box; the base `OpsBlock` default is `False`.
Each generated constraint is named `pass_through_{name}_eq` (e.g.
`pass_through_flow_vol_phase_eq`).

Subclasses that genuinely govern a variable exclude it and wire their own
relationship:

- `Pump` (hydraulic relation) excludes `pressure` — a pump *raises* pressure
  between its ports rather than passing it through.
- `Tank` excludes the flow-basis variable — its holdup difference equation
  governs flow instead.

Pass-through is **not** a physical stream. It is bookkeeping that copies
otherwise-ungoverned properties across one unit so the unit closes.

## Bypass — a flow-diversion stream (M08)

A **bypass** is a real physical stream: a fraction of a flow routed *around* a
unit's energy relation instead of through it. `flexops.logic.bypass.add_bypass(unit,
flow_var, bypass_max)` attaches:

- `bypass_flow[t]` — a `Var` bounded `[0, bypass_max]`, the diverted flow;
- `treated_flow[t]` — the quantity the unit's energy relation should consume in
  place of the raw flow;
- `treated_flow_eq[t]`: `treated_flow[t] == flow_var[t] - bypass_flow[t]`.

This is genuinely different behavior — it changes what the unit processes. In
v0 it only introduces the `treated_flow` quantity; rewiring Ports/Arcs for a
physical bypass stream is out of scope (that is the arc layer, below).

## The arc/topology layer (M09, future)

Neither of the above connects one unit to another. Flowsheet-level connectivity
— arcs between unit ports, network conservation, and any plant-scale routing —
lives in the arc/topology layer introduced in M09. A flowsheet-level bypass
(diverting a stream around a *unit block* by rewiring arcs) belongs there, not
in the intra-unit pass-through of this page.

## Summary

| Term | Scope | Builds | Milestone |
| --- | --- | --- | --- |
| Pass-through | Within one unit | `pass_through_{name}_eq` (`outlet == inlet`) | M04 (now) |
| Bypass stream | Around one unit's energy relation | `bypass_flow` / `treated_flow` | M08 |
| Arc layer | Between units (flowsheet) | Ports/Arcs, conservation | M09 (future) |

Reserve **bypass** for the flow-diversion stream; use **pass-through** for the
intra-unit property copying.
