# Resource nodes

flex-pse describes a process as a resource-task network, a graph with two kinds
of node. A **task** is a unit. A **resource** is anything units exchange, like
water, gas or electricity. Edges only join a task to a resource, so tasks reach
each other through the resources they share.

A `resource` element adds one of these nodes to a plant or network. Its balance
is strict: at every step, the flows on its edges sum to zero.

## Edges

An edge is a named view of a variable the unit already has, with a sign. A
positive coefficient supplies the resource and a negative one consumes it. Unit
models don't change to take part. Their existing equations already relate the
variables that become edges, so `power_electrical == energy_intensity *
flow_out` works as the task balance between electricity and the product stream.

Edges come from these sources:

| Source | Resource | Coefficient |
| --- | --- | --- |
| {meth}`~flexops.core.ops_block.OpsBlockData.attach_port` on an outlet | as given | +1 |
| {meth}`~flexops.core.ops_block.OpsBlockData.attach_port` on an inlet | as given | −1 |
| `register_boundary_flow(kind=FEED)` | its `resource` | +1 |
| `register_boundary_flow(kind=PRODUCT)` | its `resource` | −1 |
| `register_power(kind=ELECTRICAL)` | `"electricity"` | −1 |
| `register_fuel_usage(var, fuel_name)` | `fuel_name` | −1 |
| {meth}`~flexops.core.ops_block.OpsBlockData.register_resource` | as given | as given |

`power_electrical` is positive when a unit draws power, so the −1 makes a draw
consume electricity, and an export supplies it. Thermal power isn't mapped,
because a heat duty carries a temperature and summing heat at different
temperatures needs its own treatment.

The node's block holds `edges`, the Expression `flow[edge, t]` equal to `coeff *
var[t - offset]`, and the Constraint `balance[t]`. A node adds no flow variables.
`offset` is a delay in steps, and a term that would land before the horizon
starts adds nothing. Each term gets converted to the node's `units` on its own,
so when one doesn't fit, the error names the unit that registered it.

## Connecting units through a node

A connection's `to` can name a resource element instead of another unit's port.
Many ports can share one node, which is what a direct connection can't do:

```json
{"kind": "unit", "name": "site.well", "unit_model_class": "Feed",
 "connections": [{"port": "outlet_a", "to": "site.raw"}]},
{"kind": "unit", "name": "site.pump", "unit_model_class": "Pump",
 "connections": [{"port": "inlet", "to": "site.raw"},
                 {"port": "outlet", "to": "site.treated"}]},
{"kind": "resource", "name": "site.raw", "units": "m^3/hr"}
```

`name` is the dotted path of the node's block, and it has to sit under a plant or
network that also holds the attached units. `resource` is the key to collect and
defaults to the last part of `name`. Every member of an indexed unit attaches.
To do the same thing in code, call `attach_port` on each unit, then
{func}`~flexops.core.resources.add_resource_balance`.

A node has one state. If the property package carries pressure or temperature,
the node gets a `pressure[t]` or `temperature[t]` variable, and every attached
port equals it. To mix streams at different states, use a
{class}`~flexops.unit_models.mixer.Mixer`, which is a task with its own mixing
rule.

## Adding a candidate unit

Because units meet at nodes, a new unit joins a process by attaching to the nodes
it uses, with no edits to any other element. Add a second pump attached to
`site.raw` and `site.treated`, and the balances pick up its edges. The solver can
then share or move the flow between the pumps. With a direct connection, the same
change needs a splitter, a mixer and new arcs.

## Arcs

A direct connection stays an `Arc`. An arc is a resource with exactly two edges,
since its equality is that resource's balance, so the math doesn't change when a
stream moves to a named node. A port takes either an arc or a node. A term on a
port an `Arc` connects is refused, because balancing it again would count it
twice.

{func}`~flexops.core.resources.resource_graph` returns the whole graph, with each
resource marked as one of:

- `balanced`: it has a node.
- `arc`: a private two-edge resource, named after the `Arc`.
- `open`: units register it, but nothing balances it. Electricity is usually open
  in a plant costed through a tariff, where the bill stands in for the supply.

## Limits and storage

A node has no bounds, so a resource can't appear or vanish at a step. Limits
belong on the tasks at the edge of the process:

- A `Feed` supplies a resource, and `max_withdrawal` caps it per period or over
  the horizon.
- A `Product` takes a resource out, and `min_demand` sets an obligation.
- A {class}`~flexops.unit_models.grid_connection.GridConnection`'s `max_import`
  caps grid import.

A node has no level either. Storage is a task with holdup, so a
{class}`~flexops.unit_models.storage.tank.Tank` or
{class}`~flexops.unit_models.storage.battery.BatteryModel` sits between two nodes,
and its capacity can be a design variable.

## Things to watch for

A balance that another equation implies makes the constraint Jacobian rank
deficient. LP and MILP presolve can drop such a row, but IPOPT has to regularize
around it. On a two-pump closed recycle, where one balance is implied by the
others, IPOPT reports an optimal solve while missing the true flow by about
0.04%. Break a recycle with a task that has a real degree of freedom, like a
`Tank`.

Parallel units on a package that carries pressure or temperature do the same
thing. If two units pass pressure straight through between the same two nodes,
each one ties those nodes' pressures together, so one of the equalities is
implied. The same process built with Arcs, a `Splitter` and a `Mixer` has this
too.

Don't put a strict electricity balance on a plant that has a `GridConnection`.
The grid unit ties import and export to the net draw without being an edge, so
the balance would force the draws alone to zero.

A node only adds a block and its constraints. It never changes a unit's equations.
