# Resources and tasks

flex-pse can describe a process as a resource-task network, a graph with two kinds
of node. A **resource** is anything the process exchanges, like water, fuel or
electricity. A **task** is an operation that converts or moves resources. Tasks reach
each other through the resources they share.

A {class}`~flexops.core.resources.Resource` block adds a resource node. A task is
either one unit connected to resources, or a {class}`~flexops.core.task_block.Task`
block grouping several connected units.

An ordinary flowsheet with `Arc`s directly between units still works. Resources and
tasks are something you opt in to, and {func}`~flexops.core.resources.validate_rtn`
checks a model that means to be pure.

## Tasks

Units and tasks don't map one to one. Flocculation and filtration can form a single
pretreatment task:

```python
m.site.pretreat = Task()
m.site.pretreat.floc = Pump(property_package=m.properties, energy_intensity=low)
m.site.pretreat.filter = Pump(property_package=m.properties, energy_intensity=high)
m.site.pretreat.link = Arc(
    source=m.site.pretreat.floc.outlet, destination=m.site.pretreat.filter.inlet
)
m.site.into = Arc(source=m.site.raw.outlet, destination=m.site.pretreat.floc.inlet)
m.site.out = Arc(source=m.site.pretreat.filter.outlet, destination=m.site.clean.inlet)
```

A `Task` sits directly on a `PlantBlock`, and the plant's totals include the units
inside it. A task's place in the network is its set of resource connections, which
`resource_graph(block).connections(task)` returns as `(resource, supplies)` pairs.
Another task with the same set can replace it, whatever units it holds inside.

## Property blocks

A resource takes a property block, which sets its units and how it connects:

| Property block | Example | Connects by | Edges |
| --- | --- | --- | --- |
| **Stream**, any package with a flow | {class}`~flexops.properties.simple_aqueous.SimpleAqueousFlow` | `Arc`s to its ports | One per arc |
| **Utility**, a {class}`~flexops.properties.simple_utility.SimpleUtility` | electricity, natural gas | Collected automatically | One per registered term |

```python
m.properties = SimpleAqueousFlow()
m.power = SimpleUtility(resource="electricity", units="kW")
m.fuel = SimpleUtility(resource="natural_gas", units="m^3/hr")

m.site.raw = Resource(property_package=m.properties)
m.site.gas = Resource(property_package=m.fuel)
```

An `Arc` to a stream must join a unit on the same package class, and every state on
the resource's port must exist on the unit's port in the same units. An `Arc` writes
an equality with no conversion, so anything else is an error.

## Balance types

`balance` says how what enters a resource relates to what leaves it at each step:

| {class}`~flexcore.config.schema.BalanceType` | Relation | Use it for |
| --- | --- | --- |
| `BALANCED`, the default | in = out | A resource inside the process |
| `EXPORT` | in ≥ out | A product or waste, whose surplus leaves the model |
| `IMPORT` | in ≤ out | A feed, whose shortfall enters the model |
| `UNBOUNDED` | none | An infinite source and sink, like a two-way grid |

Feeds, products and other boundaries are these settings, so a process doesn't need a
separate boundary block:

```python
m.site.seawater = Resource(property_package=m.properties, balance=BalanceType.IMPORT)
m.site.permeate = Resource(property_package=m.properties, balance=BalanceType.EXPORT)
m.site.grid = Resource(property_package=m.power, balance=BalanceType.UNBOUNDED)
```

`Feed` and `Product` units are still how a boundary carries limits or a price.

## Streams

A stream has `inlet` and `outlet` ports, and units connect to them with ordinary
Arcs:

```python
m.raw = Resource(property_package=m.properties, balance=BalanceType.IMPORT)
m.treated = Resource(property_package=m.properties, balance=BalanceType.EXPORT)
m.site.pump = Pump(property_package=m.properties, energy_intensity=intensity)
m.into_pump = Arc(source=m.raw.outlet, destination=m.site.pump.inlet)
m.out_of_pump = Arc(source=m.site.pump.outlet, destination=m.treated.inlet)
pyo.TransformationFactory("network.expand_arcs").apply_to(m)
```

In a spec, a connection's `to` can name the resource. An inlet takes from it and an
outlet supplies it, and every member of an indexed unit connects:

```json
{"kind": "unit", "name": "site.well", "unit_model_class": "Feed",
 "connections": [{"port": "outlet_a", "to": "site.raw"}]},
{"kind": "unit", "name": "site.pump", "unit_model_class": "Pump",
 "connections": [{"port": "inlet", "to": "site.raw"},
                 {"port": "outlet", "to": "site.treated"}]},
{"kind": "resource", "name": "site.raw", "property_package": "properties"},
{"kind": "resource", "name": "site.treated", "property_package": "properties",
 "balance": "export"}
```

Many arcs can share a port. Each arc with a shared port gets its own flow variable,
and the node's inflow is the sum of the arcs entering it. The split between arcs
is free at every step.

A side with no arcs carries zero flow, so a resource can't appear from nothing or
vanish into nothing unless its balance type allows it. A `BALANCED` stream with a
supplier and no consumer forces that supply to zero.

A node has one state. If the property package carries pressure or temperature, the
inlet and outlet states are held equal, and every connected port equals them. To
mix streams at different states, use a
{class}`~flexops.unit_models.mixer.Mixer`, which is a task with its own mixing rule.

## Utilities

A utility has no ports. It collects every term registered under its package's
`resource` name by units under its parent block when it's built, so build it after
those units:

```python
m.fuel = SimpleUtility(resource="natural_gas", units="m^3/hr")
m.site.boiler = Boiler(...)
m.site.gas = Resource(property_package=m.fuel)
```

An edge is a named view of a variable the unit already has, with a sign. A positive
coefficient supplies the resource and a negative one consumes it.

| Source | Registered under | Coefficient |
| --- | --- | --- |
| `register_boundary_flow(kind=FEED)` | its `resource` | +1 |
| `register_boundary_flow(kind=PRODUCT)` | its `resource` | −1 |
| `register_power(kind=ELECTRICAL)` | `"electricity"` | −1 |
| `register_fuel_usage(var, fuel_name)` | `fuel_name` | −1 |
| {meth}`~flexops.core.ops_block.OpsBlockData.register_resource` | as given | as given |

`power_electrical` is positive when a unit draws power, so the −1 makes a draw
consume electricity, and an export supplies it. Thermal power isn't mapped, because
a heat duty carries a temperature and summing heat at different temperatures needs
its own treatment.

The block holds `edges`, the Expression `flow[edge, t]` equal to `coeff *
var[t - offset]`, and the Constraint `balance[t]`. A utility adds no flow variables.
`offset` is a delay in steps, and a term that would land before the horizon starts
adds nothing. An `UNBOUNDED` utility keeps its edges and has no `balance`. Each term
gets converted to the package's `units` on its own, so when one doesn't fit, the error names the unit that registered it. A term that's a flow
an `Arc` already conserves is refused, because balancing it again would count it
twice. That covers a flow on a connected port, and a `Feed`'s or `Product`'s metered
total when its own ports are connected.

## Verifying the model

A balance reflects the model when it was built, and a model can change afterward.
{func}`~flexops.core.resources.verify_resources` compares each resource with the
model as it is now. `build_model` runs it, and so does every solve through
{func}`~flexcore.solvers.get_solver`. A solve through Pyomo's `SolverFactory`
directly doesn't, so call it yourself in that case.

It raises `FlexConfigError` when:

- A stream's arc runs the wrong way through a port, or joins a unit on another
  package class or with different units.
- A stream's arc joins a deactivated unit, or a utility's terms differ from its
  edges. A utility's terms change when a unit is added, deactivated or reactivated
  after it was built.
- A utility counts a flow an `Arc` conserves.
- An electricity utility sits beside a `GridConnection`.

A one-sided balance forces its flows to zero: a `BALANCED` utility whose edges all have
one sign, a `BALANCED` stream with arcs on one side only, an `EXPORT` with nothing
supplying it, or an `IMPORT` with nothing consuming it. Like on-site power nothing accounts for, covered
below, that's normal while a model is being put together, so `build_model` warns with
{class}`~flexcore.exceptions.FlexResourceWarning`. It's almost always a mistake in a
model you solve, so a solve raises instead. Pass `error_type="warn"` or
`error_type="error"` to choose when you call it yourself. A stream with no arcs at all
forces nothing and is left alone.

To switch a unit off, fix its flow to zero rather than deactivating it. Deactivating
removes the unit's equations but leaves its flow in the balance, free to take any
value, so a resource could appear from nothing.

Expand arcs once, after every arc exists. A second expansion on a port with
expanded arcs raises, since the first expansion's equations can't take a new arc.

## Electricity and gas

Electricity and gas often have no `Resource`. `FlexCosting` sums what units register
across the model and bills that total, so a model that only prices its power needs no
balance, and a registered resource with no balance is unconstrained.

A {class}`~flexops.unit_models.grid_connection.GridConnection` constrains a plant's
total. Its `eq_net_power` equation sets import minus export equal to the plant's net
draw, which is already that plant's power balance. A grid is optional, but an
electricity `Resource` beside one raises, because it would repeat the balance without
the grid's terms. An `UNBOUNDED` electricity `Resource` also works as a grid, with no
import limits or tariff.

A coefficient alone can't tell supply from draw on a signed variable. A battery and a
pump both have a coefficient of −1, yet the discharging battery supplies the pump. So an
electricity `Resource` is judged by what its units can do. It has a supply if a unit is
marked `supplies_power` (`BatteryModel`, `GenericRenewables`, `Combustor`) or a term has
a positive coefficient, and it has a consumer if another edge can draw. Without both,
a `BALANCED` one forces its flows to zero, so it warns at build and raises at solve like
any other one-sided balance. A battery alone counts as one-sided, since there's nothing for
it to supply.

On-site power or storage needs something to account for what it supplies, or that power
disappears. A `GridConnection`, an electricity `Resource`, or a `FlexCosting` that bills
electricity all count. Costing counts because it bills net exports at the import price
unless `export_prices` says otherwise, which is net metering, so exports are unlimited
unless a model limits them. With none of the three, a model warns at build and raises at
solve. A `FlexCosting` that only prices gas doesn't count.

A custom unit that can supply power sets the class attribute `supplies_power = True`.

## Adding a candidate unit

Because units meet at resources, a new unit joins a process by connecting to the
streams it uses, with no edits to any other element. Add a second pump connected to
`site.raw` and `site.treated`, and the solver can share or move the flow between the
pumps. With a direct connection, the same change needs a splitter, a mixer and new
arcs.

## Arcs between units

A direct connection between two units is still an `Arc`, with the same equality as
before. Outside a task it's a resource with exactly two edges, so the math doesn't
change when a stream moves to a named resource. An undirected arc only works between
two ports that have no other arcs.

{func}`~flexops.core.resources.resource_graph` walks the resources and arcs to return
the whole graph. Its tasks are `Task` blocks and lone units, and an arc inside a `Task`
is left out, since it's internal to that task. Each resource is marked as one of:

- `stream`: a `Resource` reached by arcs, under its block's name. Asked about a
  plant, it includes a stream outside the plant that the plant's units connect to.
- `utility`: a `Resource` collecting registered terms, under its block's name.
- `arc`: a private two-edge resource, named after an `Arc` between tasks.
- `open`: terms units register that no utility collects, under the registered name.
  Electricity is usually open in a plant costed through a tariff, where the bill
  stands in for the supply.

An edge on a shared port has no variable of its own until arcs are expanded, so its
`var` is `None` before then.

## Validating a pure network

There's no special arc type for a resource-task network, and nothing enforces one
unless you ask. {func}`~flexops.core.resources.validate_rtn` walks the model's arcs and
raises `FlexConfigError`, listing every problem, when:

- An `Arc` joins two resources. Represent a bypass between resources as a task.
- An `Arc` joins two tasks, or a task and a lone unit.
- An `Arc` joins two units that aren't grouped in one `Task`.
- A `Resource` sits inside a `Task`.

It sets `model.rtn_validated` to the outcome, which RTN-specific features can require.
A model that passed is checked again at every solve through
{func}`~flexcore.solvers.get_solver`, so a later change can't leave the status stale
there.

## Limits and storage

A resource has no bounds, so its balance type alone decides how it can appear or
vanish at a step. Limits belong on the tasks at the edge of the process:

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
around it. A closed recycle with no free boundary, like two pumps feeding each
other, is the worst case: it has as many equalities as variables, so IPOPT can treat
it as a square system. It can then report an optimal solve for a point that doesn't
minimize the objective, and how close that point lands depends on the starting
values. Break a recycle with a task that has a real degree of freedom, like a `Tank`.

Parallel units on a package that carries pressure or temperature do the same
thing. If two units pass pressure straight through between the same two streams,
each one ties those streams' pressures together, so one of the equalities is
implied. The same process built with Arcs, a `Splitter` and a `Mixer` has this
too.

`UnitModelTestHarness` skips these checks, since it tests a unit alone with no plant
around it.

A resource only adds a block, its constraints and, for a stream, the state and arc
variables it needs. It never changes a unit's equations.
