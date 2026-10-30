# Resource balances

A resource is anything a plant has to balance, like a water obligation, a gas
supply, or a grid connection with a capacity limit. A `resource` element adds
a balance for one of these to a plant or network, using terms its units
already register. This is the resource-task network idea: units are tasks that
add to or take from resources.

## Where the terms come from

Units register their power draw, fuel use and boundary flows when they get
built. {func}`~flexops.core.registration.iter_resource_terms` reads those
registrations and gives each one a sign. A positive coefficient supplies the
resource and a negative one consumes it.

| Registration | Resource | Coefficient |
| --- | --- | --- |
| `register_boundary_flow(kind=FEED)` | its `resource` | +1 |
| `register_boundary_flow(kind=PRODUCT)` | its `resource` | −1 |
| `register_power(kind=ELECTRICAL)` | `"electricity"` | −1 |
| `register_fuel_usage(var, fuel_name)` | `fuel_name` | −1 |
| {meth}`~flexops.core.ops_block.OpsBlockData.register_resource` | as given | as given |

`power_electrical` is positive when a unit draws power, so the −1 makes a draw
consume electricity. An export is negative power, so it supplies electricity.
A `Feed` of `natural_gas` and a unit burning `natural_gas` land on the same
resource with opposite signs, and one balance on `natural_gas` ties them
together. Thermal power isn't mapped, because a heat duty carries a
temperature and summing heat at different temperatures needs its own
treatment.

Use `register_resource` for anything the table doesn't cover, like a
recovered stream or a utility the unit doesn't meter itself. The term at step
`t` is `coeff * var[t - offset]`. `offset` is a delay in steps. Nothing sets
it yet, and a term that would land before the horizon starts adds nothing.

## Flow and stock resources

A **flow** resource doesn't accumulate. At each step the net contribution
`net[t]` is the sum of the terms, and it has to stay between `lower` and
`upper`. If you set neither bound, `net[t]` is zero, which is a strict node
balance. If you set only one, the other side is open.

A **stock** resource accumulates the same way a
{class}`~flexops.unit_models.storage.tank.Tank` does. `level[0]` equals
`initial`, and for `t >= 1`, `level[t] = level[t-1] + dt * (sum of the terms
at t)`. Flows at `t = 0` don't count, since `initial` is the level at the
start of that step. The level stays between `lower` and `upper`. A stock needs
`initial` and a flow can't have one.

Both kinds are written in `units`, a rate for a flow and a quantity for a
stock. Each term gets converted to those units on its own, so when one doesn't
fit, the error names the unit that registered it. Power in kW and flow in
m³/hr can share a balance as long as `units` and the coefficients make them
dimensionally consistent.

## Examples

Make the gas a plant draws equal the gas it burns, at every step:

```json
{"kind": "resource", "name": "site.natural_gas", "units": "m^3/hr"}
```

Cap grid import at 5 MW. A draw is a negative contribution, so the cap is a
lower bound on the net:

```json
{"kind": "resource", "name": "site.grid_cap", "resource": "electricity",
 "units": "kW", "lower": {"value": -5000, "units": "kW"}}
```

Keep a reservoir between 0 and 500 m³, starting at 120 m³:

```json
{"kind": "resource", "name": "site.reservoir", "resource": "water",
 "units": "m^3", "stock": true, "initial": 120,
 "lower": 0, "upper": 500}
```

`name` is the dotted path of the balance block, and it has to sit under a
plant or network. `resource` is the key to collect and defaults to the last
part of `name`. To do the same thing in code, call
{func}`~flexops.core.resources.add_resource_balance`.

## Things to watch for

Arcs already conserve the flow they connect. If a term is a flow on a port an
`Arc` connects, the balance refuses it, because balancing it again would count
it twice. Register the metered quantity instead, like a `Feed` or `Product`, a
power draw, or a variable that isn't on a connected port.

Don't put a strict electricity balance on a plant that has a
`GridConnection`. The grid unit already ties import and export to the net
draw, so a second strict balance can over-constrain the model. A bounded
electricity resource on a plant without a grid unit is the intended use.

A balance only adds a block and its constraints. It never changes a unit.
