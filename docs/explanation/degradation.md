# Degradation penalties

Flexible operation wears equipment. A pump that follows a volatile tariff
cycles its speed, a reverse osmosis train that load-shifts swings its feed
pressure, and a battery that arbitrages cycles its state of charge. A
schedule that ignores that wear looks cheaper than it is.
{func}`~flexops.logic.degradation.add_degradation` attaches a **priced wear penalty** to
a unit, so the optimizer weighs each move against the maintenance it costs.

It is the soft counterpart of {func}`~flexops.logic.ramp.add_ramp_rate`. A ramp
limit forbids fast change outright. A degradation penalty charges for change
and lets the economics decide.

## The penalty

A penalty is a sum of **terms**. Each term watches one time-indexed
variable {math}`g` of the unit, measures a wear driver, subtracts a free
deadband {math}`a`, and charges what remains at a price {math}`p`. Every term is
linear, so a model that was an LP stays an LP.

| Kind | Charged excess {math}`e[t]` | Price is per | Typical use |
|---|---|---|---|
| `variation` | {math}`\max(0,\ \lvert g[t] - g[t-w]\rvert - a)` | unit of change | pump speed or power cycling, membrane pressure swings, on/off cycling (variation of a status) |
| `deviation` | {math}`\max(0,\ \lvert g[t] - g_\text{ref}\rvert - a)` | unit held for one hour | running away from the best efficiency point |
| `exceedance` | {math}`\max(0,\ g[t] - g_\text{hi}) + \max(0,\ g_\text{lo} - g[t])` | unit held for one hour | flux above critical flux, pressure above nominal |
| `throughput` | {math}`\max(0,\ g[t] - a)` | unit held for one hour | processed volume or energy |

A variation is an event: changing by 5 kW costs the same whether the step is
15 minutes or an hour. The other kinds are levels: running 1 bar above
nominal for an hour costs four times as much as for 15 minutes. The wear
cost rate (in currency per hour) is therefore

```{math}
r[t] = \sum_{\text{variation}} \frac{p_k\, e_k[t]}{\Delta t}
     + \sum_{\text{other kinds}} p_k\, e_k[t],
\qquad
C = \sum_t r[t]\,\Delta t .
```

Prices are stated directly in currency per unit of the driver (for example
dollars per kW of change). That is the number an operator or maintenance
contract usually provides. Converting a fraction-of-life model, where a
full cycle consumes a known share of a replacement, is a matter of
multiplying that share by the replacement cost.

## Deadbands, covered cost, and budgets

There are three ways to make some wear free or to cap it. They answer
different questions.

**Per-step deadband** (`deadband` on a term). Each step may move up to
{math}`a` for free; only the excess is charged. This is a deadband. It models wear
that is negligible below a threshold (small drive trims, pressure drift
within control noise, battery cycling slower than calendar aging), and it
rewards spreading moves out over time. Inside the deadband many schedules
cost the same, so the solver may return a slightly jittery profile there.
Set the deadband to zero for strict smoothing.

An aggregate floor of the form {math}`D[t] = \max(\text{cycling}[t], D_\text{shelf})`,
used in some battery arbitrage studies, is the same idea: it equals a
constant plus a deadband. The constant does not change the optimal
schedule and would double count maintenance already carried in the fixed
operating cost, so only the deadband is modeled.

**Covered cost** (`covered_cost` on the penalty). The first
dollars of wear over the period are free, for example because a service
contract already covers a number of starts per month. Only the total
matters, not when the wear happens:


```{math}
\text{billable} = \max(0,\ C - A).
```


**Horizon budget** (`horizon_budget` on the penalty). A hard cap,
{math}`C \le B`, with no price attached. Use it for vendor limits such as a
maximum number of motor starts or a warranty cap on pressure cycles. It can
be combined with a price or used alone.

Both horizon values are stated per `period_hours` (730 for a month) and
prorated to the modeled horizon. Without `period_hours` they apply to the
modeled horizon as given.

## Billing

The billable amount is spread evenly over the horizon as the series
`<name>_billed_rate` and registered on {class}`~flexops.costing.flex_costing.FlexCosting`
as the scalar operating cost `<unit>_<name>`, attributed to the unit. It
appears in the operating cost total and in the cost report next to energy
and fixed costs. Monthly maintenance cost is a reporting conversion of that
total.

## Built components

For a penalty named `wear`, term `k` builds `wear_k_quantity` (the
watched quantity), `wear_k_excess` (the hinge), the constraints
`wear_k_up`/`wear_k_down`, and mutable parameters `wear_k_price`,
`wear_k_deadband`, and the reference or bounds. The penalty builds
`wear_rate`, `wear_total`, `wear_billable`, and, if a budget is given,
`wear_budget`. Every parameter is registered, so
`unit.update_parameters({"wear_0_price": 3.0})` retunes a built model in
place.

## Richer wear relationships

`wear_k_quantity_relation` (the quantity equals the variable) and
`wear_rate_relation` (the rate equals the priced sum) are registered
relations, so {meth}`~flexops.core.ops_block.OpsBlockData.swap_relation`
can replace either with a fitted surrogate without rebuilding anything. Two
examples:

- **Off best efficiency point, variable speed.** The best efficiency flow
  scales with speed, {math}`Q_\text{bep} = Q_r\,\omega/\omega_r`. Swap the
  quantity to {math}`Q - Q_r\,\omega/\omega_r` and use a `deviation` term with
  reference 0.
- **Depth-of-discharge battery wear.** Swap the quantity to a piecewise fit
  of the damage curve {math}`\psi(s)` and use a `variation` term. The penalty then
  charges {math}`\lvert\psi(s_t) - \psi(s_{t-1})\rvert`. This is exact but no
  longer convex, so the model becomes a MILP or a nonconvex NLP.

## Caveats

- **Tightness.** Each hinge is an inequality. It equals the true excess only
  while a price or a budget pushes it down. A penalty that is neither billed
  nor budgeted logs a warning, because its values would be loose.
- **The first step.** A variation term has no history before the horizon,
  so steps {math}`t < w` carry no charge. The watched quantity is registered as
  rolling state so a rolling-horizon driver can carry the last values into
  the next window.
- **Scaling.** Keep prices so a horizon's wear cost is of the same order as
  its other costs; tiny coefficients make poorly scaled rows for
  interior-point solvers.

## Example

In Python:

```python
from flexcore.config.schema import DegradationSpec, DegradationTerm, DegradationTermSpec
from flexops.logic import add_degradation

spec = DegradationSpec(
    name="membrane_wear",
    terms=[
        DegradationTermSpec(
            kind=DegradationTerm.VARIATION,
            variable="outlet_state.pressure",
            price=40.0,       # dollars per bar of change
            deadband=0.2,    # bar per step that is free
        ),
        DegradationTermSpec(
            kind=DegradationTerm.EXCEEDANCE,
            variable="outlet_state.pressure",
            price=15.0,       # dollars per bar-hour above nominal
            upper=60.0,
        ),
    ],
    covered_cost=500.0,  # dollars per month covered by the service contract
    period_hours=730.0,
)
rate, total = add_degradation(m.plant.ro, spec, costing=m.costing)
```

The same penalty in a config sits under the unit's `degradation` list:

```json
"degradation": [
  {
    "name": "membrane_wear",
    "terms": [
      {"kind": "variation", "variable": "outlet_state.pressure", "price": 40.0, "deadband": 0.2},
      {"kind": "exceedance", "variable": "outlet_state.pressure", "price": 15.0, "upper": 60.0}
    ],
    "covered_cost": 500.0,
    "period_hours": 730.0
  }
]
```

Values are in the variable's own units (the example assumes pressure is
declared in bar), and prices are in the model currency.
