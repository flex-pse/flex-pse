# Why the reported cost isn't the solver's objective

The electricity cost you see in a run's report always comes from
{meth}`~flexops.costing.flex_costing.FlexCostingData.report_cost`. It's
computed **after** the model has solved, from the power values the solve
actually produced. It's never read off the solver's own internal objective
value.

Those two numbers aren't the same. The gap between them is deliberate.

## The objective is a solver aid, not a bill

Some tariff structures are awkward for a solver to handle directly. Take a
tiered energy surcharge that only kicks in once monthly consumption crosses
a threshold. That kind of rule introduces a jump or a non convexity that
slows the optimization down, or in the worst case, makes it unsolvable in
reasonable time. So the cost expression built into the objective is a
**simplified, relaxed** version of the real tariff. It stays that way so
the optimization problem stays tractable, an LP or MILP a fast open source
solver can close in seconds, instead of a much harder nonconvex program.

That simplified expression is a proxy the optimizer minimizes to find a
good operating schedule. It was never meant to be read as a bill. The
relaxation usually drops or under counts the tiered surcharge, so the
objective's value typically sits **at or below** the true cost of the
schedule it produced. Reading it as the cost would be misleading.

## The report is the true cost of what actually happened

Once the solver picks a schedule, `report_cost` evaluates the **real**
tariff, the full, un relaxed cost function, tiers and all, against the
power values that schedule actually settles on. This is the number that
matches what a utility bill would show for that schedule. It's the only
number flex-pse presents as "the cost."

The raw solver objective never gets surfaced as the reported cost, and
there's no supported way to pull it out of a `CostReport`. You can only
reach it by reading `pyo.value(model.objective)` directly off the solved
model, and that's for someone debugging the optimization itself, not
reading a bill.

## Where the numbers come from

flex-pse doesn't do any tariff math of its own. Every dollar figure, whether
energy charges, demand charges, tiers or fixed fees, comes from
[EECO](https://pypi.org/project/eeco/). The module
`flexops.costing.opex` is the glue. It loads tariffs, hands EECO the
power series in the shape it expects, gives the results stable flex-pse
names, and turns EECO's errors into flex-pse exceptions. It's also the only
file that imports `eeco`, so when EECO's API changes there's one place to fix.

EECO gets called twice in a run:

- **While building the model.** {func}`~flexops.costing.add_operating_cost`
  asks EECO for the relaxed cost expression that goes into the objective.
  That's the proxy described above.
- **After the solve.** {func}`~flexops.costing.evaluate_cost` and
  {func}`~flexops.costing.evaluate_fuel_cost` run the full tariff on the
  power values the solve settled on. That's the number `report_cost` gives you.

## Tiered charges and `consumption_estimate`

Tiers are where the proxy and the bill most often disagree. EECO can price
the top tier of a charge exactly, as long as it has a single rate. Any other
tier needs to know roughly how much you'll consume over the horizon, because
which tier you land in depends on it. Without that number, EECO quietly
drops the tier from the objective.

So if your tariff has tiers, set `consumption_estimate`, either on
`CostingConfig` or on `FlexCosting`. It maps each utility (`"electric"` or
`"gas"`) to the total you expect to use over the horizon, in kWh or m³. If a
tariff has tiers for a utility and you leave out its estimate, flex-pse logs
a warning so the missing cost doesn't go unnoticed.

Even with an estimate, the tier is only priced approximately, and the proxy
can end up a little above or below the real bill. To see by how much, call
{meth}`~flexops.costing.flex_costing.FlexCostingData.relaxation_gap` with the
solved model and its solver results.

## Units, time and demand response

A few conventions hold everywhere in the costing code:

- **Units.** Electrical power is always in kW and fuel is always a volume
  flow in m³/hr, because that's how fuel is metered and billed. EECO turns
  them into energy using the timestep, so `dt_hours` gets passed in exactly
  once. Don't multiply by it yourself. flex-pse doesn't apply a heating value.
  If a tariff prices gas by energy, EECO does that conversion with its own
  assumption.
- **Time zones.** EECO reads charge windows off the local wall-clock month,
  weekday and hour, with no time zone handling. flex-pse matches that and
  uses naive local timestamps throughout. A time-zone-aware index is rejected
  with a `FlexDataError`.
- **Demand response.** For now, a DR file is only loaded and stored. It
  doesn't change the objective, because EECO doesn't have a DR API yet.
