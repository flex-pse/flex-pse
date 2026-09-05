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
